import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from filelock import FileLock
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from evaluation.evaluate_native_reasoning import generate, qwen_sampling_parameters
from evaluation.llm_judge import judge_responses, summarize_judgments

LAB = Path(__file__).resolve().parents[1]
ROOT = LAB / 'outputs/base_calibration_v1'
OLD = LAB / 'outputs/knowledge_experiments_v1/validation_screen/base/validation'
TASKS = ('vqa', 'knowledge', 'case', 'context')
PROFILES = {
    'base_nonthinking_32k': dict(enable_thinking=False, budget=32768, enforce_thinking_budget=False),
    'base_thinking_sample_2k': dict(enable_thinking=True, budget=2048, enforce_thinking_budget=True),
    'base_thinking_sample_32k': dict(enable_thinking=True, budget=32768, enforce_thinking_budget=False),
    'base_thinking_sample_bounded_6k': dict(enable_thinking=True, budget=6144, enforce_thinking_budget=False, stop_repetition=True),
}


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profiles', nargs='+', choices=tuple(PROFILES), default=['base_thinking_sample_bounded_6k'])
    parser.add_argument('--limit-per-task', type=int, default=20)
    parser.add_argument('--output-root', type=Path, default=LAB / 'outputs/base_calibration_bounded_v2')
    args = parser.parse_args()
    if not 1 <= args.limit_per_task <= 20:
        parser.error('limit-per-task must be between 1 and 20')
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(root / 'calibration.lock', timeout=0):
        run(root, args)


def run(root, args):
    data_path = LAB / 'data/knowledge_experiments_v1/validation.jsonl'
    sources = {(r['task'], r['source_id']): r for r in read_rows(data_path)}
    selected = {task: read_rows(OLD / f'{task}.jsonl')[:args.limit_per_task] for task in TASKS}
    manifest = json.loads((data_path.parent / 'manifest.json').read_text(encoding='utf-8'))
    config = dict(data_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
        profile_definitions={name:PROFILES[name] for name in args.profiles}, selected_ids={t:[r['source_id'] for r in rows] for t,rows in selected.items()},
        generation_seed_policy='reuse each original sample_seed', image_max_edge=768, gpu_memory_fraction=.8,
        judge_model='gpt-5.5', source_url='https://huggingface.co/Qwen/Qwen3.5-2B',
        sampling_profiles={str((thinking,image)):qwen_sampling_parameters(thinking,image) for thinking in (True,False) for image in (True,False)})
    if any(PROFILES[name].get('stop_repetition') for name in args.profiles):
        config['repetition_stop'] = dict(min_tokens=512, check_every=64, window_tokens=2048,
            normalized_line_min_chars=50, occurrences=6, hard_cap_label='overlong_suspected_repetition')
    config_path = root / 'run_config.json'
    if config_path.exists() and json.loads(config_path.read_text(encoding='utf-8')) != config:
        raise ValueError('Resume configuration differs from existing results')
    save(config_path, config)
    state = dict(status='running', pid=os.getpid(), started_at=datetime.now(timezone.utc).isoformat(), completed=[])
    save(root / 'status.json', state)
    torch.cuda.set_per_process_memory_fraction(.8)
    processor = AutoProcessor.from_pretrained(LAB / 'models/Qwen3.5-2B', do_resize=False)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(LAB / 'models/Qwen3.5-2B', dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda').eval()
    for name in args.profiles:
        out = root / name / 'validation'
        out.mkdir(parents=True, exist_ok=True)
        if (out / 'summary.json').exists():
            json.loads((out / 'summary.json').read_text(encoding='utf-8'))
            state['completed'].append(name)
            continue
        profile = PROFILES[name]
        summary = dict(name=name, split='validation', data_sha256=config['data_sha256'], adapter_dir=None,
            settings=profile, sampling_profile='qwen35', judge_model=config['judge_model'], tasks=list(TASKS),
            score_formula='task = 50 * mean judge score; overall = equal-weight mean tasks', seed=42)
        started = time.monotonic()
        for task in TASKS:
            raw_dir = out / 'samples' / task
            raw_dir.mkdir(parents=True, exist_ok=True)
            results = []
            task_rows = [sources[(task, old['source_id'])] for old in selected[task]]
            for index, (row, old) in enumerate(zip(task_rows, selected[task], strict=True), 1):
                state.update(profile=name, task=task, sample_index=index, phase='generating', updated_at=datetime.now(timezone.utc).isoformat())
                save(root / 'status.json', state)
                path = raw_dir / (row['source_id'] + '.json')
                if path.exists():
                    result = json.loads(path.read_text(encoding='utf-8'))
                else:
                    torch.manual_seed(old['sample_seed'])
                    begin = time.monotonic()
                    result = generate(model, processor, row, manifest['system_prompt'], profile['budget'],
                        decoding='sample', image_max_edge=768, sampling_profile='qwen35',
                        enable_thinking=profile['enable_thinking'], enforce_thinking_budget=profile['enforce_thinking_budget'],
                        stop_repetition=profile.get('stop_repetition', False))
                    result.update(source_id=row['source_id'], reference=row['reference'], sample_seed=old['sample_seed'],
                        generation_seconds=time.monotonic()-begin)
                    save(path, result)
                results.append(result)
                print(f'{name} {task} {index}/{len(task_rows)} tokens={result["generated_tokens"]} stop={result["stop_reason"]} seconds={result["generation_seconds"]:.1f}', flush=True)
            (out / f'{task}_generations.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in results),encoding='utf-8')
            state.update(phase='judging',updated_at=datetime.now(timezone.utc).isoformat())
            save(root / 'status.json', state)
            for start in range(0, len(results), 4):
                judgments = judge_responses(task_rows[start:start+4],results[start:start+4],out/'judge_calls',config['judge_model'])
                for result, judgment in zip(results[start:start+4], judgments, strict=True):
                    result['judge'] = judgment
            (out / f'{task}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in results),encoding='utf-8')
            summary[task] = summarize_judgments([r['judge'] for r in results])
        summary['overall_score_0_to_100'] = sum(summary[t]['score_0_to_100'] for t in TASKS)/4
        summary['current_process_seconds'] = time.monotonic()-started
        save(out/'summary.json',summary)
        state['completed'].append(name)
        save(root/'status.json',state)
        print(json.dumps(summary),flush=True)
    state.update(status='complete',phase='complete',finished_at=datetime.now(timezone.utc).isoformat())
    save(root/'status.json',state)


if __name__ == '__main__':
    main()
