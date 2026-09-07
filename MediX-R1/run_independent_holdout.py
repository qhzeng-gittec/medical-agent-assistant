"""Frozen, paired holdout evaluation of the existing A-F adapters; no training."""
import argparse
import gc
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import diagnose_reasoning_effects as diagnostic
from data_io import load_jsonl, save_jsonl

LAB = Path(__file__).resolve().parent
ROOT = LAB / 'outputs/independent_holdout_20260906'
SOURCE = LAB / 'data/knowledge_experiments_v1'
ARMS = {'knowledge': ['A', 'B', 'C', 'D', 'D_half'], 'context': ['E', 'F']}
SEED = 20260906


def save(path, value):
    diagnostic.save(path, value)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT / 'plan.json').exists():
        return
    pools = {s: load_jsonl(SOURCE / f'{s}.jsonl') for s in ['train', 'validation', 'test']}
    rows = [dict(r, cohort='independent_project_test') for r in pools['test'] if r['task'] in ARMS]
    rows.sort(key=lambda r: (r['task'], r['source_id']))
    overlap = {}
    for field in ['source_id', 'source_group']:
        ids = {r[field] for r in rows}
        overlap[field] = {s: len(ids & {r[field] for r in pools[s]}) for s in ['train', 'validation']}
        assert not any(overlap[field].values()), overlap
    old_outputs = LAB / 'outputs/reasoning_effects_v1/diagnostics/questions'
    prior_ids = {p.stem for p in old_outputs.glob('*.json')}
    assert not prior_ids & {r['source_id'] for r in rows}
    assert Counter(r['task'] for r in rows) == {'knowledge': 150, 'context': 97}
    save_jsonl(ROOT / 'evaluation.jsonl', rows)
    checkpoints = {}
    for arm in 'ABCDEF':
        path = diagnostic.CHECKPOINTS / diagnostic.ARMS[arm] / 'final_adapter/adapter_model.safetensors'
        checkpoints[arm] = sha(path)
    save(ROOT / 'plan.json', dict(seed=SEED, frozen_before_generation=True,
         task_counts=dict(Counter(r['task'] for r in rows)), arms=ARMS, expected_outputs=944,
         contrasts={'knowledge': ['B-A', 'D-C', 'C-A', 'D-B', 'D_half-D'], 'context': ['F-E']},
         metrics=['answer_score', 'reasoning_score', 'joint_score'],
         primary_metric='answer_score; reasoning and joint scores reported separately',
         protocol=dict(enable_thinking=True, decoding='greedy', batch_size=4, total_budget=2048, reasoning_budget=1792),
         evaluation_sha256=sha(ROOT / 'evaluation.jsonl'), checkpoint_sha256=checkpoints,
         source_sha256={s: sha(SOURCE / f'{s}.jsonl') for s in pools}, overlap=overlap,
         knowledge_subjects=dict(Counter(r['subject'] for r in rows if r['task']=='knowledge')),
         context_labels=dict(Counter(r['reference'] for r in rows if r['task']=='context')),
         limitations=['Project holdout, not a new external benchmark; no claim of absence from base pretraining.',
                      'No overlap with original training/validation or prior full-reasoning diagnostic; other historical use not exhaustively excluded.',
                      'One trained seed; paired question intervals exclude training and judge randomness.',
                      'Data-added arms also differ in optimizer steps; FFN inference scaling is not lower-LR retraining.',
                      'No new VQA or case evaluation; cannot infer four-task overall rankings.']))
    print('Frozen: 150 knowledge + 97 context questions; 944 model responses.', flush=True)


def generate():
    import torch
    from peft import PeftModel
    from peft.tuners.lora.layer import LoraLayer
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
    from generate_reasoning_batch import generate_batch

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(.35)
    rows = load_jsonl(ROOT / 'evaluation.jsonl')
    plan = json.loads((ROOT / 'plan.json').read_text(encoding='utf-8'))
    assert sha(ROOT / 'evaluation.jsonl') == plan['evaluation_sha256']
    for task, arms in ARMS.items():
        subset = [r for r in rows if r['task'] == task]
        for arm in arms:
            parent = 'D' if arm == 'D_half' else arm
            adapter = diagnostic.CHECKPOINTS / diagnostic.ARMS[parent] / 'final_adapter'
            assert sha(adapter / 'adapter_model.safetensors') == plan['checkpoint_sha256'][parent]
            out = ROOT / 'generation' / arm
            batches = [subset[i:i+4] for i in range(0, len(subset), 4)]
            pending = [b for b in batches if not all((out/'samples'/f"{r['source_id']}.json").exists() for r in b)]
            if not pending:
                continue
            processor = AutoProcessor.from_pretrained(LAB / 'models/Qwen3.5-2B', do_resize=False)
            model = Qwen3_5ForConditionalGeneration.from_pretrained(
                LAB / 'models/Qwen3.5-2B', dtype=torch.bfloat16, attn_implementation='sdpa')
            model = PeftModel.from_pretrained(model, adapter)
            ffn_count = 0
            for name, module in model.named_modules():
                if isinstance(module, LoraLayer) and '.mlp.' in name:
                    ffn_count += 1
                    if arm == 'D_half':
                        module.scaling['default'] *= .5
            assert ffn_count == (0 if parent in 'AC' else 72)
            model.to('cuda').eval()
            save(out/'config.json', dict(parent=parent, ffn_scale=.5 if arm=='D_half' else 1,
                 ffn_modules=ffn_count, evaluation_sha256=plan['evaluation_sha256'],
                 checkpoint_sha256=plan['checkpoint_sha256'][parent], protocol=plan['protocol']))
            print(f'{arm}: {sum(map(len, pending))} pending', flush=True)
            for index, batch in enumerate(pending, 1):
                torch.manual_seed(SEED)
                responses = generate_batch(model, processor, batch)
                for row, response in zip(batch, responses, strict=True):
                    response.update(source_id=row['source_id'], task=task, batch_source_ids=[r['source_id'] for r in batch])
                    save(out/'samples'/f"{row['source_id']}.json", response)
                save(ROOT/'status.json', dict(stage='generation', arm=arm, batch=index, batches=len(pending)))
                print(f'{arm} batch {index}/{len(pending)} tokens={[r["generated_tokens"] for r in responses]}', flush=True)
            assert sha(adapter/'adapter_model.safetensors') == plan['checkpoint_sha256'][parent]
            save(out/'complete.json', dict(samples=len(subset), checkpoint_unchanged=True))
            del model, processor
            gc.collect()
            torch.cuda.empty_cache()
    save(ROOT/'generation_complete.json', dict(outputs=944))


def score():
    diagnostic.ROOT = ROOT
    rows = load_jsonl(ROOT/'evaluation.jsonl')
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = []
        for row in rows:
            responses = {arm: json.loads((ROOT/'generation'/arm/'samples'/f"{row['source_id']}.json").read_text(encoding='utf-8')) for arm in ARMS[row['task']]}
            futures.append(pool.submit(diagnostic.score_question, row, responses, 'diagnostics'))
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            print(f'Judged {index}/{len(rows)} {result["source_id"]}', flush=True)
            save(ROOT/'status.json', dict(stage='scoring', complete=index, total=len(rows)))


def summarize():
    rows = [json.loads(p.read_text(encoding='utf-8')) for p in sorted((ROOT/'diagnostics/questions').glob('*.json'))]
    assert len(rows) == 247
    rng = np.random.default_rng(SEED)
    groups, contrasts, conflicts = {}, {}, []
    for task, arms in ARMS.items():
        subset = [r for r in rows if r['task'] == task and r['reference_valid']]
        for row in subset:
            final_scores = {}
            for arm, model in row['models'].items():
                final_scores.setdefault(model['final_answer'], set()).add(model['diagnosis']['answer_score'])
            if any(len(v)>1 for v in final_scores.values()):
                conflicts.append(row['source_id'])
        for arm in arms:
            models = [r['models'][arm] for r in subset]
            groups[f'{task}/{arm}'] = dict(n=len(models),
                **{metric:50*float(np.mean([m['diagnosis'][metric] for m in models])) for metric in ['answer_score','reasoning_score','joint_score']},
                answer_correct=sum(m['diagnosis']['answer_score']==2 for m in models),
                forced_ends=sum(m['forced_reasoning_end'] for m in models),
                length_stops=sum(m['stop_reason']=='length' for m in models),
                mean_generated_tokens=float(np.mean([m['generated_tokens'] for m in models])))
        pairs = [('B','A'),('D','C'),('C','A'),('D','B'),('D_half','D')] if task=='knowledge' else [('F','E')]
        for hi, lo in pairs:
            metrics = {}
            for metric in ['answer_score','reasoning_score','joint_score']:
                delta = np.array([50*(r['models'][hi]['diagnosis'][metric]-r['models'][lo]['diagnosis'][metric]) for r in subset])
                boots = delta[rng.integers(0,len(delta),size=(20000,len(delta)))].mean(axis=1)
                metrics[metric] = dict(delta=float(delta.mean()), ci95=np.quantile(boots,[.025,.975]).tolist(),
                     improved=int((delta>0).sum()), worsened=int((delta<0).sum()), unchanged=int((delta==0).sum()))
            contrasts[f'{task}/{hi}-{lo}'] = metrics
    result = dict(groups=groups, contrasts=contrasts, invalid_questions=[r['source_id'] for r in rows if not r['reference_valid']],
                  same_final_score_conflicts=conflicts, nominal_intervals_not_multiple_comparison_adjusted=True)
    save(ROOT/'results.json', result)
    save(ROOT/'status.json', dict(stage='complete', questions=len(rows), outputs=944, requires_score_conflict_review=bool(conflicts)))
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare','generate','score','summarize','all'])
    args = parser.parse_args()
    if args.stage == 'all':
        for function in [prepare, generate, score, summarize]:
            function()
    else:
        globals()[args.stage]()
