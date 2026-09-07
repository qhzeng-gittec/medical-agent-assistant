"""Run and compare the medical-bridge pilot on 90 fresh validation questions."""
import argparse
import hashlib
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from common.io import load_jsonl, save_jsonl
from evaluation.llm_judge import judge_responses, summarize_judgments
from experiments.run_rationale_pilot import MODEL, diagnostic, generate_arm, score_arm
from common.io import save

LAB = Path(__file__).resolve().parents[1]
DATA = LAB / 'data/rationale_pilot_v2'
ROOT = LAB / 'outputs/rationale_pilot_v2'
NAMES = ('unchanged_C', 'original', 'medical_bridge')
FIELDS = ('final_answer_correct', 'reasoning_fact_error', 'reasoning_answer_contradiction')


def score_stream(name):
    rows = load_jsonl(DATA / 'validation.jsonl')
    positions = {r['source_id']: i for i, r in enumerate(rows)}
    out = ROOT / 'evaluation' / name
    jobs = []
    for task in ('knowledge', 'vqa'):
        group = [r for r in rows if r['task'] == task]
        for i in range(0, len(group), 4):
            jobs.append(('joint', group[i:i+4]))
        if task == 'knowledge':
            for i in range(0, len(group), 6):
                jobs.append(('diagnostic', group[i:i+6]))
    jobs.sort(key=lambda job: max(positions[r['source_id']] for r in job[1]))
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = []
        for kind, batch in jobs:
            paths = [out / 'samples' / (r['source_id'] + '.json') for r in batch]
            deadline = time.monotonic() + 900
            while not all(path.exists() for path in paths):
                if time.monotonic() > deadline:
                    raise TimeoutError(f'Generation did not produce the next batch for {name}: {paths}')
                time.sleep(2)
            responses = [json.loads(path.read_text(encoding='utf-8')) for path in paths]
            if kind == 'joint':
                future = pool.submit(judge_responses, batch, responses, out / 'judge_calls', MODEL)
            else:
                future = pool.submit(diagnostic, batch, responses, out, True)
            futures.append(future)
            print(f'{name} queued {kind} through sample {max(positions[r["source_id"]] for r in batch)+1}', flush=True)
        for i, future in enumerate(futures, 1):
            future.result()
            print(f'{name} scored batch {i}/{len(futures)}', flush=True)
    deadline = time.monotonic() + 900
    while not (out / 'generation_complete.json').exists():
        if time.monotonic() > deadline:
            raise TimeoutError(f'Generation completion marker missing for {name}')
        time.sleep(2)
    # All calls are now cached; assemble the standard comparable artifacts.
    score_arm(name, DATA, ROOT, strict_diagnostics=True)


def summarize():
    plan = json.loads((DATA / 'experiment_plan.json').read_text(encoding='utf-8'))
    source = {r['source_id']: r for r in load_jsonl(DATA / 'validation.jsonl')}
    primary_ids = set(plan['primary_ids'])
    historical_ids = set(plan['historical_ids'])
    assert primary_ids.isdisjoint(historical_ids)
    models, records, diagnostics = {}, {}, {}
    for name in NAMES:
        out = ROOT / 'evaluation' / name
        records[name] = {r['source_id']: r for r in load_jsonl(out / 'knowledge.jsonl')}
        diagnostics[name] = {r['id']: r for r in load_jsonl(out / 'knowledge_diagnostics.jsonl')}
        assert set(records[name]) == set(diagnostics[name]) == primary_ids | historical_ids
        raw = {r['source_id']: r for r in load_jsonl(out / 'knowledge_generations.jsonl')}
        assert {sid: {k: v for k, v in r.items() if k != 'judge'} for sid, r in records[name].items()} == raw
        groups = {}
        for label, ids in [('primary_90', primary_ids), ('historical_20', historical_ids)]:
            rs = [records[name][sid] for sid in sorted(ids)]
            groups[label] = summarize_judgments([r['judge'] for r in rs])
            groups[label]['mean_generated_tokens'] = statistics.mean(r['generated_tokens'] for r in rs)
            groups[label]['length_stops'] = sum(r['stop_reason'] == 'length' for r in rs)
            groups[label]['forced_reasoning_ends'] = sum(r['forced_reasoning_end'] for r in rs)
            groups[label]['empty_final_answers'] = sum(not r['final_answer'].strip() for r in rs)
            groups[label]['diagnostics'] = {}
            for field in FIELDS:
                values = [diagnostics[name][sid][field] for sid in ids if diagnostics[name][sid][field] is not None]
                groups[label]['diagnostics'][field] = dict(count=sum(values), assessable=len(values), rate=sum(values) / len(values) if values else None)
        groups['vqa'] = summarize_judgments([r['judge'] for r in load_jsonl(out / 'vqa.jsonl')])
        groups['primary_by_subject'] = {
            subject: summarize_judgments([r['judge'] for sid, r in records[name].items() if sid in primary_ids and source[sid]['subject'] == subject])
            for subject in sorted(plan['primary_subject_counts'])}
        models[name] = groups
    comparisons = {}
    for other in ('unchanged_C', 'original'):
        ids = sorted(sid for sid in primary_ids if records['medical_bridge'][sid]['judge']['score'] is not None and records[other][sid]['judge']['score'] is not None)
        assert ids
        differences = np.array([50 * (records['medical_bridge'][sid]['judge']['score'] - records[other][sid]['judge']['score']) for sid in ids])
        rng = np.random.default_rng(42)
        bootstrap_parts = []
        for subject in sorted(plan['primary_subject_counts']):
            subject_deltas = differences[[i for i, sid in enumerate(ids) if source[sid]['subject'] == subject]]
            assert len(subject_deltas), f'No paired observations in {subject}'
            bootstrap_parts.append(subject_deltas[rng.integers(0, len(subject_deltas), size=(10000, len(subject_deltas)))].mean(axis=1) * len(subject_deltas) / len(ids))
        boots = np.stack(bootstrap_parts).sum(axis=0)
        comparisons[other] = dict(paired_samples=len(ids), mean_score_delta=float(differences.mean()),
                                  wins=int((differences > 0).sum()), ties=int((differences == 0).sum()), losses=int((differences < 0).sum()),
                                  stratified_paired_bootstrap_95_ci=np.quantile(boots, [.025, .975]).tolist())
    save_jsonl(ROOT / 'sample_comparisons.jsonl', [dict(source_id=sid, primary=sid in primary_ids,
                question=source[sid]['user_text'], reference=source[sid]['reference'], subject=source[sid]['subject'],
                models={name: dict(records[name][sid], diagnostic=diagnostics[name][sid]) for name in NAMES})
                for sid in sorted(primary_ids | historical_ids)])
    baseline = Path(plan['initial_adapter']) / 'adapter_model.safetensors'
    assert hashlib.sha256(baseline.read_bytes()).hexdigest() == plan['initial_adapter_sha256']
    result = dict(plan=plan, models=models, primary_paired_comparisons=comparisons,
                  limitations='Single seed, LLM grading, validation development only. Bootstrap excludes training-seed and judge variability. Strict contradiction rubric is shared within v2; do not directly compare contradiction counts with v1.')
    save(ROOT / 'results.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('generate', 'score', 'score-stream', 'summarize'))
    parser.add_argument('--name', choices=NAMES)
    args = parser.parse_args()
    if args.phase != 'summarize' and args.name is None:
        parser.error('--name is required')
    if args.phase == 'generate':
        generate_arm(args.name, DATA, ROOT)
    elif args.phase == 'score':
        score_arm(args.name, DATA, ROOT, strict_diagnostics=True)
    elif args.phase == 'score-stream':
        score_stream(args.name)
    else:
        summarize()


if __name__ == '__main__':
    main()
