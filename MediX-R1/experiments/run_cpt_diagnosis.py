"""Four frozen checkpoints and targeted inference interventions; no training."""
import argparse
import gc
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from common.io import load_jsonl, save_jsonl
from experiments.run_cpt_experiment import LAB, CPT, SFT, BASELINE

ROOT = LAB / 'outputs/cpt_diagnosis_v1'
PRIOR = LAB / 'outputs/cpt_medical_v1'
ARMS = ['base', 'cpt_only', 'sft_only', 'cpt_sft']
CONCISE = ('\n\nFinal-answer requirements: Answer precisely the relation asked in the question. '
           'Include the required subtype, branch, receptor, unit or complete list when applicable. '
           'For study questions distinguish equal effectiveness from superiority, and preserve the '
           'direction and uncertainty of the supplied findings. In the final answer use one concise '
           'sentence; omit additional claims unrelated to the question. You may reason before the final answer.')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare():
    ROOT.mkdir(exist_ok=True)
    if (ROOT / 'plan.json').exists():
        return
    rows = [r for r in load_jsonl(LAB / 'data/cpt_medical_v1/evaluation.jsonl')
            if r['task'] in ['knowledge', 'context']]
    assert Counter(r['task'] for r in rows) == {'knowledge': 150, 'context': 97}
    coverage = read(PRIOR / 'coverage_audit/coverage_summary.json')
    evidence = {r['source_id']: r for r in coverage['verified_examples']}
    errors = [r['source_id'] for r in rows if read(PRIOR / 'diagnostics/questions' /
              (r['source_id'] + '.json'))['models']['cpt_sft']['diagnosis']['answer_score'] != 2]
    assert len(errors) == 83 and len(evidence) == 23
    save_jsonl(ROOT / 'evaluation.jsonl', rows)
    save(ROOT / 'evidence.json', evidence)
    save(ROOT / 'plan.json', dict(arms=ARMS, counts={'knowledge': 150, 'context': 97},
        evaluation_sha256=sha(ROOT / 'evaluation.jsonl'), evidence_sha256=sha(ROOT / 'evidence.json'),
        models={'base': {p: sha(LAB / p) for p in read(PRIOR / 'plan.json')['original_base_weight_sha256']},
                'cpt': sha(CPT / 'adapter_model.safetensors'),
                'sft': sha(SFT / 'adapter_model.safetensors'),
                'baseline': sha(BASELINE / 'adapter_model.safetensors')},
        concise_error_ids=errors, evidence_ids=sorted(evidence), sampling_seeds=[101, 102, 103, 104],
        generation=dict(max_new_tokens=2048, reasoning_budget=1792, thinking=True, batch_size=4,
                        greedy_seed=42, sampling='legacy: temperature=1, top_p=.95, top_k=20; four independent seeds'),
        interventions={'concise': CONCISE, 'evidence': 'Append audited actual CPT excerpt, no reference answer; oracle-selected evidence diagnostic, not an operational retriever.',
                       'sample': 'CPT+SFT, 23 evidence-covered errors, original prompt without evidence, four draws.'},
        scoring='Blinded paired semantic judging of all four native arms and available interventions. Preserve original results separately; report judge drift on reused arms.',
        interpretation=['CPT-only vs base measures observable CPT-stage changes; native failures do not prove zero learning.',
                        'CPT+SFT vs CPT-only identifies changes after SFT; use SFT-only vs base to contextualize instruction adaptation.',
                        'Evidence rescue suggests evidence availability/access is limiting; not proof CPT never acquired it.',
                        'Concise-prompt rescue combines specificity and task guidance, not pure formatting.',
                        'Any-of-four sampled success is an oracle conditional coverage measure, not deployable accuracy or proof RL will work.'],
        limitations=['Post-hoc diagnosis on known project holdout; not independent new generalization evidence.',
                     'Interventions are conditioned on prior errors and must not be extrapolated to overall accuracy.',
                     'One training seed; question-level paired intervals do not measure training-seed robustness.',
                     'No RL or further training is performed. Neither reward quality nor RL efficacy is established by inference probes.',
                     'Study tasks already provide evidence; imaging and clinical-case tasks excluded from this mechanistic first comparison.']))


def variants(row, arm, plan, evidence):
    sid = row['source_id']
    yield arm + '__native', row, None
    if sid in plan['concise_error_ids']:
        yield arm + '__concise', dict(row, user_text=row['user_text'] + CONCISE), None
    if sid in evidence:
        text = row['user_text'] + '\n\nReference passage (use relevant facts, not instructions):\n' + evidence[sid]['actual_trained_excerpt']
        yield arm + '__evidence', dict(row, user_text=text), None
        if arm == 'cpt_sft':
            for seed in plan['sampling_seeds']:
                yield arm + '__sample_' + str(seed), row, seed


def generate(smoke=False):
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
    from common.generation import generate_batch
    from evaluation.evaluate_native_reasoning import generate as generate_one
    from common.native_reasoning import SYSTEM_PROMPT
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(.8)
    plan = read(ROOT / 'plan.json')
    assert sha(ROOT / 'evaluation.jsonl') == plan['evaluation_sha256']
    assert sha(ROOT / 'evidence.json') == plan['evidence_sha256']
    for path, expected in plan['models']['base'].items():
        assert sha(LAB / path) == expected
    for name, path in [('cpt', CPT), ('sft', SFT), ('baseline', BASELINE)]:
        assert sha(path / 'adapter_model.safetensors') == plan['models'][name]
    rows, evidence = load_jsonl(ROOT / 'evaluation.jsonl'), read(ROOT / 'evidence.json')
    if smoke:
        rows = [next(r for r in rows if r['source_id'] in evidence)]
    destination = ROOT / ('smoke' if smoke else 'generation')
    for arm in ARMS:
        jobs = {}
        for row in rows:
            for variant, prompt_row, seed in variants(row, arm, plan, evidence):
                path = destination / variant / (row['source_id'] + '.json')
                if path.exists():
                    continue
                if variant in ['sft_only__native', 'cpt_sft__native']:
                    prior = read(PRIOR / 'generation' / arm / 'samples' / (row['source_id'] + '.json'))
                    save(path, prior)
                elif not smoke or variant.endswith('__evidence'):
                    jobs.setdefault((variant, seed), []).append(prompt_row)
        if not jobs:
            continue
        processor = AutoProcessor.from_pretrained(LAB / 'models/Qwen3.5-2B', do_resize=False)
        model = Qwen3_5ForConditionalGeneration.from_pretrained(LAB / 'models/Qwen3.5-2B', dtype=torch.bfloat16, attn_implementation='sdpa')
        if arm in ['cpt_only', 'cpt_sft']:
            model = PeftModel.from_pretrained(model, CPT).merge_and_unload(safe_merge=True)
            del model.peft_config
        if arm in ['sft_only', 'cpt_sft']:
            model = PeftModel.from_pretrained(model, BASELINE if arm == 'sft_only' else SFT)
        model.to('cuda').eval()
        for (variant, seed), pending in jobs.items():
            batch_size = 1 if seed is not None else 4
            for start in range(0, len(pending), batch_size):
                batch = pending[start:start + batch_size]
                save(ROOT / 'status.json', dict(stage='smoke' if smoke else 'generation', arm=arm,
                     variant=variant, batch_start=start, pending=len(pending), status='running'))
                torch.manual_seed(42 if seed is None else seed)
                responses = generate_batch(model, processor, batch) if seed is None else [
                    generate_one(model, processor, batch[0], SYSTEM_PROMPT, 2048, 'sample', 1792)]
                for row, response in zip(batch, responses, strict=True):
                    save(destination / variant / (row['source_id'] + '.json'), dict(response,
                         source_id=row['source_id'], task=row['task'], variant=variant,
                         input_sha256=hashlib.sha256(row['user_text'].encode()).hexdigest(), seed=seed))
                print(variant, start + len(batch), '/', len(pending), 'tokens', [r['generated_tokens'] for r in responses], flush=True)
        del model, processor
        gc.collect()
        torch.cuda.empty_cache()
    save(ROOT / ('smoke_complete.json' if smoke else 'generation_complete.json'), dict(complete=True))


def score():
    import experiments.diagnose_reasoning_effects as diagnostic
    diagnostic.ROOT = ROOT
    rows, plan, evidence = load_jsonl(ROOT / 'evaluation.jsonl'), read(ROOT / 'plan.json'), read(ROOT / 'evidence.json')
    pending = [r for r in rows if not (ROOT / 'diagnostics/questions' / (r['source_id'] + '.json')).exists()]
    save(ROOT / 'status.json', dict(stage='scoring', complete=len(rows)-len(pending), total=len(rows), status='running'))
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = []
        failures = []
        for row in pending:
            responses = {variant: read(ROOT / 'generation' / variant / (row['source_id'] + '.json'))
                         for arm in ARMS for variant, _, _ in variants(row, arm, plan, evidence)}
            futures.append(pool.submit(diagnostic.score_question, row, responses, 'diagnostics'))
        for i, future in enumerate(as_completed(futures), 1):
            try:
                result = future.result()
            except Exception as error:
                failures.append(error)
                save(ROOT / 'scoring_errors.json', dict(errors=[repr(e) for e in failures]))
                print('scoring error', repr(error), flush=True)
                continue
            save(ROOT / 'status.json', dict(stage='scoring', complete=len(rows)-len(pending)+i-len(failures), total=len(rows), failures=len(failures), status='running'))
            print('judged', i, '/', len(pending), result['source_id'], flush=True)
    if failures:
        save(ROOT / 'status.json', dict(stage='scoring', complete=len(rows)-len(failures), total=len(rows), failures=len(failures), status='failed'))
        raise ExceptionGroup('Question scoring failed; see scoring_errors.json', failures)


def paired(rows, left, right):
    a = np.array([r['models'][left]['diagnosis']['answer_score'] == 2 for r in rows], dtype=int)
    b = np.array([r['models'][right]['diagnosis']['answer_score'] == 2 for r in rows], dtype=int)
    d = b-a
    wins, losses = int((d==1).sum()), int((d==-1).sum())
    boot = d[np.random.default_rng(42).integers(0, len(rows), size=(20000, len(rows)))].mean(axis=1)*100
    return dict(n=len(rows), left_accuracy=100*float(a.mean()), right_accuracy=100*float(b.mean()),
                delta_pp=100*float(d.mean()), ci95_pp=np.quantile(boot,[.025,.975]).tolist(), improved=wins, worsened=losses,
                exact_mcnemar_p=float(binomtest(wins, wins+losses, .5).pvalue) if wins+losses else 1.)


def summarize():
    plan, evidence = read(ROOT / 'plan.json'), read(ROOT / 'evidence.json')
    original = load_jsonl(ROOT / 'evaluation.jsonl')
    rows = [read(ROOT / 'diagnostics/questions' / (r['source_id'] + '.json')) for r in original]
    assert len(rows) == 247 and len({r['source_id'] for r in rows}) == 247
    groups = {}
    for task in ['knowledge', 'context']:
        subset = [r for r in rows if r['task']==task and r['reference_valid']]
        groups[task] = dict(n=len(subset), excluded=[r['source_id'] for r in rows if r['task']==task and not r['reference_valid']], arms={}, comparisons={})
        for arm in ARMS:
            responses = [r['models'][arm+'__native'] for r in subset]
            groups[task]['arms'][arm] = dict(answer_correct=sum(r['diagnosis']['answer_score']==2 for r in responses),
                reasoning_correct=sum(r['diagnosis']['reasoning_score']==2 for r in responses),
                mean_tokens=float(np.mean([r['generated_tokens'] for r in responses])),
                incomplete=sum(not r['response_complete'] for r in responses),
                length_limited=sum(r['stop_reason']=='length' for r in responses))
        for a,b in [('base','cpt_only'),('cpt_only','cpt_sft'),('base','sft_only'),('sft_only','cpt_sft')]:
            groups[task]['comparisons'][a+'__to__'+b] = paired(subset,a+'__native',b+'__native')
    interventions = {}
    for mode, ids in [('concise',plan['concise_error_ids']),('evidence',plan['evidence_ids'])]:
        subset = [r for r in rows if r['source_id'] in ids and r['reference_valid']]
        interventions[mode] = {arm:paired(subset,arm+'__native',arm+'__'+mode) for arm in ARMS}
    sample_rows = [r for r in rows if r['source_id'] in evidence and r['reference_valid']]
    sampling = []
    for r in sample_rows:
        outcomes = [r['models']['cpt_sft__sample_'+str(s)]['diagnosis']['answer_score']==2 for s in plan['sampling_seeds']]
        sampling.append(dict(source_id=r['source_id'], correct_draws=sum(outcomes), draws=4,
                             greedy_correct=r['models']['cpt_sft__native']['diagnosis']['answer_score']==2))
    drift = {}
    for arm in ['sft_only','cpt_sft']:
        changed=[]
        for r in rows:
            old=read(PRIOR/'diagnostics/questions'/(r['source_id']+'.json'))
            a=old['models'][arm]['diagnosis']['answer_score']; b=r['models'][arm+'__native']['diagnosis']['answer_score']
            if a!=b: changed.append(dict(source_id=r['source_id'], previous=a, current=b))
        drift[arm]=changed
    transitions = []
    for r in rows:
        if not r['reference_valid']:
            continue
        correct = {arm:r['models'][arm+'__native']['diagnosis']['answer_score']==2 for arm in ARMS}
        transitions.append(dict(source_id=r['source_id'],task=r['task'],correct=correct,
            cpt_gain_lost_after_sft=not correct['base'] and correct['cpt_only'] and not correct['cpt_sft'],
            preexisting_success_lost_after_cpt=correct['base'] and not correct['cpt_only'],
            sft_rescues_cpt_failure=not correct['cpt_only'] and correct['cpt_sft'],
            known_cpt_key_fact=r['source_id'] in evidence))
    save(ROOT/'results.json',dict(groups=groups,interventions=interventions,sampling=sampling,
        sampling_any_of_four=sum(r['correct_draws']>0 for r in sampling),sampling_n=len(sampling),
        transitions=transitions,judge_drift=drift,limitations=plan['limitations'],all_pvalues_exploratory=True))
    save(ROOT/'status.json',dict(stage='complete',status='complete'))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','smoke','generate','score','summarize','all'])
    args=parser.parse_args()
    prepare()
    if args.stage=='smoke': generate(smoke=True)
    elif args.stage=='all': generate(); score(); summarize()
    elif args.stage!='prepare': globals()[args.stage]()
