"""Descriptive output-style diagnostics; no new training or test-score changes."""
import json
import random
import statistics
from collections import Counter
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
ROOT = LAB / 'outputs/independent_holdout_20260906'
OUT = ROOT / 'output_style'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def lengths(values):
    return dict(mean=statistics.mean(values), median=statistics.median(values),
                min=min(values), max=max(values))


def main():
    questions = {p.stem: read(p) for p in (ROOT / 'diagnostics/questions').glob('*.json')}
    with (ROOT / 'evaluation.jsonl').open(encoding='utf-8') as stream:
        sources = {r['source_id']: r for r in map(json.loads, stream)}
    loss = {arm: {r['source_id']: r for r in read(ROOT / 'reference_loss_segments' / f'{arm}.json') if r['cohort'] == 'test'}
            for arm in ['A', 'B', 'C', 'D', 'D_half', 'E', 'F']}
    groups, pairs = {}, {}
    for task, arms in [('knowledge', ['A', 'B', 'C', 'D', 'D_half']), ('context', ['E', 'F'])]:
        subset = [q for q in questions.values() if q['task'] == task]
        for arm in arms:
            rows = [q['models'][arm] for q in subset]
            groups[f'{task}/{arm}'] = dict(
                n=len(rows), reasoning_words=lengths([len(r['reasoning'].split()) for r in rows]),
                answer_words=lengths([len(r['final_answer'].split()) for r in rows]),
                answer_at_most_5_words=sum(len(r['final_answer'].split()) <= 5 for r in rows),
                unique_normalized_answers=len({r['final_answer'].strip().lower() for r in rows}),
                repeated_answers=Counter(r['final_answer'].strip().lower() for r in rows).most_common(5),
                flags=dict(Counter(flag for r in rows for flag in r['diagnosis']['flags'])),
                by_answer_score={str(score): dict(
                    n=sum(r['diagnosis']['answer_score'] == score for r in rows),
                    median_answer_words=statistics.median([len(r['final_answer'].split()) for r in rows if r['diagnosis']['answer_score'] == score]),
                    median_reasoning_words=statistics.median([len(r['reasoning'].split()) for r in rows if r['diagnosis']['answer_score'] == score]))
                    for score in [0, 1, 2] if any(r['diagnosis']['answer_score'] == score for r in rows)})
        groups[f'{task}/reference'] = dict(
            reasoning_words=lengths([len(sources[q['source_id']]['reasoning_content'].split()) for q in subset]),
            answer_words=lengths([len(sources[q['source_id']]['target'].split()) for q in subset]))
    excerpts = []
    for task, candidate, baseline in [('knowledge','B','A'), ('knowledge','D','C'), ('knowledge','C','A'), ('knowledge','D','B'), ('context','F','E')]:
        subset = [q for q in questions.values() if q['task'] == task]
        changes = []
        for q in subset:
            left, right = q['models'][baseline], q['models'][candidate]
            changes.append(dict(source_id=q['source_id'], answer_delta=right['diagnosis']['answer_score']-left['diagnosis']['answer_score'],
                answer_length_delta=len(right['final_answer'].split())-len(left['final_answer'].split()),
                reasoning_length_delta=len(right['reasoning'].split())-len(left['reasoning'].split()),
                same_answer=right['final_answer'].strip().lower()==left['final_answer'].strip().lower()))
        pairs[f'{task}/{candidate}-{baseline}'] = {
            label: dict(n=len(selected), shorter_answers=sum(r['answer_length_delta']<0 for r in selected),
                        shorter_reasoning=sum(r['reasoning_length_delta']<0 for r in selected),
                        unchanged_answer_text=sum(r['same_answer'] for r in selected))
            for label, selected in [('improved',[r for r in changes if r['answer_delta']>0]),
                                    ('worsened',[r for r in changes if r['answer_delta']<0]),
                                    ('same_score',[r for r in changes if r['answer_delta']==0])]}
        if candidate not in ('C','D') or baseline not in ('A','B'):
            continue
        for sign, label in [(-1,'worsened'),(1,'improved')]:
            pool = sorted([q for q in subset if sign*(q['models'][candidate]['diagnosis']['answer_score']-q['models'][baseline]['diagnosis']['answer_score'])>0],key=lambda q:q['source_id'])
            for q in random.Random(20260906).sample(pool, min(4,len(pool))):
                excerpts.append(dict(contrast=f'{candidate}-{baseline}',direction=label,source_id=q['source_id'],
                    question=q['question'],target=sources[q['source_id']]['target'],gold_reasoning=sources[q['source_id']]['reasoning_content'],
                    models={arm:dict(answer=q['models'][arm]['final_answer'],reasoning=q['models'][arm]['reasoning'],
                                    score=q['models'][arm]['diagnosis']['answer_score'],judge=q['models'][arm]['diagnosis']['explanation'],
                                    gold_answer_nll=loss[arm][q['source_id']]['answer_nll_mean']) for arm in [baseline,candidate]}))
    low_loss_errors = {}
    for arm in ['C','D']:
        wrong = [q for q in questions.values() if q['task']=='knowledge' and q['models'][arm]['diagnosis']['answer_score']==0]
        low_loss_errors[arm] = dict(wrong_n=len(wrong),
            gold_answer_nll_below_half=sum(loss[arm][q['source_id']]['answer_nll_mean']<.5 for q in wrong),
            examples=[dict(source_id=q['source_id'],question=q['question'],target=sources[q['source_id']]['target'],
                           gold_reasoning=sources[q['source_id']]['reasoning_content'],
                           answer=q['models'][arm]['final_answer'],reasoning=q['models'][arm]['reasoning'],
                           gold_answer_nll=loss[arm][q['source_id']]['answer_nll_mean'])
                      for q in sorted(wrong,key=lambda q:loss[arm][q['source_id']]['answer_nll_mean'])[:4]])
    OUT.mkdir(exist_ok=True)
    result = dict(groups=groups,pairs=pairs,low_loss_errors=low_loss_errors,
                  limitations=['Whitespace word counts, not token counts.',
                               'Length association is descriptive, not a randomized causal intervention.',
                               'Answer NLL conditions on gold reasoning and prior gold answer tokens.',
                               'Error groups use existing model judge labels; no new expert certification.'])
    (OUT / 'statistics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT / 'paired_excerpts.json').write_text(json.dumps(excerpts,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
