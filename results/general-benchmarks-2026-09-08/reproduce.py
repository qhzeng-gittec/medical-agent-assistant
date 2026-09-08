"""Verify published evidence and recompute paired metrics without model inference."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def lines(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def verify(bootstrap=False):
    manifest = read(ROOT/'publication_manifest.json')
    for name, digest in manifest['files'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
    questions = lines(ROOT/'data/evaluation.jsonl')
    assert len(questions) == len({q['id'] for q in questions}) == 1500
    expected = read(ROOT/'results.json')
    vectors = {}
    for arm in ('baseline', 'cpt_sft'):
        likelihoods = lines(ROOT/f'runs/{arm}/likelihood.jsonl')
        chats = lines(ROOT/f'runs/{arm}/chat.jsonl')
        assert len(likelihoods) == len(chats) == 1500
        by_score = {row['id']: row for row in likelihoods}
        by_chat = {row['id']: row for row in chats}
        assert by_score.keys() == by_chat.keys() == {q['id'] for q in questions}
        for name in ('mmlu', 'arc_challenge', 'hellaswag'):
            subset = [q for q in questions if q['benchmark'] == name]
            assert len(subset) == 500
            values = {k: [] for k in ('primary_correct', 'raw_correct', 'norm_correct',
                      'free_correct', 'ranked_correct', 'strict_format', 'stopped_on_eos')}
            for question in subset:
                score, chat = by_score[question['id']], by_chat[question['id']]
                assert score['gold'] == chat['gold'] == question['gold']
                raw = max(range(len(score['sum_logprobs'])), key=score['sum_logprobs'].__getitem__)
                normalized = [s/len(c) for s,c in zip(score['sum_logprobs'],question['candidates'],strict=True)]
                assert all(math.isclose(a,b,abs_tol=1e-9) for a,b in zip(normalized,score['char_normalized_logprobs'],strict=True))
                norm = max(range(len(normalized)),key=normalized.__getitem__)
                assert raw == score['selected'] and norm == score['selected_norm']
                assert score['raw_correct'] == (raw == question['gold'])
                assert score['norm_correct'] == (norm == question['gold'])
                assert score['primary_correct'] == ((raw if name == 'mmlu' else norm) == question['gold'])
                ranked = max(range(len(chat['letter_logits'])),key=chat['letter_logits'].__getitem__)
                assert chat['ranked_correct'] == (ranked == question['gold'])
                assert chat['free_correct'] == (chat['letter'] == 'ABCDE'[question['gold']])
                for key in values:
                    values[key].append(int((score if key in score else chat)[key]))
            for key, vector in values.items():
                assert sum(vector) == expected['benchmarks'][name][arm]['counts'][key]
                assert math.isclose(100*sum(vector)/500,expected['benchmarks'][name][arm]['percentages'][key],abs_tol=1e-9)
            vectors[name,arm] = values
    pvalues = {}
    for name in ('mmlu','arc_challenge','hellaswag'):
        subset = [q for q in questions if q['benchmark'] == name]
        for key in ('primary_correct','raw_correct','norm_correct','free_correct','ranked_correct'):
            differences = [b-a for a,b in zip(vectors[name,'baseline'][key],vectors[name,'cpt_sft'][key],strict=True)]
            fixed, lost = differences.count(1), differences.count(-1)
            comparison = expected['comparisons'][name][key]
            assert (fixed,lost) == (comparison['repaired'],comparison['regressed'])
            assert math.isclose(sum(differences)/5,comparison['delta_pp'],abs_tol=1e-9)
            assert [q['id'] for q,d in zip(subset,differences,strict=True) if d == 1] == comparison['repaired_ids']
            assert [q['id'] for q,d in zip(subset,differences,strict=True) if d == -1] == comparison['regressed_ids']
            n = fixed+lost
            p = min(1.,2*sum(math.comb(n,i) for i in range(min(fixed,lost)+1))/2**n) if n else 1.
            assert math.isclose(p,comparison['mcnemar_exact_p'],rel_tol=1e-9,abs_tol=1e-12)
            if key == 'primary_correct':
                pvalues[name] = p
            if bootstrap:
                import numpy as np
                samples = np.random.default_rng(20260908).choice(differences,size=(10000,500),replace=True).mean(axis=1)*100
                assert np.allclose(np.quantile(samples,[.025,.975]),comparison['paired_bootstrap95_pp'])
    previous = 0.
    for index,name in enumerate(sorted(pvalues,key=pvalues.__getitem__)):
        previous = max(previous,min(1.,(3-index)*pvalues[name]))
        assert math.isclose(previous,expected['comparisons'][name]['primary_correct']['holm_adjusted_p'],rel_tol=1e-9)
    with (ROOT/'paired_predictions.csv').open(encoding='utf-8-sig',newline='') as handle:
        paired = list(csv.DictReader(handle))
    assert len(paired) == 1500 and {r['id'] for r in paired} == {r['id'] for r in questions}
    for name in ('mmlu','arc_challenge','hellaswag'):
        for arm in ('baseline','cpt_sft'):
            assert sum(row[f'{arm}_primary_correct'] == 'True' for row in paired if row['benchmark'] == name) == expected['benchmarks'][name][arm]['counts']['primary_correct']
    print(json.dumps(dict(verified=True,independent_questions=1500,models=2,likelihood_records=3000,
        chat_records=3000,paired_csv_rows=len(paired),bootstrap_recomputed=bootstrap,
        primary_scores={name:{arm:expected['benchmarks'][name][arm]['percentages']['primary_correct'] for arm in ('baseline','cpt_sft')} for name in pvalues}),indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bootstrap',action='store_true')
    verify(parser.parse_args().bootstrap)
