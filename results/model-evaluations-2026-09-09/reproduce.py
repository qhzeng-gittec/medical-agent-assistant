"""Verify published evidence and recompute counts without calling a model."""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def lines(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def main():
    manifest = read(ROOT / 'manifest.json')
    for name, expected in manifest['files'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    questions = {q['source_id']: q for q in lines(ROOT / 'medical600/questions.jsonl')}
    assert len(questions) == 600
    records = lines(ROOT / 'medical600/predictions.jsonl')
    summary = read(ROOT / 'medical600/summary.json')
    arms = {}
    for arm in summary['completed_arms']:
        rows = [row for row in records if row['arm'] == arm]
        by_id = {row['source_id']: row for row in rows}
        assert len(rows) == len(by_id) == 600 and by_id.keys() == questions.keys()
        arms[arm] = by_id
        for row in rows:
            assert row['reference_label'] == questions[row['source_id']]['answer_label']
            assert row['ranked_letter'] == max(row['letter_logits'], key=row['letter_logits'].get)
            strict = re.fullmatch(r'\s*([ABCD])\s*[.)]?\s*', row['prediction'])
            explicit = re.match(r'^\s*(?i:(?:final\s+)?answer)\s*:\s*\*{0,2}([ABCDabcd])\*{0,2}(?:[.)]|\s*$)', row['prediction'])
            if explicit is None:
                explicit = re.match(r'^\s*\*{0,2}([ABCD])\*{0,2}(?:[.)]|\s*$)', row['prediction'])
            assert row['strict_letter'] == (strict.group(1) if strict else None)
            assert row['explicit_letter'] == (explicit.group(1).upper() if explicit else None)
            for metric, predicted in [('ranked_correct', 'ranked_letter'), ('free_correct', 'explicit_letter'), ('strict_correct', 'strict_letter')]:
                assert row[metric] == (row[predicted] == row['reference_label'])
        for field in ['ranked_correct', 'free_correct', 'strict_correct']:
            assert sum(row[field] for row in rows) == summary['summaries'][arm][field]
        for subject, group in summary['summaries'][arm]['subjects'].items():
            subset = [row for row in rows if row['subject'] == subject]
            assert len(subset) == group['n']
            for metric in ['ranked_correct', 'free_correct', 'strict_correct']:
                assert sum(row[metric] for row in subset) == group[metric]
    for contrast in summary['comparisons']:
        before, after, metric = arms[contrast['before']], arms[contrast['after']], contrast['metric']
        repaired = sum(not before[sid][metric] and after[sid][metric] for sid in questions)
        regressed = sum(before[sid][metric] and not after[sid][metric] for sid in questions)
        assert repaired == contrast['repaired'] and regressed == contrast['regressed']
        assert abs((repaired - regressed) / 6 - contrast['delta_pp']) < 1e-10
    comparison = read(REPO / 'MediX-R1/reports/gspo_comparison.json')
    scores = {}
    for arm in ['sft', 'rl']:
        rows = lines(ROOT / f'vqa-gspo/{arm}_scored.jsonl')
        assert len(rows) == len({r['source_id'] for r in rows}) == 93
        assert len({r['source_group'] for r in rows}) == 16
        scores[arm] = {r['source_id']: r['judge']['score'] for r in rows}
        assert all(r['judge']['judge_model'] == 'gpt-5.5' for r in rows)
        assert abs(50 * sum(scores[arm].values()) / 93 - comparison[arm]['score_0_to_100']) < 1e-10
        for score, verdict in [(0, 'incorrect'), (1, 'partial'), (2, 'correct')]:
            assert sum(v == score for v in scores[arm].values()) == comparison[arm]['verdicts'][verdict]
    assert scores['sft'].keys() == scores['rl'].keys()
    assert sum(scores['rl'][sid] > value for sid, value in scores['sft'].items()) == comparison['improved']
    assert sum(scores['rl'][sid] < value for sid, value in scores['sft'].items()) == comparison['worsened']
    families = read(ROOT.parent / '2026-09-08/training/mechanism_diagnosis_20260908/retention/family_states.json')
    assert len(families) == 91
    assert sum(row['partition'] == 'training' for row in families) == 45
    assert sum(row['partition'] == 'heldout' for row in families) == 46
    for stage, expected in [('base', 39), ('cpt', 38), ('sft', 45), ('reference', 48), ('rl', 48)]:
        assert sum(row['models'][stage]['both_correct'] for row in families) == expected
    print(f"Verified {len(manifest['files'])} file hashes; 2,400 MCQ predictions, all 12 paired contrasts, 186 VQA scores and 91 fact families.")


if __name__ == '__main__':
    main()
