"""Recount published raw scores, verify file hashes, or regenerate derived tables."""
import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def analyze(records):
    rows = []
    for record in records:
        assert record['reference_valid']
        valid = record['usable'] and bool(record['facts']) and all(f['reference_valid'] for f in record['facts'])
        for model in ['reference', 'rl']:
            pairs = [[f['responses'][model][w]['judgment']['score'] for w in ['question', 'paraphrase']]
                     for f in record['facts']] if valid else []
            assert all(s in [0, 1, 2] for p in pairs for s in p)
            if not valid:
                category = 'no_valid_probes'
            elif all(p == [2, 2] for p in pairs):
                category = 'all_probed_facts_correct'
            elif all(p == [0, 0] for p in pairs):
                category = 'all_probed_facts_wrong'
            elif any(p == [0, 0] for p in pairs):
                category = 'some_facts_consistently_wrong'
            else:
                category = 'partial_or_wording_sensitive'
            application = valid and not record['primitive'] and record['audit']['jointly_sufficient'] and record['audit']['separable_application']
            review = record['manual_application_review']
            original = record['originals'][model]
            rows.append(dict(source_id=record['source_id'], model=model, task=record['task'],
                original_score=original['judgment']['score'], reasoning_score=original['reasoning_score'],
                category=category, fact_scores=pairs, primitive=record['primitive'], application_eligible=application,
                manual_application_failure=review['include_application_failure'] if review else None,
                question=record['question']))
    result = {}
    for model in ['reference', 'rl']:
        result[model] = {}
        for task in ['all', 'knowledge', 'case']:
            group = [r for r in rows if r['model']==model and (task=='all' or r['task']==task)]
            result[model][task] = {}
            for label, scores in [('not_full_credit', [0, 1]), ('zero_only', [0]), ('partial_only', [1]), ('all_questions', [0, 1, 2])]:
                subset = [r for r in group if r['original_score'] in scores]
                counts = Counter(r['category'] for r in subset)
                known = [r for r in subset if r['category']=='all_probed_facts_correct']
                mixed = [r for r in subset if r['category']=='partial_or_wording_sensitive']
                assert sum(counts.values()) == len(subset)
                result[model][task][label] = dict(n=len(subset), counts=dict(counts),
                    percentages={k: round(100*v/len(subset), 2) for k,v in counts.items()},
                    known_subtypes=dict(primitive=sum(r['primitive'] for r in known),
                        application_eligible=sum(r['application_eligible'] for r in known),
                        other=sum(not r['primitive'] and not r['application_eligible'] for r in known),
                        reviewed_application_failure=sum(r['application_eligible'] and r['manual_application_failure'] is True for r in known),
                        disputed_application=sum(r['application_eligible'] and r['manual_application_failure'] is False for r in known)),
                    mixed_with_zero_and_full_wording=sum(any(set(p)=={0,2} for p in r['fact_scores']) for r in mixed),
                    correct_answer_imperfect_reasoning=sum(r['original_score']==2 and r['reasoning_score']<2 for r in subset))
    assert len({r['source_id'] for r in records}) == len(records) == 312
    assert sum(len(r['facts']) for r in records) == 427
    return result, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true', help='Regenerate summaries, CSV and package checksums.')
    args = parser.parse_args()
    records = [json.loads(line) for line in (ROOT/'records.jsonl').read_text(encoding='utf-8').split('\n') if line]
    summary, rows = analyze(records)
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    csv_bytes = buffer.getvalue().encode('utf-8-sig')
    if args.write:
        (ROOT/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf-8', newline='\n')
        (ROOT/'question_breakdown.csv').write_bytes(csv_bytes)
        manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.iterdir()) if p.is_file() and p.name!='manifest.json'}
        (ROOT/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8', newline='\n')
    else:
        assert summary == json.loads((ROOT/'summary.json').read_text(encoding='utf-8')), 'Summary differs from raw scores'
        assert csv_bytes == (ROOT/'question_breakdown.csv').read_bytes(), 'CSV differs from raw scores'
        manifest = json.loads((ROOT/'manifest.json').read_text(encoding='utf-8'))
        assert set(manifest) == {p.name for p in ROOT.iterdir() if p.is_file() and p.name!='manifest.json'}
        for name, digest in manifest.items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
    print(json.dumps(dict(verified_questions=len(records), model_question_rows=len(rows),
        prerequisite_facts=427, final_model_errors=summary['rl']['all']['not_full_credit'],
        package_files=len(manifest)), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
