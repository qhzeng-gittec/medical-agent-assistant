"""Verify the published current Agent results against original records, without API calls."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent


def main():
    def read(name):
        return json.loads((ROOT/name).read_text(encoding='utf-8'))

    manifest = read('manifest.json')
    for name,digest in manifest['files'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
    cases = {c['case_id']: c for c in read('cases.json')}
    rows = read('case_results.json')
    assert len(rows) == len(cases) == 72
    assert {r['case_id'] for r in rows} == set(cases)
    domains = {}
    execution, grading = Counter(), Counter()
    with zipfile.ZipFile(ROOT/'evidence.zip') as z:
        assert set(z.namelist()) == set(manifest['archive_entries'])
        for name,digest in manifest['archive_entries'].items():
            assert hashlib.sha256(z.read(name)).hexdigest() == digest, name
        for row in rows:
            run = json.loads(z.read(row['source_run']))
            grade = json.loads(z.read(row['source_grade']))
            cid = row['case_id']
            assert run['case_id'] == grade['case_id'] == cid
            trace = row['source_run'].replace('/runs/','/traces/')
            assert hashlib.sha256(z.read(trace)).hexdigest() == run['trace_sha256']
            if '/recovered_grades/' in row['source_grade']:
                original = row['source_grade'].replace('/recovered_grades/','/grades/')
                assert hashlib.sha256(z.read(original)).hexdigest() == grade['recovery']['original_grade_sha256']
            assert row['turns'] == [{k:t[k] for k in ['turn','user','answer']} for t in run['turns']]
            assert row['checks'] == grade['checks']
            assert row['judge_results'] == grade['judge_results']
            success = False
            if grade['status'] == 'graded':
                votes = [{c['id']:c['verdict'] for c in j['checks']} for j in grade['judge_results']]
                assert len(votes) == 2
                checks = [[v[c['id']] for v in votes] for c in grade['checks']]
                applicable = [v == ['pass','pass'] for v in checks if v != ['not_applicable','not_applicable']]
                success = bool(applicable) and all(applicable)
                assert success == grade['all_applicable_passed']
            assert success == row['all_applicable_passed']
            execution[run['status']] += 1
            grading[grade['status']] += 1
            group = domains.setdefault(cases[cid]['domain'], {'planned':0,'completed':0,'graded':0,'passed':0})
            group['planned'] += 1
            group['completed'] += run['status'] == 'completed'
            group['graded'] += grade['status'] == 'graded'
            group['passed'] += success
    summary = read('summary.json')
    assert domains == summary['domains']
    assert dict(execution) == summary['execution']
    assert dict(grading) == summary['grading']
    assert sum(g['passed'] for g in domains.values()) == summary['all_applicable_passed'] == 67
    assert summary['rate_over_planned'] == 67/72
    print(json.dumps({'verified':True,'cases':72,'execution':dict(execution),'grading':dict(grading),
                      'all_applicable_passed':67,'domains':domains},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
