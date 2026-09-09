"""Verify the selected evidence and recompute paired results without API calls."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets-dir', required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'manifest.json').read_text(encoding='utf-8'))
    asset = args.assets_dir / manifest['attachment']['filename']
    assert hashlib.sha256(asset.read_bytes()).hexdigest() == manifest['attachment']['sha256']
    with zipfile.ZipFile(asset) as archive:
        def raw(name):
            info = manifest['files'][name]
            return (ROOT / name).read_bytes() if info['storage'] == 'repository' else archive.read(name)

        def read(name):
            return json.loads(raw(name))

        for name, info in manifest['files'].items():
            assert hashlib.sha256(raw(name)).hexdigest() == info['sha256'], name
        for name, digest in manifest['supporting_files'].items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
        result = {}
        for phase in ('reg', 'fresh'):
            cases = read(f'{phase}/experiment.json')['cases']
            summary = read(f'{phase}/summary.json')
            assert summary['complete']
            states = {}
            result[phase] = {}
            for arm in ('before', 'after'):
                states[arm] = {}
                statuses = Counter()
                for cid in cases:
                    prefix = f'{phase}/{arm}/{cid}/'
                    runs = [n for n in manifest['files'] if n.startswith(prefix + 'runs/')]
                    assert len(runs) == 1, (phase, arm, cid)
                    name = Path(runs[0]).name
                    run = read(runs[0])
                    assert run['status'] == 'completed'
                    assert hashlib.sha256(raw(prefix + 'traces/' + name)).hexdigest() == run['trace_sha256']
                    original = prefix + 'grades/' + name
                    recovered = prefix + 'recovered_grades/' + name
                    grade = read(recovered if recovered in manifest['files'] else original)
                    if recovered in manifest['files']:
                        assert grade['recovery']['original_grade_sha256'] == hashlib.sha256(raw(original)).hexdigest()
                    statuses[grade['status']] += 1
                    if grade['status'] != 'graded':
                        states[arm][cid] = None
                        continue
                    votes = [{c['id']: c['verdict'] for c in j['checks']} for j in grade['judge_results']]
                    assert len(votes) == 2
                    checks = [[v[c['id']] for v in votes] for c in grade['checks']]
                    applicable = [v == ['pass', 'pass'] for v in checks if v != ['not_applicable', 'not_applicable']]
                    passed = bool(applicable) and all(applicable)
                    assert passed == grade['all_applicable_passed']
                    states[arm][cid] = passed
                passed = sum(v is True for v in states[arm].values())
                assert passed == summary['arms'][arm]['all_applicable_passed']
                assert dict(statuses) == summary['arms'][arm]['grading_status']
                result[phase][arm] = {'passed': passed, 'planned': len(cases), 'grading': dict(statuses)}
            paired = Counter()
            for cid in cases:
                b, a = states['before'][cid], states['after'][cid]
                category = ('incomplete_pair' if b is None or a is None else 'both_pass' if b and a
                            else 'improved' if a else 'regressed' if b else 'neither_pass')
                paired[category] += 1
            assert dict(paired) == summary['paired']
            result[phase]['paired'] = dict(paired)
        print(json.dumps({'verified_files': len(manifest['files']), 'results': result}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
