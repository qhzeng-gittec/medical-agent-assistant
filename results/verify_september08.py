"""Verify the September 8 file snapshot and recompute selected public metrics."""
import hashlib
import json
import argparse
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent / '2026-09-08'
ARCHIVE = None


def read(name):
    path = ROOT / name
    return json.loads(path.read_text(encoding='utf-8') if path.exists() else ARCHIVE.read(name).decode('utf-8'))


def main():
    global ARCHIVE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets-dir', required=True, type=Path, help='Directory containing the downloaded September 8 ZIP')
    args = parser.parse_args()
    manifest = read('manifest.json')
    attachment = manifest['attachment']
    archive_path = args.assets_dir / attachment['filename']
    assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == attachment['sha256']
    ARCHIVE = zipfile.ZipFile(archive_path)
    for item in manifest['files']:
        path = ROOT / item['path']
        assert path.resolve().is_relative_to(ROOT.resolve()), item['path']
        raw = ARCHIVE.read(item['path']) if item.get('storage') == 'attachment' else path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == item['sha256'], item['path']

    folder = 'training/mechanism_diagnosis_20260908/'
    states = read(folder + 'mechanisms/states.json')
    eligible = [s for s in states if s['reference_valid']]
    assert len(eligible) == len({s['source_id'] for s in eligible}) == 312
    before = [s['scores']['reference__original'] == 2 for s in eligible]
    after = [s['scores']['rl__original'] == 2 for s in eligible]
    counts = dict(before=sum(before), after=sum(after), wins=sum(not b and a for b,a in zip(before,after)),
                  losses=sum(b and not a for b,a in zip(before,after)))
    summary = read(folder + 'mechanisms/analysis.json')['all']['contrasts']['rl__original']
    assert counts == {k: summary[k] for k in counts} == dict(before=134, after=132, wins=4, losses=6)

    rows = [r for r in read(folder + 'rl_frozen_eval/states.json') if r['split'] == 'train' and r['reference_valid']]
    assert len(rows) == len({r['source_id'] for r in rows}) == 128
    frozen = read(folder + 'rl_frozen_eval/analysis.json')['train']['all']
    for model in ('r0', 'r2', 'r5'):
        assert all(len(r['models'][model]['samples']) == 8 for r in rows)
        greedy = sum(r['models'][model]['greedy']['answer'] == 2 for r in rows)
        pass8 = sum(any(s['answer'] == 2 for s in r['models'][model]['samples']) for r in rows)
        assert greedy == frozen['models'][model]['greedy_answer_correct']
        assert pass8 == round(frozen['models'][model]['pass8']['mean'] * len(rows))
    contrast = frozen['contrasts']['r0__r5']['greedy_answer']
    assert (contrast['before'], contrast['after'], contrast['wins'], contrast['losses']) == (79,84,5,0)
    assert contrast['exact_mcnemar_two_sided_p'] == 2 * 0.5**5

    arch = read('agent/architecture_compare_20260908_v2/comparison_summary.json')
    assert sum(g['runs'] for g in arch['groups'].values()) == 32
    assert sum(g['automatic_grades_completed'] for g in arch['groups'].values()) == 17
    serial, parallel = [arch['scheduler'][k]['mean_seconds'] for k in ('serial','swarm')]
    assert abs(100*(1-parallel/serial) - arch['scheduler']['latency_reduction_percent']) < 1e-8
    improvement = read('agent/improvement_v2/summary.json')
    assert sum(g['completed'] for g in improvement['groups']) == 48
    telemetry = read('agent/holdout_v1_live_20260908/telemetry_summary.json')
    assert telemetry['coverage']['expected_runs'] == 324
    assert telemetry['coverage']['canonical_completed_records'] == 277
    completed = [read(r['path']) for r in manifest['files'] if r['path'].startswith('agent/holdout_v1_live_20260908/runs/')]
    errors = [read(r['path']) for r in manifest['files'] if r['path'].startswith('agent/holdout_v1_live_20260908/errors/')]
    assert len(completed) == 277 and all(r['status'] == 'completed' for r in completed)
    assert len(errors) == 23 and all(r['status'] == 'error' for r in errors)
    assert Counter(r['model'] for r in completed if r['repetition'] == 1) == {
        'minimax/minimax-m2.5': 69, 'qwen/qwen3.5-27b': 69, 'gemini-3.5-flash': 68}
    print(f"Verified {len(manifest['files'])} snapshot files; recomputed 312-question RL pairs, 128-question greedy/pass@8, and Agent summary counts.")
    ARCHIVE.close()


if __name__ == '__main__':
    main()
