"""Verify downloaded result archives and recompute the principal reported scores."""

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
from zipfile import ZipFile


def equal(actual, expected, label):
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-8):
        raise ValueError(f'{label}: recomputed {actual}, reported {expected}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets-dir', type=Path, required=True)
    args = parser.parse_args()
    index = json.loads(Path(__file__).with_name('index.json').read_text(encoding='utf-8'))
    verified, scores = 0, []
    for asset in index['archives']:
        path = args.assets_dir / asset['name']
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != asset['sha256']:
            raise ValueError(f'Archive checksum mismatch: {path.name}')
        with ZipFile(path) as archive:
            names = set(archive.namelist())
            expected_names = {name for name in names if name.endswith('/PUBLICATION_MANIFEST.json')}
            for name in sorted(expected_names):
                manifest = json.loads(archive.read(name))
                for record in manifest['files']:
                    payload = archive.read(record['path'])
                    if len(payload) != record['bytes'] or hashlib.sha256(payload).hexdigest() != record['sha256']:
                        raise ValueError(f"Result checksum mismatch: {record['path']}")
                    expected_names.add(record['path'])
                    verified += 1
            if names != expected_names:
                raise ValueError(f'Unmanifested archive members: {asset["name"]}')

            def read(name):
                return json.loads(archive.read(name))

            if asset['name'] == 'agent-evaluations.zip':
                prefix = 'agent/full_system_v1/'
                summary = read(prefix + 'campaign_summary.json')
                rows = summary['runs']
                equal(sum(len(row['judges']) for row in rows), summary['valid_judgements'], 'Agent valid judgments')
                for stats in summary['model_stats']:
                    subset = [row for row in rows if read(prefix + 'runs/' + row['run_id'] + '.json')['model'] == stats['model']]
                    equal(len(subset), stats['tasks'], 'Agent tasks')
                    passed = sum(len(row['judges']) == 2 and row['execution'] == 'completed' and all(check['verdict'] == 'pass' for judge in row['judges'] for check in judge['checks']) for row in subset)
                    critical = sum(any(judge['critical_failed'] for judge in row['judges']) for row in subset)
                    equal(passed, stats['both_judges_all_pass_cases'], 'Agent both-judge passes')
                    equal(critical, stats['any_judge_critical_failure_cases'], 'Agent critical failures')
                scores.append('Agent: 48 tasks and 96 judgments')

            if asset['name'] == 'model-evaluations.zip':
                for experiment, directory in [('independent_holdout_20260906', 'diagnostics/questions'), ('format_factorial_v1', 'judge/questions')]:
                    prefix = f'training/{experiment}/'
                    expected = read(prefix + 'results.json')['groups']
                    groups = collections.defaultdict(list)
                    for name in names:
                        if name.startswith(prefix + directory + '/') and name.endswith('.json'):
                            question = read(name)
                            if not question.get('reference_valid', True):
                                continue
                            for arm, model in question['models'].items():
                                task = model.get('task', question.get('task'))
                                groups[f'{task}/{arm}'].append(model['diagnosis'])
                    for group, result in expected.items():
                        rows = groups[group]
                        equal(len(rows), result['n'], f'{experiment}/{group}/n')
                        for metric in ['answer_score', 'reasoning_score', 'joint_score']:
                            equal(50 * mean(row[metric] for row in rows), result[metric], f'{experiment}/{group}/{metric}')
                    scores.append(experiment)
                comparison = read('training/gspo_vqa_rl_v1/comparison.json')
                pairs = comparison['pairs']
                equal(len(pairs), comparison['paired_scorable_questions'], 'GSPO pair count')
                equal(50 * mean(row['before'] for row in pairs), comparison['sft']['score_0_to_100'], 'GSPO SFT score')
                equal(50 * mean(row['after'] for row in pairs), comparison['rl']['score_0_to_100'], 'GSPO RL score')
                scores.append('GSPO paired scores')
    print(json.dumps({'verified_files': verified, 'recomputed': scores}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
