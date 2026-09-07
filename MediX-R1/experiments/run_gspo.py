"""Run the recorded SFT baseline, GSPO training, and paired VQA evaluation."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from common.io import load_jsonl, save

LAB = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=LAB / 'outputs/gspo_vqa_rl_v1')
    parser.add_argument('--resume-from', type=Path)
    args = parser.parse_args()
    root = args.output.resolve()
    initial = LAB / 'outputs/knowledge_experiments_v1/seed42/vqa_knowledge_attention/final_adapter'
    training = root / 'training'
    stages = [
        ('baseline', 'evaluation.evaluate_vqa_rl', ['--adapter-dir', str(initial), '--output-dir', str(root / 'evaluation/sft')]),
        ('training', 'training.gspo', ['--output-dir', str(training), '--prompts', '1000', '--prompts-per-rollout', '8',
          '--group-size', '4', '--judge-model', 'gpt-5.5', '--judge-batch-size', '16', '--judge-workers', '2',
          '--generation-batch-size', '4', '--train-micro-batch-size', '4'] +
         (['--resume-from', str(args.resume_from.resolve())] if args.resume_from else [])),
        ('evaluation', 'evaluation.evaluate_vqa_rl', ['--adapter-dir', str(training / 'final_adapter'), '--output-dir', str(root / 'evaluation/rl')]),
    ]
    root.mkdir(parents=True, exist_ok=True)
    markers = {'baseline': root / 'evaluation/sft/summary.json',
               'training': training / 'summary.json', 'evaluation': root / 'evaluation/rl/summary.json'}
    for name, module, arguments in stages:
        if markers[name].exists():
            continue
        command = [sys.executable, '-u', '-m', module, *arguments]
        save(root / 'status.json', {'stage': name, 'status': 'running', 'command': command})
        subprocess.run(command, cwd=LAB, check=True)
    before = load_jsonl(root / 'evaluation/sft/scored.jsonl')
    after = load_jsonl(root / 'evaluation/rl/scored.jsonl')
    pairs = []
    for old, new in zip(before, after, strict=True):
        if old['source_id'] != new['source_id']:
            raise ValueError('Paired evaluation source IDs differ')
        if old['judge']['score'] is not None and new['judge']['score'] is not None:
            pairs.append({'source_id': old['source_id'], 'before': old['judge']['score'], 'after': new['judge']['score']})
    comparison = {'status': 'completed', 'paired_scorable_questions': len(pairs),
                  'paired_score_delta_0_to_100': 50 * sum(p['after'] - p['before'] for p in pairs) / len(pairs) if pairs else None,
                  'pairs': pairs, 'limitation': 'One training seed; development VQA comparison scored by a model.'}
    save(root / 'comparison.json', comparison)
    save(root / 'status.json', {'status': 'completed'})
    print(json.dumps(comparison))


if __name__ == '__main__':
    main()
