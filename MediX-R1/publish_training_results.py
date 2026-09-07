"""Publish the frozen project holdout as the current training results."""
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from markdown_it import MarkdownIt

LAB = Path(__file__).resolve().parent
SOURCE = LAB / 'outputs/independent_holdout_20260906'
OUT = LAB / 'outputs/knowledge_experiments_v1/analysis'
REPORT = SOURCE / '训练结果分析_独立留出复测.md'
NAMES = {
    'A': 'VQA / Attention', 'B': 'VQA / Attention+FFN',
    'C': 'VQA+知识 / Attention', 'D': 'VQA+知识 / Attention+FFN',
    'D_half': 'D，FFN 推理缩放 0.5',
    'E': 'VQA+知识+病例 / Attention+FFN',
    'F': 'VQA+知识+病例+上下文 / Attention+FFN',
}
METRICS = ('answer_score', 'reasoning_score', 'joint_score')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def table(headers, rows):
    return '\n'.join('| ' + ' | '.join(map(str, row)) + ' |'
                     for row in [headers, ['---'] * len(headers), *rows])


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    plan = read_json(SOURCE / 'plan.json')
    results = read_json(SOURCE / 'results.json')
    audit = read_json(SOURCE / 'holdout_audit.json')
    evaluation = SOURCE / 'evaluation.jsonl'
    assert hashlib.sha256(evaluation.read_bytes()).hexdigest() == plan['evaluation_sha256']
    with evaluation.open(encoding='utf-8') as stream:
        sources = [json.loads(line) for line in stream]
    assert Counter(row['task'] for row in sources) == plan['task_counts']
    assert len({row['source_id'] for row in sources}) == 247
    for split in ('train', 'validation'):
        with (LAB / f'data/knowledge_experiments_v1/{split}.jsonl').open(encoding='utf-8') as stream:
            pool = [json.loads(line) for line in stream]
        for field in ('source_id', 'source_group'):
            assert not {row[field] for row in sources} & {row[field] for row in pool}
    questions = {p.stem: read_json(p) for p in (SOURCE / 'diagnostics/questions').glob('*.json')}
    assert questions.keys() == {row['source_id'] for row in sources}
    assert not results['invalid_questions'] and not results['same_final_score_conflicts']
    groups = []
    for task, arms in plan['arms'].items():
        subset = [q for q in questions.values() if q['task'] == task]
        for arm in arms:
            scores = [q['models'][arm]['diagnosis'] for q in subset]
            expected = results['groups'][f'{task}/{arm}']
            assert expected['n'] == len(scores)
            for metric in METRICS:
                assert all(score[metric] in (0, 1, 2) for score in scores)
                assert math.isclose(50 * sum(s[metric] for s in scores) / len(scores), expected[metric])
            assert sum(s['answer_score'] == 2 for s in scores) == expected['answer_correct']
            groups.append(dict(task=task, arm=arm, model=NAMES[arm], **expected))
    assert sum(g['n'] for g in groups) == 944
    pairs = []
    for task, contrasts in plan['contrasts'].items():
        for contrast in contrasts:
            candidate, baseline = contrast.split('-')
            subset = [q for q in questions.values() if q['task'] == task]
            for q in sorted(subset, key=lambda q: q['source_id']):
                left, right = q['models'][baseline], q['models'][candidate]
                record = dict(task=task, contrast=contrast, source_id=q['source_id'],
                              question=q['question'], reference=q['reference'],
                              baseline=baseline, candidate=candidate)
                for metric in METRICS:
                    record[f'baseline_{metric}'] = left['diagnosis'][metric]
                    record[f'candidate_{metric}'] = right['diagnosis'][metric]
                    record[f'delta_{metric}'] = right['diagnosis'][metric] - left['diagnosis'][metric]
                record.update(baseline_answer=left['final_answer'], candidate_answer=right['final_answer'],
                              baseline_reasoning=left['reasoning'], candidate_reasoning=right['reasoning'],
                              baseline_judge=left['diagnosis']['explanation'], candidate_judge=right['diagnosis']['explanation'])
                pairs.append(record)
            current = [p for p in pairs if p['task'] == task and p['contrast'] == contrast]
            for metric in METRICS:
                assert math.isclose(50 * sum(p[f'delta_{metric}'] for p in current) / len(current),
                                    results['contrasts'][f'{task}/{contrast}'][metric]['delta'], abs_tol=1e-12)
    assert len(pairs) == 847
    score_table = table(['任务', '模型', '答案分', '解释分', '联合分', '最终答案满分数'], [
        [('知识' if g['task'] == 'knowledge' else '上下文') + f"，{g['n']} 题",
         f"{g['arm']}：{g['model']}", *[f'{g[m]:.2f}' for m in METRICS],
         f"{g['answer_correct']}/{g['n']}"] for g in groups])
    report = REPORT.read_text(encoding='utf-8')
    start = report.index('| 任务 | 模型 |')
    end = report.index('\n\n', start)
    report = report[:start] + score_table + report[end:]
    contrast_rows = []
    for key, values in results['contrasts'].items():
        cells = [key]
        for metric in ('answer_score', 'joint_score'):
            value = values[metric]
            low, high = value['ci95']
            cells.append(f"{value['delta']:+.2f} [{low:+.2f}, {high:+.2f}]")
        contrast_rows.append(cells)
    start = report.index('| 对照 | 答案分差')
    end = report.index('\n\n', start)
    report = report[:start] + table(['对照', '答案分差及名义 95% 配对区间', '联合分差及名义 95% 配对区间'], contrast_rows) + report[end:]
    REPORT.write_text(report, encoding='utf-8')
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / 'report_metrics.json', dict(
        evaluation='independent_project_test_20260906', split='test',
        source=str(SOURCE / 'results.json'), evaluation_sha256=plan['evaluation_sha256'],
        task_counts=plan['task_counts'], metric_scale='0-100 normalized ordinal scores, not exact accuracy',
        overall_score=None, unmeasured=['base', 'vqa', 'case', 'four_task_overall'],
        training_loss_probes_in_test_scores=False, **results))
    write_csv(OUT / 'sample_comparisons.csv', pairs)
    write_json(OUT / 'sample_comparisons.json', pairs)
    write_csv(LAB / 'outputs/analysis/metrics.csv', groups)
    write_json(LAB / 'outputs/analysis/aggregate.json', read_json(OUT / 'report_metrics.json'))
    report_paths = [OUT / 'training_report_2026-09-06.md',
                    LAB / 'outputs/analysis/训练实验完整复盘与核查报告.md',
                    LAB / 'outputs/analysis/experiment_report.md',
                    LAB / 'outputs/reasoning_effects_v1/full_reasoning_diagnostic_report.md']
    for path in report_paths:
        path.write_text(report, encoding='utf-8')
    html = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>医疗 LoRA 留出测试结果</title>'
    html += '<style>body{max-width:1100px;margin:40px auto;padding:0 24px;font:16px/1.75 sans-serif;color:#222}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border:1px solid #ddd;padding:8px;text-align:left}th{background:#f4f6f8}a{color:#1765a5}code{overflow-wrap:anywhere}</style><body>'
    html += MarkdownIt('commonmark').enable('table').render(report) + '</body></html>'
    for path in [OUT / 'training_report_2026-09-06.html',
                 LAB / 'outputs/reasoning_effects_v1/full_reasoning_diagnostic_report.html']:
        path.write_text(html, encoding='utf-8')
    lines = ['# 留出测试逐题对照', '',
             '150 道知识题、97 道上下文题，共 847 对预定比较。以下同时列出改善和退步；分差为 0/1/2 原始评分差。各对照重复使用同一批题，不能把 847 对当作 847 道独立题。', '',
             '答案和解释由 gpt-5.6-sol 匿名评分，不等同医学专家审核。完整题目、两组输出和评分理由见同目录 sample_comparisons.csv/json。', '']
    for task, contrasts in plan['contrasts'].items():
        for contrast in contrasts:
            current = [p for p in pairs if p['task'] == task and p['contrast'] == contrast]
            lines += [f'## {task} / {contrast}', '']
            for label, condition in [('退步', lambda p: any(p[f'delta_{m}'] < 0 for m in METRICS)),
                                     ('改善', lambda p: any(p[f'delta_{m}'] > 0 for m in METRICS))]:
                selected = [p for p in current if condition(p)]
                lines += [f'### {label}：{len(selected)} 对', '',
                          table(['样本与完整证据', '答案分差', '解释分差', '联合分差'], [
                              [f"[{p['source_id']}](<{(SOURCE / 'diagnostics/questions' / (p['source_id'] + '.json')).as_posix()}>)",
                               *[p[f'delta_{m}'] for m in METRICS]] for p in selected]), '']
    (OUT / 'regression_cases.md').write_text('\n'.join(lines), encoding='utf-8')
    write_json(OUT / 'publication_verification.json', dict(
        passed=True, questions=247, outputs=944, paired_rows=847,
        recomputed_from_question_scores=True, train_validation_overlap=0,
        source_results_sha256=hashlib.sha256((SOURCE / 'results.json').read_bytes()).hexdigest(),
        evaluation_sha256=plan['evaluation_sha256'], checkpoints_unchanged=audit['checkpoints_unchanged']))
    print('Published current holdout: 247 questions, 944 outputs, 847 paired comparisons; scores and split separation verified.')


if __name__ == '__main__':
    main()
