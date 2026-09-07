"""Verify current report copies, evidence links and published metrics."""
import csv
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote

LAB = Path(__file__).resolve().parent
ROOT = LAB / 'outputs/reasoning_effects_v1'
SOURCE = LAB / 'outputs/independent_holdout_20260906'
OUT = LAB / 'outputs/knowledge_experiments_v1/analysis'


class Inspect(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.tables = 0

    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            self.links.append(dict(attrs)['href'])
        if tag == 'table':
            self.tables += 1


report = (SOURCE / '训练结果分析_独立留出复测.md').read_text(encoding='utf-8')
copies = [OUT / 'training_report_2026-09-06.md', ROOT / 'full_reasoning_diagnostic_report.md',
          LAB / 'outputs/analysis/训练实验完整复盘与核查报告.md', LAB / 'outputs/analysis/experiment_report.md']
assert all(p.read_text(encoding='utf-8') == report for p in copies)
html = (ROOT / 'full_reasoning_diagnostic_report.html').read_text(encoding='utf-8')
assert html == (OUT / 'training_report_2026-09-06.html').read_text(encoding='utf-8')
parser = Inspect()
parser.feed(html)
assert parser.tables == report.count('\n| ---') == 5
assert '\ufffd' not in html
for link in parser.links:
    if not link.startswith(('https://', 'http://')):
        assert Path(unquote(link)).exists(), link
for phrase in ('七组完整结果', '61.875', '67.500', '推荐使用 `all_concise`'):
    assert phrase not in report
with (OUT / 'sample_comparisons.csv').open(encoding='utf-8-sig', newline='') as stream:
    pairs = list(csv.DictReader(stream))
assert len(pairs) == 847
assert len({p['source_id'] for p in pairs}) == 247
metrics = json.loads((OUT / 'report_metrics.json').read_text(encoding='utf-8'))
raw = json.loads((SOURCE / 'results.json').read_text(encoding='utf-8'))
assert metrics['groups'] == raw['groups'] and metrics['contrasts'] == raw['contrasts']
assert metrics['overall_score'] is None and not metrics['training_loss_probes_in_test_scores']
for path in [OUT / 'regression_cases.md', LAB / 'outputs/analysis/实验结论与面试复盘.md']:
    for link in re.findall(r'\]\(<([^>]+)>\)', path.read_text(encoding='utf-8')):
        assert Path(link).exists(), link
for path in [LAB / 'publish_training_results.py', OUT / 'build_report.py', OUT / 'analyze_report.py', ROOT / 'build_report.py', Path(__file__)]:
    compile(path.read_text(encoding='utf-8'), str(path), 'exec')
result = dict(passed=True, current_report_copies=4, html_tables=parser.tables,
              questions=247, outputs=944, paired_rows=847, evidence_links_valid=True,
              current_metrics_match_frozen_results=True)
(OUT / 'report_verification.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps(result))
