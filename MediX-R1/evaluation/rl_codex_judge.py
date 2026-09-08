"""Budgeted, cached Codex GPT-5.5 judgments for multi-round medical RL."""
import copy
import hashlib
import json
import re
from common.codex_judge import MODEL, judge_request
import threading
from pathlib import Path

from experiments.diagnose_reasoning_effects import FLAGS, RUBRIC, SCORE_SCHEMA
from data_processing.download_cpt_sources import save

LAB = Path(__file__).resolve().parents[1]


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def parse_judgment(raw, expected):
    if raw['requested_model'] != MODEL or raw['finish_reason'] != 'stop':
        raise ValueError('Codex model mismatch or truncated judgment; cached raw response retained.')
    visible = raw['message']['content'].strip()
    while visible.startswith('<thought>'):
        _, separator, visible = visible.partition('</thought>')
        if not separator:
            raise ValueError('Incomplete thought envelope.')
        visible = visible.strip()
    fenced = re.fullmatch(r'```(?:json)?\s*([\s\S]*?)\s*```', visible, re.I)
    result = json.loads(fenced[1] if fenced else visible)
    if type(result['reference_valid']) is not bool:
        raise ValueError('Invalid reference validity field.')
    candidates = result['candidates']
    if len(candidates) != len(expected) or {c['id'] for c in candidates} != set(expected):
        raise ValueError('Judge candidate IDs differ from the request.')
    for c in candidates:
        for field in ['answer_score', 'reasoning_score']:
            if type(c[field]) is not int or c[field] not in [0, 1, 2]:
                raise ValueError('Invalid judgment score.')
        if not set(c['flags']) <= set(FLAGS) or any(f['type'] not in FLAGS for f in c['findings']):
            raise ValueError('Invalid judgment finding.')
        c['joint_score'] = min(c['answer_score'], c['reasoning_score'])
    return result


class CodexJudge:
    def __init__(self, root, max_calls):
        self.root, self.max_calls = Path(root)/'codex_gpt55', max_calls
        self.lock = threading.Lock()
        for name in ['requests', 'attempts', 'calls']:
            (self.root/name).mkdir(parents=True, exist_ok=True)

    def score(self, row, responses):
        unique = {}
        order = []
        for r in responses:
            text = {k:r[k] for k in ['reasoning', 'final_answer']}
            key = digest(text)[:20]
            order.append(key)
            unique[key] = dict(id=key, **text)
        schema = copy.deepcopy(SCORE_SCHEMA)
        schema['properties']['candidates']['items']['properties']['id']['enum'] = sorted(unique)
        request = dict(rubric=RUBRIC, payload=dict(task=row['task'], question=row['user_text'],
            reference=row['reference'], reference_explanation=row.get('reference_explanation', ''),
            candidates=[unique[k] for k in sorted(unique)]), schema=schema,
            thinking_level='medium', max_output_tokens=4096)
        if len(json.dumps(request, ensure_ascii=False)) > 30000:
            raise ValueError('Judge request exceeds the frozen input-character budget.')
        key = digest(dict(model=MODEL, request=request))
        call = self.root/'calls'/f'{key}.json'
        with self.lock:
            if not call.exists():
                attempt = self.root/'attempts'/f'{key}.json'
                if attempt.exists():
                    raise RuntimeError('Unresolved prior Codex attempt; inspect it before retrying.')
                if len(list((self.root/'attempts').glob('*.json'))) >= self.max_calls:
                    raise RuntimeError('Frozen Codex request budget exhausted.')
                save(self.root/'requests'/f'{key}.json', request)
                save(attempt, dict(state='submitted', source_id=row['source_id']))
        if call.exists():
            raw = json.loads(call.read_text(encoding='utf-8'))
        else:
            raw = judge_request(request, self.root/'structured'/f'{key}.json')
            save(call, raw)
            save(self.root/'attempts'/f'{key}.json', dict(state='response_saved', source_id=row['source_id']))
        recovery = self.root/'recoveries'/f'{key}.json'
        if recovery.exists():
            replacement = json.loads(recovery.read_text(encoding='utf-8'))['replacement_call_sha256']
            if raw['finish_reason'] != 'length' or not re.fullmatch('[0-9a-f]{64}', replacement):
                raise ValueError('Only a confirmed truncated response can use an audited replacement.')
            if json.loads((self.root/'requests'/f'{replacement}.json').read_text(encoding='utf-8')) != request:
                raise ValueError('Replacement judgment request differs from the frozen request.')
            raw = json.loads((self.root/'calls'/f'{replacement}.json').read_text(encoding='utf-8'))
            key = replacement
        result = parse_judgment(raw, unique)
        indexed = {c['id']:c for c in result['candidates']}
        return [dict(indexed[k], reference_valid=result['reference_valid'], call_sha256=key) for k in order]

    def usage(self):
        totals = dict(requests=len(list((self.root/'attempts').glob('*.json'))), responses=0,
                      prompt_tokens=0, completion_tokens=0, total_tokens=0, missing_usage=0)
        for path in (self.root/'calls').glob('*.json'):
            raw = json.loads(path.read_text(encoding='utf-8'))
            totals['responses'] += 1
            if not raw.get('usage'):
                totals['missing_usage'] += 1
            else:
                for key in ['prompt_tokens', 'completion_tokens', 'total_tokens']:
                    totals[key] += raw['usage'].get(key, 0)
        return totals
