"""Bounded, inspectable OpenRouter calls for this benchmark only."""
import json
import os
import threading
import time
from pathlib import Path

import httpx

MODEL = 'qwen/qwen3.5-27b'
EMBED = 'qwen/qwen3-embedding-8b'


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


class API:
    def __init__(self, directory, max_requests=1000):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.key = os.environ['OPENROUTER_API_KEY']
        self.lock = threading.Lock()
        self.max_requests = max_requests
        self.calls = len(list(self.directory.glob('*.json')))
        self.halted = False

    def call(self, name, endpoint, payload):
        path = self.directory / (name + '.json')
        if path.exists():
            record = read(path)
            if record['request'] != payload:
                raise ValueError(f'Frozen request changed: {name}')
            if record['status'] != 'completed':
                raise RuntimeError(f'Inspect preserved incomplete request before resuming: {name}')
            return record['response']
        with self.lock:
            if self.halted:
                raise RuntimeError('API access halted; inspect the first provider error')
            if self.calls >= self.max_requests:
                raise RuntimeError('Benchmark API request cap reached')
            self.calls += 1
        record = {'request': payload, 'status': 'started'}
        write(path, record)
        started = time.monotonic()
        try:
            record['transport_attempts'] = []
            for attempt in range(3):
                try:
                    response = httpx.post('https://openrouter.ai/api/v1/' + endpoint,
                                          headers={'Authorization': 'Bearer ' + self.key},
                                          json=payload, timeout=180)
                    record['transport_attempts'].append({'attempt':attempt+1,'http_status':response.status_code})
                    if response.status_code >= 500 and attempt < 2:
                        time.sleep(1)
                        continue
                    break
                except httpx.TransportError as error:
                    record['transport_attempts'].append({'attempt':attempt+1,'error_type':type(error).__name__})
                    if attempt == 2:
                        raise
                    time.sleep(1)
            record['http_status'] = response.status_code
            if response.is_error:
                record['provider_message'] = response.text[:1500].replace(self.key, '[REDACTED]')
            response.raise_for_status()
            data = response.json()
            if 'error' in data:
                raise ValueError('Provider returned an error envelope')
            record.update(status='completed', response=data)
            return data
        except Exception as error:
            if isinstance(error, httpx.HTTPStatusError) and error.response.status_code in {401, 402}:
                self.halted = True
            record.update(status='error', error_type=type(error).__name__)
            raise
        finally:
            record['seconds'] = round(time.monotonic() - started, 4)
            write(path, record)

    def json(self, name, instruction, content, model=MODEL, max_tokens=12000):
        data = self.call(name, 'chat/completions', {
            'model': model, 'messages': [{'role': 'system', 'content': instruction},
                                       {'role': 'user', 'content': json.dumps(content, ensure_ascii=False)}],
            'temperature': 0.2, 'max_tokens': max_tokens,
            'reasoning': {'effort': 'low'} if model.startswith('minimax/') else {'enabled': False},
            'response_format': {'type': 'json_object'}})
        choice = data['choices'][0]
        if choice['finish_reason'] != 'stop':
            raise ValueError(f'Incomplete generation: {name}')
        content = choice['message']['content'].strip()
        if content.startswith('```json') and content.endswith('```'):
            content = content[len('```json'):-3].strip()
        return json.loads(content)

    def embed(self, name, texts):
        data = self.call(name, 'embeddings', {'model': EMBED, 'input': texts, 'encoding_format': 'float'})
        rows = sorted(data['data'], key=lambda r: r['index'])
        if [r['index'] for r in rows] != list(range(len(texts))):
            raise ValueError('Embedding indices differ from input batch')
        return [r['embedding'] for r in rows]
