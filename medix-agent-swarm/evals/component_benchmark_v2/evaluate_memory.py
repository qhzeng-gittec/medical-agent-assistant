"""Real Mem0 OSS extraction, persistence and scoped dense retrieval on fictional users."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import threading
import time
import warnings

import httpx
from loguru import logger
from openai import OpenAI

from api import read, write
from evaluate_rag import lines, verify_freeze

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('frozen_memory', HERE / 'runtime' / 'long_term.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
STOP = threading.Event()
COST_LOCK = threading.Lock()
COST = 0.0


def anchor_coverage(groups, hits):
    # Literal, case-insensitive alternatives; semantic support is separately reviewed.
    text = '\n'.join(h['content'] for h in hits).casefold()
    return [any(term.casefold() in text for term in group) for group in groups]


def run_case(case, output, stores):
    global COST
    path = output / 'cases' / (case['id'] + '.json')
    if path.exists():
        saved = read(path)
        if saved['status'] != 'completed':
            raise RuntimeError(f'Preserved incomplete case requires inspection: {case["id"]}')
        return saved
    if STOP.is_set():
        raise RuntimeError('Batch halted after provider access error or budget cap')
    row = {'case': case, 'status': 'started', 'api_calls': [], 'queries': []}
    write(path, row)
    memory = None
    issues = []
    started = time.monotonic()

    def response_hook(response):
        global COST
        response.read()
        payload = json.loads(response.request.content)
        entry = {'endpoint': response.request.url.path, 'model': payload.get('model'),
                 'status': response.status_code}
        if response.is_success:
            data = response.json()
            entry.update(id=data.get('id'), usage=data.get('usage'))
            if data.get('choices'):
                entry['finish_reason'] = data['choices'][0].get('finish_reason')
                entry['response_content'] = data['choices'][0]['message'].get('content')
                if entry['finish_reason'] != 'stop':
                    issues.append('incomplete_model_response')
            with COST_LOCK:
                COST += (data.get('usage') or {}).get('cost', 0) or 0
                if COST >= 10:
                    STOP.set()
        else:
            if response.status_code in {401, 402}:
                issues.append(f'HTTP_{response.status_code}')
                STOP.set()
        row['api_calls'].append(entry)

    def request_hook(request):
        if STOP.is_set():
            raise RuntimeError('API access stopped or $10 memory-run cap reached')

    def open_memory():
        instance = MODULE.LongTermMemory(config={'storage_path': str(stores / case['id']),
                                                 'app_id': 'component-v2', 'threshold': 0.0})
        if not instance.enabled:
            raise RuntimeError(instance.disabled_reason)
        for provider in (instance.client.embedding_model, instance.client.llm):
            provider.client.close()
            provider.client = OpenAI(api_key=os.environ['OPENROUTER_API_KEY'],
                                     base_url='https://openrouter.ai/api/v1', timeout=120, max_retries=2,
                                     http_client=httpx.Client(event_hooks={'request': [request_hook], 'response': [response_hook]}))
        generate=instance.client.llm.generate_response
        def tracked_generate(*args, **kwargs):
            try:
                return generate(*args, **kwargs)
            except Exception as error:
                issues.append('extraction_'+type(error).__name__)
                raise
        instance.client.llm.generate_response=tracked_generate
        return instance

    try:
        memory = open_memory()
        for i, turn in enumerate(case['turns']):
            memory.add_session_summary(case['id'], f'session-{i+1}', turn['user'], turn['assistant'],
                                       metadata={'evaluation': 'synthetic-component-v2'})
            if issues:
                raise RuntimeError('Provider issue in extraction; inspect API records')
            write(path, row)
        memory.close()
        memory = open_memory()
        row['reopened_before_queries'] = True
        for query in case['queries']:
            before = time.monotonic()
            hits = memory.search_similar_sessions(query['query'], case['id'], 10)
            row['queries'].append({'query': query, 'hits': hits, 'seconds': round(time.monotonic()-before, 4)})
        # A same-topic request for an undisclosed administrative value is deliberately hard for similarity filtering.
        unknown = case['queries'][0]['query'] + ' 对应门诊收费发票的完整编号是什么？'
        row['unanswerable'] = {'query': unknown, 'hits': memory.search_similar_sessions(unknown, case['id'], 10)}
        row['unknown_user_hits'] = memory.search_similar_sessions(case['queries'][0]['query'], case['id']+'-unknown', 10)
        memory.app_id = 'component-v2-other-app'
        row['other_app_hits'] = memory.search_similar_sessions(case['queries'][0]['query'], case['id'], 10)
        if issues:
            raise RuntimeError('Provider issue during retrieval; inspect API records')
        row['status'] = 'completed'
        return row
    except Exception as error:
        row.update(status='error', error_type=type(error).__name__, provider_issues=issues)
        return row
    finally:
        row['seconds'] = round(time.monotonic()-started, 4)
        write(path, row)
        if memory:
            memory.close()


def main(args):
    global COST
    args.output=args.output.resolve()
    args.stores=args.stores.resolve()
    logger.remove()
    warnings.filterwarnings('ignore', message='Payload indexes have no effect')
    freeze = verify_freeze(HERE / 'data')
    cases = [c for c in lines(HERE / 'data' / 'mem0_cases.jsonl') if c['split'] == args.split]
    protocol = {'dataset_freeze': freeze, 'runtime_sha256': sha256((HERE/'runtime'/'long_term.py').read_bytes()).hexdigest(),
                'backend': 'Mem0 OSS 2.0.20 / local Qdrant / actual API extraction and embeddings',
                'llm_model': 'qwen/qwen3.5-27b', 'embedding_model': 'qwen/qwen3-embedding-8b',
                'collection_threshold': 0.0, 'baseline_threshold': 0.3, 'k': [3, 10],
                'scope': 'Two authored sessions per synthetic user; reopen storage before querying; unknown user and app isolation.',
                'workers': args.workers, 'per_invocation_cost_cap_usd': 10}
    protocol['openai_sdk_max_retries']=2
    protocol['harness_sha256']=sha256(Path(__file__).read_bytes()).hexdigest()
    old = args.output / 'protocol.json'
    if old.exists() and read(old) != protocol:
        raise ValueError('Changed protocol; select another output directory')
    write(old, protocol)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(run_case, c, args.output, args.stores): c for c in cases}
        for i, job in enumerate(as_completed(jobs), 1):
            result = job.result()
            print(json.dumps({'case': result['case']['id'], 'completed_in_split': i, 'split': args.split,
                              'status': result['status']}), flush=True)
    print(json.dumps({'split': args.split, 'scenarios': len(cases), 'new_api_cost_usd': round(COST, 6)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['dev', 'test'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stores', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    main(parser.parse_args())
