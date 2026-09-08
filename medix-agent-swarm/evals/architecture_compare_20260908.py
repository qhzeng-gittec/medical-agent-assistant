"""Live, frozen architecture comparison; no product source is modified."""
import argparse
import asyncio
import copy
import json
import os
from pathlib import Path
import random
import shutil
import statistics
import sys
import threading
import time
import types

from loguru import logger
import campaign_gateway as wire
from campaign_gateway import Gateway, ModelAdapter, dump, sha
from campaign_run import PROJECT, DATA, build_system, parse_json_answer
from campaign_rag import LocalRAG
from core.evidence_store import EvidenceStore, use_evidence_store
from core.rag_context import compact_rag_result
from memory.patient_profile import PROFILE_UPDATE_TOOL
from swarm.supervisor_agent import MedicalSupervisorAgent

ROOT = Path(__file__).parent / 'results/architecture_compare_20260908_v2'
CASES = Path(__file__).with_name('architecture_cases_20260908.json')
TARGET = 'minimax/minimax-m2.5'
JUDGE = 'qwen/qwen3.5-27b'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


class BenchGateway(Gateway):
    def __init__(self, root, limit=.2):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.key = os.environ['OPENROUTER_API_KEY']
        self.lock = threading.RLock()
        self.limit, self.spent, self.reserved, self.halt_reason = limit, 0., 0., None
        self.ledger = root / 'api_ledger.jsonl'
        if self.ledger.exists():
            raise RuntimeError('Run ledger already exists; do not silently rerun')
        self.catalog = read(ROOT / 'models_snapshot.json')

    async def chat(self, model, messages, trace, role, tools=None, tool_choice='auto', max_tokens=4096):
        payload = {'model': model, 'messages': messages, 'max_tokens': max_tokens,
                   'temperature': .2, 'reasoning': {'effort': 'low'},
                   'provider': {'require_parameters': True}}
        if role == 'judge':
            payload['reasoning'] = {'enabled': False}
        if tools:
            payload.update(tools=tools, tool_choice=tool_choice)
        return await asyncio.to_thread(self.request, 'chat/completions', payload, trace, role)


class SingleAgent(MedicalSupervisorAgent):
    """Same state/safety harness; all original Skills available to one model loop."""
    def get_system_prompt(self):
        return super().get_system_prompt() + (
            '\n本组由你独立承担症状分析、风险识别、生活方式咨询及医学证据核查，直接调用底层工具。'
            '不存在可委派的子Agent。你可使用全部专业工具；检查来源适用性，不虚构来源或评级。'
            '按任务需要选择工具，不必机械遍历；已有资料足够时直接完成。')

    def _available_tools(self, round_number, user_id=None):
        if round_number == 1:
            self.raw_calls = 0
        return (self.raw_tools if self.raw_calls < 24 else []) + ([PROFILE_UPDATE_TOOL] if user_id else [])

    async def _execute_calls(self, calls, question, context, prior_records, round_number,
                             allowed_names, user_id=None, session_id=None, profile_updates=None):
        records = {}
        for i, call in enumerate(calls):
            if call.name == 'update_patient_profile':
                records[i] = await self._execute_memory_call(call, question, context, round_number,
                    allowed_names, user_id, session_id, profile_updates)
        async def raw(call):
            if call.name not in allowed_names or self.raw_calls >= 24:
                return self._error_record(call, round_number, 'PolicyDenied', 'Tool unavailable or budget exhausted')
            self.raw_calls += 1
            started = time.monotonic()
            result = await self.raw_owners[call.name].execute_tool(call.name, call.arguments)
            self.bench_trace.append({'event': 'single_tool', 'name': call.name,
                'started': started, 'ended': time.monotonic(), 'result': copy.deepcopy(result)})
            result = compact_rag_result(result, self.raw_messages, call.id)
            self.raw_messages.append({'role': 'tool', 'tool_call_id': call.id, 'content': json.dumps(result, ensure_ascii=False)})
            return {'tool_call_id': call.id, 'tool_name': call.name, 'agent_id': 'single_agent',
                    'round': round_number, 'success': result.get('success', True), 'result': result}
        indices = [i for i in range(len(calls)) if i not in records]
        records.update(zip(indices, await asyncio.gather(*(raw(calls[i]) for i in indices))))
        return [records[i] for i in range(len(calls))]

    async def process(self, *args, **kwargs):
        self.raw_messages = []
        with use_evidence_store(EvidenceStore()):
            return await super().process(*args, **kwargs)


def system(gateway, trace, runroot, arm):
    shutil.copy2(ROOT / 'corpus_vectors.json', runroot / 'corpus_vectors.json')
    kb = LocalRAG(gateway, DATA / 'corpus.jsonl', trace)
    supervisor, profile = build_system(gateway, TARGET, trace, kb, runroot / 'profiles')
    for name, worker in supervisor.workers.items():
        original = worker.process
        async def timed(data, original=original, name=name):
            event = {'event': 'worker_span', 'name': name, 'started': time.monotonic()}
            trace.append(event)
            try:
                return await original(data)
            finally:
                event['ended'] = time.monotonic()
        worker.process = timed
    original_batch = supervisor._execute_calls
    async def batch(*args, **kwargs):
        event = {'event': 'dispatch_batch', 'workers': [c.name for c in args[0] if c.name in supervisor.workers],
                 'started': time.monotonic()}
        trace.append(event)
        try:
            return await original_batch(*args, **kwargs)
        finally:
            event['ended'] = time.monotonic()
    supervisor._execute_calls = batch
    if arm == 'single':
        supervisor.__class__ = SingleAgent
        del supervisor._execute_calls
        supervisor.max_rounds = 10
        supervisor.raw_tools = [tool for worker in supervisor.workers.values() for tool in worker.get_tools_for_llm()]
        supervisor.raw_owners = {name: worker for worker in supervisor.workers.values() for name in worker.skill_registry.skills}
        assert len(supervisor.raw_tools) == len(supervisor.raw_owners) == 7
        supervisor.bench_trace = trace
    elif arm == 'serial':
        original_call = supervisor._execute_call
        lock = asyncio.Lock()
        async def serial(*args, **kwargs):
            async with lock:
                return await original_call(*args, **kwargs)
        supervisor._execute_call = serial
    return supervisor, profile


def metrics(trace):
    requests = [r for r in trace if r['event'] == 'api_request']
    usage = [r.get('response', {}).get('usage', {}) for r in requests if r['role'] != 'embedding']
    batches = [r for r in trace if r['event'] == 'dispatch_batch']
    return {'llm_requests': sum(r['role'] != 'embedding' for r in requests),
        'prompt_tokens': sum(r.get('prompt_tokens', 0) for r in usage),
        'completion_tokens': sum(r.get('completion_tokens', 0) for r in usage),
        'cost_usd': sum(r['cost_usd'] for r in requests),
        'retrievals': sum(r['event'] == 'retrieval' for r in trace),
        'workers_called': sum(r['event'] == 'worker_span' for r in trace),
        'multi_worker_batches': sum(len(r['workers']) > 1 for r in batches),
        'provider_errors': sum(r.get('status') != 'ok' for r in requests),
        'dispatch_seconds': sum(r['ended'] - r['started'] for r in batches)}


async def execute(case_id, arm, repetition, scheduler=False):
    freeze = read(ROOT / 'protocol.json')
    assert freeze['cases_sha256'] == sha(CASES)
    assert all(sha(PROJECT / name) == digest for name, digest in freeze['product_sha256'].items())
    assert freeze['harness_sha256'] == sha(Path(__file__))
    case = next(c for c in read(CASES)['scheduler' if scheduler else 'cases'] if c['id'] == case_id)
    runid = f'{case_id}_{arm}_r{repetition}'
    runroot = ROOT / ('scheduler' if scheduler else 'runs') / runid
    gateway = BenchGateway(runroot)
    trace, turns = [], []
    record = {'run_id': runid, 'case_id': case_id, 'arm': arm, 'repetition': repetition,
              'category': case.get('category', 'fixed_dispatch'), 'status': 'started', 'turns': turns}
    try:
        supervisor, profile = system(gateway, trace, runroot, arm)
        started = time.monotonic()
        if scheduler:
            from core import ToolCall
            calls = [ToolCall(f'fixed-{i}', name, {'task': task}) for i, (name, task) in enumerate(case['tasks'])]
            result = await supervisor._execute_calls(calls, case['question'], {}, [], 1, set(supervisor.workers))
            record['records'] = result
        else:
            for text in case['turns']:
                turnstart = time.monotonic()
                result = await asyncio.wait_for(supervisor.process(text, user_id=runid, session_id=runid), 240)
                turns.append({'user': text, 'answer': result['answer'], 'profile_after': profile.get_context(runid),
                    'seconds': time.monotonic()-turnstart, 'result': result})
        record.update(status='completed', seconds=time.monotonic()-started)
    except Exception as error:
        record.update(status='error', error=f'{type(error).__name__}: {error}'.replace(gateway.key, '[REDACTED]'))
        raise
    finally:
        record['metrics'] = metrics(trace)
        dump(runroot / 'trace.json', trace)
        dump(runroot / 'result.json', record)
        print(json.dumps({k: record.get(k) for k in ('run_id','status','seconds','error','metrics')}, ensure_ascii=False), flush=True)


async def grade(runid):
    result = read(ROOT / 'runs' / runid / 'result.json')
    if result['status'] != 'completed':
        raise RuntimeError('Cannot grade incomplete execution as answer quality')
    case = next(c for c in read(CASES)['cases'] if c['id'] == result['case_id'])
    root = ROOT / 'grades' / runid
    gateway = BenchGateway(root, .04)
    trace = []
    criteria = case['criteria']
    messages = [{'role':'system','content':
        '你是匿名任务完成评审。只按逐项要求评价给定对话与最终档案，不奖励篇幅、标题或形式。'
        '医学参考以附带知识库为准。工具是否实际保存以档案状态为准。不得从回答风格猜测系统架构。'
        '对每项返回pass/fail/uncertain与简短中文理由。仅返回JSON：'
        '{"checks":[{"id":0,"verdict":"pass","reason":"..."}]}，id从0开始覆盖全部要求。'},
        {'role':'user','content':json.dumps({'requirements':dict(enumerate(criteria)),
          'dialogue':[{'user':t['user'],'answer':t['answer'],'profile_after':t['profile_after']} for t in result['turns']],
          'reference_corpus':[json.loads(x) for x in (DATA/'corpus.jsonl').read_text(encoding='utf-8').splitlines()]},ensure_ascii=False)}]
    data = await gateway.chat(JUDGE, messages, trace, 'judge', max_tokens=4096)
    choice = data['choices'][0]
    if choice['finish_reason'] != 'stop':
        raise RuntimeError('Judge output incomplete')
    obj = parse_json_answer(choice['message']['content'])
    checks = sorted(obj['checks'], key=lambda r:r['id'])
    assert [r['id'] for r in checks] == list(range(len(criteria)))
    assert all(r['verdict'] in {'pass','fail','uncertain'} for r in checks)
    dump(root/'trace.json',trace)
    dump(root/'result.json',{'run_id':runid,'judge':JUDGE,'checks':checks,
         'all_pass':all(r['verdict']=='pass' for r in checks),
         'criteria_pass':sum(r['verdict']=='pass' for r in checks),'criteria_count':len(criteria)})
    print(json.dumps({'graded':runid,'all_pass':all(r['verdict']=='pass' for r in checks)},ensure_ascii=False),flush=True)


async def prepare():
    if (ROOT/'protocol.json').exists():
        raise RuntimeError('Protocol already frozen')
    wire.MODELS[:] = [TARGET,JUDGE]
    gateway = Gateway(ROOT, .2)
    existing = Path(__file__).parent/'results/full_system_v1/corpus_vectors.json'
    shutil.copy2(existing, ROOT/'corpus_vectors.json')
    LocalRAG(gateway, DATA/'corpus.jsonl', [])
    trace=[]
    data=await gateway.chat(TARGET,[{'role':'user','content':'请只回复OK。'}],trace,'probe')
    dump(ROOT/'probe.json',trace)
    if not data['choices'][0]['message'].get('content'):
        raise RuntimeError('Empty preflight answer')
    products={str(p.relative_to(PROJECT)):sha(p) for folder in ('agents','core','swarm','memory','constraints','validation','.claude')
              for p in (PROJECT/folder).rglob('*') if p.suffix in {'.py','.yaml','.md'} and '__pycache__' not in p.parts}
    dump(ROOT/'protocol.json',{'target':TARGET,'judge':JUDGE,'cases_sha256':sha(CASES),
        'harness_sha256':sha(Path(__file__)),'product_sha256':products,'temperature':.2,'max_tokens':4096,
        'cases':8,'repetitions':2,'arms':['single','swarm'],'scheduler_repetitions':3,
        'single_budget':'10 model rounds, 24 raw-tool calls/turn; parallel tool calls allowed',
        'swarm_budget':'unchanged product: 4 supervisor rounds, 2 raw tools per worker invocation',
        'services':'real model and query embeddings; existing 15-document exact cosine index; no live Mem0 or external web service',
        'scheduler':'same three real worker tasks; only worker concurrency changes; no supervisor planning/synthesis in timing',
        'budget_upper_bound_usd':9.5,'grading':'anonymous single automatic judge plus manual audit; not clinical accuracy',
        'selection':'8 authored development cases; 2 per simple/compound/risk/profile; not independent clinical generalization'})
    print('PREFLIGHT_OK',flush=True)


if __name__=='__main__':
    logger.remove()
    logger.add(sys.stderr,level='ERROR',diagnose=False,backtrace=False)
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['prepare','run','scheduler','grade'])
    parser.add_argument('--case');parser.add_argument('--arm');parser.add_argument('--rep',type=int,default=1)
    parser.add_argument('--runid')
    args=parser.parse_args()
    if args.mode=='prepare': asyncio.run(prepare())
    elif args.mode=='grade': asyncio.run(grade(args.runid))
    else: asyncio.run(execute(args.case,args.arm,args.rep,args.mode=='scheduler'))
