import asyncio
from types import SimpleNamespace
from architecture_compare_20260908 import SingleAgent
from core import ToolCall
from core.rag_context import document_block


def test_single_repeated_document_preserves_tool_call_reference():
    async def execute(name, arguments):
        return {'success': True, 'documents': [document_block({'id':'D1','content':'evidence',
            'metadata':{'source':'test'},'score':1.0})]}
    agent=SingleAgent.__new__(SingleAgent)
    agent.raw_calls=0
    agent.raw_messages=[]
    agent.bench_trace=[]
    agent.raw_owners={'lookup':SimpleNamespace(execute_tool=execute)}
    calls=[ToolCall('first','lookup',{}),ToolCall('second','lookup',{})]
    records=asyncio.run(agent._execute_calls(calls,'question',{},[],1,{'lookup'}))
    assert len(records)==2 and all(r['success'] for r in records)
    assert agent.raw_messages[0]['tool_call_id']=='first'
    assert 'reference' in records[1]['result']['documents'][0]


def test_single_budget_denies_execution():
    agent=SingleAgent.__new__(SingleAgent)
    agent.raw_calls=24
    agent.raw_messages=[]
    agent.bench_trace=[]
    agent.workers={}
    agent.raw_owners={}
    records=asyncio.run(agent._execute_calls([ToolCall('blocked','lookup',{})],'question',{},[],1,{'lookup'}))
    assert not records[0]['success']
    assert records[0]['result']['error_type']=='PolicyDenied'
    assert not agent.bench_trace
