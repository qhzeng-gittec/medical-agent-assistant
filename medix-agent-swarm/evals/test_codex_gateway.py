import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import codex_gateway
from campaign_gemini_memory import CodexGateway, MODEL
from campaign_gateway import BudgetExceeded


def test_tool_roundtrip_uses_codex_and_preserves_conversation(tmp_path, monkeypatch):
    commands=[]
    prompts=[]
    replies=[dict(content=None,tool_calls=[dict(name='echo',arguments='{"value":"OK"}')]),
             dict(content='OK',tool_calls=[])]
    monkeypatch.setattr(codex_gateway.shutil,'which',lambda name:'codex.cmd')
    def run(command, **kwargs):
        commands.append(command)
        prompts.append(kwargs['input'])
        Path(command[command.index('--output-last-message')+1]).write_text(json.dumps(replies.pop(0)),encoding='utf-8')
        return SimpleNamespace(returncode=0,stderr='')
    monkeypatch.setattr(codex_gateway.subprocess,'run',run)
    tools=[dict(type='function',function=dict(name='echo',parameters=dict(type='object')))]
    messages=[dict(role='system',content='Original medical prompt'),dict(role='user',content='Call echo')]
    trace=[]
    first=codex_gateway.request(tmp_path,dict(model=MODEL,messages=messages,tools=tools,tool_choice='required'),trace,'probe')
    call=first['choices'][0]['message']['tool_calls'][0]
    assert call['function']==dict(name='echo',arguments='{"value":"OK"}')
    messages += [first['choices'][0]['message'],dict(role='tool',tool_call_id=call['id'],content='OK')]
    final=codex_gateway.request(tmp_path,dict(model=MODEL,messages=messages),trace,'probe')
    assert final['choices'][0]['message']['content']=='OK'
    assert final['choices'][0]['finish_reason']=='stop'
    assert all(c[c.index('--model')+1]=='gpt-5.5' for c in commands)
    assert all('forced_login_method="chatgpt"' in c for c in commands)
    assert 'Original medical prompt' in prompts[0] and call['id'] in prompts[1]
    assert all(e['provider']=='codex_chatgpt' for e in trace)


def test_request_cap_blocks_before_codex(tmp_path, monkeypatch):
    gateway=CodexGateway(tmp_path,max_requests=0)
    monkeypatch.setattr('campaign_gemini_memory.codex_request',lambda *a:pytest.fail('Must not call Codex'))
    with pytest.raises(BudgetExceeded):
        asyncio.run(gateway.chat(MODEL,[],[],'probe'))


def test_transport_failure_is_visible_and_never_retried(tmp_path, monkeypatch):
    monkeypatch.setattr(codex_gateway.shutil,'which',lambda name:'codex.cmd')
    calls=[]
    def run(*args,**kwargs):
        calls.append(1)
        return SimpleNamespace(returncode=1,stderr='authentication failure')
    monkeypatch.setattr(codex_gateway.subprocess,'run',run)
    trace=[]
    with pytest.raises(RuntimeError,match='Codex exited 1'):
        codex_gateway.request(tmp_path,dict(model=MODEL,messages=[]),trace,'probe')
    assert len(calls)==1 and trace[0]['status']=='error'
    assert len(list(tmp_path.glob('codex_gpt55_calls/*/receipt.json')))==1


def test_unknown_tool_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(codex_gateway.shutil,'which',lambda name:'codex.cmd')
    def run(command,**kwargs):
        Path(command[command.index('--output-last-message')+1]).write_text(
            json.dumps(dict(content=None,tool_calls=[dict(name='unavailable',arguments='{}')])),encoding='utf-8')
        return SimpleNamespace(returncode=0,stderr='')
    monkeypatch.setattr(codex_gateway.subprocess,'run',run)
    with pytest.raises(ValueError,match='unavailable'):
        codex_gateway.request(tmp_path,dict(model=MODEL,messages=[]),[],'probe')
