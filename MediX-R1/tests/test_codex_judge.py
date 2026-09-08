import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import common.medical_teacher as medical_teacher
import common.codex_judge as codex_judge


def test_cli_uses_chatgpt_account_model_schema_and_isolated_directory(tmp_path):
    output=tmp_path/'answer.json'
    schema=dict(type='object',properties=dict(ok=dict(type='boolean')),required=['ok'],additionalProperties=False)
    def run(command,**kwargs):
        assert command[command.index('--model')+1]=='gpt-5.5'
        assert 'forced_login_method="chatgpt"' in command
        assert command[command.index('--sandbox')+1]=='read-only'
        assert Path(kwargs['cwd']).is_dir() and Path(kwargs['cwd'])!=tmp_path
        assert json.loads(Path(command[command.index('--output-schema')+1]).read_text())==schema
        output.write_text('{"ok":true}',encoding='utf-8')
        return SimpleNamespace(returncode=0,stderr='',stdout=json.dumps(dict(type='turn.completed',usage=dict(input_tokens=10,output_tokens=2))))
    with patch.object(medical_teacher.shutil,'which',return_value='codex.cmd'),patch.object(medical_teacher.subprocess,'run',side_effect=run):
        result=codex_judge.judge_request(dict(rubric='Fixture',payload={},schema=schema),output)
    assert result['requested_model']=='gpt-5.5'
    assert result['provider']=='codex_chatgpt'
    assert result['usage']['total_tokens']==12
    assert result['thinking_level']=='medium'


def test_failed_cli_cannot_reuse_an_old_output(tmp_path):
    output=tmp_path/'answer.json'
    output.write_text('{"ok":true}')
    with patch.object(medical_teacher.shutil,'which',return_value='codex.cmd'),patch.object(medical_teacher.subprocess,'run',return_value=SimpleNamespace(returncode=1,stderr='Failed',stdout='')) as call:
        with pytest.raises(RuntimeError,match='exited 1'):
            medical_teacher.call_teacher('Fixture',[],{},output)
        call.assert_called_once()
    assert output.with_suffix('.log').read_text()=='Failed'


def test_rl_cache_separates_historical_judge(tmp_path):
    from evaluation.rl_codex_judge import CodexJudge
    (tmp_path/'calls').mkdir()
    (tmp_path/'calls/gemini.json').write_text('{"requested_model":"gemini-3.5-flash"}')
    judge=CodexJudge(tmp_path,1)
    assert judge.root==tmp_path/'codex_gpt55'
    assert judge.usage()['responses']==0
