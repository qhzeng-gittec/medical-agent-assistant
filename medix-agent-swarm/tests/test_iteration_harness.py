import json
import sys
from pathlib import Path
from threading import Lock
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))
from iteration_grade import validated_response
from system_iteration import LocalMemory
from holdout_services import post_with_deadline
import holdout_services
import httpx
import asyncio
import portalocker
import system_iteration


def test_memory_snapshot_uses_scoped_mem0_listing_api():
    memory = LocalMemory.__new__(LocalMemory)
    memory._lock = Lock()
    memory.app_id = "test-app"
    memory.client = Mock()
    memory.client.get_all.return_value = {"results": [{"memory": "user fact"}]}
    assert memory.snapshot("u1")["memories"]["results"] == [{"memory": "user fact"}]
    memory.client.get_all.assert_called_once_with(filters={"user_id": "u1", "agent_id": "test-app"}, top_k=100)


@pytest.mark.parametrize("elapsed,raises", [(1, False), (181, True)])
def test_stream_keepalives_cannot_extend_elapsed_deadline(monkeypatch, elapsed, raises):
    response = Mock()
    response.iter_raw.return_value = iter([b'{"ok":', b'true}'])
    response.status_code, response.headers = 200, {}
    response.request = httpx.Request('POST', 'https://example.test')
    context = Mock(__enter__=Mock(return_value=response), __exit__=Mock(return_value=False))
    monkeypatch.setattr(holdout_services.httpx, 'stream', Mock(return_value=context))
    times = iter([0, 1, elapsed])
    monkeypatch.setattr(holdout_services.time, 'monotonic', lambda: next(times))
    if raises:
        with pytest.raises(TimeoutError, match='elapsed-time'):
            post_with_deadline('https://example.test', {}, {}, 180)
    else:
        assert post_with_deadline('https://example.test', {}, {}, 180).json() == {'ok': True}
    context.__exit__.assert_called_once()


def test_second_case_process_is_rejected_before_any_api_access(tmp_path, monkeypatch):
    for name in ('history', 'evidence', 'consultation'):
        (tmp_path / f'{name}.jsonl').write_text('{"case_id":"case-1"}\n' if name == 'history' else '', encoding='utf-8')
    monkeypatch.setattr(system_iteration, 'DATA', tmp_path)
    monkeypatch.setattr(system_iteration, 'freeze', Mock(return_value={'sources': {}}))
    gateway = Mock(side_effect=AssertionError('Must not connect while the case is locked'))
    monkeypatch.setattr(system_iteration, 'IterationGateway', gateway)
    monkeypatch.setattr(sys, 'argv', ['system_iteration', '--output', str(tmp_path),
                                    '--cases', 'case-1', '--worker-case', 'case-1'])
    with portalocker.Lock(str(tmp_path / 'case-1.lock'), timeout=0):
        with pytest.raises(portalocker.exceptions.LockException):
            asyncio.run(system_iteration.main())
    gateway.assert_not_called()


@pytest.mark.parametrize("evidence", [["t1.answer"], [{"id": "t1.answer"}]])
def test_id_shape_normalization_preserves_verdict_and_resolves_real_evidence(evidence):
    raw = {"checks": [{"id": "c1", "verdict": "pass", "reason": "Observed answer",
                       "requires_durable_write": False, "evidence": evidence}]}
    payload = {"evidence_catalogue": [{"id": "t1.answer", "pointer": "/observations/answer"}],
               "observations": {"answer": "actual answer"}}
    checks, _ = validated_response(json.dumps(raw), [{"id": "c1"}], payload)
    assert checks[0]["verdict"] == "pass"
    assert checks[0]["evidence"][0]["value"] == "actual answer"
    raw["checks"][0]["evidence"] = ["invented-id"]
    with pytest.raises(ValueError, match="observed catalogue"):
        validated_response(json.dumps(raw), [{"id": "c1"}], payload)
