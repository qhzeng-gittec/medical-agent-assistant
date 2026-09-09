import json
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))
import holdout_services


@pytest.mark.parametrize("error_type", [httpx.ConnectError, httpx.ConnectTimeout])
def test_gateway_retries_connection_establishment_with_original_payload(tmp_path, monkeypatch, error_type):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    model = "qwen/qwen3.5-27b"
    (tmp_path / "provider_catalog.json").write_text(json.dumps({model: {"pricing": {"prompt": "0", "completion": "0"}}}))
    calls = []

    def post(url, payload, headers, timeout):
        calls.append(payload)
        if len(calls) < 3:
            raise error_type("connection establishment failed")
        return httpx.Response(200, json={"usage": {"cost": 0}, "choices": []}, request=httpx.Request("POST", url))

    monkeypatch.setattr(holdout_services, "post_with_deadline", post)
    monkeypatch.setattr(holdout_services.time, "sleep", lambda seconds: None)
    gateway = holdout_services.HoldoutGateway(tmp_path)
    payload = {"model": model, "messages": [{"role": "user", "content": "original"}]}
    trace = []
    gateway.request("chat/completions", payload, trace, "supervisor")
    assert calls == [payload, payload, payload]
    assert [event["status"] for event in trace] == ["error", "error", "ok"]
    assert [event["attempt"] for event in trace] == [1, 2, 3]
    assert len({event["request_id"] for event in trace}) == 3


def test_gateway_does_not_retry_uncertain_read_timeout(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    model = "qwen/qwen3.5-27b"
    (tmp_path / "provider_catalog.json").write_text(json.dumps({model: {"pricing": {"prompt": "0", "completion": "0"}}}))

    def post(*args, **kwargs):
        raise httpx.ReadTimeout("response not received")

    monkeypatch.setattr(holdout_services, "post_with_deadline", post)
    gateway = holdout_services.HoldoutGateway(tmp_path)
    trace = []
    with pytest.raises(httpx.ReadTimeout):
        gateway.request("chat/completions", {"model": model}, trace, "supervisor")
    assert len(trace) == 1 and trace[0]["status"] == "error"


def test_gateway_stops_after_three_connection_failures(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    model = "qwen/qwen3.5-27b"
    (tmp_path / "provider_catalog.json").write_text(json.dumps({model: {"pricing": {"prompt": "0", "completion": "0"}}}))

    def post(*args, **kwargs):
        raise httpx.ConnectError("connection unavailable")

    monkeypatch.setattr(holdout_services, "post_with_deadline", post)
    monkeypatch.setattr(holdout_services.time, "sleep", lambda seconds: None)
    trace = []
    with pytest.raises(httpx.ConnectError, match="connection unavailable"):
        holdout_services.HoldoutGateway(tmp_path).request("chat/completions", {"model": model}, trace, "supervisor")
    assert len(trace) == 3
    assert all(event["status"] == "error" for event in trace)
