import asyncio
import json

import httpx
import pytest

from campaign_gemini_memory import GoogleGateway, MODEL
from campaign_gateway import ApiAccessBlocked, BudgetExceeded, ModelAdapter


@pytest.fixture
def gateway(tmp_path, monkeypatch):
    monkeypatch.setenv("MEDIX_GEMINI_API_KEY", "synthetic-test-credential")
    monkeypatch.setenv("MEDIX_GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai")
    monkeypatch.setenv("MEDIX_GEMINI_MODEL", MODEL)
    return GoogleGateway(tmp_path)


def response(payload, status=200):
    return httpx.Response(status, json=payload, request=httpx.Request("POST", "https://example.test/chat/completions"))


def test_preserves_nested_signature_and_original_medical_messages(gateway, monkeypatch):
    signature = {"google": {"thought_signature": "opaque-test-signature"}}
    tool = {"id": "call1", "type": "function", "function": {"name": "echo", "arguments": '{"value":"OK"}'},
            "extra_content": signature}
    received = []

    def post(url, **kwargs):
        received.append(kwargs["json"])
        return response({"choices": [{"finish_reason": "tool_calls", "message": {
            "role": "assistant", "content": None, "tool_calls": [tool]}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}})

    monkeypatch.setattr("campaign_gemini_memory.httpx.post", post)
    adapter = ModelAdapter(gateway, MODEL, [], "supervisor")
    messages = [{"role": "system", "content": "Unchanged medical system prompt"}, {"role": "user", "content": "question"}]
    result = asyncio.run(adapter.chat_with_tools(messages, [{"type": "function", "function": {"name": "echo"}}]))
    assert result.wire_message["tool_calls"][0]["extra_content"] == signature
    assert received[0]["messages"] == messages
    assert received[0]["reasoning_effort"] == "low"
    assert "response_format" not in received[0] and "provider" not in received[0]
    assert gateway.spent == pytest.approx(.025 + .00015 + .00018)
    assert gateway.reserved == pytest.approx(0)


def test_budget_refuses_before_network(gateway, monkeypatch):
    gateway.limit = gateway.spent
    monkeypatch.setattr("campaign_gemini_memory.httpx.post", lambda *a, **k: pytest.fail("Network must not run"))
    with pytest.raises(BudgetExceeded):
        asyncio.run(gateway.chat(MODEL, [{"role": "user", "content": "test"}], [], "probe"))
    assert gateway.requests == 0


def test_cost_includes_thinking_when_total_exceeds_visible_completion(gateway, monkeypatch):
    monkeypatch.setattr("campaign_gemini_memory.httpx.post", lambda *a, **k: response({
        "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "OK"}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 620}}))
    trace = []
    asyncio.run(gateway.chat(MODEL, [], trace, "probe"))
    assert trace[0]["cost_usd"] == pytest.approx(100 * 1.5e-6 + 520 * 9e-6)
    assert trace[0]["cost_status"] == "estimated_standard_list_price_not_invoice"


def test_http_failure_is_redacted_retains_reserve_and_stops_followups(gateway, monkeypatch):
    monkeypatch.setattr("campaign_gemini_memory.httpx.post", lambda *a, **k: response({"error": gateway.key}, 429))
    trace = []
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(gateway.chat(MODEL, [], trace, "probe"))
    assert gateway.key not in json.dumps(trace)
    assert trace[0]["cost_status"] == "unknown_cost_reserved"
    assert gateway.spent > .025
    with pytest.raises(ApiAccessBlocked):
        asyncio.run(gateway.chat(MODEL, [], [], "probe"))
    assert gateway.requests == 1


def test_unresolved_request_cannot_be_resumed(gateway):
    gateway.append({"request_id": "unfinished", "status": "started", "charged_or_reserved_usd": .1})
    with pytest.raises(RuntimeError, match="Unresolved"):
        GoogleGateway(gateway.ledger.parent)


def test_wrong_host_is_rejected(gateway, monkeypatch):
    monkeypatch.setenv("MEDIX_GEMINI_BASE_URL", "https://wrong-host.test")
    with pytest.raises(ValueError, match="endpoint"):
        GoogleGateway(gateway.ledger.parent)
