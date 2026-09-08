import asyncio
import json
import threading

import numpy as np
import pytest

from campaign_gateway import ApiAccessBlocked, Gateway, ModelAdapter
from campaign_rag import LocalRAG
from campaign_run import DATA, parse_json_answer, patient_reply
from campaign_grade import protocol_checks, validate_verdict
from campaign_regrade import anonymous_profile, numbered_dialogue, regrade, validate_ids


class FakeGateway:
    def __init__(self, root, response=None):
        self.root, self.response = root, response
        self.calls = []

    def request(self, endpoint, payload, trace, role):
        self.calls.append(payload)
        return {"data": [{"index": i, "embedding": [1.0, float(i)]}
                         for i in range(len(payload["input"]))]}

    async def chat(self, *args, **kwargs):
        return self.response


def test_dataset_has_patient_tasks_and_no_exam_options():
    cases = [json.loads(s) for s in (DATA / "cases.private.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(cases) == len({c["case_id"] for c in cases}) == 24
    assert all(c["criteria"] and c["opening"] for c in cases)
    assert not any("selected_options" in c["opening"] for c in cases)


def test_real_vector_search_path_uses_query_embedding_and_metadata_filter(tmp_path):
    path = tmp_path / "corpus.jsonl"
    docs = [{"id": "a", "content": "A", "metadata": {"type": "patient_information"}},
            {"id": "b", "content": "B", "metadata": {"type": "clinical_guideline"}}]
    path.write_text("\n".join(json.dumps(d) for d in docs), encoding="utf-8")
    gateway = FakeGateway(tmp_path)
    rag = LocalRAG(gateway, path, [])
    results = rag.search("query", top_k=1, filter_type="clinical_guideline")
    assert results[0]["id"] == "b"
    assert results[0]["score"] == pytest.approx(1 / np.sqrt(2))
    assert len(gateway.calls) == 2
    with pytest.raises(ValueError):
        rag.search("query", top_k=1.5)


def test_truncated_model_response_is_not_success(tmp_path):
    gateway = FakeGateway(tmp_path, {"choices": [{"finish_reason": "length", "message": {"content": None}}]})
    client = ModelAdapter(gateway, "test", [], "supervisor")
    with pytest.raises(RuntimeError, match="Incomplete"):
        asyncio.run(client.chat_with_tools([]))


def test_preserves_provider_reasoning_details_for_tool_roundtrip(tmp_path):
    message = {"role": "assistant", "content": None,
               "reasoning_details": [{"type": "reasoning.encrypted", "data": "opaque-signature"}],
               "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "echo", "arguments": "{}"}}]}
    gateway = FakeGateway(tmp_path, {"choices": [{"finish_reason": "tool_calls", "message": message}]})
    response = asyncio.run(ModelAdapter(gateway, "test", [], "worker").chat_with_tools([]))
    assert response.wire_message["reasoning_details"] == message["reasoning_details"]
    assert response.tool_calls[0].id == "c1"


def test_patient_selector_cannot_invent_fact_values(tmp_path):
    gateway = FakeGateway(tmp_path, {"choices": [{"finish_reason": "stop", "message": {
        "content": '{"needs_reply":true,"fact_ids":[99]}'}}]})
    with pytest.raises(ValueError, match="invented"):
        asyncio.run(patient_reply(gateway, {"patient_facts": ["only fact"]}, "question", [], []))


def test_parse_json_does_not_silently_accept_prose():
    assert parse_json_answer('```json\n{"a":1}\n```') == {"a": 1}
    assert parse_json_answer('```json{"a":1}\n```') == {"a": 1}
    assert parse_json_answer('```\n{"a":1}\n```') == {"a": 1}
    with pytest.raises(json.JSONDecodeError):
        parse_json_answer('Here is the answer: {"a":1}')


def test_judge_cannot_invent_answer_evidence():
    verdict = {"checks": [{"id": "c1", "verdict": "pass", "reason": "reason",
                           "evidence": [{"turn": 1, "quote": "invented"}]}]}
    with pytest.raises(ValueError, match="quote"):
        validate_verdict(verdict, [{"id": "c1"}], [{"answer": "actual"}])


def test_protocol_checks_detect_missing_body_and_orphan_results():
    messages = [{"role": "tool", "tool_call_id": "missing", "content": json.dumps({
        "documents": [{"block_id": "b1", "reference": {"tool_call_id": "old"}}]})}]
    result = protocol_checks([{"event": "api_request", "role": "supervisor", "payload": {"messages": messages}}])
    assert {row["type"] for row in result["failures"]} == {"orphan_tool_result", "reference_without_visible_body"}


def test_id_grading_extracts_real_answer_lines_and_rejects_invention():
    turns = [{"user": "question", "answer": "Line one\n\nLine three", "profile_after": {},
              "session_id": "s", "user_id": "u"}]
    dialogue, lines = numbered_dialogue(turns)
    assert dialogue[0]["answer_lines"][1]["id"] == "T1L3"
    content = '{"checks":[{"id":"c1","verdict":"fail","reason":"reason","evidence_ids":["T1L3"]}]}'
    assert validate_ids(content, [{"id": "c1"}], lines)[0]["evidence"] == [{"turn": 1, "quote": "Line three"}]
    with pytest.raises(ValueError, match="invented"):
        validate_ids(content.replace("T1L3", "T1L99"), [{"id": "c1"}], lines)


def test_judge_profile_does_not_leak_target_model_in_session_id():
    profile = {"allergies": [{"value": "penicillin", "source_session_id": "minimax-model-run-id"}]}
    assert anonymous_profile(profile) == {"allergies": [{"value": "penicillin"}]}


def test_protocol_audit_catches_tool_finish_without_any_tool_call():
    event = {"event": "api_request", "role": "research_agent", "request_id": "req1", "payload": {"messages": []},
             "response": {"choices": [{"finish_reason": "tool_calls", "message": {"content": None}}]}}
    assert protocol_checks([event])["response_contract_failures"] == [
        {"type": "tool_calls_finish_without_calls", "role": "research_agent", "request_id": "req1"}]


def test_gateway_does_not_send_more_requests_after_billing_pause():
    gateway = object.__new__(Gateway)
    gateway.catalog = {"test": {"pricing": {"prompt": "0", "completion": "0"}}}
    gateway.lock = threading.RLock()
    gateway.halt_reason = "OpenRouter HTTP 402"
    trace = []
    with pytest.raises(ApiAccessBlocked, match="402"):
        gateway.request("chat/completions", {"model": "test"}, trace, "judge")
    assert trace == []


def test_serial_grader_skips_reserved_future_job_and_valid_grade(tmp_path):
    result = {"run_id": "case"}
    name = "case__judge_test.json"
    original = tmp_path / "grades_v2" / name
    retry = tmp_path / "grades_v2_retries" / name
    original.parent.mkdir()
    retry.parent.mkdir()
    original.write_text('{"status":"judge_error"}', encoding="utf-8")
    retry.write_text('{"status":"running","claim_owner":"worker-b"}', encoding="utf-8")
    gateway = FakeGateway(tmp_path)
    asyncio.run(regrade(gateway, result, {}, "test", retry_errors=True))
    asyncio.run(regrade(gateway, result, {}, "test", retry_errors=True, claim_owner="worker-c"))
    assert json.loads(retry.read_text(encoding="utf-8"))["status"] == "running"
    original.write_text('{"status":"graded"}', encoding="utf-8")
    asyncio.run(regrade(gateway, result, {}, "test", retry_errors=True, claim_owner="worker-b"))
    assert gateway.calls == []
