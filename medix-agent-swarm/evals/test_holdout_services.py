"""Calibration-only service contract tests; no held-out cases or network calls."""

import copy
from types import SimpleNamespace

import pytest

from holdout_services import FixtureKnowledge, FixtureMemory, LiveMemory, RecordingMemoryClient, SERVICE_CONFIG


def environment(memory_mode="controlled", rag_mode="controlled"):
    return {"memory_mode": memory_mode, "rag_mode": rag_mode,
            "documents": [{"id": "toy-doc", "content": "PUBLIC_TOY_SOURCE", "metadata": {"source": "fixture"}}],
            "memory_records": [
                {"id": "a", "user_key": "self", "source_role": "user", "content": "USER_SOURCE"},
                {"id": "b", "user_key": "self", "source_role": "assistant", "content": "ASSISTANT_SOURCE"},
                {"id": "c", "user_key": "other", "source_role": "user", "content": "OTHER_USER"}]}


def test_disabled_memory_cannot_enable_from_secret_environment(monkeypatch):
    monkeypatch.setenv("MEM0_API_KEY", "public-toy-not-a-key")
    memory = FixtureMemory(environment("off"), {"self": "one", "other": "two"}, [])
    assert memory.enabled is False
    assert memory.search_similar_sessions("anything", "one") == []
    assert memory.snapshot("one")["write_observation_available"] is False


def test_controlled_identity_and_speaker_reach_product_fields():
    memory = FixtureMemory(environment(), {"self": "one", "other": "two"}, [])
    rows = memory.search_similar_sessions("not keyword matched", "one", 5)
    assert len(rows) == 2
    assert rows[0]["metadata"] == {"user_statement": "USER_SOURCE"}
    assert rows[1]["metadata"] == {"assistant_response": "ASSISTANT_SOURCE"}
    assert memory.search_similar_sessions("anything", "unknown", 5) == []


def test_seed_bootstrap_does_not_consume_event_fixture():
    memory = FixtureMemory(environment(), {"self": "one", "other": "two"}, [])
    memory.seed_replay = True
    assert memory.search_similar_sessions("anything", "one") == []


def test_controlled_snapshots_are_not_successful_persistence():
    memory = FixtureMemory(environment(), {"self": "one", "other": "two"}, [])
    before = copy.deepcopy(memory.records)
    memory.add_session_summary(user_id="one", question="new info", answer="saved")
    assert memory.records == before
    assert memory.snapshot("one")["write_observation_available"] is False


def test_controlled_timeout_remains_observable_before_raise():
    trace = []
    kb = FixtureKnowledge(environment(rag_mode="retrieval_timeout"), trace)
    with pytest.raises(TimeoutError):
        kb.search("PUBLIC_TOY_QUERY")
    assert trace == [{"event": "retrieval", "query": "PUBLIC_TOY_QUERY", "top_k": 5,
                      "filter_type": None, "status": "retrieval_timeout", "documents": []}]


def test_controlled_candidates_preserve_source_and_label_not_semantic():
    trace = []
    kb = FixtureKnowledge(environment(), trace)
    rows = kb.search("an unrelated query")
    assert rows[0]["metadata"] == {"source": "fixture"}
    assert trace[-1]["ranking"] == "authored_order_not_similarity"


def test_integer_float_limit_matches_real_backend_contract():
    kb = FixtureKnowledge(environment(), [])
    assert len(kb.search("toy", top_k=1.0)) == 1
    for invalid in (True, 1.5, 0, 11):
        with pytest.raises(ValueError):
            kb.search("toy", top_k=invalid)


def test_failed_live_write_is_not_retried_or_marked_pending():
    calls = []
    def failed(**kwargs):
        calls.append(kwargs)
        raise TimeoutError("uncertain network outcome")
    owner = SimpleNamespace(sdk=SimpleNamespace(add=failed), pending=[], call=lambda op, params, fn: fn())
    with pytest.raises(TimeoutError):
        RecordingMemoryClient(owner).add(messages=[{"role": "user", "content": "toy"}])
    assert len(calls) == 1 and owner.pending == []


def test_unknown_mem0_receipt_cannot_mean_success():
    memory = LiveMemory.__new__(LiveMemory)
    memory.pending = [{"unexpected": True}]
    memory.trace = []
    with pytest.raises(ValueError, match="Unknown Mem0 write receipt"):
        memory.settle()


def test_async_completion_is_observed_not_guessed():
    memory = LiveMemory.__new__(LiveMemory)
    memory.pending, memory.trace = [{"event_id": "toy-event"}], []
    memory.call = lambda op, params, fn: {"status": "SUCCEEDED"}
    result = memory.settle()
    assert result["status"] == "settled" and result["events"] == 1
    assert memory.pending == []


def test_unlimited_spend_still_has_bounded_requests_and_output():
    assert SERVICE_CONFIG["budget_cap_usd"] is None
    assert SERVICE_CONFIG["request_timeout_seconds"] == 180
    assert SERVICE_CONFIG["target_max_output_tokens"] == 8192
