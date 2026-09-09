import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from memory.long_term import LongTermMemory, LongTermMemoryError


def memory_client():
    client = Mock()
    client.add.return_value = {"results": [{"id": "memory-123", "event": "ADD"}]}
    client.embedding_model.embed.return_value = [1.0, 0.0]
    client.vector_store.search.return_value = [
        SimpleNamespace(id="m1", score=0.8, payload={"data": "用户暂停了晨跑。", "source_session_id": "s1"}),
        SimpleNamespace(id="m2", score=0.7, payload={"data": "用户暂停了晨跑。"}),
        SimpleNamespace(id="m3", score=0.6, payload={"data": "用户经常上夜班。"}),
        SimpleNamespace(id="m4", score=0.2, payload={"data": "低相关信息"}),
    ]
    return client


def test_add_turn_uses_real_roles_and_user_scope():
    client = memory_client()
    memory = LongTermMemory(
        config={"app_id": "medix-test", "threshold": 0.4},
        client=client,
    )

    operation_id = memory.add_session_summary(
        user_id="user-1",
        session_id="session-1",
        question="我经常上夜班，最近睡不好",
        answer="可以先记录一周睡眠时间。",
    )

    assert operation_id == "memory-123"
    call = client.add.call_args.kwargs
    assert call["user_id"] == "user-1"
    assert call["agent_id"] == "medix-test"
    assert "app_id" not in call and "run_id" not in call
    assert [message["role"] for message in call["messages"]] == ["user", "assistant"]
    assert call["metadata"]["type"] == "consultation_event"
    assert call["metadata"]["source_session_id"] == "session-1"
    assert call["metadata"]["user_statement"] == "我经常上夜班，最近睡不好"
    assert call["metadata"]["user_statement_truncated"] is False
    assert "assistant_response" not in call["metadata"]


def test_source_excerpt_is_bounded_and_cannot_be_replaced_by_metadata():
    client = memory_client()
    memory = LongTermMemory(config={}, client=client)
    question = "原话" * 2500
    memory.add_session_summary("user-1", "session-1", question, "答复", metadata={
        "user_statement": "伪造出处", "user_statement_truncated": False,
    })
    metadata = client.add.call_args.kwargs["metadata"]
    assert metadata["user_statement"] == question[:4000]
    assert metadata["user_statement_truncated"] is True


def test_memory_context_preserves_source_truncation_and_recording_time():
    from swarm.supervisor_agent import MedicalSupervisorAgent
    context = MedicalSupervisorAgent._memory_context([{
        "content": "摘要", "timestamp": "2026-09-09T00:00:00Z", "metadata": {
            "user_statement": "原话片段", "user_statement_truncated": True,
        },
    }])
    assert context == [{"memory": "摘要", "recorded_at": "2026-09-09T00:00:00Z",
                        "user_statement": "原话片段", "user_statement_truncated": True}]


def test_search_uses_current_filters_and_deduplicates_content():
    client = memory_client()
    memory = LongTermMemory(
        config={"app_id": "medix-test", "threshold": 0.4},
        client=client,
    )

    results = memory.search_similar_sessions("怎么开始运动", "user-2", limit=4)

    client.embedding_model.embed.assert_called_once_with("怎么开始运动", "search")
    client.search.assert_not_called()
    call = client.vector_store.search.call_args.kwargs
    assert call == {
        "query": "怎么开始运动",
        "vectors": [1.0, 0.0],
        "filters": {"user_id": "user-2", "agent_id": "medix-test"},
        "top_k": 4,
    }
    assert [item["memory_id"] for item in results] == ["m1", "m3"]


def test_enabled_mem0_failures_are_not_silently_hidden():
    client = memory_client()
    client.add.side_effect = TimeoutError("write timeout")
    client.embedding_model.embed.side_effect = TimeoutError("search timeout")
    memory = LongTermMemory(config={}, client=client)

    with pytest.raises(LongTermMemoryError, match="search timeout"):
        memory.search_similar_sessions("query", "user-3")
    with pytest.raises(LongTermMemoryError, match="write timeout"):
        memory.add_session_summary("user-3", "session-3", "question", "answer")


def test_missing_api_key_disables_optional_memory(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("MEM0_API_KEY", "unused-cloud-key")

    memory = LongTermMemory(config={})

    assert memory.enabled is False
    assert memory.disabled_reason == "OPENROUTER_API_KEY is not configured"
    memory.close()


@pytest.mark.parametrize("query,user_id,limit", [("", "u", 3), ("q", "", 3), ("q", "u", 0)])
def test_invalid_search_scope_rejected_before_api(query, user_id, limit):
    client = memory_client()
    memory = LongTermMemory(config={}, client=client)
    with pytest.raises(ValueError):
        memory.search_similar_sessions(query, user_id, limit)
    client.embedding_model.embed.assert_not_called()


def test_closing_injected_client_does_not_close_callers_resources():
    client = memory_client()
    memory = LongTermMemory(config={}, client=client)
    memory.close()
    memory.close()
    assert memory.enabled is False
    client.close.assert_not_called()


def test_real_local_storage_restart_user_app_isolation_and_thread_execution(tmp_path, monkeypatch):
    """Real Mem0/Qdrant/SQLite; network model outputs are deterministic test doubles."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "unit-test-not-a-real-key")
    monkeypatch.delenv("MEM0_LOCAL_PATH", raising=False)
    config = {"storage_path": str(tmp_path / "mem0"), "app_id": "app-a", "embedding_dims": 2}
    memory = LongTermMemory(config=config)
    from mem0.embeddings.openai import OpenAIEmbedding
    from mem0.llms.openai import OpenAILLM

    monkeypatch.setattr(OpenAIEmbedding, "embed", lambda *args, **kwargs: [1.0, 0.0])
    monkeypatch.setattr(OpenAIEmbedding, "embed_batch", lambda self, texts, *args: [[1.0, 0.0] for _ in texts])
    monkeypatch.setattr(OpenAILLM, "generate_response", lambda *args, **kwargs: json.dumps(
        {"memory": [{"text": "用户从九月开始上夜班。", "attributed_to": "user"}]}, ensure_ascii=False))

    try:
        assert memory.client.embedding_model.config.model == "qwen/qwen3-embedding-8b"
        assert str(memory.client.embedding_model.client.base_url).startswith("https://openrouter.ai/")
        assert memory.client.llm.config.model == "qwen/qwen3.5-27b"
        assert "identify who each event concerns" in memory.client.custom_instructions
        memory_id = asyncio.run(asyncio.to_thread(
            memory.add_session_summary, "user-a", "session-one", "九月起上夜班。", "收到。"))
        assert memory_id
    finally:
        memory.close()

    reopened = LongTermMemory(config=config)
    try:
        hits = asyncio.run(asyncio.to_thread(reopened.search_similar_sessions, "工作作息", "user-a"))
        assert [hit["memory_id"] for hit in hits] == [memory_id]
        assert hits[0]["metadata"]["source_session_id"] == "session-one"
        assert hits[0]["metadata"]["user_statement"] == "九月起上夜班。"
        assert hits[0]["metadata"]["user_statement_truncated"] is False
        assert reopened.search_similar_sessions("工作作息", "user-b") == []
    finally:
        reopened.close()

    other_app = LongTermMemory(config={**config, "app_id": "app-b"})
    try:
        assert other_app.search_similar_sessions("工作作息", "user-a") == []
    finally:
        other_app.close()
    assert (tmp_path / "mem0/history.sqlite3").exists()
