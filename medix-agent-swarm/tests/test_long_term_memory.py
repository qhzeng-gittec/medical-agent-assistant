import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from memory.long_term import LongTermMemory, LongTermMemoryError


class Mem0ClientStub:
    def __init__(self):
        self.add_calls = []
        self.search_calls = []

    def add(self, **kwargs):
        self.add_calls.append(kwargs)
        return {"status": "PENDING", "event_id": "event-123"}

    def search(self, **kwargs):
        self.search_calls.append(kwargs)
        return {
            "results": [
                {"id": "m1", "memory": "用户之前因膝盖不适停止晨跑。", "score": 0.8},
                {"id": "m2", "memory": "用户之前因膝盖不适停止晨跑。", "score": 0.7},
                {"id": "m3", "memory": "用户经常上夜班。", "score": 0.6},
            ]
        }


class FailingClient:
    def add(self, **kwargs):
        raise TimeoutError("write timeout")

    def search(self, **kwargs):
        raise TimeoutError("search timeout")


def test_add_turn_uses_real_roles_and_user_scope():
    client = Mem0ClientStub()
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

    assert operation_id == "event-123"
    call = client.add_calls[0]
    assert call["user_id"] == "user-1"
    assert call["app_id"] == "medix-test"
    assert call["run_id"] == "session-1"
    assert [message["role"] for message in call["messages"]] == ["user", "assistant"]
    assert call["metadata"]["type"] == "consultation_event"
    assert call["metadata"]["user_statement"] == "我经常上夜班，最近睡不好"
    assert call["metadata"]["assistant_response"] == "可以先记录一周睡眠时间。"
    assert "custom_instructions" in call


def test_search_uses_current_filters_and_deduplicates_content():
    client = Mem0ClientStub()
    memory = LongTermMemory(
        config={"app_id": "medix-test", "threshold": 0.4},
        client=client,
    )

    results = memory.search_similar_sessions("怎么开始运动", "user-2", limit=3)

    call = client.search_calls[0]
    assert call == {
        "query": "怎么开始运动",
        "filters": {"AND": [{"user_id": "user-2"}, {"app_id": "medix-test"}]},
        "top_k": 3,
        "threshold": 0.4,
    }
    assert [item["memory_id"] for item in results] == ["m1", "m3"]


def test_enabled_mem0_failures_are_not_silently_hidden():
    memory = LongTermMemory(config={}, client=FailingClient())

    with pytest.raises(LongTermMemoryError, match="search timeout"):
        memory.search_similar_sessions("query", "user-3")
    with pytest.raises(LongTermMemoryError, match="write timeout"):
        memory.add_session_summary("user-3", "session-3", "question", "answer")


def test_missing_api_key_disables_optional_memory(monkeypatch):
    monkeypatch.delenv("MEM0_API_KEY", raising=False)

    memory = LongTermMemory(config={})

    assert memory.enabled is False
    assert memory.disabled_reason == "MEM0_API_KEY is not configured"
