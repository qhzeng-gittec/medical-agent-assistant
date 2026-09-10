import asyncio
import json
from unittest.mock import AsyncMock

import httpx
import pytest

from agents import ResearchAgent
from core.skill_loader import load_skill_function
from core.rag_context import compact_rag_result


def configure(monkeypatch, handler):
    execute = load_skill_function("deep-research", "research", "deep_research")
    client_class = httpx.AsyncClient
    monkeypatch.setattr(execute.__globals__["httpx"], "AsyncClient",
                        lambda **kwargs: client_class(transport=httpx.MockTransport(handler), **kwargs))
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    return execute


def test_live_sources_preserve_urls_limits_and_no_model_answer(monkeypatch):
    def handler(request):
        payload = json.loads(request.content)
        assert request.headers["Authorization"] == "Bearer test-key"
        assert payload["query"] == "site:cdc.gov original query"
        assert payload["include_answer"] is False
        assert payload["max_results"] == 15
        return httpx.Response(200, json={"request_id": "r1", "results": [
            {"title": "Source", "url": "https://cdc.gov/example", "content": "excerpt",
             "raw_content": "x" * 6001},
            {"title": "Other", "url": "https://example.org", "content": "retrieved excerpt"},
        ]})
    execute = configure(monkeypatch, handler)
    result = asyncio.run(execute("site:cdc.gov original query"))
    assert result["documents"][0]["metadata"]["source"] == "https://cdc.gov/example"
    assert len(result["documents"][0]["content"]) == 6000
    assert result["documents"][0]["metadata"]["truncated"] is True
    assert result["documents"][1]["metadata"]["content_kind"] == "search_excerpt"
    assert result["documents"][1]["metadata"]["published_date"] is None
    assert "test-key" not in json.dumps(result)
    first = compact_rag_result(result, [], "call1")
    messages = [{"role": "tool", "tool_call_id": "call1", "content": json.dumps(first)}]
    repeated = compact_rag_result(result, messages, "call2")
    assert repeated["documents"][0]["reference"] == {"tool_call_id": "call1"}
    assert "content" not in repeated["documents"][0]


@pytest.mark.parametrize("size", [1, 3, 5])
def test_search_size_controls_requested_and_returned_sources(monkeypatch, size):
    def handler(request):
        assert json.loads(request.content)["max_results"] == size * 3
        return httpx.Response(200, json={"results": [
            {"title": f"Source {index}", "url": f"https://example.org/{index}", "content": "excerpt"}
            for index in range(20)
        ]})

    execute = configure(monkeypatch, handler)
    result = asyncio.run(execute("question", max_iterations=size))
    assert len(result["documents"]) == size * 3


@pytest.mark.parametrize("status", [401, 429, 500])
def test_service_failure_is_visible_through_registered_tool(monkeypatch, status):
    execute = configure(monkeypatch, lambda request: httpx.Response(status))
    agent = ResearchAgent(llm_client=AsyncMock())
    agent.skill_registry.skills["deep_research"]["function"] = execute
    result = asyncio.run(agent.execute_tool("deep_research", {"query": "question"}))
    assert result["success"] is False
    assert str(status) in result["error"]
    assert "test-key" not in json.dumps(result)


def test_empty_search_is_not_a_medical_conclusion(monkeypatch):
    execute = configure(monkeypatch, lambda request: httpx.Response(200, json={"results": []}))
    result = asyncio.run(execute("question"))
    assert result["status"] == "no_results"
    assert result["documents"] == []
    assert "不能据此排除疾病" in result["answer"]


def test_missing_key_fails_before_network(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    execute = load_skill_function("deep-research", "research", "deep_research")
    with pytest.raises(RuntimeError, match="TAVILY_API_KEY"):
        asyncio.run(execute("question"))


def test_model_visible_schema_matches_search_bounds():
    agent = ResearchAgent(llm_client=AsyncMock())
    tool = next(t for t in agent.get_tools_for_llm() if t["function"]["name"] == "deep_research")
    schema = tool["function"]["parameters"]
    assert schema["properties"]["max_iterations"]["type"] == "integer"
    assert schema["properties"]["max_iterations"]["enum"] == [1, 2, 3, 4, 5]
    assert "15 sources" in schema["properties"]["max_iterations"]["description"]
    assert "query" in schema["required"]


@pytest.mark.parametrize("arguments", [{"query": ""}, {"query": "x", "max_iterations": 0},
                                       {"query": "x", "max_iterations": 6},
                                       {"query": "x", "max_iterations": True}])
def test_invalid_search_input_is_rejected(arguments):
    execute = load_skill_function("deep-research", "research", "deep_research")
    with pytest.raises(ValueError):
        asyncio.run(execute(**arguments))
