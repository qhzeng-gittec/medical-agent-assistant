import asyncio
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core import EvidenceStore, SkillRegistry, SkillParameter, use_evidence_store


def registry(owner, function):
    value = SkillRegistry(owner=owner)
    value.register(
        name="search_knowledge",
        function=function,
        description="test search",
        parameters=[SkillParameter("query", "string", "query", True)],
    )
    return value


def test_identical_calls_are_reused_within_one_agent():
    calls = []

    async def search(query):
        calls.append(query)
        return {"answer": f"evidence for {query}"}

    async def run():
        store = EvidenceStore()
        first = registry("research_agent", search)
        second = registry("research_agent", search)
        with use_evidence_store(store):
            result_a = await first.execute("search_knowledge", query="胸痛风险")
            result_b = await second.execute("search_knowledge", query="胸痛风险")
        return result_a, result_b, store

    result_a, result_b, store = asyncio.run(run())
    assert result_a == result_b
    assert calls == ["胸痛风险"]
    assert store.summary()["executed_calls"] == 1
    assert store.summary()["cache_hits"] == 1
    assert store.summary()["entries"][0]["consumers"] == [
        "research_agent",
    ]


def test_parallel_identical_calls_use_single_flight():
    calls = []

    async def search(query):
        calls.append(query)
        await asyncio.sleep(0.02)
        return {"answer": query}

    async def run():
        store = EvidenceStore()
        first = registry("research_agent", search)
        second = registry("research_agent", search)
        with use_evidence_store(store):
            results = await asyncio.gather(
                first.execute("search_knowledge", query="同一查询"),
                second.execute("search_knowledge", query="同一查询"),
            )
        return results, store

    results, store = asyncio.run(run())
    assert results == [{"answer": "同一查询"}, {"answer": "同一查询"}]
    assert calls == ["同一查询"]
    assert store.summary()["cache_hits"] == 1


def test_different_evidence_needs_are_not_merged():
    calls = []

    async def search(query):
        calls.append(query)
        return {"answer": query}

    async def run():
        store = EvidenceStore()
        searcher = registry("research_agent", search)
        with use_evidence_store(store):
            await searcher.execute("search_knowledge", query="胸痛风险")
            await searcher.execute("search_knowledge", query="胸痛饮食")
        return store

    store = asyncio.run(run())
    assert calls == ["胸痛风险", "胸痛饮食"]
    assert store.summary()["executed_calls"] == 2
    assert store.summary()["cache_hits"] == 0
