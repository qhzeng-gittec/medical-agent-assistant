import asyncio
import copy
import json

import pytest

from agents.base_agent import BaseAgent
from core import LLMClient, LLMResponse, SkillParameter, ToolCall
from core.rag_context import compact_rag_result, document_block
from core.skill_loader import load_skill_function


def document(content="模拟知识库正文", version="v1", score=0.9):
    return {"id": "B12", "content": content,
            "metadata": {"source": "测试资料", "version": version}, "score": score}


def tool_message(call_id, result):
    return {"role": "tool", "tool_call_id": call_id,
            "content": json.dumps(result, ensure_ascii=False)}


def test_different_queries_share_body_but_preserve_results_and_sources():
    first = {"query": "问题A", "documents": [document_block(document())]}
    second = {"query": "问题B", "documents": [document_block(document(score=0.8))]}
    compacted = compact_rag_result(second, [tool_message("q1", first)], "q2")
    assert compacted["query"] == "问题B"
    assert compacted["documents"][0]["reference"] == {"tool_call_id": "q1"}
    assert "content" not in compacted["documents"][0]
    assert "content" in second["documents"][0]
    assert first["documents"][0]["metadata"]["source"] == "测试资料"


@pytest.mark.parametrize("changed", [document("不同正文"), document(version="v2")])
def test_changed_body_or_metadata_is_not_reused(changed):
    first = {"documents": [document_block(document())]}
    second = {"documents": [document_block(changed)]}
    result = compact_rag_result(second, [tool_message("q1", first)], "q2")
    assert "content" in result["documents"][0]


def test_removed_or_truncated_body_is_reinjected():
    raw = {"documents": [document_block(document())]}
    reference_only = compact_rag_result(raw, [tool_message("q1", raw)], "q2")
    for history in ([], [tool_message("q2", reference_only)]):
        assert compact_rag_result(raw, history, "q3") == raw
    cropped = copy.deepcopy(raw)
    cropped["documents"][0]["content"] = "模拟"
    assert compact_rag_result(raw, [tool_message("q1", cropped)], "q3") == raw


def test_same_batch_duplicates_and_excerpt_boundaries():
    block = document_block(document())
    result = compact_rag_result({"documents": [block, block]}, [], "q1")
    assert "content" in result["documents"][0]
    assert result["documents"][1]["reference"] == {"tool_call_id": "q1"}
    excerpt = {"documents": [document_block(document(), max_chars=2)]}
    full = {"documents": [block]}
    assert compact_rag_result(full, [tool_message("q1", excerpt)], "q2") == full


class ScriptedClient:
    create_tool_message = LLMClient.create_tool_message

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def chat_with_tools(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        return self.responses.pop(0)


def search_response(call_id, query="问题A"):
    return LLMResponse(None, [ToolCall(call_id, "search_knowledge", {"query": query})], "tool_calls")


def final_response():
    return LLMResponse("最终结论，仅供参考。", [], "stop")


class SearchAgent(BaseAgent):
    def __init__(self, client, search, agent_id="research_agent"):
        self.search = search
        super().__init__(agent_id, {"max_iterations": 4}, client)

    def register_tools(self):
        self.skill_registry.register("search_knowledge", self.search, "检索资料",
                                     [SkillParameter("query", "string", "查询", True)])

    def get_system_prompt(self):
        return "检索资料后交付最终结论。"


def test_real_loop_exact_cache_and_references_reset_each_invocation():
    searches = []

    async def search(query):
        searches.append(query)
        return {"query": query, "documents": [document_block(document())]}

    client = ScriptedClient([
        search_response("q1"), search_response("q2"), final_response(),
        search_response("q3"), final_response(),
    ])
    agent = SearchAgent(client, search)

    async def run():
        await agent.process({"question": "第一次"})
        await agent.process({"question": "第二次"})

    asyncio.run(run())
    assert searches == ["问题A", "问题A"]
    messages = client.calls[2]["messages"]
    assert [item["tool_call_id"] for item in messages if item["role"] == "tool"] == ["q1", "q2"]
    assert "reference" in json.loads(messages[-1]["content"])["documents"][0]
    assert client.calls[2]["tools"]
    new_messages = client.calls[4]["messages"]
    assert len(new_messages) == 4
    assert "content" in json.loads(new_messages[-1]["content"])["documents"][0]


@pytest.mark.parametrize("budget,expected", [(None, ["A", "B", "C"]), (2, ["A", "B"])])
def test_third_tool_call_executes_by_default_and_explicit_cap_preserves_protocol(budget, expected):
    searches = []

    async def search(query):
        searches.append(query)
        return {"query": query, "documents": []}

    client = ScriptedClient([
        search_response("q1", "A"),
        LLMResponse(None, [ToolCall("q2", "search_knowledge", {"query": "B"}),
                           ToolCall("q3", "search_knowledge", {"query": "C"})], "tool_calls"),
        final_response(),
    ])
    agent = SearchAgent(client, search)
    agent.loop.max_tool_calls = budget
    result = asyncio.run(agent.process({"question": "需要三份资料"}))
    assert result["answer"]
    assert searches == expected
    replies = [m for m in client.calls[-1]["messages"] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in replies] == ["q1", "q2", "q3"]
    if budget is None:
        assert json.loads(replies[-1]["content"])["query"] == "C"
        assert client.calls[-1]["tools"]
    else:
        assert json.loads(replies[-1]["content"])["error"] == "ToolCallBudgetExceeded"
        assert client.calls[-1]["tools"] is None


def test_text_tool_envelope_is_repaired_without_executing_text_and_evidence_is_deduplicated():
    searches = []

    async def search(query):
        searches.append(query)
        return {"documents": [document_block(document())]}

    client = ScriptedClient([
        LLMResponse('<tool_call><function=search_knowledge>untrusted text</function></tool_call>', [], 'stop'),
        search_response('q1', 'actual query'), search_response('q2', 'actual query'), final_response(),
    ])
    result = asyncio.run(SearchAgent(client, search).process({"question": "查资料"}))
    assert searches == ['actual query']
    assert result['answer'] == '最终结论，仅供参考。'
    assert len(result['evidence']) == 1
    assert result['evidence'][0]['content'] == document()['content']
    assert client.calls[1]['tools']
    assert not any(m['role'] == 'tool' for m in client.calls[1]['messages'])


def test_repeated_text_tool_envelopes_never_escape_as_a_final_deliverable():
    invalid = LLMResponse('<tool_call>invalid</tool_call>', [], 'stop')
    client = ScriptedClient([invalid] * 5)
    with pytest.raises(RuntimeError, match='Tool protocol'):
        asyncio.run(SearchAgent(client, None).process({'question': '查资料'}))


def test_parallel_agent_invocations_do_not_share_cache_or_references():
    searches = []

    async def search(query):
        searches.append(query)
        await asyncio.sleep(0)
        return {"documents": [document_block(document())]}

    clients = [ScriptedClient([search_response("q1"), final_response()]) for _ in range(2)]
    agents = [SearchAgent(client, search) for client in clients]

    async def run():
        await asyncio.gather(*(agent.process({"question": "查资料"}) for agent in agents))

    asyncio.run(run())
    assert len(searches) == 2
    for client in clients:
        result = json.loads(client.calls[1]["messages"][-1]["content"])
        assert "content" in result["documents"][0]


def test_batch_budget_preserves_a_result_for_every_call():
    searches = []

    async def search(query):
        searches.append(query)
        return {"documents": [document_block(document())]}

    batch = LLMResponse(None, [
        ToolCall(f"q{i}", "search_knowledge", {"query": str(i)}) for i in range(3)
    ], "tool_calls")
    client = ScriptedClient([batch, final_response()])
    agent = SearchAgent(client, search)
    agent.loop.max_tool_calls = 2
    asyncio.run(agent.process({"question": "查资料"}))
    assert searches == ["0", "1"]
    results = [m for m in client.calls[1]["messages"] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in results] == ["q0", "q1", "q2"]
    assert json.loads(results[-1]["content"])["error"] == "ToolCallBudgetExceeded"


def test_broken_result_stops_instead_of_resending_unmatched_call():
    async def search(query):
        return {"documents": [{"content": "missing block identity"}]}

    client = ScriptedClient([search_response("q1"), final_response()])
    with pytest.raises(KeyError, match="block_id"):
        asyncio.run(SearchAgent(client, search).process({"question": "查资料"}))
    assert len(client.calls) == 1


def test_offline_example_captures_real_context_growth():
    from examples.context_growth import BODY_A, build_trace

    trace = asyncio.run(build_trace())
    snapshots = trace["snapshots"]
    assert [len(s["request"]["messages"]) for s in snapshots] == [2, 4, 2, 4, 6, 6, 2, 8, 2, 2, 4, 2, 4, 6]
    research_messages = snapshots[4]["request"]["messages"]
    assert sum(BODY_A in (m.get("content") or "") for m in research_messages) == 1
    duplicate = json.loads(research_messages[-1]["content"])["documents"][0]
    assert duplicate["reference"] == {"tool_call_id": "search-1"}
    for snapshot in snapshots:
        if snapshot["agent"] not in {"research_agent", "supervisor"}:
            assert BODY_A not in json.dumps(snapshot["request"]["messages"], ensure_ascii=False)
            for message in snapshot["request"]["messages"]:
                assert "search-1" not in str(message.get("content"))
    second_turn = json.loads(snapshots[8]["request"]["messages"][1]["content"])
    assert len(second_turn["context"]["recent_history"]) == 2
    assert trace["turns"][1]["agents_involved"] == []
    third_turn = json.loads(snapshots[9]["request"]["messages"][1]["content"])
    assert third_turn["context"]["patient_profile"]["medications"][0]["status"] == "active"
    assert json.loads(snapshots[10]["request"]["messages"][-1]["content"])["result"]["updates"][0]["status"] == "stopped"
    assert len(third_turn["context"]["recent_history"]) == 4
    assert len(trace["knowledge_base_calls"]) == 3
    assert trace["knowledge_base_calls"][0] == trace["knowledge_base_calls"][2]
    new_result = json.loads(snapshots[12]["request"]["messages"][-1]["content"])
    assert new_result["documents"][0]["content"] == BODY_A
    assert [m["role"] for m in trace["persisted_conversation"]] == ["user", "assistant"] * 3


@pytest.mark.parametrize("skill,script,function,arguments", [
    ("search-knowledge", "search", "search_knowledge", {"query": "测试"}),
    ("clinical-guideline", "guideline", "clinical_guideline", {"query": "测试"}),
    ("recommend-lifestyle", "lifestyle", "recommend_lifestyle", {"diagnosis": "测试"}),
    ("disease-code", "code", "disease_code", {"disease_name": "测试"}),
    ("assess-risk", "risk", "assess_risk", {"symptoms": "咳嗽"}),
    ("analyze-symptoms", "symptoms", "analyze_symptoms", {"symptoms": "咳嗽"}),
])
def test_real_kb_tools_return_structured_bodies_once(monkeypatch, skill, script, function, arguments):
    class FakeKB:
        def search(self, **kwargs):
            return [document("唯一的模拟知识库原文")]

    execute = load_skill_function(skill, script, function)
    monkeypatch.setitem(execute.__globals__, "_kb_instance", FakeKB())
    result = asyncio.run(execute(**arguments))
    assert result["documents"]
    assert "唯一的模拟知识库原文" not in result["answer"]
    assert result["documents"][0]["content"] == "唯一的模拟知识库原文"


def test_lifestyle_returns_candidates_not_validated_advice(monkeypatch):
    class CandidateKB:
        def search(self, **kwargs):
            assert kwargs["top_k"] == 3
            return [document("另一疾病的材料", score=.2), {**document("更相关的候选材料"), "id": "d2"}]
    execute = load_skill_function("recommend-lifestyle", "lifestyle", "recommend_lifestyle")
    monkeypatch.setitem(execute.__globals__, "_kb_instance", CandidateKB())
    result = asyncio.run(execute(diagnosis="目标疾病"))
    assert result["status"] == "candidates"
    assert len(result["documents"]) == 2
    assert "不代表已匹配" in result["answer"]


def test_empty_lifestyle_retrieval_is_explicit(monkeypatch):
    class EmptyKB:
        def search(self, **kwargs):
            return []
    execute = load_skill_function("recommend-lifestyle", "lifestyle", "recommend_lifestyle")
    monkeypatch.setitem(execute.__globals__, "_kb_instance", EmptyKB())
    result = asyncio.run(execute(diagnosis="目标疾病"))
    assert result["status"] == "no_results" and result["documents"] == []
