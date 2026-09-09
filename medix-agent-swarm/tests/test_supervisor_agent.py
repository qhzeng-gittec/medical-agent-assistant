import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest


PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents import ConsultationAgent, DiagnosticAgent, ResearchAgent
from core import LLMResponse, SkillParameter, SkillRegistry, ToolCall
from memory import LongTermMemoryError
from swarm.supervisor_agent import MedicalSupervisorAgent


@pytest.mark.parametrize("fail", [False, True])
def test_public_entrypoint_releases_local_memory_after_each_request(monkeypatch, fail):
    from swarm import supervisor_agent
    instance = Mock()
    instance.process = AsyncMock(return_value={"answer": "done"})
    if fail:
        instance.process.side_effect = RuntimeError("processing failed")
    monkeypatch.setattr(supervisor_agent, "MedicalSupervisorAgent", lambda: instance)
    if fail:
        with pytest.raises(RuntimeError, match="processing failed"):
            asyncio.run(supervisor_agent.process_medical_question("question"))
    else:
        assert asyncio.run(supervisor_agent.process_medical_question("question")) == {"answer": "done"}
    instance.long_term_memory.close.assert_called_once_with()


class ScriptedLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def chat_with_tools(self, messages, tools=None, tool_choice="auto", **kwargs):
        self.calls.append({
            "messages": [dict(message) for message in messages],
            "tools": tools,
            "tool_choice": tool_choice,
        })
        return self.responses.pop(0)


class MemoryStub:
    def __init__(self):
        self.enabled = True
        self.messages = []
        self.summaries = []
        self.searches = []
        self.search_results = []

    def get_recent_messages(self, session_id, limit=10):
        return []

    def search_similar_sessions(self, query, user_id, limit=3):
        self.searches.append((query, user_id, limit))
        return self.search_results

    def add_message(self, session_id, role, content):
        self.messages.append((session_id, role, content))

    def add_session_summary(self, **kwargs):
        self.summaries.append(kwargs)


class FailingLongTermMemory(MemoryStub):
    def search_similar_sessions(self, query, user_id, limit=3):
        raise LongTermMemoryError("Mem0 search failed: unavailable")

    def add_session_summary(self, **kwargs):
        raise LongTermMemoryError("Mem0 add failed: unavailable")


class ProfileStub:
    def __init__(self):
        self.context = {}
        self.updates = []
        self.update_calls = []

    def apply_updates(self, user_id, message, session_id, updates):
        self.update_calls.append((user_id, message, session_id))
        return list(self.updates)

    def get_context(self, user_id):
        return self.context.get(user_id, {})


class ConcurrencyProbe:
    def __init__(self):
        self.active = 0
        self.max_active = 0
        self.lock = asyncio.Lock()

    async def enter(self):
        async with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)

    async def leave(self):
        async with self.lock:
            self.active -= 1


class WorkerStub:
    def __init__(self, agent_id, answer, probe=None):
        self.agent_id = agent_id
        self.answer = answer
        self.probe = probe
        self.inputs = []

    async def process(self, input_data):
        self.inputs.append(input_data)
        if self.probe:
            await self.probe.enter()
            await asyncio.sleep(0.02)
            await self.probe.leave()
        return {"answer": self.answer}


class EvidenceWorker:
    def __init__(self, agent_id, search):
        self.agent_id = agent_id
        self.skill_registry = SkillRegistry(owner=agent_id)
        self.skill_registry.register(
            "search_knowledge",
            search,
            "test search",
            [SkillParameter("query", "string", "query", True)],
        )

    async def process(self, input_data):
        return await self.skill_registry.execute(
            "search_knowledge",
            query="糖尿病饮食",
        )


def response(*calls, content=None):
    return LLMResponse(content=content, tool_calls=list(calls), finish_reason="tool_calls" if calls else "stop")


def call(call_id, name, task):
    return ToolCall(id=call_id, name=name, arguments={"task": task})


def build_supervisor(llm, workers, max_rounds=4):
    short_memory = MemoryStub()
    long_memory = MemoryStub()
    profiles = ProfileStub()
    supervisor = MedicalSupervisorAgent(
        llm_client=llm,
        workers=workers,
        short_term_memory=short_memory,
        long_term_memory=long_memory,
        patient_profiles=profiles,
        max_rounds=max_rounds,
        worker_timeout=1,
    )
    return supervisor, short_memory, long_memory


def test_supervisor_repairs_text_call_and_receives_bounded_actual_evidence():
    workers = {f"call_{role}": WorkerStub(role, "done") for role in (
        "consultation_agent", "diagnostic_agent", "research_agent")}
    worker = workers["call_research_agent"]
    worker.process = AsyncMock(return_value={"answer": "supported answer", "evidence": [
        {"document_id": "d1", "content": "actual source", "metadata": {"source": "source-url"}},
    ]})
    llm = ScriptedLLM([
        response(content="<tool_call>not an executable call</tool_call>"),
        response(call("r1", "call_research_agent", "verify")),
        response(content="finished"),
    ])
    supervisor, _, _ = build_supervisor(llm, workers)
    result = asyncio.run(supervisor.process("verify", session_id="protocol-test"))
    worker.process.assert_awaited_once()
    assert result["answer"] == "finished"
    returned = json.loads(llm.calls[-1]["messages"][-1]["content"])["result"]
    assert returned["evidence"][0]["content"] == "actual source"
    assert returned["evidence"][0]["metadata"]["source"] == "source-url"
    bounded = supervisor._bounded_result({"answer": "answer", "evidence": [
        {"document_id": "d1", "content": "abcdef", "metadata": {"source": "original"}}]}, max_chars=3)
    assert bounded["evidence"][0]["content"] == "abc"
    assert bounded["evidence_truncated"] is True


def test_diagnostic_first_then_parallel_workers():
    probe = ConcurrencyProbe()
    diagnostic = WorkerStub("diagnostic_agent", "高风险，需要进一步核验")
    consultation = WorkerStub("consultation_agent", "建议立即就医", probe)
    research = WorkerStub("research_agent", "指南支持立即评估", probe)
    workers = {
        "call_consultation_agent": consultation,
        "call_diagnostic_agent": diagnostic,
        "call_research_agent": research,
    }
    llm = ScriptedLLM([
        response(call("diag-1", "call_diagnostic_agent", "评估胸痛和呼吸困难的风险")),
        response(
            call("consult-1", "call_consultation_agent", "给出立即行动建议"),
            call("research-1", "call_research_agent", "独立核验临床指南"),
        ),
        response(content="该症状存在紧急风险，建议立即就医。"),
    ])
    supervisor, short_memory, long_memory = build_supervisor(llm, workers)

    result = asyncio.run(supervisor.process(
        "我胸痛并且呼吸困难，应该怎么办？",
        session_id="s1",
        user_id="user-1",
    ))

    first_tool_names = [tool["function"]["name"] for tool in llm.calls[0]["tools"]]
    assert set(first_tool_names) == set(workers) | {"update_patient_profile", "search_patient_history"}
    assert llm.calls[0]["tool_choice"] == "auto"
    assert [item["agent_id"] for item in result["call_trace"]] == [
        "diagnostic_agent",
        "consultation_agent",
        "research_agent",
    ]
    assert probe.max_active == 2
    assert result["swarm_enabled"] is True
    assert result["subtasks_completed"] == 3
    assert consultation.inputs[0]["context"]["prior_agent_findings"][0]["agent"] == "diagnostic_agent"
    assert research.inputs[0]["context"]["original_question"].startswith("我胸痛")
    assert [item["agent"] for item in consultation.inputs[0]["context"]["prior_agent_findings"]] == ["diagnostic_agent"]
    assert [item["agent"] for item in research.inputs[0]["context"]["prior_agent_findings"]] == ["diagnostic_agent"]
    assert [role for _, role, _ in short_memory.messages] == ["user", "assistant"]
    assert len(long_memory.summaries) == 1
    assert long_memory.summaries[0]["user_id"] == "user-1"

    second_call_messages = llm.calls[1]["messages"]
    assert second_call_messages[-2]["role"] == "assistant"
    assert second_call_messages[-1]["role"] == "tool"
    assert second_call_messages[-1]["tool_call_id"] == "diag-1"
    third_call_messages = llm.calls[2]["messages"]
    parallel_results = third_call_messages[-2:]
    assert [message["tool_call_id"] for message in parallel_results] == ["consult-1", "research-1"]


def test_simple_question_uses_one_worker_and_stays_single_agent():
    workers = {
        "call_consultation_agent": WorkerStub("consultation_agent", "高血压是持续血压升高。"),
        "call_diagnostic_agent": WorkerStub("diagnostic_agent", "unused"),
        "call_research_agent": WorkerStub("research_agent", "unused"),
    }
    llm = ScriptedLLM([
        response(call("consult-1", "call_consultation_agent", "解释什么是高血压")),
        response(content="高血压是血压持续升高的慢性疾病。"),
    ])
    supervisor, _, _ = build_supervisor(llm, workers)

    result = asyncio.run(supervisor.process("什么是高血压？", session_id="s2"))

    assert result["agents_involved"] == ["consultation_agent"]
    assert result["swarm_enabled"] is False
    assert not workers["call_diagnostic_agent"].inputs
    assert not workers["call_research_agent"].inputs


def test_user_scoped_long_term_memory_is_injected_into_context():
    workers = {
        "call_consultation_agent": WorkerStub("consultation_agent", "咨询结果"),
        "call_diagnostic_agent": WorkerStub("diagnostic_agent", "unused"),
        "call_research_agent": WorkerStub("research_agent", "unused"),
    }
    llm = ScriptedLLM([
        response(call("consult-1", "call_consultation_agent", "制定可执行建议")),
        response(content="建议从能够坚持的运动开始。"),
    ])
    supervisor, _, long_memory = build_supervisor(llm, workers)
    long_memory.search_results = [{
        "content": "用户以前晨跑时因膝盖不适而停止。",
        "score": 0.82,
    }]

    result = asyncio.run(supervisor.process(
        "我想开始运动",
        session_id="session-7",
        user_id="user-7",
    ))

    payload = json.loads(llm.calls[0]["messages"][1]["content"])
    assert payload["context"]["historical_memories"] == [{
        "memory": "用户以前晨跑时因膝盖不适而停止。",
    }]
    assert long_memory.searches == [("我想开始运动", "user-7", 3)]
    assert long_memory.summaries[0]["user_id"] == "user-7"
    assert result["long_term_memory_enabled"] is True
    assert result["recalled_memories"] == 1


def test_structured_patient_profile_is_injected_before_mem0_context():
    workers = {
        "call_consultation_agent": WorkerStub("consultation_agent", "咨询结果"),
        "call_diagnostic_agent": WorkerStub("diagnostic_agent", "unused"),
        "call_research_agent": WorkerStub("research_agent", "unused"),
    }
    llm = ScriptedLLM([
        response(call("consult-1", "call_consultation_agent", "回答用药问题")),
        response(content="请让医生结合过敏史评估。"),
    ])
    profiles = ProfileStub()
    profiles.context["user-9"] = {
        "allergies": [{"value": "青霉素", "status": "active", "source": "user_reported"}]
    }
    profiles.updates = [{"category": "allergies", "value": "青霉素"}]
    supervisor = MedicalSupervisorAgent(
        llm_client=llm,
        workers=workers,
        short_term_memory=MemoryStub(),
        long_term_memory=MemoryStub(),
        patient_profiles=profiles,
        worker_timeout=1,
    )

    result = asyncio.run(supervisor.process(
        "我对青霉素过敏，这个药能吃吗？",
        session_id="session-9",
        user_id="user-9",
    ))

    payload = json.loads(llm.calls[0]["messages"][1]["content"])
    assert payload["context"]["patient_profile"]["allergies"][0]["value"] == "青霉素"
    assert profiles.update_calls == []  # Reading context never writes a new fact.
    assert result["patient_profile_enabled"] is True
    assert result["profile_updates"] == []


def test_supervisor_does_not_share_evidence_cache_between_workers():
    real_searches = []

    async def search(query):
        real_searches.append(query)
        await asyncio.sleep(0.02)
        return {"answer": "共享证据"}

    workers = {
        "call_consultation_agent": EvidenceWorker("consultation_agent", search),
        "call_diagnostic_agent": WorkerStub("diagnostic_agent", "unused"),
        "call_research_agent": EvidenceWorker("research_agent", search),
    }
    llm = ScriptedLLM([
        response(
            call("consult-1", "call_consultation_agent", "生成饮食建议"),
            call("research-1", "call_research_agent", "检索饮食证据"),
        ),
        response(content="建议结合个人情况制定饮食方案。"),
    ])
    supervisor = MedicalSupervisorAgent(
        llm_client=llm,
        workers=workers,
        short_term_memory=MemoryStub(),
        long_term_memory=MemoryStub(),
        patient_profiles=ProfileStub(),
        worker_timeout=1,
    )

    result = asyncio.run(supervisor.process("糖尿病怎么吃？", session_id="evidence-1"))

    assert real_searches == ["糖尿病饮食", "糖尿病饮食"]
    assert "evidence_store" not in result


def test_supervisor_can_answer_without_delegation():
    workers = {
        "call_consultation_agent": WorkerStub("consultation_agent", "unused"),
        "call_diagnostic_agent": WorkerStub("diagnostic_agent", "unused"),
        "call_research_agent": WorkerStub("research_agent", "unused"),
    }
    llm = ScriptedLLM([response(content="请把需要压缩的文字发给我。")])
    supervisor, memory, _ = build_supervisor(llm, workers)
    result = asyncio.run(supervisor.process("帮我压缩一段文字", session_id="direct"))
    assert result["subtasks_completed"] == 0
    assert result["agents_involved"] == []
    assert llm.calls[0]["tool_choice"] == "auto"
    assert all(not worker.inputs for worker in workers.values())
    assert [role for _, role, _ in memory.messages] == ["user", "assistant"]


def test_only_final_answer_crosses_worker_boundary():
    class NoisyWorker(WorkerStub):
        async def process(self, input_data):
            return {
                "answer": "最终结论，来源为示例资料，适用范围有限。",
                "documents": [{"content": "PRIVATE_DOCUMENT_BODY"}],
                "query": "PRIVATE_QUERY",
                "messages": [{"content": "PRIVATE_TRANSCRIPT"}],
                "iterations": 3,
            }

    workers = {
        "call_consultation_agent": WorkerStub("consultation_agent", "整理完成"),
        "call_diagnostic_agent": WorkerStub("diagnostic_agent", "unused"),
        "call_research_agent": NoisyWorker("research_agent", "unused"),
    }
    llm = ScriptedLLM([
        response(call("r1", "call_research_agent", "查资料")),
        response(call("c1", "call_consultation_agent", "整理已有结论")),
        response(content="回答完成"),
    ])
    supervisor, _, _ = build_supervisor(llm, workers)
    result = asyncio.run(supervisor.process("请查资料并整理", session_id="private"))
    context = workers["call_consultation_agent"].inputs[0]["context"]
    assert "available_evidence" not in context
    assert context["prior_agent_findings"][0]["result"] == {
        "answer": "最终结论，来源为示例资料，适用范围有限。",
    }
    assert "PRIVATE_" not in json.dumps([llm.calls, context, result])


@pytest.mark.parametrize("question,answer", [
    ("我没有胸痛或呼吸困难，请只整理这条记录。", "记录：您否认胸痛或呼吸困难。"),
    ("材料中写着‘确诊为’，请原样引用，不是在说我患病。", "原文：确诊为。"),
    ("我胸痛且呼吸困难，现在应该先做什么？", "请立即拨打120，不要等待在线咨询。"),
    ("他刚刚突然说不出完整的话，一侧手臂也抬不起来。", "请立即联系急救，不要等待进一步检索。"),
])
def test_model_can_answer_without_keyword_routing_or_rewriting(question, answer):
    workers = {
        "call_consultation_agent": WorkerStub("consultation_agent", "unused"),
        "call_diagnostic_agent": WorkerStub("diagnostic_agent", "unused"),
        "call_research_agent": WorkerStub("research_agent", "unused"),
    }
    llm = ScriptedLLM([response(content=answer)])
    supervisor, _, _ = build_supervisor(llm, workers)
    result = asyncio.run(supervisor.process(question, session_id="semantic"))
    assert result["answer"] == answer
    assert not result["call_trace"]
    assert all(not worker.inputs for worker in workers.values())
    assert {tool["function"]["name"] for tool in llm.calls[0]["tools"]} == set(workers)
    assert llm.calls[0]["tool_choice"] == "auto"
    assert json.loads(llm.calls[0]["messages"][1]["content"]) == {"question": question, "context": {}}


def test_mem0_failures_are_reported_without_losing_the_answer():
    workers = {
        "call_consultation_agent": WorkerStub("consultation_agent", "咨询结果"),
        "call_diagnostic_agent": WorkerStub("diagnostic_agent", "unused"),
        "call_research_agent": WorkerStub("research_agent", "unused"),
    }
    llm = ScriptedLLM([
        response(call("consult-1", "call_consultation_agent", "回答问题")),
        response(content="可以先从规律作息开始。"),
    ])
    supervisor = MedicalSupervisorAgent(
        llm_client=llm,
        workers=workers,
        short_term_memory=MemoryStub(),
        long_term_memory=FailingLongTermMemory(),
        patient_profiles=ProfileStub(),
        worker_timeout=1,
    )

    result = asyncio.run(supervisor.process(
        "怎样保持健康？",
        session_id="session-8",
        user_id="user-8",
    ))

    assert result["answer"] == "可以先从规律作息开始。"
    assert result["memory_warnings"] == [
        "Mem0 search failed: unavailable",
        "Mem0 add failed: unavailable",
    ]


def test_tool_failure_is_returned_with_matching_call_id():
    workers = {
        "call_consultation_agent": WorkerStub("consultation_agent", "unused"),
        "call_diagnostic_agent": WorkerStub("diagnostic_agent", "unused"),
        "call_research_agent": WorkerStub("research_agent", "unused"),
    }
    llm = ScriptedLLM([
        response(ToolCall(id="bad-1", name="call_research_agent", arguments={})),
        response(content="信息不足，建议补充具体问题。"),
    ])
    supervisor, _, _ = build_supervisor(llm, workers)

    result = asyncio.run(supervisor.process("请帮我查资料", session_id="s3"))

    assert result["call_trace"][0]["success"] is False
    assert result["call_trace"][0]["result"]["error_type"] == "InvalidArguments"
    tool_message = llm.calls[1]["messages"][-1]
    assert tool_message["tool_call_id"] == "bad-1"


def test_round_limit_forces_final_answer():
    workers = {
        "call_consultation_agent": WorkerStub("consultation_agent", "咨询结果"),
        "call_diagnostic_agent": WorkerStub("diagnostic_agent", "unused"),
        "call_research_agent": WorkerStub("research_agent", "unused"),
    }
    llm = ScriptedLLM([
        response(call("consult-1", "call_consultation_agent", "回答问题")),
        response(content="基于咨询结果给出最终回答。"),
    ])
    supervisor, _, _ = build_supervisor(llm, workers, max_rounds=1)

    result = asyncio.run(supervisor.process("如何保持健康？", session_id="s4"))

    assert result["answer"] == "基于咨询结果给出最终回答。"
    assert llm.calls[-1]["tools"] is None


def test_model_can_choose_parallel_first_round_despite_symptom_words():
    probe = ConcurrencyProbe()
    workers = {
        "call_consultation_agent": WorkerStub("consultation_agent", "整理结果", probe),
        "call_diagnostic_agent": WorkerStub("diagnostic_agent", "unused"),
        "call_research_agent": WorkerStub("research_agent", "来源核验", probe),
    }
    llm = ScriptedLLM([
        response(call("c1", "call_consultation_agent", "整理既往材料"),
                 call("r1", "call_research_agent", "独立核验资料来源")),
        response(content="材料整理及来源核验完成。"),
    ])
    supervisor, _, _ = build_supervisor(llm, workers)

    result = asyncio.run(supervisor.process("请整理教材中的胸痛案例并独立核验文献，不是本人症状。", session_id="s5"))

    assert all(item["success"] for item in result["call_trace"])
    assert probe.max_active == 2
    assert not workers["call_diagnostic_agent"].inputs
    for name in ("call_consultation_agent", "call_research_agent"):
        assert workers[name].inputs[0]["context"]["prior_agent_findings"] == []


def test_each_real_worker_exposes_only_role_specific_skills():
    fake_llm = ScriptedLLM([])

    consultation = ConsultationAgent(llm_client=fake_llm)
    diagnostic = DiagnosticAgent(llm_client=fake_llm)
    research = ResearchAgent(llm_client=fake_llm)

    assert set(consultation.skill_registry.get_all()) == {"recommend_lifestyle"}
    assert set(diagnostic.skill_registry.get_all()) == {
        "assess_risk", "analyze_symptoms", "disease_code",
    }
    assert set(research.skill_registry.get_all()) == {
        "clinical_guideline", "deep_research", "search_knowledge",
    }


def memory_call(call_id, name, **arguments):
    return ToolCall(call_id, name, arguments)


def test_memory_write_precedes_same_round_worker_and_pairs_results(tmp_path):
    from memory import PatientProfileStore
    workers = {f"call_{role}_agent": WorkerStub(f"{role}_agent", "完成")
               for role in ("consultation", "diagnostic", "research")}
    proposal = {"category": "allergies", "value": "某药", "status": "active", "evidence": "我对某药过敏"}
    llm = ScriptedLLM([
        response(call("worker", "call_consultation_agent", "只解释用户问题"),
                 memory_call("write", "update_patient_profile", updates=[proposal])),
        response(content="已记录您的自述。"),
    ])
    supervisor, _, _ = build_supervisor(llm, workers)
    supervisor.patient_profiles = PatientProfileStore(tmp_path)
    result = asyncio.run(supervisor.process("我对某药过敏。", user_id="u", session_id="s"))
    context = workers["call_consultation_agent"].inputs[0]["context"]
    assert "case_context" not in context
    assert context["user_context"]["patient_profile"]["allergies"][0]["value"] == "某药"
    assert result["agents_involved"] == ["consultation_agent"]
    assert result["subtasks_completed"] == 1
    assert len(result["profile_updates"]) == 1
    assert [m["tool_call_id"] for m in llm.calls[1]["messages"] if m["role"] == "tool"] == ["worker", "write"]
    assert supervisor.patient_profiles.get_context("other-user") == {}


def test_history_search_is_bound_to_current_user_and_keeps_event_source():
    workers = {f"call_{role}_agent": WorkerStub(f"{role}_agent", "unused")
               for role in ("consultation", "diagnostic", "research")}
    llm = ScriptedLLM([response(memory_call("history", "search_patient_history", query="工作作息", limit=5)),
                       response(content="记录显示您轮班工作。")])
    supervisor, _, memory = build_supervisor(llm, workers)
    memory.search_results = [{"memory_id": "m1", "content": "用户轮班工作", "score": .8,
                              "metadata": {"user_statement": "我轮班工作", "assistant_response": "建议记录作息"}}]
    result = asyncio.run(supervisor.process("整理一下我的作息记录", user_id="u", session_id="s"))
    assert memory.searches == [("整理一下我的作息记录", "u", 3), ("工作作息", "u", 5)]
    tool = json.loads(llm.calls[1]["messages"][-1]["content"])["result"]
    assert tool["memories"][0]["assistant_response"] == "建议记录作息"
    assert "score" not in tool["memories"][0]
    assert result["agents_involved"] == []
    assert result["recalled_memories"] == 1


def test_memory_tools_deny_missing_user_or_identity_override():
    workers = {f"call_{role}_agent": WorkerStub(f"{role}_agent", "unused")
               for role in ("consultation", "diagnostic", "research")}
    for user, arguments, error_type in [(None, {"query": "病史"}, "PolicyDenied"),
                                        ("u", {"query": "病史", "user_id": "victim"}, "ValueError"),
                                        ("u", {"query": "病史", "limit": 99}, "ValueError")]:
        llm = ScriptedLLM([response(memory_call("bad", "search_patient_history", **arguments)), response(content="未执行越权查询")])
        supervisor, _, memory = build_supervisor(llm, workers)
        result = asyncio.run(supervisor.process("查历史", user_id=user, session_id="s"))
        assert result["call_trace"][0]["result"]["error_type"] == error_type
        assert all(query != "病史" for query, _, _ in memory.searches)


def test_personal_history_keywords_do_not_require_a_consultation_agent():
    workers = {f"call_{role}_agent": WorkerStub(f"{role}_agent", "unused")
               for role in ("consultation", "diagnostic", "research")}
    llm = ScriptedLLM([response(content="需要补充个人历史。")])
    supervisor, _, _ = build_supervisor(llm, workers)
    asyncio.run(supervisor.process("整理我的运动和作息历史。", session_id="s"))
    payload = json.loads(llm.calls[0]["messages"][1]["content"])
    assert "required_agents_from_safety_policy" not in payload


def test_disabled_memory_search_is_not_advertised_or_executed():
    workers = {f"call_{role}_agent": WorkerStub(f"{role}_agent", "unused")
               for role in ("consultation", "diagnostic", "research")}
    llm = ScriptedLLM([response(memory_call("bad", "search_patient_history", query="病史")), response(content="历史服务未启用")])
    supervisor, _, memory = build_supervisor(llm, workers)
    memory.enabled = False
    result = asyncio.run(supervisor.process("查历史", user_id="u", session_id="s"))
    assert "search_patient_history" not in [t["function"]["name"] for t in llm.calls[0]["tools"]]
    assert result["call_trace"][0]["result"]["error_type"] == "PolicyDenied"
    assert not memory.searches
