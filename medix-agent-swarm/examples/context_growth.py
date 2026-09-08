"""Offline trace: real harness and tool adapters, scripted model and KB responses."""

import asyncio
import copy
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from loguru import logger

from agents import ConsultationAgent, DiagnosticAgent, ResearchAgent
from core import LLMClient, LLMResponse, ToolCall
from memory import PatientProfileStore
from swarm.supervisor_agent import MedicalSupervisorAgent


BODY_A = (
    "【合成资料A，不是临床指南】本资料用于演示糖尿病运动咨询的资料组织方式。"
    "记录分为用户已确认的事实、尚未提供的信息和需要继续核对的资料来源。"
    "已确认的自述应保留来源，未提供的信息不能自动填写为不存在。"
    "资料的标题不能代替正文，概括后的结论也不能代替适用范围的核对。"
    "此处没有提供具体运动强度、时长或用药方案，因为本例只验证软件的数据传递。"
    "若同一段资料被不同查询命中，文档内容和版本不变，但检索相关度可能发生变化。"
    "系统应保留新的查询记录，同时避免在同一个模型输入中再次粘贴这段正文。"
)
BODY_B = "【合成资料B】用户更正药物状态后，应区分旧对话记录和最新自述；这是软件测试材料，不提供临床结论。"


def reply(text):
    return LLMResponse(text, [], "stop")


def invoke(call_id, name, **arguments):
    return LLMResponse(None, [ToolCall(call_id, name, arguments)], "tool_calls")


class TraceClient:
    create_tool_message = LLMClient.create_tool_message

    def __init__(self, agent_id, responses, trace, clock):
        self.agent_id = agent_id
        self.responses = list(responses)
        self.trace = trace
        self.clock = clock

    async def chat_with_tools(self, **request):
        response = self.responses.pop(0)
        self.trace.append({
            "step": len(self.trace) + 1,
            "user_turn": self.clock["turn"],
            "agent": self.agent_id,
            "request": copy.deepcopy(request),
            "scripted_response": {
                "content": response.content,
                "tool_calls": [{"id": call.id, "name": call.name, "arguments": call.arguments}
                               for call in response.tool_calls],
            },
        })
        return response


class ConversationMemory:
    enabled = False  # No external Mem0 calls in this offline example.

    def __init__(self):
        self.sessions = {}

    def get_recent_messages(self, session_id, limit=50):
        return copy.deepcopy(self.sessions.get(session_id, [])[-limit:])

    def add_message(self, session_id, role, content):
        self.sessions.setdefault(session_id, []).append({"role": role, "content": content})


class ExampleKB:
    def __init__(self):
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        result = [{"id": "B12", "content": BODY_A,
                   "metadata": {"source": "合成资料A", "version": "v1"}, "score": 0.92}]
        if "更正" in kwargs["query"]:
            result.append({"id": "B27", "content": BODY_B,
                           "metadata": {"source": "合成资料B", "version": "v1"}, "score": 0.85})
        return result


async def build_trace():
    trace = []
    clock = {"turn": 1}
    research_answer = "依据合成资料A、B，可以区分用户已提供的事实和待确认信息。资料仅用于软件演示，不构成个体化医学判断。"
    final_answer = "已整理：已知信息、待确认信息和资料来源。依据是合成资料A、B，仅供演示，不提供具体运动或用药方案。"
    supervisor_client = TraceClient("supervisor", [
        invoke("profile-1", "update_patient_profile", updates=[
            {"category": "medications", "value": "二甲双胍", "status": "active", "evidence": "目前服用二甲双胍"}]),
        invoke("delegate-r1", "call_research_agent", task="核对运动咨询的资料范围和用户信息更正的处理"),
        invoke("delegate-c1", "call_consultation_agent", task="仅将研究结论整理成易读说明，不增加医学判断"),
        reply(final_answer),
        reply("已知信息需区分来源。缺失信息需要确认。本例资料仅供软件演示。"),
        invoke("profile-2", "update_patient_profile", updates=[
            {"category": "medications", "value": "二甲双胍", "status": "stopped", "evidence": "我已经停用二甲双胍了"}]),
        invoke("delegate-r2", "call_research_agent", task="用户已停药，请重新核对资料适用边界，不提供用药判断"),
        reply("已按你的最新自述将药物状态更新为停用。旧回答需要按新事实重新核对；本例不提供临床结论。"),
    ], trace, clock)
    research_client = TraceClient("research_agent", [
        invoke("search-1", "search_knowledge", query="糖尿病运动资料适用范围"),
        invoke("search-2", "search_knowledge", query="运动咨询资料 用户信息更正"),
        reply(research_answer),
        invoke("search-3", "search_knowledge", query="糖尿病运动资料适用范围"),
        reply("已重新查看合成资料A。只能确认信息需要按新状态核对，无法给出个体化判断。仅供演示。"),
    ], trace, clock)
    research = ResearchAgent(llm_client=research_client)
    consultation = ConsultationAgent(llm_client=TraceClient(
        "consultation_agent", [reply(final_answer)], trace, clock,
    ))
    diagnostic = DiagnosticAgent(llm_client=TraceClient("diagnostic_agent", [], trace, clock))
    knowledge_base = ExampleKB()
    search_globals = research.skill_registry.get("search_knowledge")["function"].__globals__
    original_kb = search_globals["_kb_instance"]
    search_globals["_kb_instance"] = knowledge_base
    memory = ConversationMemory()
    questions = [
        "我52岁，我被医生确诊为2型糖尿病，目前服用二甲双胍。请帮我查运动资料并整理资料的适用范围。",
        "把刚才的回答缩短成三句话。",
        "我已经停用二甲双胍了。请重新核对前面资料的适用边界。",
    ]
    try:
        with TemporaryDirectory(prefix="medix-context-example-") as directory:
            supervisor = MedicalSupervisorAgent(
                llm_client=supervisor_client,
                workers={"call_research_agent": research, "call_consultation_agent": consultation,
                         "call_diagnostic_agent": diagnostic},
                short_term_memory=memory, long_term_memory=memory,
                patient_profiles=PatientProfileStore(Path(directory)),
            )
            outcomes = []
            for turn, question in enumerate(questions, 1):
                clock["turn"] = turn
                result = await supervisor.process(question, session_id="demo-session", user_id="demo-user")
                outcomes.append({"turn": turn, "question": question, "answer": result["answer"],
                                 "agents_involved": result["agents_involved"]})
    finally:
        search_globals["_kb_instance"] = original_kb

    return {
        "example": "真实Harness消息快照；模型回复与知识库为固定模拟，Mem0禁用；不是医疗评测。",
        "snapshots": trace,
        "turns": outcomes,
        "knowledge_base_calls": knowledge_base.calls,
        "persisted_conversation": memory.sessions["demo-session"],
    }


if __name__ == "__main__":
    logger.remove()
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(asyncio.run(build_trace()), ensure_ascii=False, indent=2))
