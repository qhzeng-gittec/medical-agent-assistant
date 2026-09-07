"""迭代式医疗 Supervisor：按观察结果顺序或并行调用专业 Agent。"""

import asyncio
import json
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger

from agents import ConsultationAgent, DiagnosticAgent, ResearchAgent
from constraints import ConstraintValidator
from core import LLMClient, LLMResponse, ToolCall
from memory import (
    LongTermMemory,
    LongTermMemoryError,
    PatientProfileError,
    PatientProfileStore,
    RecentHistoryBudget,
    ShortTermMemory,
)


DEFAULT_DISCLAIMER = "以上信息仅供参考，不能替代专业医生的诊断和治疗。如有疑虑，请及时就医。"


SUBAGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "call_diagnostic_agent",
            "description": "分析症状、识别危险信号并评估紧急程度。症状或严重程度问题应优先单独调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "给诊断 Agent 的具体任务"},
                    "case_context": {"type": "string", "description": "原始病例事实，不要只传推测结论"},
                },
                "required": ["task"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "call_consultation_agent",
            "description": "提供健康科普、行动建议和生活方式指导；不负责确诊。",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "给健康咨询 Agent 的具体任务"},
                    "case_context": {"type": "string", "description": "用户问题、病例事实和必要的既有发现"},
                },
                "required": ["task"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "call_research_agent",
            "description": "独立检索临床指南和医学证据，核验其他 Agent 的结论。",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "给医学研究 Agent 的具体研究任务"},
                    "case_context": {"type": "string", "description": "原始事实及需要独立核验的已有发现"},
                },
                "required": ["task"],
                "additionalProperties": False,
            },
        },
    },
]


class MedicalSupervisorAgent:
    """面向用户的唯一总控 Agent，以专业 Agent 作为工具迭代调用。"""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        workers: Optional[Dict[str, Any]] = None,
        short_term_memory: Optional[ShortTermMemory] = None,
        long_term_memory: Optional[LongTermMemory] = None,
        patient_profiles: Optional[PatientProfileStore] = None,
        recent_history_budget: Optional[RecentHistoryBudget] = None,
        max_rounds: int = 4,
        worker_timeout: float = 90.0,
    ):
        if max_rounds < 1:
            raise ValueError("max_rounds must be at least 1")
        if worker_timeout <= 0:
            raise ValueError("worker_timeout must be positive")
        self.llm_client = llm_client or LLMClient()
        self.max_rounds = max_rounds
        self.worker_timeout = worker_timeout
        self.short_term_memory = (
            short_term_memory
            if short_term_memory is not None
            else ShortTermMemory(storage_type="memory")
        )
        self.long_term_memory = (
            long_term_memory if long_term_memory is not None else LongTermMemory()
        )
        self.patient_profiles = patient_profiles or PatientProfileStore()
        self.recent_history_budget = recent_history_budget or RecentHistoryBudget()
        self.validator = ConstraintValidator()

        self.workers = workers if workers is not None else {
            "call_consultation_agent": ConsultationAgent(),
            "call_diagnostic_agent": DiagnosticAgent(),
            "call_research_agent": ResearchAgent(),
        }
        schema_names = {tool["function"]["name"] for tool in SUBAGENT_TOOLS}
        if set(self.workers) != schema_names:
            raise ValueError("Subagent tool schemas and handlers must match exactly")

        self.consultation_agent = self.workers["call_consultation_agent"]
        self.diagnostic_agent = self.workers["call_diagnostic_agent"]
        self.research_agent = self.workers["call_research_agent"]

    def get_system_prompt(self) -> str:
        return """你是医疗助手的唯一 Supervisor，负责选择专业 Agent、观察结果并给用户统一答复。

工作规则：
1. 先判断已有信息是否足够。改写、解释已有答案或必要追问可以直接完成；需要新的专业分析时再委派。
2. 症状、严重程度或是否就医的问题，第一轮只调用 DiagnosticAgent；拿到风险评估后再决定下一步。
3. 健康科普或生活方式问题通常只调用 ConsultationAgent。
4. 指南、文献、最新进展或证据核验问题调用 ResearchAgent。
5. 已有诊断结果后，行动建议和循证核验若互不依赖，可以在同一轮并行调用。
6. 给 ResearchAgent 同时传原始事实，要求独立核验，避免被诊断结论锚定。
7. 信息不足且会改变风险判断时，停止调用并直接向用户提出必要的追问。
8. 不做明确诊断，不开具体处方；高危症状必须明确建议及时就医。
9. 已获得足够信息时停止调用工具，综合各 Agent 结果生成简洁、无冲突的最终答复。
10. patient_profile 是带来源和状态的用户自述事实，其优先级高于普通历史记忆；如与当前输入冲突，以当前输入为准并提醒用户确认。
11. historical_memories 是 Mem0 语义召回的情景记忆，可能过期或不准确；不能单独作为高风险决策、诊断、处方或过敏判断的依据。
12. 跨 Agent 只传最终结果，不共享检索query、召回正文和内部工具过程。需要核验来源时委派 ResearchAgent，并要求结论保留来源和局限性。
"""

    async def process(
        self,
        question: str,
        context: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        started_at = datetime.now()
        if user_id is not None and (not isinstance(user_id, str) or not user_id.strip()):
            raise ValueError("user_id must be a non-empty string when provided")
        session_id = session_id or f"{started_at:%Y%m%d-%H%M%S}-{str(uuid.uuid4())[:8]}"
        enhanced_context, memory_warnings, profile_updates = await self._build_context(
            question, context, session_id, user_id
        )
        required_agents = self.validator.get_required_agents(question)
        force_diagnostic = "diagnostic_agent" in required_agents
        messages = self._initial_messages(question, enhanced_context, required_agents)
        records: List[Dict[str, Any]] = []
        final_answer = ""

        for round_number in range(1, self.max_rounds + 1):
            tools = self._available_tools(round_number, force_diagnostic)
            response = await self.llm_client.chat_with_tools(
                messages=messages,
                tools=tools,
                tool_choice="required" if round_number == 1 and force_diagnostic else "auto",
                temperature=0.2,
            )
            messages.append(self._assistant_message(response))

            if not response.has_tool_calls():
                if round_number == 1 and force_diagnostic:
                    raise RuntimeError("高风险问题第一轮必须调用诊断 Agent")
                final_answer = response.content or ""
                break

            round_records = await self._execute_calls(
                response.tool_calls,
                question,
                enhanced_context,
                records,
                round_number,
                {tool["function"]["name"] for tool in tools},
            )
            records.extend(round_records)
            messages.extend(self._tool_messages(response.tool_calls, round_records))
        else:
            final_answer = await self._force_final_answer(messages)

        if not final_answer.strip():
            raise RuntimeError("Supervisor finished without a final answer")

        final_answer = self._enforce_output_safety(final_answer, force_diagnostic)
        await self._save_memory(
            user_id,
            session_id,
            question,
            final_answer,
            records,
            started_at,
            memory_warnings,
        )
        return self._build_result(
            session_id,
            final_answer,
            records,
            started_at,
            user_id,
            enhanced_context,
            memory_warnings,
            profile_updates,
        )

    async def _build_context(
        self,
        question: str,
        context: Optional[Dict[str, Any]],
        session_id: str,
        user_id: Optional[str],
    ) -> tuple[Dict[str, Any], List[str], List[Dict[str, Any]]]:
        enhanced = dict(context or {})
        warnings: List[str] = []
        profile_updates: List[Dict[str, Any]] = []
        if user_id:
            try:
                profile_updates = await asyncio.to_thread(
                    self.patient_profiles.update_from_user_message,
                    user_id,
                    question,
                    session_id,
                )
                patient_profile = await asyncio.to_thread(
                    self.patient_profiles.get_context,
                    user_id,
                )
                if patient_profile:
                    enhanced["patient_profile"] = patient_profile
            except PatientProfileError as error:
                logger.warning(str(error))
                warnings.append(str(error))
        recent_messages = self.short_term_memory.get_recent_messages(session_id, limit=50)
        recent = self.recent_history_budget.select(recent_messages)
        if recent:
            enhanced["recent_history"] = [
                {"role": message.get("role", ""), "content": message.get("content", "")}
                for message in recent
            ]
        if user_id and getattr(self.long_term_memory, "enabled", True):
            try:
                similar = await asyncio.to_thread(
                    self.long_term_memory.search_similar_sessions,
                    question,
                    user_id,
                    3,
                )
                if similar:
                    enhanced["historical_memories"] = [
                        {"memory": item["content"], "score": item.get("score")}
                        for item in similar
                    ]
            except LongTermMemoryError as error:
                logger.warning(str(error))
                warnings.append(str(error))
        return enhanced, warnings, profile_updates

    def _initial_messages(
        self,
        question: str,
        context: Dict[str, Any],
        required_agents: List[str],
    ) -> List[Dict[str, Any]]:
        payload = {
            "question": question,
            "context": context,
            "required_agents_from_safety_policy": required_agents,
        }
        return [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, default=str)},
        ]

    def _available_tools(self, round_number: int, force_diagnostic: bool) -> List[Dict[str, Any]]:
        if round_number == 1 and force_diagnostic:
            return [tool for tool in SUBAGENT_TOOLS if tool["function"]["name"] == "call_diagnostic_agent"]
        return SUBAGENT_TOOLS

    async def _execute_calls(
        self,
        calls: List[ToolCall],
        question: str,
        context: Dict[str, Any],
        prior_records: List[Dict[str, Any]],
        round_number: int,
        allowed_names: set[str],
    ) -> List[Dict[str, Any]]:
        seen = set()
        tasks = []
        for call in calls:
            duplicate = call.name in seen
            seen.add(call.name)
            tasks.append(self._execute_call(
                call,
                question,
                context,
                prior_records,
                round_number,
                duplicate,
                allowed_names,
            ))
        return await asyncio.gather(*tasks)

    async def _execute_call(
        self,
        call: ToolCall,
        question: str,
        context: Dict[str, Any],
        prior_records: List[Dict[str, Any]],
        round_number: int,
        duplicate: bool,
        allowed_names: set[str],
    ) -> Dict[str, Any]:
        if call.name not in self.workers:
            return self._error_record(call, round_number, "UnknownTool", f"未知 Subagent：{call.name}")
        if call.name not in allowed_names:
            return self._error_record(call, round_number, "PolicyDenied", "当前阶段不允许调用该 Subagent")
        if duplicate:
            return self._error_record(call, round_number, "DuplicateCall", "同一轮不能重复调用同一 Subagent")

        task = call.arguments.get("task") if isinstance(call.arguments, dict) else None
        case_context = call.arguments.get("case_context", "") if isinstance(call.arguments, dict) else ""
        if not isinstance(task, str) or not task.strip():
            return self._error_record(call, round_number, "InvalidArguments", "task 必须是非空字符串")
        if not isinstance(case_context, str):
            return self._error_record(call, round_number, "InvalidArguments", "case_context 必须是字符串")

        worker_context = {
            "original_question": question,
            "case_context": case_context or question,
            "user_context": context,
            "prior_agent_findings": [
                {"agent": item["agent_id"], "result": item["result"]}
                for item in prior_records if item["success"]
            ],
        }
        try:
            result = await asyncio.wait_for(
                self.workers[call.name].process({"question": task, "context": worker_context}),
                timeout=self.worker_timeout,
            )
            if not isinstance(result, dict):
                raise TypeError("Subagent result must be a dictionary")
            return {
                "tool_call_id": call.id,
                "tool_name": call.name,
                "agent_id": self.workers[call.name].agent_id,
                "round": round_number,
                "success": True,
                "result": self._bounded_result(result),
            }
        except asyncio.TimeoutError:
            return self._error_record(call, round_number, "TimeoutError", f"Subagent 超过 {self.worker_timeout} 秒")
        except Exception as error:
            logger.exception(f"Subagent {call.name} failed")
            return self._error_record(call, round_number, type(error).__name__, str(error))

    def _error_record(
        self,
        call: ToolCall,
        round_number: int,
        error_type: str,
        message: str,
    ) -> Dict[str, Any]:
        worker = self.workers.get(call.name)
        return {
            "tool_call_id": call.id,
            "tool_name": call.name,
            "agent_id": getattr(worker, "agent_id", call.name),
            "round": round_number,
            "success": False,
            "result": {"error_type": error_type, "message": message},
        }

    def _bounded_result(self, result: Dict[str, Any], max_chars: int = 8000) -> Dict[str, Any]:
        # Only the final deliverable crosses the worker boundary, never its transcript.
        answer = result.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Subagent 必须返回非空的最终 answer")
        bounded = {"answer": answer[:max_chars]}
        for key in ("warning", "error"):
            if key in result:
                bounded[key] = str(result[key])[:500]
        if len(answer) > max_chars:
            bounded["truncated"] = True
        return bounded

    def _assistant_message(self, response: LLMResponse) -> Dict[str, Any]:
        message: Dict[str, Any] = {"role": "assistant", "content": response.content or None}
        if response.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments, ensure_ascii=False),
                    },
                }
                for call in response.tool_calls
            ]
        return message

    def _tool_messages(
        self,
        calls: List[ToolCall],
        records: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        by_id = {record["tool_call_id"]: record for record in records}
        return [
            {
                "role": "tool",
                "tool_call_id": call.id,
                "name": call.name,
                "content": json.dumps(by_id[call.id], ensure_ascii=False, default=str),
            }
            for call in calls
        ]

    async def _force_final_answer(self, messages: List[Dict[str, Any]]) -> str:
        messages.append({
            "role": "user",
            "content": "已达到 Supervisor 调用轮数上限。请基于现有结果立即给出最终答复，不再调用工具。",
        })
        response = await self.llm_client.chat_with_tools(
            messages=messages,
            tools=None,
            temperature=0.2,
        )
        return response.content or ""

    def _enforce_output_safety(self, answer: str, high_risk: bool) -> str:
        answer = answer.replace("您患有", "可能存在").replace("确诊为", "建议检查以确认")
        if high_risk and not any(word in answer for word in ("就医", "急诊", "医院", "120")):
            warning = "⚠️ 你描述的症状可能存在紧急风险，建议立即就医或拨打急救电话120，不要延误。\n\n"
            answer = warning + answer
        return answer

    async def _save_memory(
        self,
        user_id: Optional[str],
        session_id: str,
        question: str,
        answer: str,
        records: List[Dict[str, Any]],
        started_at: datetime,
        memory_warnings: List[str],
    ) -> None:
        self.short_term_memory.add_message(session_id, "user", question)
        self.short_term_memory.add_message(session_id, "assistant", answer)
        if not user_id or not getattr(self.long_term_memory, "enabled", True):
            return
        try:
            await asyncio.to_thread(
                self.long_term_memory.add_session_summary,
                user_id=user_id,
                session_id=session_id,
                question=question,
                answer=answer,
                metadata={
                    "mode": "iterative_supervisor",
                    "agents_count": len({item["agent_id"] for item in records if item["success"]}),
                    "total_time": (datetime.now() - started_at).total_seconds(),
                },
            )
        except LongTermMemoryError as error:
            logger.warning(str(error))
            memory_warnings.append(str(error))

    def _build_result(
        self,
        session_id: str,
        answer: str,
        records: List[Dict[str, Any]],
        started_at: datetime,
        user_id: Optional[str],
        context: Dict[str, Any],
        memory_warnings: List[str],
        profile_updates: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        agents = list(dict.fromkeys(item["agent_id"] for item in records if item["success"]))
        suggestions = re.findall(r"\d+\.\s*([^\n]+)", answer)[:5]
        return {
            "answer": answer,
            "session_id": session_id,
            "mode": "iterative_supervisor",
            "swarm_enabled": len(agents) > 1,
            "agents_involved": agents,
            "subtasks_completed": sum(1 for item in records if item["success"]),
            "supervisor_rounds": max((item["round"] for item in records), default=0) + 1,
            "total_time": (datetime.now() - started_at).total_seconds(),
            "timeout_occurred": any(
                item["result"].get("error_type") == "TimeoutError"
                for item in records if not item["success"]
            ),
            "call_trace": records,
            "route_reason": "Supervisor 根据每轮 Subagent 结果动态决定后续调用",
            "long_term_memory_enabled": bool(
                user_id and getattr(self.long_term_memory, "enabled", True)
            ),
            "recalled_memories": len(context.get("historical_memories", [])),
            "patient_profile_enabled": bool(user_id),
            "profile_updates": profile_updates,
            "memory_warnings": memory_warnings,
            "suggestions": suggestions,
            "disclaimer": DEFAULT_DISCLAIMER,
        }


async def process_medical_question(
    question: str,
    context: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    supervisor = MedicalSupervisorAgent()
    return await supervisor.process(question, context, session_id, user_id)
