"""Run patient tasks through the real project Supervisor and Worker classes."""

import argparse
import asyncio
import copy
import json
import sys
import time
from pathlib import Path

from loguru import logger

from campaign_gateway import EMBED_MODEL, Gateway, MODELS, ModelAdapter, dump, sha
from campaign_rag import LocalRAG
from campaign_grade import grade_run


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(1, str(PROJECT.parent))
DATA = Path(__file__).parent / "campaign_v1"
OUTPUT = Path(__file__).parent / "results" / "full_system_v1"


def parse_json_answer(content: str) -> dict:
    text = content.strip()
    if text.startswith("```") and text.endswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(text)


async def patient_reply(gateway, case, answer, already, trace):
    if not case["patient_facts"]:
        return None
    messages = [{"role": "system", "content":
                 "你是问诊测试的患者事实选择器，不负责医学判断。只判断医生是否在向患者提问。"
                 "从给定事实中选择与提问相关的条目编号，不能创造任何新事实，不能主动展开全部病历。"
                 "如果医生没有问题，needs_reply=false。若有问题但事实中无答案，needs_reply=true且fact_ids为空。"
                 "返回JSON：{\"needs_reply\":true或false,\"fact_ids\":[0,1]}。不要解释。"},
                {"role": "user", "content": json.dumps({"doctor_reply": answer,
                 "facts": dict(enumerate(case["patient_facts"])), "already_revealed": already}, ensure_ascii=False)}]
    data = await gateway.chat(MODELS[1], messages, trace, "patient_selector", max_tokens=4096)
    choice = data["choices"][0]
    if choice["finish_reason"] != "stop":
        raise RuntimeError("Patient selector response incomplete")
    obj = parse_json_answer(choice["message"]["content"])
    if not isinstance(obj["needs_reply"], bool) or not isinstance(obj["fact_ids"], list):
        raise ValueError("Invalid patient selector shape")
    ids = obj["fact_ids"]
    if any(type(i) is not int or not 0 <= i < len(case["patient_facts"]) for i in ids):
        raise ValueError("Patient selector invented fact ID")
    if not obj["needs_reply"]:
        return None
    already.extend(i for i in ids if i not in already)
    return " ".join(case["patient_facts"][i] for i in dict.fromkeys(ids)) or "这个情况我不知道，还没有检查。"


async def unavailable_deep_research(query: str, max_iterations=2):
    return {"success": False, "status": "error", "error": "ExternalDeepResearchNotConfigured",
            "answer": "本次测试未配置独立外网搜索服务；可使用现有本地知识库工具，不能声称已完成外网研究。"}


def build_system(gateway, model, trace, kb, profile_path):
    from agents import ConsultationAgent, DiagnosticAgent, ResearchAgent
    from memory import LongTermMemory, PatientProfileStore, ShortTermMemory
    from swarm.supervisor_agent import MedicalSupervisorAgent

    workers = {}
    for cls, role, tool in [(DiagnosticAgent, "diagnostic_agent", "call_diagnostic_agent"),
                             (ConsultationAgent, "consultation_agent", "call_consultation_agent"),
                             (ResearchAgent, "research_agent", "call_research_agent")]:
        worker = cls(llm_client=ModelAdapter(gateway, model, trace, role))
        # Only the provider wire-message preservation is adapted; routing/prompts are untouched.
        original_formatter = worker.loop._create_assistant_message_with_tools
        worker.loop._create_assistant_message_with_tools = (
            lambda response, original=original_formatter: copy.deepcopy(getattr(response, "wire_message", None))
            or original(response))
        for name, spec in worker.skill_registry.skills.items():
            if name == "deep_research":
                spec["function"] = unavailable_deep_research
            elif "_kb_instance" in spec["function"].__globals__:
                spec["function"].__globals__["_kb_instance"] = kb
        workers[tool] = worker
    profile = PatientProfileStore(profile_path)
    supervisor = MedicalSupervisorAgent(
        llm_client=ModelAdapter(gateway, model, trace, "supervisor"), workers=workers,
        short_term_memory=ShortTermMemory(storage_type="memory"),
        long_term_memory=LongTermMemory(config={}), patient_profiles=profile,
        max_rounds=4, worker_timeout=120)
    original_formatter = supervisor._assistant_message
    supervisor._assistant_message = (
        lambda response: copy.deepcopy(getattr(response, "wire_message", None)) or original_formatter(response))
    return supervisor, profile


def trace_metrics(trace: list) -> dict:
    apis = [e for e in trace if e["event"] == "api_request"]
    target = [e for e in apis if e["role"] in {"supervisor", "diagnostic_agent", "research_agent", "consultation_agent"}]
    usage = [e.get("response", {}).get("usage", {}) for e in target]
    calls, violations = 0, []
    for event in target:
        payload = event["payload"]
        advertised = {t["function"]["name"] for t in payload.get("tools", [])}
        for choice in event.get("response", {}).get("choices", []):
            for call in choice["message"].get("tool_calls", []):
                calls += 1
                if call["function"]["name"] not in advertised:
                    violations.append(call["function"]["name"])
    observations = [e for e in trace if e["event"] == "tool_observation"]
    blocks = [b for e in observations for b in e["result"].get("documents", [])]
    return {"model_requests": len(target), "requested_tool_calls": calls,
            "retrieval_backend_executions": sum(e["event"] == "retrieval" for e in trace),
            "document_bodies_added": sum("content" in b for b in blocks),
            "document_references_added": sum("reference" in b for b in blocks),
            "document_body_characters": sum(len(b.get("content", "")) for b in blocks),
            "prompt_tokens": sum(u.get("prompt_tokens", 0) for u in usage),
            "completion_tokens": sum(u.get("completion_tokens", 0) for u in usage),
            "target_cost_usd": sum(e["cost_usd"] for e in target),
            "all_api_cost_usd": sum(e["cost_usd"] for e in apis),
            "model_requested_unadvertised_tools": violations,
            "provider_error_count": sum(e.get("status") == "error" for e in target),
            "incomplete_response_count": sum(c.get("finish_reason") not in {"stop", "tool_calls"}
                for e in target for c in e.get("response", {}).get("choices", []))}


async def run_case(gateway, model, case, repetition):
    run_id = f"{case['case_id']}__{model.replace('/', '_')}__r{repetition}"
    destination = gateway.root / "runs" / f"{run_id}.json"
    if destination.exists():
        return json.loads(destination.read_text(encoding="utf-8"))
    trace, turns = [], []
    started = time.monotonic()
    result = {"run_id": run_id, "case_id": case["case_id"], "model": model,
              "category": case["category"], "environment": case["environment"],
              "repetition": repetition, "status": "running", "turns": turns,
              "dataset_sha256": sha(DATA / "cases.private.jsonl"), "embedding_model": EMBED_MODEL}
    user = f"{run_id}-user-a"
    session = f"{run_id}-session-1"
    revealed = []
    try:
        kb = LocalRAG(gateway, DATA / "corpus.jsonl", trace, case["environment"])
        supervisor, profile = build_system(gateway, model, trace, kb, gateway.root / "profiles" / run_id)

        async def turn(text):
            trace.append({"event": "turn_start", "turn": len(turns) + 1,
                          "user_id": user, "session_id": session, "text": text})
            start = time.monotonic()
            before = profile.get_context(user)
            response = await supervisor.process(text, user_id=user, session_id=session)
            turns.append({"user": text, "answer": response["answer"], "system_result": response,
                          "user_id": user, "session_id": session,
                          "profile_before": before, "profile_after": profile.get_context(user),
                          "latency_seconds": round(time.monotonic() - start, 3)})
            trace.append({"event": "turn_end", "turn": len(turns)})
            return response["answer"]

        answer = await turn(case["opening"])
        for _ in range(case["max_patient_replies"]):
            reply = await patient_reply(gateway, case, answer, revealed, trace)
            if reply is None:
                break
            answer = await turn(reply)
        for index, followup in enumerate(case["scripted_followups"], 2):
            if followup.get("new_user"):
                user = f"{run_id}-user-b"
            if followup.get("new_session") or followup.get("new_user"):
                session = f"{run_id}-session-{index}"
            answer = await turn(followup["text"])
        result["status"] = "completed"
        result["patient_fact_ids_revealed"] = revealed
    except Exception as error:
        result.update(status="error", error_type=type(error).__name__,
                      error=str(error).replace(gateway.key, "[REDACTED]"))
    finally:
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        result["metrics"] = trace_metrics(trace)
        dump(gateway.root / "traces" / f"{run_id}.json", trace)
        dump(destination, result)
    print(json.dumps({"run_id": run_id, "status": result["status"], "turns": len(turns),
                      "elapsed": result["elapsed_seconds"], "spent": round(gateway.spent, 4)}, ensure_ascii=False), flush=True)
    return result


async def probe(gateway):
    results = []
    for model in MODELS:
        trace = []
        try:
            tools = [{"type": "function", "function": {"name": "echo", "description": "Echo a value",
                     "parameters": {"type": "object", "properties": {"value": {"type": "string"}},
                                    "required": ["value"], "additionalProperties": False}}}]
            messages = [{"role": "user", "content": "Call echo with value OK, then reply with the returned value."}]
            data = await gateway.chat(model, messages, trace, "probe", tools, "required", 2048)
            choice = data["choices"][0]
            if choice["finish_reason"] != "tool_calls":
                raise ValueError(f"Expected tool_calls, got {choice['finish_reason']}")
            message = choice["message"]
            messages.append({k: v for k, v in message.items() if k in {"role", "content", "tool_calls", "reasoning", "reasoning_details"}})
            for call in message["tool_calls"]:
                if call["function"]["name"] != "echo" or json.loads(call["function"]["arguments"]) != {"value": "OK"}:
                    raise ValueError("Unexpected echo arguments")
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": '{"value":"OK"}'})
            final = await gateway.chat(model, messages, trace, "probe", max_tokens=2048)
            content = final["choices"][0]["message"].get("content")
            if final["choices"][0]["finish_reason"] != "stop" or not content or "OK" not in content:
                raise ValueError("No completed final echo")
            results.append({"model": model, "passed": True, "answer": content})
        except Exception as error:
            results.append({"model": model, "passed": False, "error": str(error).replace(gateway.key, "[REDACTED]")})
        dump(gateway.root / "probes" / f"{model.replace('/', '_')}.json", trace)
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)
    dump(gateway.root / "probe_summary.json", results)
    return results


async def main(args):
    logger.remove()
    logger.add(sys.stderr, level="ERROR", diagnose=False, backtrace=False)
    gateway = Gateway(args.output, args.budget)
    if args.probe:
        await probe(gateway)
        return
    sources = list((PROJECT / "agents").glob("*.py")) + list((PROJECT / "core").glob("*.py"))
    sources += list((PROJECT / "swarm").glob("*.py")) + list((PROJECT / "memory").glob("*.py"))
    sources += list((PROJECT / ".claude/skills").rglob("*.py"))
    snapshot = {str(p.relative_to(PROJECT)): sha(p) for p in sources}
    snapshot_path = gateway.root / "product_source_hashes.json"
    if snapshot_path.exists() and json.loads(snapshot_path.read_text(encoding="utf-8")) != snapshot:
        raise ValueError("Product source changed; use a different campaign output directory")
    dump(snapshot_path, snapshot)
    protocol = {
        "dataset_sha256": sha(DATA / "cases.private.jsonl"), "corpus_sha256": sha(DATA / "corpus.jsonl"),
        "temperature": 0.2, "reasoning_effort": "low", "max_output_tokens": 2048,
        "supervisor_rounds": 4, "worker_tool_budget": 2, "models": MODELS,
        "embedding_model": EMBED_MODEL, "budget_usd": args.budget,
        "adaptations": ["real API embeddings + local exact cosine replaces unavailable Milvus",
                        "provider reasoning_details preserved by evaluation transport",
                        "deep_research returns explicit not-configured error, never fake success",
                        "Mem0 Platform disabled: no credentials; Profile persistence is real",
                        "fine-tuned medical model not connected: no serving endpoint configured"],
        "clinical_review": "pending", "patient_selector": MODELS[1],
        "patient_selector_max_output_tokens": 4096,
        "patient_selector_reasoning": "disabled; target Agent models remain low effort",
        "parallel_patient_tasks": args.concurrency,
        "execution_isolation": "one event loop per patient task in separate threads",
        "query_instruction": "English medical retrieval task prefix; documents have no prefix"}
    dump(gateway.root / "run_protocol.json", protocol)
    dump(gateway.root / "protocols" / f"r{args.repetition}.json", protocol)
    cases = [json.loads(line) for line in (DATA / "cases.private.jsonl").read_text(encoding="utf-8").splitlines()]
    planned_case_ids = [case["case_id"] for case in cases]
    if args.cases:
        cases = [case for case in cases if case["case_id"] in args.cases.split(",")]
    models = MODELS if not args.model else [args.model]
    probe_path = gateway.root / "probe_summary.json"
    if probe_path.exists():
        unavailable = {row["model"] for row in json.loads(probe_path.read_text(encoding="utf-8"))
                       if not row["passed"] and "403" in row.get("error", "")}
        models = [model for model in models if model not in unavailable]
        dump(gateway.root / "unavailable_models.json", [{"model": model,
             "status": "provider_access_denied", "planned_case_ids": planned_case_ids,
             "reason": "Preflight HTTP 403; not retried or counted as clinical/model failure"}
             for model in sorted(unavailable)])
    semaphore = asyncio.Semaphore(args.concurrency)

    async def execute(case, model):
        async with semaphore:
            # Synchronous embedding inside a Skill must not stall another patient's event loop.
            result = await asyncio.to_thread(lambda: asyncio.run(run_case(gateway, model, case, args.repetition)))
            if args.grade:
                await grade_run(gateway, case, result)

    await asyncio.gather(*(execute(case, model) for case in cases for model in models))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--budget", type=float, default=5)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--cases")
    parser.add_argument("--repetition", type=int, default=1)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--grade", action="store_true")
    parser.add_argument("--concurrency", type=int, choices=range(1, 9), default=4)
    asyncio.run(main(parser.parse_args()))
