"""Development regression iterations using real workers and the existing blinded rubric.

Known cases are development data, not an unseen holdout. Each CLI invocation owns
one arm, runs cases serially, and freezes its source before any paid execution.
"""
import argparse
import asyncio
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import httpx
import portalocker
from loguru import logger
from openai import OpenAI

from campaign_gateway import dump, sha
from campaign_run import PROJECT, build_system, trace_metrics
from iteration_grade import grade_run
from holdout_run import execution_case, patient_reply, annotate_events
from holdout_services import FixtureKnowledge, FixtureMemory, HoldoutGateway
from campaign_rag import LocalRAG
from memory.long_term import LongTermMemory

CASE_IDS = ["HH-011", "HE-005", "HE-012", "HC-001", "HC-002", "HC-003"]
DATA = Path(__file__).parent / "holdout_v1_20260908/sealed/authors"


class IterationGateway(HoldoutGateway):
    async def chat(self, model, messages, trace, role, tools=None, tool_choice="auto", max_tokens=8192):
        max_tokens = 16384 if role == "rubric_judge" else 4096 if role == "patient_selector" else 8192
        return await super().chat(model, messages, trace, role, tools, tool_choice, max_tokens)


class LocalMemory(LongTermMemory):
    """The product Mem0 implementation; only transport metering/fixtures are adapted."""
    backend_label = "actual_local_mem0_oss"

    def __init__(self, gateway, trace, user_ids, path):
        os.environ["MEM0_LOCAL_PATH"] = str(path)
        super().__init__({"storage_path": str(path)})
        self.trace, self.user_ids, self.seed_replay = trace, user_ids, False

        def transport(request):
            endpoint = "embeddings" if request.url.path.endswith("embeddings") else "chat/completions"
            role = "memory_embedding" if endpoint == "embeddings" else "memory_extraction"
            data = gateway.request(endpoint, json.loads(request.content), trace, role)
            return httpx.Response(200, json=data, request=request)

        for component in (self.client.llm, self.client.embedding_model):
            component.client.close()
            component.client = OpenAI(api_key=gateway.key, base_url="https://openrouter.ai/api/v1",
                                      max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(transport)))

    def add_session_summary(self, **kwargs):
        if self.seed_replay:
            return None
        result = super().add_session_summary(**kwargs)
        self.trace.append({"event": "memory_write", "user_id": kwargs["user_id"], "operation_id": result})
        return result

    def search_similar_sessions(self, query, user_id, limit=3):
        if self.seed_replay:
            return []
        result = super().search_similar_sessions(query, user_id, limit)
        self.trace.append({"event": "memory_search", "query": query, "user_id": user_id,
                           "limit": limit, "memories": copy.deepcopy(result)})
        return result

    def prepare_fixture(self, seed_history):
        for history in seed_history:
            self.client.add(messages=history["messages"], user_id=self.user_ids[history["user_key"]],
                            agent_id=self.app_id, metadata={"source_session_id": history["session"]})

    def snapshot(self, user_id):
        with self._lock:
            records = self.client.get_all(filters={"user_id": user_id, "agent_id": self.app_id}, top_k=100)
        return {"backend": self.backend_label, "write_observation_available": True, "memories": records}


def freeze(root, cases, model, budget, purpose="known-case development regression"):
    files = sorted(p for folder in ("agents", "core", "memory", "swarm", ".claude/skills", "knowledge", "research")
                   for p in (PROJECT / folder).rglob("*") if p.suffix in {".py", ".md", ".yaml", ".json"})
    files += [Path(__file__), *[PROJECT / "evals" / name for name in (
        "campaign_run.py", "campaign_gateway.py", "campaign_rag.py", "holdout_run.py", "holdout_services.py", "holdout_grade.py", "iteration_grade.py")], PROJECT / "evals/campaign_v1/corpus.jsonl"]
    manifest = {str(p.relative_to(PROJECT)): sha(p) for p in files}
    protocol = {"purpose": purpose, "case_ids": [c["case_id"] for c in cases],
                "case_sha256": hashlib.sha256(json.dumps(cases, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
                "model": model, "max_tool_calls": budget, "worker_iterations": 5, "supervisor_rounds": 4,
                "worker_timeout_seconds": 120, "turn_timeout_seconds": 600,
                "target_max_output_tokens": 8192, "judge_max_output_tokens": 16384,
                "judges": ["minimax/minimax-m2.5", "qwen/qwen3.5-27b"],
                "case_isolation": "one case per fresh child process",
                "grader": "iteration-v1-explicit-timing-and-evidence-shape",
                "memory": "local Mem0 for live_mem0 cases; otherwise original fixtures",
                "rag": "original case modes: controlled snippets or live embeddings over frozen corpus; no external web search",
                "sources": manifest}
    path = root / "protocol.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != protocol:
            raise ValueError("Inputs changed; choose a new arm directory")
    else:
        dump(path, protocol)
        dump(root / "cases.private.json", cases)
        for file in files:
            dest = root / "source" / file.relative_to(PROJECT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(file.read_bytes())
    return protocol


async def run_case(gateway, raw, model, budget, protocol):
    case = execution_case(raw)
    run_id = f"{case['case_id']}__{model.replace('/', '_')}__r1"
    dest = gateway.root / "runs" / f"{run_id}.json"
    if dest.exists():
        result = json.loads(dest.read_text(encoding="utf-8"))
        return result, json.loads((gateway.root / "traces" / f"{run_id}.json").read_text(encoding="utf-8"))
    trace, turns, seeds, revealed = [], [], [], []
    # Short physical paths avoid MAX_PATH in SQLite and atomic profile saves.
    state = gateway.root / "state" / case["case_id"]
    users = {key: f"iteration-{case['case_id']}-{key}" for key in ("self", "other")}
    result = {"run_id": run_id, "case_id": case["case_id"], "model": model, "repetition": 1,
              "status": "running", "turns": turns, "seed_turns": seeds, "user_ids": users,
              "protocol": protocol, "environment": case["environment"], "patient_fact_ids_revealed": revealed}
    memory = None
    started = time.monotonic()
    try:
        env = case["environment"]
        if env["rag_mode"] == "live_rag":
            # Serialize creation of the shared immutable corpus index across case processes.
            with portalocker.Lock(str(gateway.root / "corpus.lock"), timeout=240):
                kb = LocalRAG(gateway, PROJECT / "evals/campaign_v1/corpus.jsonl", trace)
            kb.backend_label = "live_qwen_api_embeddings_local_cosine"
        else:
            kb = FixtureKnowledge(env, trace)
        memory = (LocalMemory(gateway, trace, users, state / "mem0") if env["memory_mode"] == "live_mem0"
                  else FixtureMemory(env, users, trace))
        result["environment_backends"] = {"rag": kb.backend_label, "memory": memory.backend_label,
                                           "profile": "actual_product_PatientProfileStore"}
        supervisor, profile = build_system(gateway, model, trace, kb, state / "profiles", memory)
        for worker in supervisor.workers.values():
            worker.loop.max_tool_calls = budget

        async def turn(text, user, session, phase="target", prefix=None):
            offset, start = len(trace), time.monotonic()
            rows = seeds if phase == "seed" else turns
            before, raw_before = profile.get_context(user), profile._load(user)
            response = await asyncio.wait_for(supervisor.process(text, user_id=user, session_id=session), 600)
            if not response.get("answer", "").strip():
                raise ValueError("No user-visible answer")
            row = {"turn": len(rows) + 1, "user": text, "answer": response["answer"],
                   "user_id": user, "session_id": session, "profile_before": before,
                   "profile_record_before": raw_before, "profile_after": profile.get_context(user),
                   "profile_record_after": profile._load(user), "system_result": response,
                   "event_memory_after": memory.snapshot(user), "latency_seconds": time.monotonic() - start}
            if phase == "seed":
                row["authored_history_before"] = copy.deepcopy(prefix)
            rows.append(row)
            annotate_events(trace, offset, phase, len(rows))
            dump(gateway.root / "progress" / f"{run_id}.json", result)
            dump(gateway.root / "progress" / f"{run_id}.trace.json", trace)
            print(json.dumps({"case": case["case_id"], "phase": phase, "turn": len(rows),
                              "seconds": round(row["latency_seconds"], 1)}, ensure_ascii=False), flush=True)
            return response["answer"]

        memory.seed_replay = True
        for history in case["seed_history"]:
            prefix = []
            session = f"{users[history['user_key']]}-seed-{history['session']}"
            for message in history["messages"]:
                if message["role"] == "user":
                    supervisor.short_term_memory.clear_session(session)
                    for previous in prefix:
                        supervisor.short_term_memory.add_message(session, previous["role"], previous["content"])
                    await turn(message["content"], users[history["user_key"]], session, "seed", prefix)
                prefix.append(message)
        memory.seed_replay = False
        await asyncio.to_thread(memory.prepare_fixture, case["seed_history"])
        result["seed_event_memory_after"] = {key: memory.snapshot(user) for key, user in users.items()}
        user, session = users["self"], f"{run_id}-target-1"
        answer = await turn(case["interaction"]["opening"], user, session)
        for _ in range(case["interaction"]["max_patient_replies"]):
            reply = await patient_reply(gateway, case["patient_facts"], answer, revealed, trace)
            if reply is None:
                break
            answer = await turn(reply, user, session)
        for index, followup in enumerate(case["interaction"]["scripted_followups"], 2):
            if followup["new_user"]:
                user = users["other"]
            if followup["new_session"] or followup["new_user"]:
                session = f"{run_id}-target-{index}"
            await turn(followup["text"], user, session)
        result["status"] = "completed"
    except Exception as error:
        result.update(status="error", error=gateway.redact(error), error_type=type(error).__name__)
    finally:
        if isinstance(memory, LocalMemory):
            memory.close()
        result["elapsed_seconds"] = time.monotonic() - started
        result["metrics"] = trace_metrics(trace)
        trace_path = gateway.root / "traces" / f"{run_id}.json"
        dump(trace_path, trace)
        result["trace_sha256"] = sha(trace_path)
        dump(dest, result)
    print(json.dumps({"case": case["case_id"], "status": result["status"],
                      "error": result.get("error"), "metrics": result["metrics"]}, ensure_ascii=False), flush=True)
    return result, trace


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="qwen/qwen3.5-27b")
    parser.add_argument("--tool-budget", type=int)
    parser.add_argument("--cases", nargs="+", default=CASE_IDS)
    parser.add_argument("--worker-case")
    parser.add_argument("--purpose", default="known-case development regression")
    parser.add_argument("--dataset", type=Path, help="Explicit JSON case array; frozen into each run protocol")
    args = parser.parse_args()
    logger.remove()
    args.output.mkdir(parents=True, exist_ok=True)
    all_cases = json.loads(args.dataset.read_text(encoding="utf-8")) if args.dataset else [json.loads(line) for name in ("history", "evidence", "consultation")
                 for line in (DATA / f"{name}.jsonl").read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    cases = [next(c for c in all_cases if c["case_id"] == cid) for cid in args.cases]
    protocol = freeze(args.output, cases, args.model, args.tool_budget, args.purpose)
    if not args.worker_case:
        for case in cases:
            command = [sys.executable, str(Path(__file__).resolve()), "--output", str(args.output.resolve()),
                       "--model", args.model, "--purpose", args.purpose, "--cases", *args.cases, "--worker-case", case["case_id"]]
            if args.tool_budget is not None:
                command += ["--tool-budget", str(args.tool_budget)]
            if args.dataset:
                command += ["--dataset", str(args.dataset.resolve())]
            process = await asyncio.create_subprocess_exec(*command)
            if await process.wait():
                raise RuntimeError(f"Case worker exited unsuccessfully: {case['case_id']}")
        return
    if args.worker_case not in args.cases:
        raise ValueError("Worker case is outside the frozen selection")
    # OS-owned locks release on process exit, including interrupted runs.
    with portalocker.Lock(str(args.output / f"{args.worker_case}.lock"), timeout=0):
        gateway = IterationGateway(args.output)
        for case in [c for c in cases if c["case_id"] == args.worker_case]:
            for relative, digest in protocol["sources"].items():
                if sha(PROJECT / relative) != digest:
                    raise ValueError(f"Source changed during execution: {relative}")
            result, trace = await run_case(gateway, case, args.model, args.tool_budget, protocol)
            grade = await grade_run(gateway, case, result, trace)
            print(json.dumps({"case": case["case_id"], "grade": grade["status"],
                              "all_pass": grade["all_applicable_passed"], "spent_usd": gateway.spent}), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
