"""Paired old/new product runs on a fixed synthetic development set."""

import argparse
import asyncio
import copy
import json
import sys
import time
from pathlib import Path

from loguru import logger

from campaign_gateway import Gateway, MODELS, dump, sha
from campaign_gemini_memory import CodexGateway, MODEL as CODEX_MODEL
from campaign_grade import protocol_checks
from campaign_rag import LocalRAG
from campaign_run import DATA, OUTPUT, PROJECT, build_system, trace_metrics


ROOT = OUTPUT.parent / "improvement_codex_v2"
CASES = Path(__file__).with_name("improvement_cases.json")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


class ComparisonCodex(CodexGateway):
    def __init__(self, embeddings):
        super().__init__(ROOT, embedding_gateway=embeddings, max_requests=120)


class ControlledMemory:
    """A bounded deterministic store tests the Agent's decision to request more history, not Mem0 recall quality."""
    enabled = True

    def __init__(self, case, user, trace):
        self.case, self.user, self.trace = case, user, trace

    def search_similar_sessions(self, query, user_id, limit=3):
        if user_id != self.user:
            raise ValueError("Cross-user controlled-memory access")
        items = self.case.get("memory", []) + self.case.get("additional_memory", [])
        result = copy.deepcopy(items[:limit])
        self.trace.append({"event": "controlled_history_search", "query": query, "limit": limit,
                           "user_id": user_id, "memories": result})
        return result

    def add_session_summary(self, **kwargs):
        self.trace.append({"event": "cloud_write_suppressed", "reason": "paired read-side experiment"})
        return None


def storage_checks(case, turns):
    final = turns[-1]["profile_after"]
    active = lambda category: [f for f in final.get(category, []) if f["status"] == "active"]
    checks = {}
    if case["id"] in {"N01", "N06"}:
        checks["no_question_or_quotation_written_as_allergy"] = not active("allergies")
        checks["no_profile_write"] = not turns[-1]["result"]["profile_updates"]
    if case["id"] == "N02":
        checks["no_relative_condition_or_medication_as_active_self_fact"] = not active("conditions") and not active("medications")
    if case["id"] == "N03":
        checks["explicit_allergy_denial_saved"] = any("头孢" in f["value"] and f["status"] == "denied" for f in final.get("allergies", []))
        checks["no_active_allergy"] = not active("allergies")
    if case["id"] in {"N04", "N05"}:
        name = "氯沙坦" if case["id"] == "N04" else "泮托拉唑"
        meds = final.get("medications", [])
        checks["one_named_medication_stopped"] = len(meds) == 1 and name in meds[0]["value"] and meds[0]["status"] == "stopped"
        if case["id"] == "N04":
            checks["final_question_does_not_write"] = not turns[-1]["result"]["profile_updates"]
    if case["id"] in {"N07", "N08"}:
        checks["history_question_does_not_create_current_user_facts"] = not turns[-1]["result"]["profile_updates"]
    return checks


async def run(gateway, model, case, variant):
    run_id = f"{case['id']}__{model.replace('/', '_')}__{variant}"
    path = ROOT / "runs" / f"{run_id}.json"
    if path.exists():
        previous = read(path)
        if previous["status"] != "completed":
            raise RuntimeError(f"Previous {run_id} did not complete; inspect before retry")
        return
    trace, turns = [], []
    user, session = run_id + "-synthetic-user", run_id + "-session"
    record = {"run_id": run_id, "case_id": case["id"], "variant": variant, "model": model,
              "dataset_sha256": sha(CASES), "status": "running", "turns": turns,
              "memory_backend": "controlled_record_pool_not_live_Mem0", "single_sample": True}
    dump(path, record)
    started = time.monotonic()
    try:
        memory = ControlledMemory(case, user, trace)
        kb = LocalRAG(gateway, DATA / "corpus.jsonl", trace)
        supervisor, profile = build_system(gateway, model, trace, kb, ROOT / "profiles" / run_id, memory)
        import agents, memory as memory_module, constraints
        expected = ROOT / "baseline_product" if variant == "baseline" else PROJECT
        for module in (agents, memory_module, constraints):
            if not Path(module.__file__).resolve().is_relative_to(expected.resolve()):
                raise RuntimeError(f"Product import is not from selected variant: {module.__file__}")
        for question in case["turns"]:
            before = profile.get_context(user)
            turn_started = time.monotonic()
            result = await asyncio.wait_for(supervisor.process(question, user_id=user, session_id=session), timeout=180)
            turns.append({"question": question, "answer": result["answer"], "result": result,
                          "profile_before": before, "profile_after": profile.get_context(user),
                          "seconds": time.monotonic() - turn_started})
        record.update(status="completed", storage_checks=storage_checks(case, turns))
    except Exception as error:
        record.update(status="error", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        record.update(seconds=time.monotonic() - started, metrics=trace_metrics(trace), protocol=protocol_checks(trace))
        dump(ROOT / "traces" / f"{run_id}.json", trace)
        dump(path, record)
        print(json.dumps({"run_id": run_id, "status": record["status"], "seconds": round(record["seconds"], 1),
                          "storage_checks": record.get("storage_checks")}, ensure_ascii=False), flush=True)


def report():
    rows = [read(p) for p in sorted((ROOT / "runs").glob("*.json"))]
    groups = []
    for model in MODELS[:2] + [CODEX_MODEL]:
        for variant in ("baseline", "candidate"):
            selected = [r for r in rows if r["model"] == model and r["variant"] == variant]
            done = [r for r in selected if r["status"] == "completed"]
            groups.append({"model": model, "variant": variant, "completed": len(done), "planned": 8,
                           "storage_checks_passed": sum(all(r["storage_checks"].values()) for r in done),
                           "model_requests": sum(r["metrics"]["model_requests"] for r in selected),
                           "prompt_tokens": sum(r["metrics"]["prompt_tokens"] for r in selected),
                           "requested_tools": sum(r["metrics"]["requested_tool_calls"] for r in selected),
                           "retrieval_executions": sum(r["metrics"]["retrieval_backend_executions"] for r in selected),
                           "case_mean_seconds": sum(r["seconds"] for r in done) / len(done) if done else None})
    dump(ROOT / "summary.json", {"groups": groups, "protocol_failures": sum(len(r["protocol"]["failures"]) for r in rows),
                                 "automated_storage_checks_are_not_semantic_or_clinical_grades": True})
    lines = ["# 改前与改后的原始执行记录", "", "8 个合成开发案例，每模型每版本各一次。未做临床审核。"
             "档案检查只评数据状态，不代表回答语义正确。个人记忆使用控制数据池，在线 Mem0 另测。", ""]
    for row in rows:
        lines += [f"## {row['run_id']}", "", f"状态：{row['status']}", ""]
        for turn in row["turns"]:
            lines += ["用户：" + turn["question"], "", turn["answer"], "", "本轮结束后的档案：", "", "```json",
                      json.dumps(turn["profile_after"], ensure_ascii=False, indent=2), "```", ""]
    (ROOT / "改前改后原始记录.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(groups, ensure_ascii=False), flush=True)


async def main(args):
    logger.remove()
    logger.add(sys.stderr, level="ERROR", backtrace=False, diagnose=False)
    if args.report_only:
        report()
        return
    selected_root = ROOT / "baseline_product" if args.variant == "baseline" else PROJECT
    sys.path.insert(0, str(selected_root))
    sources = [p for folder in ("agents", "core", "swarm", "memory", "constraints", ".claude")
               for p in (selected_root / folder).rglob("*") if p.suffix in {".py", ".yaml", ".md"} and "__pycache__" not in p.parts]
    protocol = {"variant": args.variant, "dataset_sha256": sha(CASES),
                "sources": {str(p.relative_to(selected_root)): sha(p) for p in sources},
                "corpus_sha256": sha(DATA / "corpus.jsonl"), "temperature": .2, "max_tokens": 2048,
                "memory": "controlled record pool, bounded by requested limit; no cloud writes",
                "single_sample_per_case": True, "codex_max_requests": 120, "openrouter_total_cap_usd": 4}
    protocol_path = ROOT / f"protocol_{args.variant}.json"
    if protocol_path.exists() and read(protocol_path) != protocol:
        raise ValueError("Frozen product or dataset changed during comparison")
    dump(protocol_path, protocol)
    embeddings = Gateway(OUTPUT, limit=4)
    gateway = ComparisonCodex(embeddings) if args.model == CODEX_MODEL else embeddings
    try:
        for case in read(CASES)["cases"]:
            await run(gateway, args.model, case, args.variant)
    finally:
        report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["baseline", "candidate"], default="candidate")
    parser.add_argument("--model", choices=MODELS[:2] + [CODEX_MODEL], default=CODEX_MODEL)
    parser.add_argument("--report-only", action="store_true")
    asyncio.run(main(parser.parse_args()))
