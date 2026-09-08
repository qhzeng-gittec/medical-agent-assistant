"""Paired, live Agent inference using frozen real Mem0 retrievals; no cloud write-back."""

import argparse
import asyncio
import copy
import json
import time
from collections import Counter

from campaign_gateway import ApiAccessBlocked, BudgetExceeded, Gateway, MODELS, dump, sha
from campaign_grade import protocol_checks
from campaign_mem0_boundary import anchor_checks
from campaign_rag import LocalRAG
from campaign_run import DATA, OUTPUT, PROJECT, build_system, trace_metrics


ROOT = OUTPUT / "memory_context_ablation_v1"
SOURCE = OUTPUT / "mem0_boundary_v1"
SELECTION = [("M01", "direct"), ("M02", "current"), ("M07", "contamination"), ("M11", "broad")]
ARMS = ["neither", "profile_only", "mem0_replay_only", "profile_and_mem0_replay"]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


class FrozenRecall:
    """Replay an unchanged observed retrieval to hold evidence equal across arms/models."""

    def __init__(self, enabled, query, user_id, memories, trace):
        self.enabled, self.query, self.user_id = enabled, query, user_id
        self.memories, self.trace = copy.deepcopy(memories), trace

    def search_similar_sessions(self, query, user_id, limit=3):
        if query != self.query or user_id != self.user_id or limit != 3:
            raise ValueError("Frozen recall scope/query differs from the recorded experiment")
        self.trace.append({"event": "frozen_mem0_recall", "source": "real_cloud_baseline_r1",
                           "query": query, "memories": copy.deepcopy(self.memories)})
        return copy.deepcopy(self.memories)

    def add_session_summary(self, *args, **kwargs):
        self.trace.append({"event": "memory_write_suppressed", "reason": "read-side attribution experiment"})
        return None


class NoProfile:
    def update_from_user_message(self, user_id, message, session_id):
        return []

    def get_context(self, user_id):
        return {}


def prepare_inputs():
    selected = []
    for case_id, query_id in SELECTION:
        path = SOURCE / "cases" / f"{case_id}.json"
        row = read(path)
        if row["status"] != "completed":
            raise ValueError(f"Source {case_id} has not completed")
        query = next(q for q in row["queries"] if q["query"]["id"] == query_id)
        selected.append({"case_id": case_id, "scenario": row["scenario"], "query": query,
                         "source_path": str(path.relative_to(PROJECT)), "source_sha256": sha(path)})
    return selected


async def execute(gateway, item, model, arm):
    run_id = f"{item['case_id']}__{model.replace('/', '_')}__{arm}"
    destination = ROOT / "runs" / f"{run_id}.json"
    if destination.exists():
        prior = read(destination)
        if prior["status"] == "running":
            raise RuntimeError(f"Unresolved prior execution {run_id}; inspect before retry")
        return
    user_id = run_id + "-synthetic-user"
    session_id = run_id + "-new-session"
    question = item["query"]["query"]["query"]
    trace = []
    row = {"run_id": run_id, "case_id": item["case_id"], "model": model, "arm": arm,
           "question": question, "status": "running", "source_sha256": item["source_sha256"],
           "source_query_id": item["query"]["query"]["id"], "synthetic_data_only": True,
           "experiment": "live_product_with_frozen_cloud_recall_not_full_live_Mem0_E2E"}
    dump(destination, row)
    started = time.monotonic()
    try:
        memory = FrozenRecall("mem0_replay" in arm, question, user_id, item["query"]["baseline"], trace)
        kb = LocalRAG(gateway, DATA / "corpus.jsonl", trace)
        supervisor, profiles = build_system(gateway, model, trace, kb, ROOT / "profiles" / run_id,
                                            long_term_memory=memory)
        if arm.startswith("profile"):
            for index, turn in enumerate(item["scenario"]["turns"]):
                profiles.update_from_user_message(user_id, turn["user"], f"seed-session-{index}")
        else:
            supervisor.patient_profiles = NoProfile()
        row["profile_before"] = supervisor.patient_profiles.get_context(user_id)
        response = await asyncio.wait_for(supervisor.process(question, user_id=user_id, session_id=session_id), 240)
        row.update(status="completed", answer=response["answer"], system_result=response)
        first = next(e for e in trace if e["event"] == "api_request" and e["role"] == "supervisor")
        supplied_context = json.loads(first["payload"]["messages"][1]["content"])["context"]
        row["context_actually_supplied"] = supplied_context
        required = item["query"]["query"]["required_terms"]
        row["answer_anchor_checks"] = anchor_checks(required, [{"content": response["answer"]}]) if required else None
        row["context_anchor_checks"] = anchor_checks(required, [{"content": json.dumps(supplied_context, ensure_ascii=False)}]) if required else None
    except Exception as error:
        row.update(status="error", error=f"{type(error).__name__}: {error}".replace(gateway.key, "[REDACTED]"))
        if isinstance(error, (BudgetExceeded, ApiAccessBlocked)) or gateway.halt_reason:
            raise
    finally:
        row["seconds"] = time.monotonic() - started
        row["metrics"] = trace_metrics(trace)
        row["protocol"] = protocol_checks(trace)
        dump(ROOT / "traces" / f"{run_id}.json", trace)
        dump(destination, row)
    print(json.dumps({"run_id": run_id, "status": row["status"], "seconds": round(row["seconds"], 1)}, ensure_ascii=False), flush=True)


def report():
    rows = [read(p) for p in sorted((ROOT / "runs").glob("*.json"))]
    summary = {"planned": len(SELECTION) * len(ARMS) * 2, "statuses": dict(Counter(r["status"] for r in rows)),
               "scope": "4 deliberately selected development cases; one sample/model/arm; not independent clinical evaluation",
               "automatic_semantic_grading": False, "cloud_recall": "frozen real r1 results", "cloud_write_back": False,
               "target_provider_reported_usd": sum(r.get("metrics", {}).get("target_cost_usd", 0) for r in rows),
               "runs": [{k: r.get(k) for k in ("run_id", "case_id", "model", "arm", "status", "seconds", "context_anchor_checks", "answer_anchor_checks", "error")} for r in rows]}
    dump(ROOT / "summary.json", summary)
    text = ["# 记忆进入 Agent 后发生什么：四组配对实验", "",
            "这不是完整在线 Mem0 端到端测评，而是读侧归因实验：固定首轮真实云检索结果，"
            "由真实 Supervisor / Worker 和 OpenRouter 模型回答相同问题。需要检索时仍使用真实 Qwen API embedding + 本地 RAG。"
            "云记忆回写被测评器显式关闭，避免前一个实验的答案污染下一个实验。", "",
            "四组：无档案无事件记忆、仅真实规则抽取的持久档案、仅 Mem0 真实召回快照、档案与召回快照同时提供。"
            "每次都是新短期会话，档案只从历史用户原话构建，未手工补足漏抽的事实。", "",
            f"计划 {summary['planned']} 次，当前状态：{summary['statuses']}。不是 32 个独立病例，而是 4 个开发病例 × 4 组 × 2 款模型。", "",
            "程序只列事实词组覆盖、执行和协议指标；出现某个词不代表语义正确，尤其需要人工检查停药、否定和来源归属。"
            "无记忆组诚实地说不知道，不等于安全失败。不能用词组覆盖直接判定哪个架构更好。", ""]
    for case_id, _ in SELECTION:
        text += [f"## {case_id}", ""]
        for row in [r for r in rows if r["case_id"] == case_id]:
            text += [f"### {row['model']} · {row['arm']}", "", f"问题：{row['question']}", "",
                     "实际提供的上下文：", "", "```json", json.dumps(row.get("context_actually_supplied"), ensure_ascii=False, indent=2), "```", "",
                     "真实回答：", "", row.get("answer", row.get("error", "仍在执行")), ""]
    (ROOT / "记忆四组对照记录.md").write_text("\n".join(text), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "runs"}, ensure_ascii=False))


async def main(report_only):
    if report_only:
        report()
        return
    items = prepare_inputs()
    protocol = {"selected_sources": [{"case_id": x["case_id"], "source_sha256": x["source_sha256"], "query_id": x["query"]["query"]["id"]} for x in items],
                "arms": ARMS, "models": MODELS[:2], "single_sample_per_arm": True,
                "source_selection": "deliberate development diagnostics chosen after viewing Mem0 r1",
                "execution": "sequential; rotate arm order by case/model index; no parallel shared tool globals",
                "memory_recall": "frozen actual Mem0 r1 baseline, identical across compared arms/models",
                "write_back": "disabled for this read-side experiment", "profile_seed": "real PatientProfileStore extraction of historical user turns",
                "corpus_sha256": sha(DATA / "corpus.jsonl"),
                "shared_budget_ledger": str((OUTPUT / "api_ledger.jsonl").relative_to(PROJECT)), "cumulative_budget_usd": 5,
                "source_hashes": {str(p.relative_to(PROJECT)): sha(p) for p in
                                  [PROJECT / "evals/campaign_run.py", PROJECT / "evals/campaign_memory_ablation.py",
                                   PROJECT / "swarm/supervisor_agent.py", PROJECT / "memory/patient_profile.py"]}}
    path = ROOT / "protocol.json"
    if path.exists() and read(path) != protocol:
        raise ValueError("Frozen ablation protocol changed; use a new experiment")
    dump(path, protocol)
    gateway = Gateway(OUTPUT, limit=5)
    try:
        for index, item in enumerate(items):
            for mi, model in enumerate(MODELS[:2]):
                offset = (index + mi) % len(ARMS)
                for arm in ARMS[offset:] + ARMS[:offset]:
                    await execute(gateway, item, model, arm)
    finally:
        report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-only", action="store_true")
    asyncio.run(main(parser.parse_args().report_only))
