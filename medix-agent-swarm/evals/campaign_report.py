"""Aggregate experiment runs into a Chinese evaluation report."""

import json
from collections import Counter
from statistics import mean

import numpy as np

from campaign_gateway import MODELS, dump, sha
from campaign_grade import TARGET_ROLES, protocol_checks
from campaign_run import DATA, OUTPUT, PROJECT


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def primary_runs(root=None):
    root = root or OUTPUT
    selected, missing = [], []
    for i in range(1, 25):
        # All adaptive patient tasks use the repaired selector, regardless of earlier scores.
        repetition = 4 if i <= 6 else 3
        for model in MODELS[:2]:
            name = f"CONSULT-{i:02}__{model.replace('/', '_')}__r{repetition}"
            path = root / "runs" / f"{name}.json"
            resumed = root / "runs" / f"{name}_resumed.json"
            if resumed.exists():
                path = resumed
            if path.exists():
                selected.append(read(path))
            else:
                missing.append(name)
    return selected, missing


def get_grades(run_id):
    grades = {p.name: read(p) for p in (OUTPUT / "grades_v2").glob(f"{run_id}__judge_*.json")}
    for path in (OUTPUT / "grades_v2_retries").glob(f"{run_id}__judge_*.json"):
        grades[path.name] = read(path)
    return list(grades.values())


def observed_metrics(run):
    trace = read(OUTPUT / "traces" / f"{run['run_id']}.json")
    metrics = protocol_checks(trace)
    retrievals = [e for e in trace if e["event"] == "retrieval"]
    exact_queries = Counter((e["query"], e.get("filter_type"), e.get("top_k")) for e in retrievals)
    worker_calls = [c for t in run["turns"] for c in t["system_result"]["call_trace"]]
    apis = [e for e in trace if e["event"] == "api_request" and e["role"] in TARGET_ROLES]
    observations = [e for e in trace if e["event"] == "tool_observation"]
    blocks = [d for e in observations for d in e["result"].get("documents", [])]
    body_by_id = {d["block_id"]: d for d in blocks if "content" in d}
    full_size, compact_size, unresolved = 0, 0, 0
    for event in observations:
        compact = event["result"]
        expanded = json.loads(json.dumps(compact))
        for index, block in enumerate(expanded.get("documents", [])):
            if "reference" in block:
                if block["block_id"] in body_by_id:
                    expanded["documents"][index] = body_by_id[block["block_id"]]
                else:
                    unresolved += 1
        compact_size += len(json.dumps(compact, ensure_ascii=False))
        full_size += len(json.dumps(expanded, ensure_ascii=False))
    return {"protocol_contexts": metrics["context_checks"], "protocol_failures": metrics["failures"],
            "response_contract_failures": metrics["response_contract_failures"],
            "context_growth": metrics["context_growth"],
            "worker_calls": len(worker_calls), "worker_failures": [c for c in worker_calls if not c["success"]],
            "exact_repeated_backend_queries_within_case": sum(n - 1 for n in exact_queries.values()),
            "tool_errors": sum(e["result"].get("success") is False for e in observations),
            "target_provider_reported_usd": sum(e["cost_usd"] for e in apis if e["cost_status"] == "provider_reported"),
            "target_cost_unconfirmed_usd": sum(e["cost_usd"] for e in apis if e["cost_status"] != "provider_reported"),
            "serialization_estimate": {"compacted_tool_characters": compact_size,
               "reexpanded_tool_characters": full_size, "character_difference": full_size - compact_size,
               "unresolved_references": unresolved,
               "meaning": "Offline same-trace serialization estimate, not measured model token/cost reduction"}}


def main():
    runs, missing = primary_runs()
    if missing:
        raise SystemExit(f"Primary matrix incomplete: {len(missing)} missing runs")
    details = {run["run_id"]: observed_metrics(run) for run in runs}
    judge_pairs, disagreements, score_rows = 0, 0, []
    for run in runs:
        grades = [g for g in get_grades(run["run_id"]) if g["status"] == "graded"]
        if len(grades) == 2:
            for a, b in zip(grades[0]["checks"], grades[1]["checks"]):
                judge_pairs += 1
                disagreements += a["verdict"] != b["verdict"]
        score_rows.append({"run_id": run["run_id"], "execution": run["status"],
            "valid_judges": len(grades),
            "any_judge_critical_failure": any(g["critical_failed"] for g in grades) if grades else None,
            "both_judges_all_pass": run["status"] == "completed" and len(grades) == 2 and all(g["all_criteria_passed"] for g in grades),
            "judges": [{"model": g["judge"], "critical_failed": g["critical_failed"], "checks": g["checks"]} for g in grades]})
    by_run = {r["run_id"]: r for r in score_rows}
    urgent_turns = []
    for run in runs:
        index = 1 if run["case_id"] == "CONSULT-24" else 0
        if run["case_id"] in {"CONSULT-07", "CONSULT-08", "CONSULT-09", "CONSULT-10", "CONSULT-24"} and len(run["turns"]) > index:
            turn = run["turns"][index]
            urgent_turns.append({"run_id": run["run_id"], "turn": index + 1,
                                 "seconds": turn["latency_seconds"],
                                 "waited_for_workers_before_final": bool(turn["system_result"]["call_trace"])})
    stats = []
    for model in MODELS[:2]:
        items = [r for r in runs if r["model"] == model]
        latencies = [t["latency_seconds"] for r in items for t in r["turns"]]
        ratings = [by_run[r["run_id"]] for r in items]
        stats.append({"model": model, "tasks": len(items),
            "execution_completed": sum(r["status"] == "completed" for r in items),
            "execution_errors": [{"case_id": r["case_id"], "error": r.get("error")} for r in items if r["status"] != "completed"],
            "completed_turns": len(latencies), "mean_turn_seconds": mean(latencies),
            "p95_turn_seconds": float(np.percentile(latencies, 95)),
            "target_model_requests": sum(r["metrics"]["model_requests"] for r in items),
            "target_prompt_tokens": sum(r["metrics"]["prompt_tokens"] for r in items),
            "target_completion_tokens": sum(r["metrics"]["completion_tokens"] for r in items),
            "target_provider_reported_usd": sum(details[r["run_id"]]["target_provider_reported_usd"] for r in items),
            "cases_with_worker_failures": sum(bool(details[r["run_id"]]["worker_failures"]) for r in items),
            "retrieval_executions": sum(r["metrics"]["retrieval_backend_executions"] for r in items),
            "document_bodies": sum(r["metrics"]["document_bodies_added"] for r in items),
            "document_references": sum(r["metrics"]["document_references_added"] for r in items),
            "protocol_violations": sum(len(details[r["run_id"]]["protocol_failures"]) for r in items),
            "response_contract_failures": sum(len(details[r["run_id"]]["response_contract_failures"]) for r in items),
            "valid_double_graded_cases": sum(r["valid_judges"] == 2 for r in ratings),
            "any_judge_critical_failure_cases": sum(r["any_judge_critical_failure"] is True for r in ratings),
            "both_judges_all_pass_cases": sum(r["both_judges_all_pass"] for r in ratings)})
    ledger = {}
    for line in (OUTPUT / "api_ledger.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        ledger[row["request_id"]] = row
    costs = Counter()
    for row in ledger.values():
        costs[row["cost_status"]] += row["charged_or_reserved_usd"]
    costs["provider_reported"] += 0.00008073
    rag = read(OUTPUT / "retrieval_smoke.json")
    tests = read(OUTPUT / "automated_test_result.json")
    valid_grades = sum(row["valid_judges"] for row in score_rows)
    summary = {"status": "primary_matrix_executed_grading_incomplete" if valid_grades < 96 else "primary_matrix_and_grading_complete",
               "model_stats": stats, "costs_by_status_usd": dict(costs),
               "valid_judgements": valid_grades, "planned_judgements": 96, "pending_judgements": 96 - valid_grades,
               "judge_comparable_criteria": judge_pairs, "judge_disagreement_criteria": disagreements,
               "urgent_turn_delivery": urgent_turns,
               "runs": score_rows, "deterministic_details": details,
               "primary_selection": "r4 for CONSULT-01..06 (checkpoint continuations where present), r3 for CONSULT-07..24; no best-of-N selection",
               "clinical_validation": False, "gemini": "provider 403; not clinically evaluated"}
    dump(OUTPUT / "campaign_summary.json", summary)
    dump(OUTPUT / "evaluation_source_hashes.json", {str(p.relative_to(PROJECT)): sha(p)
         for p in (PROJECT / "evals").glob("campaign_*.py")})
    grading_note = (
        f"**当前状态：48 个主测任务已完整执行；自动双评审只有 {valid_grades}/96 条有效评分，剩余 {96-valid_grades} 条待补。** "
        "缺失或无效评分未计入通过率。"
        "当前不发布整体医疗质量通过率，也不按部分评分给模型排名。"
        if valid_grades < 96 else
        "**当前状态：48 个主测任务和 96 条自动双评审均已完成。** 自动评分仍不代表医生审核。"
    )
    text = ["# 医疗问诊 Agent 测评结果（开发回归版）", "",
        "日期：2026-09-07。真实调用 OpenRouter；全部病例为模拟患者，不含真实患者个人信息。",
        "这份报告验证工程行为和小范围问诊表现，不是临床认证、诊断准确率或上线许可。", "",
        grading_note, "",
        "## 1. 实验目标", "",
        "这次不是让模型做医学选择题，而是把患者放进真实 Supervisor → 专业 Agent → 工具 → 回复的链路里。"
        "既看有没有问对问题和处理危险信号，也看工具调用、记忆读写、证据使用、错误恢复以及实际耗时。",
        "共 24 个场景、2 款可用被测模型，形成 48 个主测任务。第三款 Gemini 在预检返回服务商 403；没有替换型号或把拒绝访问算成能力失败。", "",
        "## 2. 测了什么，没测什么", "",
        "| 层次 | 真实执行内容 | 不应夸大的边界 |", "|---|---|---|",
        "| 工程回归 | 原项目自动测试，以及 8 个 RAG 缓存/正文引用合约场景 | 通过只证明已写断言，不证明医学正确 |",
        "| 真实 RAG | OpenRouter qwen/qwen3-embedding-8b，4096 维；本地保存向量并做余弦检索 | 没有本机 embedding 推理；不是完整 Milvus 部署压测 |",
        "| 问诊主流程 | 追问、急症、用药边界、续问、更正、用户隔离、档案持久化、异常与注入 | 24 个开发场景，不是独立测试集或真实临床患者研究 |",
        "| 长期事实 | 实际 PatientProfileStore 文件读写及新会话读取 | Mem0 Platform 无凭据，本轮未连通；不能声称长期事件记忆通过 |",
        "| 专业模型 | 同一次任务的总控和子 Agent 都使用对应被测通用模型 | 没有微调医疗模型服务端点，未验证微调收益或异构模型分工 |",
        "| 外网深度研究 | 如被调用返回明确不可用错误 | 未配置外网研究服务，不伪造成功结果 |", "",
        "知识库来自 NHS 患者信息和 CDC 成人门诊指导页面的 15 个中文摘要块。摘要由本次测评以 AI 辅助整理，"
        "未经过独立医生审核；字段中‘人工整理’不代表医生编写或审核。NHS 患者说明也不等同于正式临床指南。", "",
        "方法参考：[OpenRouter embedding 接口](https://openrouter.ai/docs/api/api-reference/embeddings/submit-an-embedding-request)、"
        "[Qwen 官方模型卡](https://huggingface.co/Qwen/Qwen3-Embedding-8B)；查询加任务说明，文档不加。"
        "[逐轨迹评估思路](https://developers.openai.com/api/docs/guides/trace-grading)用于区分最终答复和工具过程。", "",
        "## 3. 真实测量结果", "",
        "| 指标 | MiniMax M2.5 | Qwen3.5-27B |", "|---|---:|---:|"]
    fields = [("完整走到任务结尾 / 24", "execution_completed", 0),
              ("存在子 Agent 失败的任务", "cases_with_worker_failures", 0),
              ("已完成的患者轮次", "completed_turns", 0),
              ("每轮平均耗时（秒）", "mean_turn_seconds", 1),
              ("每轮 P95 耗时（秒）", "p95_turn_seconds", 1),
              ("主链路模型请求数", "target_model_requests", 0),
              ("实际检索后端执行数", "retrieval_executions", 0),
              ("工具结果中的新正文块 / 引用另见下一行", "document_bodies", 0),
              ("工具结果中的正文引用块", "document_references", 0),
              ("发往模型的上下文配对/引用违规", "protocol_violations", 0),
              ("收到的空结果/工具协议异常", "response_contract_failures", 0),
              ("两位评审均成功评分的任务", "valid_double_graded_cases", 0),
              ("主链路服务商回报费用（美元）", "target_provider_reported_usd", 5)]
    for label, key, precision in fields:
        text.append(f"| {label} | {stats[0][key]:.{precision}f} | {stats[1][key]:.{precision}f} |")
    text.extend(["", "‘走到结尾’不等于答案合格；子 Agent 失败后总控仍可能生成答复。耗时为非流式、每个已完成用户轮次到最终回复的墙钟时间，"
                 "不包括模拟患者/评审用时，也不是首 token 延迟；4 个独立患者任务并发，云服务抖动和超时上限会影响统计。未完成轮次不能被当成 0 秒。", "",
        f"单独看已预置明确危险信号的 5 类场景（07–10、24）：记录了 {len(urgent_turns)} 次目标轮回复，"
        f"其中 {sum(t['waited_for_workers_before_final'] for t in urgent_turns)} 次先等待子 Agent 链路；"
        f"最终回复耗时范围 {min(t['seconds'] for t in urgent_turns):.1f}–{max(t['seconds'] for t in urgent_turns):.1f} 秒。"
        "这是流程时延事实，不等于这些答复经过了医生的安全审核。", "",
        f"工程测试输出：`{tests['stdout'].strip().splitlines()[-1]}`。RAG 小检索集：10 个库内问题，Top-1 命中率 {rag['top1_hit_rate']:.0%}，"
        f"Top-3 命中率 {rag['top3_hit_rate']:.0%}。这是与 15 个精选块配套的冒烟集，不是大规模泛化成绩。", "",
        f"仅在目前已完成的部分评分中，两位自动评审在 {judge_pairs} 个可对应条目中有 {disagreements} 个判定不一致。样本按执行顺序缺失，不能代表全部场景。没有把模型评审伪装成人类医生结论；"
        "也不依据这么小的样本给两款模型排临床优劣。详见逐项理由和原始对话。", "",
        "## 4. 测试过程中发现的问题如何解释", "",
        "需要结合轨迹检查以下已知问题："
        "追问前是否过早宣布低风险、提示词是否强迫编造概率、风险规则能否理解否定、急救提示是否被工具链阻塞、"
        "自然语言档案更新是否真正落盘、证据是否只是检索到了却没有传给最终答复。", "",
        "重复检索与重复正文必须分开：本轮使用原项目的 worker 内完全相同参数缓存；跨 Agent 不共享中间检索。"
        "正文引用检查只证明可见上下文里复用了同一块，不能直接得出减少了多少 API 调用。"
        "summary 中的正文重新展开字符差只是一条轨迹的离线估计，不是与无去重版本配对实测的 token 节省率。", "",
        "## 5. 测评器也接受审计", "",
        "第一轮患者事实选择器的 2048 输出额度发生截断；因此将该模拟器额度统一升到 4096，"
        "把全部 6 类可交互场景、两款被测模型一起重跑为 r4，不按旧分数挑选重跑。其他 18 类固定脚本场景使用 r3。"
        "被测 Agent 仍统一 temperature=0.2、reasoning effort=low、单次输出上限 2048，专业 Agent 工具预算 2，总控轮数上限 4。",
        "这里的 low 是统一发送的请求参数，不代表不同供应商/模型采用了完全相等的内部推理量；"
        "本轮可以比较这套运行配置的表现，不能宣称已经严格隔离所有模型与供应商差异。",
        "升到 4096 后个别患者选择仍截断，随后只对这些测评器断点关闭患者模型的 reasoning，再继续未完成的患者补充。"
        "之前的医疗回答、用户事实、短期历史和档案原样恢复，没有重抽前一轮医疗回答。恢复结果以 _resumed 单独保存，原失败仍保留。",
        "初版评审自行摘抄引文时出现原文对不上，均记录为 judge_error。v2 改成选答案行 ID，程序从原记录提取证据；"
        "评审覆盖情况以本次汇总的有效评分数为准，失败记录单独保留。", "",
        "## 6. 费用", ""])
    for status, value in costs.items():
        text.append(f"- {status}：${value:.6f}")
    text.extend(["", "provider_reported 是 API 回报的费用，不是独立账单核对；estimated 是按回报 token 和目录价格估算；"
                 "unknown/reserved 是失败请求没有费用字段时保留的预算占位，不能说已实际扣款。"
                 "总额包含连通性预检、embedding、患者模拟、旧版失败、重跑和自动评审。两次额外 403 诊断请求无 usage，未据此断言免费。", "",
        "## 7. 比较协议与局限", "",
        "固定此版场景、来源和产品源码哈希，把真实失败加入回归。改一个问题后，在同一场景、相同推理预算和同一 RAG 快照下配对重跑，"
        "至少保留多次重复，再报告通过率、耗时、token 的差值。当前结果是现状基线，没有旧架构配对数据，不能编造‘提升 X%’。",
        "扩展时按病例家族划分 dev / test，冻结 holdout，并请医生独立审核病例事实、风险边界和评分标准；"
        "新增更长多轮、模糊口语、否定/更正、家属代问、年龄与孕期、药物相互作用，以及 Mem0 事件过期与隔离测试。", "",
        "## 附：每个任务的结果入口", "",
        "| 病例 | 被测模型 | 执行 | 有效评审数量 / 2 | 原始记录 |", "|---|---|---|---|---|"])
    for run in runs:
        score = by_run[run["run_id"]]
        path = (OUTPUT / "runs" / f"{run['run_id']}.json").as_posix()
        text.append(f"| {run['case_id']} | {run['model']} | {run['status']} | {score['valid_judges']}/2 | [对话 JSON]({path}) |")
    (OUTPUT / "医疗问诊Agent测评报告.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    print(json.dumps({"primary_tasks": len(runs), "costs": dict(costs), "stats": stats}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
