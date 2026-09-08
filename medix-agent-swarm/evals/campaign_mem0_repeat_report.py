"""Summarize every preserved Mem0 repetition without selecting the best result."""

import json
import re
from collections import Counter

from campaign_gateway import dump
from campaign_run import OUTPUT


ROOTS = [OUTPUT / name for name in ("mem0_boundary_v1", "mem0_boundary_r2", "mem0_boundary_r3")]
DESTINATION = OUTPUT / "mem0_repeat_summary"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(repetitions):
    query_rows = {}
    totals = []
    for label, rows in repetitions:
        complete = [row for row in rows if row["status"] == "completed"]
        queries = [q for row in complete for q in row["queries"]]
        positives = [q for q in queries if q["query"]["required_terms"]]
        negatives = [q for q in queries if q["query"].get("expect_empty")]
        totals.append({"repetition": label, "completed_scenarios": len(complete),
                       "statuses": dict(Counter(r["status"] for r in rows)),
                       "positive_queries": len(positives),
                       "storage_anchor_hits": sum(all(q["storage_anchors"]) for q in positives),
                       "baseline_anchor_hits": sum(all(q["baseline_anchors"]) for q in positives),
                       "top10_anchor_hits": sum(all(q["top10_anchors"]) for q in positives),
                       "threshold0_anchor_hits": sum(all(q["threshold0_anchors"]) for q in positives),
                       "negative_queries": len(negatives),
                       "negative_empty": sum(not q["baseline"] for q in negatives)})
        for row in complete:
            for q in row["queries"]:
                key = row["scenario"]["id"] + "/" + q["query"]["id"]
                entry = query_rows.setdefault(key, {"query": q["query"], "observations": []})
                entry["observations"].append({"repetition": label,
                     "stored_count": len(row["stored"]), "retrieved_count": len(q["baseline"]),
                     "storage_anchors": q["storage_anchors"], "baseline_anchors": q["baseline_anchors"],
                     "top10_anchors": q["top10_anchors"], "baseline_text": [m["content"] for m in q["baseline"]]})
    return {"repetitions": totals, "queries": query_rows,
            "independence": "Same 12 source scenarios repeated; queries and repetitions are not independent clinical cases",
            "grading": "Frozen bilingual lexical anchors only; review annotations do not overwrite original scores"}


def main():
    repetitions, manifests = [], []
    operations = Counter()
    for index, root in enumerate(ROOTS, 1):
        if not (root / "manifest.json").exists():
            continue
        manifests.append(read(root / "manifest.json"))
        repetitions.append((f"r{index}", [read(p) for p in sorted((root / "cases").glob("*.json"))]))
        for path in (root / "operations").glob("*.json"):
            row = read(path)
            if re.fullmatch(r"M\d+__write_\d+", row["name"]):
                kind = "add"
            elif "__event_" in row["name"]:
                kind = "event_read"
            elif row["name"].endswith("__stored"):
                kind = "list"
            else:
                kind = "search"
            operations[kind + "/" + row["status"]] += 1
    hashes = {m["dataset_sha256"] for m in manifests}
    users = [u for m in manifests for u in m["user_ids"]]
    if len(hashes) > 1 or len(set(users)) != len(users):
        raise ValueError("Repetitions have different datasets or overlapping user scopes")
    summary = summarize(repetitions)
    summary.update(planned_repetitions=3, operations=dict(operations), isolated_user_scopes=len(users))
    summary["review_annotations"] = [{"target": "r2,M06/uncertain; r3,M06/uncertain", "type": "lexical_false_negative",
        "observed": "User's doctor suggested screening for asthma, but asthma has not been diagnosed yet",
        "reason": "has not been diagnosed 不匹配冻结规则 not diagnos，但实际文本保留了未确诊状态；不能说 Mem0 丢了否定。",
        "original_scores_preserved": True, "reviewer": "assistant text inspection, not clinician validation"}]
    dump(DESTINATION / "summary.json", summary)
    text = ["# Mem0 三次重复实验：能记住什么，边界在哪里", "",
            "同一套 12 类合成场景，在不同测试用户空间重新抽取、写入、查询。全部重复都保留，未选择最佳一次。"
            "不是 36 个独立病例，也不是 36 位真实患者。", "",
            "## 先看各次结果", "",
            "以下是冻结关键词规则的原始结果，不是医学准确率。M06 第二、三次各有一处规则误报，解释见下。", "",
            "| 次数 | 完成场景 | 有目标词组的问题 | 存储词组找全 | Top-3 找全 | Top-10 找全 | 无关/隔离查询返回空 |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary["repetitions"]:
        text.append(f"| {row['repetition']} | {row['completed_scenarios']}/12 | {row['positive_queries']} | "
                    f"{row['storage_anchor_hits']} | {row['baseline_anchor_hits']} | {row['top10_anchor_hits']} | "
                    f"{row['negative_empty']}/{row['negative_queries']} |")
    text += ["", "## 具体发生了什么", "",
             "1. **明确过敏史可以召回。** 问法包含直接询问、换表达、英语和间接问药；具体各次结果在下面逐项列出。"
             "但多数隔离用户只有一两条记忆，这不能证明有几百条历史时也同样可靠。", "",
             "2. **旧记录没有自动变成当前状态。** 首次停药例同时找回开始服药和后来停药两条记录。"
             "历史都在是有用的，但最终回答仍需理解时间和当前状态；这不是一个自动维护正确现状的用户档案。", "",
             "3. **不是每个来源都是用户本人。** 首次污染例找回“助手说用户有十年高血压史”，也找回“用户只说了头痛”。"
             "Mem0 保留了助手归属，不能说它已无条件把假病史改成用户事实；但假内容确实进入了可召回集合。"
             "下游是否误信，必须另看 Agent 回答。", "",
             "4. **Top-3 容量会成为真实限制。** 首次五事实总结中，高血压、青霉素过敏、哮喘、夜班和膝伤均在库里。"
             "Top-3 只返回前三项；Top-10 同次检查覆盖五项；把阈值降为 0、仍取三条则仍缺后两项。"
             "这是开发例，不代表普遍改成 Top-10 就最优，也不代表下游已经回答正确。", "",
             "5. **评分规则本身会误报。** 第二、三次 M06 中实际文本写着 asthma has not been diagnosed yet，"
             "没有把待排查升级为确诊。但冻结的词组表只列了 not diagnos 等表达，因此得到 false。"
             "保留这个原分与注释，不把语言匹配问题归咎于 Mem0 抽取错误。", "",
             "## 不能从这些数据推出什么", "",
             "没有测试几百条历史、真实跨天延迟、完整在线问诊回写链路或医生认定的最终安全性。"
             "助手内容污染例没有目标词组，不因为空数组 all([]) 为真就算通过。", "",
             "Top-3、Top-10、低阈值按固定顺序调用。云端排序/索引状态没有冻结，结果只是参数敏感性观察。"
             "部分响应 score 低于请求 threshold；官方说明某些后处理（如开启 decay）可以出现这种行为，"
             "本轮没有核定完整项目配置，不能凭公开分数断言阈值失效，也没有增加未声明的客户端过滤。"
             "[官方分数说明](https://docs.mem0.ai/platform/features/memory-decay)。", "",
             "词组覆盖不验证主体、否定、日期或医学含义。当前是模型辅助工程文本复核，没有独立医生审核。", "",
             "## 每个问题在三次里找回了什么", ""]
    for key, entry in summary["queries"].items():
        text += [f"### {key}", "", entry["query"]["query"], ""]
        for obs in entry["observations"]:
            text += [f"{obs['repetition']}：存储 {obs['stored_count']} 条，返回 {obs['retrieved_count']} 条；"
                     f"存储词组 {obs['storage_anchors']}，Top-3 词组 {obs['baseline_anchors']}。", ""]
            text += [f"- {value}" for value in obs["baseline_text"]] or ["- 返回空集。"]
            text.append("")
    text += ["## 请求量与复现入口", "", "```json", json.dumps(dict(operations), indent=2), "```", "",
             "Mem0 请求量与 OpenRouter 模型费用分开。响应未提供账单金额，不能把未返回费用解释为已确认零收费。", "",
             "原始记录分别在 mem0_boundary_v1、mem0_boundary_r2、mem0_boundary_r3；summary.json 包含每个问题的各次输出。"
             "所有作用域均为专用合成测试用户，没有删除账号内既有资料。", ""]
    (DESTINATION / "Mem0重复实验解读.md").write_text("\n".join(text), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("repetitions", "operations", "isolated_user_scopes")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
