"""Frozen retrieval boundary probes; remote embeddings, no answer generation."""

import argparse
import json
import time
from collections import Counter

from campaign_gateway import EMBED_MODEL, Gateway, dump, sha
from campaign_rag import LocalRAG, QUERY_INSTRUCTION
from campaign_run import DATA, OUTPUT

CASES = DATA / "rag_boundary.jsonl"
ROOT = OUTPUT / "rag_boundary_v1"


def coverage(gold_groups, docs, k=3, threshold=0.0):
    ids = {doc["id"] for doc in docs[:k] if doc["score"] >= threshold}
    if not gold_groups:
        return {"returned": len(ids), "any_hit": None, "all_hit": None, "group_recall": None}
    hits = [bool(ids.intersection(group)) for group in gold_groups]
    return {"returned": len(ids), "any_hit": any(hits), "all_hit": all(hits),
            "group_recall": sum(hits) / len(hits)}


def summarize(rows):
    completed = [r for r in rows if r["status"] == "completed"]
    positives = [r for r in completed if r["case"]["answerability"] == "answerable"]
    negatives = [r for r in completed if r["case"]["answerability"] == "unanswerable"]
    groups = {}
    for name in sorted({r["case"]["group"] for r in completed}):
        items = [r for r in completed if r["case"]["group"] == name]
        gold = [r for r in items if r["case"]["gold_groups"]]
        groups[name] = {"cases": len(items), "with_gold": len(gold),
                       "top3_any_hits": sum(coverage(r["case"]["gold_groups"], r["documents"])["any_hit"] for r in gold),
                       "top3_all_hits": sum(coverage(r["case"]["gold_groups"], r["documents"])["all_hit"] for r in gold)}
    sweep = []
    for k in (1, 3, 5, 10):
        for threshold in (0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            checks = [coverage(r["case"]["gold_groups"], r["documents"], k, threshold) for r in positives]
            sweep.append({"k": k, "threshold": threshold, "positive_n": len(positives),
                          "any_hit_n": sum(c["any_hit"] for c in checks),
                          "all_hit_n": sum(c["all_hit"] for c in checks),
                          "mean_group_recall": sum(c["group_recall"] for c in checks) / len(checks) if checks else None,
                          "unanswerable_n": len(negatives),
                          "unanswerable_returned_n": sum(coverage([], r["documents"], k, threshold)["returned"] > 0 for r in negatives)})
    return {"total_attempted": len(rows), "completed": len(completed),
            "errors": [{"id": r["case"]["id"], "error": r["error"]} for r in rows if r["status"] != "completed"],
            "answerable_n": len(positives), "unanswerable_n": len(negatives),
            "groups": groups, "threshold_sweep": sweep,
            "scope": "Synthetic development probes against 15 frozen curated chunks; no clinician review; no held-out threshold validation",
            "interpretation": "Returning a chunk for an unanswerable query is not an observed hallucinated answer. Ranking hits do not prove medical reasoning."}


def report():
    saved = {p.name: json.loads(p.read_text(encoding="utf-8")) for p in (ROOT / "cases").glob("*.json")}
    saved.update({p.name: json.loads(p.read_text(encoding="utf-8")) for p in (ROOT / "cases_retries").glob("*.json")})
    rows = [saved[name] for name in sorted(saved)]
    summary = summarize(rows)
    summary["planned"] = len(CASES.read_text(encoding="utf-8").splitlines())
    summary["suite"] = CASES.stem
    summary["label_review"] = [{"case": "R25", "finding": "Original gold over-restricts the generic care question: NHS-COUGH already contains rest/fluids. Keep original 39/40 strict result; not a demonstrated missing-answer failure."}] if CASES.stem == "rag_boundary" else []
    dump(ROOT / "summary.json", summary)
    text = ["# RAG 召回边界：真实 API 测试", "",
            "这不是医学回答准确率，而是检查：所需资料有没有进入返回结果。所有向量由 OpenRouter Qwen API 生成，本机只存向量和计算余弦。",
            f"本套 {summary['planned']} 个冻结开发探针，沿用原来的 15 个 NHS/CDC 中文摘要块；没有扩大成大型知识库，也没有医生审定。", "",
            f"已执行 {summary['completed']}/{summary['planned']}，错误 {len(summary['errors'])}。普通可回答问题 {summary['answerable_n']} 个；库中没有充分答案的问题 {summary['unanswerable_n']} 个。",
            "依赖历史的原始短句、刻意错误的类型过滤、信息不足的询问单独统计，不混进正常召回率。", "",
            "补充的 rag_near_miss 套件是在首批探针完成后，针对 0.4 阈值候选构造的近义反例，用于找漏洞，不是独立随机测试集。"
            "所有缺失数字以实际冻结摘要是否包含为准，不假设源网页全文已经入库。", "",
            "标注复核：原始 R25 要求单独命中照护块，但已命中的 NHS-COUGH 本身包含休息与补液建议。"
            "因此原标签下的漏块不能直接算作回答所需信息缺失；原记录和分数保留，不通过改标签宣称优化提升。" if summary["label_review"] else "", "",
            "## 1. 各类问题的 Top-3 表现", "",
            "‘至少命中一个’容易掩盖多证据遗漏；‘全部命中’要求每组必要证据至少找回一篇。一个答案可有多篇等价文档。", "",
            "| 类别 | 数量 | 有明确目标的数量 | 至少命中一组 | 所有必要组都命中 |",
            "|---|---:|---:|---:|---:|"]
    for name, values in summary["groups"].items():
        text.append(f"| {name} | {values['cases']} | {values['with_gold']} | {values['top3_any_hits']} | {values['top3_all_hits']} |")
    text += ["", "## 2. 阈值与 Top-K：漏召回和无答案也返回的取舍", "",
             "下表在同一批实际余弦得分上离线施加固定阈值，没有为每个阈值重新请求模型。"
             "这是开发集敏感性分析，不是在独立测试集验证过的生产阈值。余弦得分不是答案正确概率。", "",
             "| K | 阈值 | 可回答：至少一组 | 可回答：全部证据 | 无答案却仍返回文档 |", "|---:|---:|---:|---:|---:|"]
    for r in summary["threshold_sweep"]:
        text.append(f"| {r['k']} | {r['threshold']:.1f} | {r['any_hit_n']}/{r['positive_n']} | {r['all_hit_n']}/{r['positive_n']} | {r['unanswerable_returned_n']}/{r['unanswerable_n']} |")
    text += ["", "## 3. 每个例子实际找回了什么", "",
             "所有 query、过滤参数、目标资料、实际得分及完整返回正文都保存在 cases/*.json；原始 API 轨迹在 traces/*.json。"
             "人工检查后的错误可显式续跑一次，保存到 cases_retries/traces_retries，旧错误不覆盖。",
             "context 字段只用于解释及标注，没有偷偷交给检索器。oracle_rewrite 是预先写好的上下文补全对照，不代表产品已经具备自动改写能力。", ""]
    for row in rows:
        case = row["case"]
        text += [f"### {case['id']} · {case['group']}", "", f"问题：{case['query']}", ""]
        if case.get("context"):
            text += [case["context"], ""]
        text += [f"前提：{case['answerability']}；类型过滤：{case.get('filter_type') or '无'}；必要资料组：{case['gold_groups'] or '库内无充分答案/需先追问'}。", ""]
        if row["status"] == "completed":
            hits = "；".join(f"{d['id']}（{d['score']:.3f}）" for d in row["documents"][:3]) or "空结果"
            text += [f"实际 Top-3：{hits}。", ""]
            if case["gold_groups"]:
                found = {d["id"] for d in row["documents"][:3]}
                missing = [g for g in case["gold_groups"] if not found.intersection(g)]
                text += [f"Top-3 未覆盖的必要资料组：{missing or '无'}。", ""]
        else:
            text += [f"执行错误：{row['error']}。不算作召回失败。", ""]
    text += ["## 4. 下一步如何用这套数据迭代", "",
             "先对照案例确认漏在什么位置：库中没有 → 补资料并重新标注；类型过滤删掉 → 修过滤；短句缺历史 → 做查询补全；"
             "多个主题竞争 Top-K → 比较分解检索；无答案仍有相似结果 → 需要证据充分性检查，不能只看是否返回内容。",
             "保持这份 v1 不变。改动后另存新结果，再在未用于调参的新病例上确认收益；不能根据这些开发探针宣称临床安全或大库性能。", ""]
    (ROOT / "RAG召回边界报告.md").write_text("\n".join(text), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("completed", "answerable_n", "unanswerable_n", "errors")}, ensure_ascii=False))


def main(report_only=False, retry_errors=False):
    if report_only:
        report()
        return
    cases = [json.loads(s) for s in CASES.read_text(encoding="utf-8").splitlines()]
    corpus_ids = {json.loads(s)["id"] for s in (DATA / "corpus.jsonl").read_text(encoding="utf-8").splitlines()}
    assert len(cases) == len({c["id"] for c in cases})
    assert all(set(group) <= corpus_ids for c in cases for group in c["gold_groups"])
    protocol = {"cases_sha256": sha(CASES), "corpus_sha256": sha(DATA / "corpus.jsonl"),
                "embedding_model": EMBED_MODEL, "query_instruction": QUERY_INSTRUCTION,
                "search_top_k": 10, "generation": False, "thresholds": [0, .3, .4, .5, .6, .7, .8],
                "declared_before_results": True, "clinical_review": False}
    protocol_path = ROOT / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
        raise ValueError("Frozen boundary protocol changed; use a new version")
    dump(protocol_path, protocol)
    gateway = Gateway(OUTPUT)
    trace = []
    rag = LocalRAG(gateway, DATA / "corpus.jsonl", trace)
    for case in cases:
        destination = ROOT / "cases" / f"{case['id']}.json"
        trace_dir = "traces"
        if destination.exists():
            old = json.loads(destination.read_text(encoding="utf-8"))
            if old["status"] == "completed" or not retry_errors:
                continue
            destination = ROOT / "cases_retries" / destination.name
            trace_dir = "traces_retries"
            if destination.exists():
                continue
        trace.clear()
        started = time.monotonic()
        row = {"case": case}
        try:
            docs = rag.search(case["query"], top_k=10, filter_type=case.get("filter_type"))
            row.update(status="completed", documents=docs,
                       at_k={str(k): coverage(case["gold_groups"], docs, k) for k in (1, 3, 5, 10)})
        except Exception as error:
            row.update(status="error", error=f"{type(error).__name__}: {error}".replace(gateway.key, "[REDACTED]"))
        row["seconds"] = time.monotonic() - started
        dump(destination, row)
        dump(ROOT / trace_dir / f"{case['id']}.json", trace)
        print(json.dumps({"case": case["id"], "status": row["status"]}), flush=True)
        if row["status"] == "error":
            report()
            raise SystemExit("Boundary probe failed; preserved error requires inspection before continuation")
    report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--suite", choices=("base", "near_miss"), default="base")
    args = parser.parse_args()
    if args.suite == "near_miss":
        CASES = DATA / "rag_near_miss.jsonl"
        ROOT = OUTPUT / "rag_near_miss_v1"
    main(args.report_only, args.retry_errors)
