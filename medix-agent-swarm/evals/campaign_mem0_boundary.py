"""Real Mem0 Platform probes through the unchanged product memory adapter."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import httpx

# Set before importing Mem0; telemetry is unrelated to this authorized test.
os.environ["MEM0_TELEMETRY"] = "false"

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "evals" / "campaign_v1" / "mem0_boundary.json"
ROOT = PROJECT / "evals" / "results" / "full_system_v1" / "mem0_boundary_v1"


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def anchor_checks(required, memories):
    text = "\n".join(str(m.get("content") or m.get("memory") or m.get("text") or "") for m in memories)
    return [any(re.search(pattern, text, re.IGNORECASE) for pattern in alternatives) for alternatives in required]


def result_items(result):
    items = result["results"] if isinstance(result, dict) else result
    if not isinstance(items, list):
        raise ValueError("Unexpected Mem0 result envelope")
    return items


class Recorder:
    def __init__(self, key):
        self.key = key

    def call(self, name, parameters, fn):
        path = ROOT / "operations" / f"{name}.json"
        if path.exists():
            previous = read(path)
            if previous["parameters"] != parameters:
                raise ValueError(f"Parameters changed for preserved operation {name}")
            if previous["status"] == "completed":
                return previous["result"]
            raise RuntimeError(f"Unresolved prior operation {name}; inspect before retry, especially writes")
        record = {"name": name, "parameters": parameters, "status": "started",
                  "started_at_utc": datetime.now(timezone.utc).isoformat()}
        dump(path, record)
        started = time.monotonic()
        try:
            result = fn()
            record.update(status="completed", result=result)
            return result
        except Exception as error:
            record.update(status="error", error=f"{type(error).__name__}: {error}".replace(self.key, "[REDACTED]"))
            raise
        finally:
            record["seconds"] = time.monotonic() - started
            dump(path, record)


class ProductClient:
    """Record actual SDK calls; preserve product add/search arguments unchanged."""

    def __init__(self, sdk, recorder):
        self.sdk, self.recorder = sdk, recorder
        self.name = ""

    def add(self, **kwargs):
        return self.recorder.call(self.name, kwargs, lambda: self.sdk.add(**kwargs))

    def search(self, **kwargs):
        return self.recorder.call(self.name, kwargs, lambda: self.sdk.search(**kwargs))


def wait_for_write(sdk, recorder, name, receipt):
    if receipt.get("status") == "SUCCEEDED" or ("results" in receipt and not receipt.get("event_id")):
        return {"status": "SUCCEEDED", "receipt": receipt}
    event_id = receipt.get("event_id")
    if not event_id:
        raise ValueError("Mem0 accepted write without a recognized completion/event receipt")
    for attempt, delay in enumerate((0, 2, 5, 10, 20, 30, 30)):
        event_path = ROOT / "operations" / f"{name}__event_{attempt}.json"
        if not event_path.exists() and delay:
            time.sleep(delay)

        def fetch():
            response = sdk.client.get(f"/v1/event/{quote(event_id, safe='')}/")
            response.raise_for_status()
            return response.json()

        event = recorder.call(f"{name}__event_{attempt}", {"event_id": event_id}, fetch)
        if event["status"] == "SUCCEEDED":
            return event
        if event["status"] == "FAILED":
            raise RuntimeError(f"Mem0 extraction failed: {event.get('error')}")
        if event["status"] not in {"PENDING", "RUNNING"}:
            raise ValueError(f"Unknown event status: {event['status']}")
    raise TimeoutError("Mem0 write still pending after bounded observation; not scored as retrieval failure")


def report():
    rows = [read(p) for p in sorted((ROOT / "cases").glob("*.json"))]
    queries = [q for row in rows for q in row.get("queries", [])]
    positives = [q for q in queries if q["query"]["required_terms"]]
    stored = [q for q in positives if all(q["storage_anchors"])]
    negatives = [q for q in queries if q["query"].get("expect_empty")]
    summary = {"scenarios_completed": sum(r["status"] == "completed" for r in rows),
               "scenarios_planned": 12, "queries_completed": len(queries),
               "positive_anchor_queries": len(positives),
               "all_anchors_present_in_storage": len(stored),
               "baseline_all_anchor_hits": sum(all(q["baseline_anchors"]) for q in positives),
               "stored_baseline_all_anchor_hits": sum(all(q["baseline_anchors"]) for q in stored),
               "negative_queries": len(negatives),
               "negative_empty_results": sum(not q["baseline"] for q in negatives),
               "errors": [{"scenario": r["scenario"]["id"], "error": r["error"]} for r in rows if r["status"] != "completed"],
               "grading": "Deterministic bilingual anchor checks plus inspectable full text; not clinician or full semantic validation",
               "mem0_platform_cost": "Not exposed by these operation responses; do not report as zero or as OpenRouter charges"}
    dump(ROOT / "summary.json", summary)
    text = ["# Mem0 长期记忆召回边界", "",
            "测试项目现用 Mem0 Platform，SDK mem0ai==2.0.20，经由原项目 LongTermMemory 的 add/search 方法。"
            "在隔离的临时测试用户/应用下写入合成对话，不使用真实患者资料，不修改产品实现或全局项目设置。", "",
            f"已完成 {summary['scenarios_completed']}/12 个场景，{len(queries)} 次问题的对照检索；执行错误 {len(summary['errors'])}。", "",
            "## 1. 怎么分清存不进去与找不出来", "",
            "先记录新增返回和立即检索，再跟踪写入事件完成，列出该用户该应用下的实际存储记录，最后检索。"
            "每一步都保留原始 JSON。baseline 是原项目阈值 0.3、Top-3；另外依次调用 Top-10/0.3 和 Top-3/0.0 作参数敏感性检查。"
            "请求每次只改一个参数，但云端索引与排序状态没有冻结，不能把差异解释成严格的单因素因果收益。",
            "评估器给事实预先写中英双语关键词组，用来辅助定位遗漏。关键词命中不证明主体、否定、时间或医学语义正确；"
            "所以完整存储文本和召回文本都列在下面供复核。无目标词的污染测试不自动算通过。", "",
            f"有明确事实词组的问题：{len(positives)}；所有词组都在存储中：{len(stored)}；"
            f"在这些已存全的查询中，baseline 找全词组 {summary['stored_baseline_all_anchor_hits']}/{len(stored)}。",
            f"预期无相关记忆/需要隔离的问题：{len(negatives)}，baseline 返回空集 {summary['negative_empty_results']}/{len(negatives)}。"
            "‘返回了无关记忆’不同于‘下游生成了错误回答’。", "",
            "## 2. 逐场景原始结果", ""]
    for row in rows:
        case = row["scenario"]
        text += [f"### {case['id']} · {case['topic']}", "", "写入的合成对话：", ""]
        for i, turn in enumerate(case["turns"], 1):
            text += [f"{i}. 用户：{turn['user']}  助手：{turn['assistant']}", ""]
        text += ["实际持久化文本：", ""]
        for memory in row.get("stored", []):
            text += [f"- {memory.get('memory') or memory.get('text')}（id={memory.get('id')}）"]
        text.append("")
        for query in row.get("queries", []):
            text += [f"问题：{query['query']['query']}", "",
                     f"词组覆盖：存储 {query['storage_anchors']}；原配置 {query['baseline_anchors']}；"
                     f"Top-10 {query['top10_anchors']}；阈值降为 0 {query['threshold0_anchors']}。", ""]
            for item in query["baseline"]:
                text += [f"- score={item.get('score')}：{item['content']}"]
            if not query["baseline"]:
                text += ["- 返回空集。"]
            if query["query"].get("review_focus"):
                text += ["", "核对重点：" + query["query"]["review_focus"]]
            text.append("")
        if row["status"] != "completed":
            text += ["执行中断：" + row["error"], ""]
    text += ["## 3. 解释边界", "",
             "当前官方 V3 新增流程为追加式抽取，新增本身不做旧记录的 UPDATE/DELETE。历史记录存在并不自动算错误，"
             "但不能把‘能够找回某条历史’等同于‘知道当前最新状态’。"
             "[新增接口](https://docs.mem0.ai/api-reference/memory/add-memories)。",
             "事件成功表示写入流程完成，不代表所有时间排序信号立即就绪。本轮是在写入后短时间内测试，不是数周后的长期稳定性实验。"
             "[事件说明](https://docs.mem0.ai/api-reference/events/get-event)。",
             "云端返回的是托管检索综合分数，不是本地 Qwen 余弦，也不能把 0.3 当成医学置信度。"
             "[搜索接口](https://docs.mem0.ai/api-reference/memory/search-memories)。",
             "实际响应中有分数低于请求阈值的结果。官方说明部分搜索阶段可在阈值筛选之后调整分数；"
             "例如开启 decay 时会发生这种情况。本轮未核定项目的全部排序配置，不据此断言具体原因或偷偷做客户端二次过滤。"
             "[分数与衰减说明](https://docs.mem0.ai/platform/features/memory-decay)。",
             "本轮没有通过 OpenRouter 控制 Mem0 云端内部 embedding/抽取模型；Mem0 自己的服务用量不包含在 OpenRouter 账本中。",
             "测试数据保留用于审计；没有删除账号中的任何既有记忆。需要清理时应只删除 manifest 列出的测试 user/app 下本次创建的 ID。", ""]
    (ROOT / "Mem0召回边界报告.md").write_text("\n".join(text), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def main(report_only=False):
    if report_only:
        report()
        return
    key = os.getenv("MEM0_API_KEY")
    if not key:
        raise SystemExit("MEM0_API_KEY missing; no platform calls made")
    from mem0 import MemoryClient

    spec = importlib.util.spec_from_file_location("medix_eval_long_term", PROJECT / "memory" / "long_term.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cases = read(DATA)["scenarios"]
    manifest_path = ROOT / "manifest.json"
    data_sha = hashlib.sha256(DATA.read_bytes()).hexdigest()
    if manifest_path.exists():
        manifest = read(manifest_path)
        if manifest["dataset_sha256"] != data_sha:
            raise ValueError("Frozen Mem0 dataset changed; create a new run")
    else:
        prefix = "medix-eval-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
        manifest = {"prefix": prefix, "app_id": prefix + "-app", "dataset_sha256": data_sha,
                    "mem0_sdk": importlib.metadata.version("mem0ai"),
                    "user_ids": [f"{prefix}-{c['id']}" for c in cases],
                    "purpose": "Synthetic boundary tests; retained for audit; not production data",
                    "baseline": {"threshold": 0.3, "top_k": 3},
                    "max_write_requests": sum(len(c["turns"]) for c in cases),
                    "managed_models": "Mem0 Platform controls its backend; not configured with OpenRouter Qwen"}
        dump(manifest_path, manifest)
    # Dedicated client keeps all test network requests bounded without changing product code.
    with httpx.Client(timeout=45) as http:
        sdk = MemoryClient(api_key=key, client=http)
        dump(ROOT / "connection.json", {"status": "authenticated", "sdk": manifest["mem0_sdk"],
                                        "at_utc": datetime.now(timezone.utc).isoformat()})
        recorder = Recorder(key)
        wrapped = ProductClient(sdk, recorder)
        app_id = manifest["app_id"]
        memory = module.LongTermMemory(config={"app_id": app_id, "threshold": 0.3}, client=wrapped)
        for case in cases:
            path = ROOT / "cases" / f"{case['id']}.json"
            if path.exists():
                continue
            user_id = f"{manifest['prefix']}-{case['id']}"
            row = {"scenario": case, "user_id": user_id, "app_id": app_id,
                   "queries": [], "write_events": [], "status": "running"}
            try:
                for index, turn in enumerate(case["turns"], 1):
                    name = f"{case['id']}__write_{index}"
                    wrapped.name = name
                    receipt_path = ROOT / "operations" / f"{name}.json"
                    if receipt_path.exists():
                        if read(receipt_path)["status"] != "completed":
                            raise RuntimeError(f"Unresolved write {name}; do not duplicate it")
                    else:
                        memory.add_session_summary(user_id, f"{user_id}-session-{index}", turn["user"], turn["assistant"],
                                                   metadata={"evaluation": "synthetic-mem0-boundary-v1"})
                    receipt = read(ROOT / "operations" / f"{name}.json")["result"]
                    wrapped.name = name + "__immediate_search"
                    immediate = memory.search_similar_sessions(case["queries"][0]["query"], user_id, limit=3)
                    event = wait_for_write(sdk, recorder, name, receipt)
                    row["write_events"].append({"receipt": receipt, "immediate_search": immediate, "event": event})
                filters = {"AND": [{"user_id": user_id}, {"app_id": app_id}]}
                snapshot = recorder.call(f"{case['id']}__stored", {"filters": filters, "page_size": 100},
                                         lambda: sdk.get_all(filters=filters, page_size=100))
                if isinstance(snapshot, dict) and snapshot.get("next"):
                    raise ValueError("Snapshot paginated; do not score partial stored memories")
                stored = result_items(snapshot)
                row["stored"] = stored
                for query in case["queries"]:
                    query_user = user_id + "-other" if query.get("scope") == "other_user" else user_id
                    query_app = app_id + "-other" if query.get("scope") == "other_app" else app_id
                    scoped = module.LongTermMemory(config={"app_id": query_app, "threshold": .3}, client=wrapped)
                    name = f"{case['id']}__query_{query['id']}"
                    wrapped.name = name + "__baseline"
                    baseline = scoped.search_similar_sessions(query["query"], query_user, limit=3)
                    query_filters = {"AND": [{"user_id": query_user}, {"app_id": query_app}]}
                    alternatives = {}
                    for label, k, threshold in (("top10", 10, .3), ("threshold0", 3, .0)):
                        params = {"query": query["query"], "filters": query_filters, "top_k": k, "threshold": threshold}
                        alternatives[label] = result_items(recorder.call(name + "__" + label, params,
                                                                          lambda p=params: sdk.search(**p)))
                    required = query["required_terms"]
                    row["queries"].append({"query": query, "baseline": baseline, **alternatives,
                                           "storage_anchors": anchor_checks(required, stored),
                                           "baseline_anchors": anchor_checks(required, baseline),
                                           "top10_anchors": anchor_checks(required, alternatives["top10"]),
                                           "threshold0_anchors": anchor_checks(required, alternatives["threshold0"])})
                row["status"] = "completed"
            except Exception as error:
                row.update(status="error", error=f"{type(error).__name__}: {error}".replace(key, "[REDACTED]"))
                dump(path, row)
                report()
                raise
            dump(path, row)
            print(json.dumps({"case": case["id"], "status": row["status"], "stored": len(stored),
                              "queries": len(row["queries"])}), flush=True)
    report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT,
                        help="Use a new directory for an independent repetition; preserve completed runs")
    args = parser.parse_args()
    ROOT = args.output.resolve()
    main(args.report_only)
