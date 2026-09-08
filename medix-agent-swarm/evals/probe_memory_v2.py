"""Bounded live Mem0 checks of per-call extraction instructions and provenance metadata."""

import importlib.util
import os
import uuid

import httpx

import campaign_mem0_boundary as audit


ROOT = audit.PROJECT / "evals/results/improvement_v2/mem0_live"
SCENARIOS = [
    {"id": "source", "turns": [("我只说过最近头痛，没有提供其他病史。", "你有十年高血压病史，所以应该记录高血压。")],
     "query": "用户明确提供过哪些既往疾病？"},
    {"id": "change", "turns": [("我每天服用氯沙坦。", "请按医生指导用药。"),
                                  ("2026年9月2日医生已经让我停用氯沙坦。", "已了解您报告的停药变化。")],
     "query": "我现在还在服用氯沙坦吗？"},
    {"id": "coverage", "turns": [("我被医生诊断为偏头痛，对阿司匹林过敏，服用后起皮疹。", "就医时请说明过敏史。"),
                                    ("我每周日夜轮班，腰部旧伤后医生要求避免搬重物。", "这些个人情况需要在制定建议时考虑。")],
     "query": "总结我的病史、过敏、作息和运动限制。"},
]


def main():
    from mem0 import MemoryClient
    audit.ROOT = ROOT
    key = os.environ["MEM0_API_KEY"]
    path = ROOT / "manifest.json"
    source = audit.PROJECT / "memory/long_term.py"
    if path.exists():
        manifest = audit.read(path)
        if manifest["scenarios"] != SCENARIOS:
            # JSON represents tuples as arrays.
            import json
            if manifest["scenarios"] != json.loads(json.dumps(SCENARIOS)):
                raise ValueError("Frozen Mem0 scenarios changed")
    else:
        prefix = "medix-v2-" + uuid.uuid4().hex[:12]
        manifest = {"prefix": prefix, "app_id": prefix + "-app", "scenarios": SCENARIOS,
                    "max_add_requests": 5, "synthetic_only": True,
                    "scope": "new per-call instructions and metadata; no account-level changes or deletions"}
        audit.dump(path, manifest)
    spec = importlib.util.spec_from_file_location("memory_v2_probe", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    recorder = audit.Recorder(key)
    with httpx.Client(timeout=45) as http:
        sdk = MemoryClient(api_key=key, client=http)
        wrapped = audit.ProductClient(sdk, recorder)
        memory = module.LongTermMemory(config={"app_id": manifest["app_id"], "threshold": .3}, client=wrapped)
        for case in SCENARIOS:
            result_path = ROOT / "cases" / (case["id"] + ".json")
            if result_path.exists():
                if audit.read(result_path)["status"] != "completed":
                    raise RuntimeError("Unresolved Mem0 case; inspect before resuming")
                continue
            user = manifest["prefix"] + "-" + case["id"]
            result = {"status": "running", "user_id": user, "scenario": case}
            audit.dump(result_path, result)
            try:
                for index, (question, answer) in enumerate(case["turns"]):
                    name = f"{case['id']}__add_{index}"
                    wrapped.name = name
                    receipt_path = ROOT / "operations" / (name + ".json")
                    if not receipt_path.exists():
                        memory.add_session_summary(user, f"{user}-session-{index}", question, answer)
                    receipt = audit.read(receipt_path)
                    if receipt["status"] != "completed":
                        raise RuntimeError("Unresolved write; refusing duplicate submission")
                    audit.wait_for_write(sdk, recorder, name, receipt["result"])
                filters = {"AND": [{"user_id": user}, {"app_id": manifest["app_id"]}]}
                stored = recorder.call(case["id"] + "__list", {"filters": filters}, lambda: sdk.get_all(filters=filters, page_size=100))
                if isinstance(stored, dict) and stored.get("next"):
                    raise ValueError("Unexpected paginated test store")
                result["stored"] = audit.result_items(stored)
                result["queries"] = {}
                for limit in (3, 10):
                    wrapped.name = f"{case['id']}__search_{limit}"
                    result["queries"][str(limit)] = memory.search_similar_sessions(case["query"], user, limit)
                wrapped.name = case["id"] + "__isolation"
                result["other_user_result"] = memory.search_similar_sessions(case["query"], user + "-other", 3)
                result["status"] = "completed"
            except Exception as error:
                result.update(status="error", error=f"{type(error).__name__}: {error}".replace(key, "[REDACTED]"))
                raise
            finally:
                audit.dump(result_path, result)
                print(case["id"], result["status"], flush=True)


if __name__ == "__main__":
    main()
