"""Read-only artifact checks; write one verification summary, never call paid APIs."""

import json
import os
from collections import Counter

from campaign_gateway import dump, sha
from campaign_improvement import ROOT, CASES, read
from campaign_run import DATA, OUTPUT, PROJECT


def ledger(path):
    latest = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        latest[row["request_id"]] = row
    return {"requests": len(latest), "usd": sum(r["charged_or_reserved_usd"] for r in latest.values()),
            "inflight": sum(r["status"] == "started" for r in latest.values()),
            "cost_statuses": dict(Counter(r["cost_status"] for r in latest.values()))}


def main():
    for variant in ("baseline", "candidate"):
        protocol = read(ROOT / f"protocol_{variant}.json")
        product = ROOT / "baseline_product" if variant == "baseline" else PROJECT
        assert protocol["dataset_sha256"] == sha(CASES)
        assert protocol["corpus_sha256"] == sha(DATA / "corpus.jsonl")
        assert all(sha(product / name) == digest for name, digest in protocol["sources"].items())
    runs = [read(p) for p in (ROOT / "runs").glob("*.json")]
    regressions = [read(p) for p in (ROOT / "regressions/runs").glob("*.json")]
    clouds = [read(p) for p in (ROOT / "mem0_live/cases").glob("*.json")]
    assert len(runs) == 48 and all(r["status"] == "completed" for r in runs)
    assert len(regressions) == 6 and all(r["status"] == "completed" for r in regressions)
    assert all(not r["protocol"]["failures"] for r in runs + regressions)
    assert all(not r["metrics"]["model_requested_unadvertised_tools"] for r in runs)
    rejected_calls = []
    for path in (ROOT / "regressions/traces").glob("*.json"):
        trace = read(path)
        apis = [e for e in trace if e["event"] == "api_request"]
        observations = {m["tool_call_id"]: json.loads(m["content"])
                        for e in apis for m in e["payload"].get("messages", []) if m["role"] == "tool"}
        for event in apis:
            advertised = {t["function"]["name"] for t in event["payload"].get("tools", [])}
            for choice in event.get("response", {}).get("choices", []):
                for call in choice["message"].get("tool_calls", []):
                    if call["function"]["name"] not in advertised:
                        result = observations[call["id"]]
                        assert result.get("error") == "ToolCallBudgetExceeded", "Inspect unadvertised tool execution"
                        rejected_calls.append({"case": path.stem, "name": call["function"]["name"],
                                               "rejected_with": result["error"]})
    assert len(clouds) == 3 and all(r["status"] == "completed" and r["other_user_result"] == [] for r in clouds)
    keys = [os.environ[name] for name in ("OPENROUTER_API_KEY", "MEM0_API_KEY")]
    assert all(keys), "Three configured keys are required for the local redaction check"
    checked = 0
    for path in ROOT.rglob("*"):
        if path.suffix not in {".json", ".jsonl", ".md", ".yaml", ".yml", ".py"}:
            continue
        content = path.read_text(encoding="utf-8-sig")
        assert not any(key in content for key in keys), f"Credential found in {path}"
        if path.suffix == ".json":
            json.loads(content)
        elif path.suffix == ".jsonl":
            for line in content.splitlines():
                json.loads(line)
        checked += 1
    charges = {
        "openrouter_all_campaigns": ledger(OUTPUT / "api_ledger.jsonl"),
        "google_previous": ledger(OUTPUT / "gemini_memory_ablation_v1/google_api_ledger.jsonl"),
        "google_improvement_and_regressions": ledger(ROOT / "google_api_ledger.jsonl"),
    }
    assert all(value["inflight"] == 0 for value in charges.values())
    openrouter_total = charges["openrouter_all_campaigns"]["usd"] + 0.00008073
    google_total = charges["google_previous"]["usd"] + charges["google_improvement_and_regressions"]["usd"] + .025
    assert openrouter_total <= 4 and google_total <= 1
    summary = {"paired_runs": len(runs), "paired_user_turns": sum(len(r["turns"]) for r in runs),
               "regression_runs": len(regressions), "mem0_scenarios": len(clouds),
               "protocol_failures": 0, "product_and_dataset_hashes_match": True,
               "regression_unadvertised_requests_rejected": rejected_calls,
               "redaction_files_checked": checked, "three_configured_keys_absent": True,
               "charges": charges, "openrouter_accounted_usd_including_probe": openrouter_total,
               "google_accounted_usd_including_unknown_probe_reserve": google_total,
               "combined_accounted_usd": openrouter_total + google_total,
               "google_prices_are_estimates_not_invoices": True,
               "clinical_review_completed": False}
    dump(ROOT / "verification.json", summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
