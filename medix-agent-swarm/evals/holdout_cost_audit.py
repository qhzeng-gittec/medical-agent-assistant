"""Offline request-ledger audit; never import or modify the frozen harness."""

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


TARGET_ROLES = {"supervisor", "diagnostic_agent", "consultation_agent", "research_agent"}
KNOWN_COST_STATUSES = {
    "provider_reported": "provider_reported",
    "estimated_standard_list_price_not_invoice": "google_estimated",
    "estimated_from_usage_and_catalog": "other_estimated",
}


def calibration_requests(root):
    paths = list((root / "calibration").glob("*.json"))
    paths += list((root / "calibration_grades").glob("**/judge_traces/*.json"))
    regression = root / "rag_regression" / "trace.json"
    if regression.exists():
        paths.append(regression)
    identifiers = set()
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        events = data if isinstance(data, list) else data.get("trace", [])
        identifiers.update(event["request_id"] for event in events
                           if event.get("event") == "api_request" and event.get("request_id"))
    return identifiers


def group_for(request_id, roles, calibration_ids):
    if len(roles) != 1:
        return "unknown"
    role = roles[0]
    if role == "embedding":
        return "embedding"
    if request_id in calibration_ids or role == "probe":
        return "calibration"
    if role in TARGET_ROLES:
        return "target"
    if role in {"patient_selector", "patient_simulator"}:
        return "selector"
    if role == "rubric_judge" or role.startswith("rubric_judge_"):
        return "judge"
    return "unknown"


def summarize(rows):
    unknown = sum(row["cost_kind"] == "unknown" for row in rows)
    amounts = {kind: round(sum(row["known_cost_usd"] or 0 for row in rows
                              if row["cost_kind"] == kind), 12)
               for kind in ("provider_reported", "google_estimated", "other_estimated")}
    subtotal = round(sum(amounts.values()), 12)
    return {"requests": len(rows), "cost_kind_counts": dict(Counter(row["cost_kind"] for row in rows)),
            "lifecycle_counts": dict(Counter(row["lifecycle"] for row in rows)),
            "unknown_cost_requests": unknown, "known_subtotal_usd": subtotal,
            "amounts_by_basis_usd": amounts, "api_cost_total_usd": None if unknown else subtotal}


def build_audit(root, stopped_pids=()):
    root = Path(root)
    stopped_pids = set(stopped_pids)
    calibration_ids = calibration_requests(root)
    records, snapshots, warnings = defaultdict(list), [], []
    paths = sorted((root / "ledgers").glob("*.jsonl"))
    legacy = root / "api_ledger.jsonl"
    if legacy.exists():
        paths.append(legacy)
    for path in paths:
        raw = path.read_bytes()
        lines = raw.decode("utf-8-sig").splitlines()
        relative = str(path.relative_to(root))
        match = re.fullmatch(r"api-(\d+)-.+\.jsonl", path.name)
        pid = int(match[1]) if match else None
        snapshots.append({"path": relative, "bytes": len(raw),
                          "sha256_at_read": hashlib.sha256(raw).hexdigest()})
        for index, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as error:
                if index == len(lines) and not raw.endswith((b"\n", b"\r")):
                    warnings.append({"path": relative, "line": index, "kind": "partial_trailing_record"})
                    continue
                raise ValueError(f"Invalid ledger JSON: {relative}:{index}") from error
            if not isinstance(entry, dict) or not isinstance(entry.get("request_id"), str):
                raise ValueError(f"Missing ledger request ID: {relative}:{index}")
            records[entry["request_id"]].append((entry, relative, pid))

    requests = []
    for request_id, entries in sorted(records.items()):
        terminal = [entry for entry, _, _ in entries if entry.get("status") not in {None, "started"}]
        observed = terminal or [entry for entry, _, _ in entries]
        roles = sorted({entry.get("role", "unknown") for entry in observed})
        pids = sorted({pid for _, _, pid in entries if pid is not None})
        unknown_reasons, amounts, kinds = set(), set(), set()
        for entry in observed:
            amount = entry.get("cost_usd", entry.get("charged_or_reserved_usd"))
            valid_amount = (type(amount) in {int, float} and math.isfinite(amount) and amount >= 0)
            if valid_amount:
                amounts.add(float(amount))
            cost_status = entry.get("cost_status", "missing")
            if cost_status not in KNOWN_COST_STATUSES:
                unknown_reasons.add(cost_status)
            else:
                kinds.add(KNOWN_COST_STATUSES[cost_status])
            if not valid_amount:
                unknown_reasons.add("missing_or_invalid_amount")
        if len(amounts) > 1 or len(kinds) > 1:
            unknown_reasons.add("conflicting_terminal_costs")
        kind = next(iter(kinds)) if len(kinds) == 1 and not unknown_reasons else "unknown"
        interrupted = not terminal and bool(pids) and set(pids).issubset(stopped_pids)
        lifecycle = ("finished" if terminal else "unfinished_interrupted" if interrupted
                     else "unfinished_in_flight_or_unresolved")
        requests.append({"request_id": request_id, "group": group_for(request_id, roles, calibration_ids),
                         "roles": roles, "providers": sorted({entry.get("provider", "unknown") for entry in observed}),
                         "models": sorted({entry.get("model", "unknown") for entry in observed}),
                         "lifecycle": lifecycle, "statuses": sorted({entry.get("status", "missing") for entry in observed}),
                         "cost_kind": kind, "cost_statuses": sorted({entry.get("cost_status", "missing") for entry in observed}),
                         "unknown_reasons": sorted(unknown_reasons),
                         "known_cost_usd": next(iter(amounts)) if kind != "unknown" else None,
                         "recorded_amount_candidates_usd": sorted(amounts),
                         "raw_entry_count": len(entries), "terminal_entry_count": len(terminal),
                         "process_ids": pids, "ledger_files": sorted({path for _, path, _ in entries})})
    groups = defaultdict(list)
    for request in requests:
        groups[request["group"]].append(request)
    summary = summarize(requests)
    summary["unparsed_partial_records"] = len(warnings)
    if warnings:
        summary["api_cost_total_usd"] = None
    return {"schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
            "scope": "All available API ledgers, including old calibration and interrupted/replaced judges",
            "snapshot_consistency": "Files read sequentially during execution; unfinished requests may finish later",
            "stopped_process_ids_supplied": sorted(stopped_pids), "ledger_snapshots": snapshots,
            "parse_warnings": warnings, "ledger_coverage_complete": not warnings, "summary": summary,
            "groups": {name: summarize(rows) for name, rows in sorted(groups.items())},
            "duplicate_terminal_request_ids": [row["request_id"] for row in requests if row["terminal_entry_count"] > 1],
            "requests": requests, "mem0_cost_included": False, "is_invoice_total": False,
            "limitations": ["Mem0 monetary costs are not exposed in these API ledgers.",
                            "A recorded numeric zero does not resolve unknown cost_status.",
                            "Google costs are list-price estimates, not provider invoices.",
                            "Request IDs deduplicate ledger records; distinct network requests may each be charged.",
                            "A started request from a supplied stopped PID may have reached the provider; its cost remains unknown.",
                            "Ledger metadata lacks run/attempt IDs and timestamps; this audit does not infer them."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Existing evaluation output directory")
    parser.add_argument("--stopped-pid", type=int, action="append", default=[])
    args = parser.parse_args()
    audit = build_audit(args.output, args.stopped_pid)
    destination = args.output / "cost_audit.json"
    destination.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"artifact": str(destination), **audit["summary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
