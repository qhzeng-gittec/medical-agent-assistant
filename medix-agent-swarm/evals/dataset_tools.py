"""Read/validate the development dataset without importing the medical application."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


DEFAULT_DATASET = Path(__file__).parent / "datasets" / "medix_dev_v0_1"


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"{path}:{number}: blank JSONL row")
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{number}: expected object")
        rows.append(row)
    return rows


def record_hash(record: dict) -> str:
    payload = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def current_user_message(case: dict, turn: int) -> dict:
    """Only this event crosses the model boundary; the runner owns prior state."""
    event = next(event for event in case["events"] if event["turn"] == turn)
    return {"role": "user", "content": event["text"]}


def indexed(rows: list[dict], key: str) -> dict:
    result = {row[key]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate {key}")
    return result


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate(root: Path, verify_hashes: bool = True) -> dict:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    cases = indexed(read_jsonl(root / "cases.jsonl"), "case_id")
    rubrics = indexed(read_jsonl(root / "private/rubrics.jsonl"), "case_id")
    fixtures = indexed(read_jsonl(root / "private/fixtures.jsonl"), "case_id")
    sources = indexed(read_jsonl(root / "private/source_records.jsonl"), "source_id")
    documents = indexed(read_jsonl(root / "documents.jsonl"), "key")
    require(cases.keys() == rubrics.keys() == fixtures.keys(), "Case/rubric/fixture IDs differ")
    families = {}
    fingerprints = set()
    valid_modes = {"exact_set", "rubric", "trace", "exact_metrics"}
    for case_id, case in cases.items():
        require(case["dataset_version"] == manifest["version"], f"{case_id}: version mismatch")
        require(case["split"] == "dev", f"{case_id}: pilot is dev only")
        family = case["family_id"]
        require(families.setdefault(family, case["split"]) == case["split"], "Family split leak")
        require(set(case["source_ids"]) <= sources.keys(), f"{case_id}: missing source")
        require(bool(case["source_ids"]) == (case["origin"] == "public_dataset_adaptation"),
                f"{case_id}: origin mismatch")
        require(not ({"answer", "gold", "checks", "initial_state", "record"} & case.keys()),
                f"{case_id}: private fields in case")
        events = case["events"]
        require([e["turn"] for e in events] == list(range(1, len(events) + 1)),
                f"{case_id}: turns not sequential")
        require(bool(events) == (case["suite"] != "component"), f"{case_id}: invalid events")
        for event in events:
            require(set(event) == {"turn", "user_id", "session_id", "text"},
                    f"{case_id}: event fields must be explicitly allowlisted")
            require(all(isinstance(event[k], str) and event[k] for k in ("user_id", "session_id", "text")),
                    f"{case_id}: invalid event")
        fingerprint = json.dumps([e["text"] for e in events], ensure_ascii=False)
        if events:
            require(fingerprint not in fingerprints, f"{case_id}: duplicate dialogue")
            fingerprints.add(fingerprint)
        checks = rubrics[case_id]["checks"]
        indexed(checks, "check_id")
        require(bool(checks), f"{case_id}: no checks")
        for check in checks:
            require(check["mode"] in valid_modes, f"{case_id}: unknown grading mode")
            require(check["weight"] > 0 and isinstance(check["critical"], bool),
                    f"{case_id}: invalid check metadata")
            require(check["turn"] in ([0] if not events else range(1, len(events) + 1)),
                    f"{case_id}: check refers to nonexistent turn")
        fixture = fixtures[case_id]
        for rule in fixture["tool_responses"]:
            require(rule["turn"] in range(1, len(events) + 1), f"{case_id}: invalid tool turn")
            require(rule["match"] == "any_arguments", f"{case_id}: unsupported matching")
            require(bool(rule["responses"]), f"{case_id}: empty responses")
            for response in rule["responses"]:
                require(response["status"] in {"ok", "error"}, f"{case_id}: invalid status")
                require(set(response.get("document_keys", [])) <= documents.keys(),
                        f"{case_id}: unknown tool document")
        invocations = {}
        for action in fixture.get("component_actions", []):
            require(action["op"] in {"call", "clear_visible_messages"}, f"{case_id}: unknown action")
            invocation = action["invocation"]
            if action["op"] == "call":
                scope = (action["worker"], action["turn"])
                require(invocations.setdefault(invocation, scope) == scope,
                        f"{case_id}: invocation crosses worker or turn")
                require(set(action["document_keys"]) <= documents.keys(), f"{case_id}: unknown document")
            else:
                require(invocation in invocations, f"{case_id}: unknown invocation to clear")
        if case["suite"] == "component":
            calls = sum(a["op"] == "call" for a in fixture["component_actions"])
            metrics = checks[0]["expectation"]
            require(len(metrics["new_bodies_per_call"]) == calls, f"{case_id}: body count mismatch")
            require(len(metrics["references_per_call"]) == calls, f"{case_id}: reference count mismatch")
    for source in sources.values():
        require(source["record_sha256"] == record_hash(source["record"]),
                f"{source['source_id']}: source record checksum mismatch")
    suites = Counter(case["suite"] for case in cases.values())
    origins = Counter(case["origin"] for case in cases.values())
    expected_counts = {"total": len(cases), **suites,
                       "public_adaptations": origins["public_dataset_adaptation"],
                       "synthetic": origins["synthetic"]}
    require(expected_counts == manifest["counts"], "Manifest counts mismatch")
    if verify_hashes:
        required = {"cases.jsonl", "documents.jsonl", "private/rubrics.jsonl",
                    "private/fixtures.jsonl", "private/source_records.jsonl"}
        require(required <= manifest["files"].keys(), "Manifest missing data checksums")
        for relative, expected in manifest["files"].items():
            path = (root / relative).resolve()
            require(path.is_relative_to(root.resolve()), "Manifest path escapes dataset")
            require(hashlib.sha256(path.read_bytes()).hexdigest() == expected,
                    f"{relative}: file checksum mismatch")
    return {"status": "valid", "cases": len(cases), "families": len(families),
            "suites": dict(suites), "events": sum(len(c["events"]) for c in cases.values()),
            "checks": sum(len(r["checks"]) for r in rubrics.values()),
            "clinical_review": manifest["clinical_review"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args()
    print(json.dumps(validate(args.dataset), ensure_ascii=False, indent=2))
