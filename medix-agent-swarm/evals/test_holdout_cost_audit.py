import json

import pytest

from holdout_cost_audit import build_audit


def ledger(root, pid, rows):
    folder = root / "ledgers"
    folder.mkdir(exist_ok=True)
    path = folder / f"api-{pid}-toy.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def event(identifier, status="ok", cost=0.25, basis="provider_reported", role="supervisor"):
    return {"request_id": identifier, "status": status, "cost_usd": cost,
            "cost_status": basis, "role": role, "provider": "toy", "model": "toy"}


def test_finished_known_replaces_started_and_duplicate_record_is_not_charged_twice(tmp_path):
    finished = event("one")
    ledger(tmp_path, 1, [event("one", "started", None, "missing"), finished])
    ledger(tmp_path, 2, [finished, event("one", "started", None, "missing")])
    audit = build_audit(tmp_path)
    assert audit["summary"]["requests"] == 1
    assert audit["summary"]["known_subtotal_usd"] == 0.25
    assert audit["summary"]["unknown_cost_requests"] == 0
    assert audit["duplicate_terminal_request_ids"] == ["one"]


def test_unknown_zero_and_stopped_unfinished_requests_remain_unknown(tmp_path):
    ledger(tmp_path, 57676, [event("failed", "error", 0, "unknown_not_zero_cost"),
                           event("interrupted", "started", None, "missing", "rubric_judge")])
    ledger(tmp_path, 2, [event("active", "started", None, "missing")])
    audit = build_audit(tmp_path, [57676])
    rows = {row["request_id"]: row for row in audit["requests"]}
    assert audit["summary"]["unknown_cost_requests"] == 3
    assert audit["summary"]["api_cost_total_usd"] is None
    assert rows["failed"]["known_cost_usd"] is None
    assert rows["failed"]["recorded_amount_candidates_usd"] == [0.0]
    assert rows["interrupted"]["lifecycle"] == "unfinished_interrupted"
    assert rows["active"]["lifecycle"] == "unfinished_in_flight_or_unresolved"


def test_terminal_unknown_takes_priority_and_preserves_prior_cost_observation(tmp_path):
    ledger(tmp_path, 1, [event("one"), event("one", "error", 0, "unknown_not_zero_cost")])
    row = build_audit(tmp_path)["requests"][0]
    assert row["cost_kind"] == "unknown"
    assert row["recorded_amount_candidates_usd"] == [0.0, 0.25]


def test_cost_bases_and_operational_groups_are_distinct(tmp_path):
    calibration = tmp_path / "calibration"
    calibration.mkdir()
    (calibration / "PUBLIC-CALIBRATION_trace.json").write_text(json.dumps([
        {"event": "api_request", "request_id": "calibration", "payload": {"private_toy": "DO_NOT_EXPORT"}}
    ]), encoding="utf-8")
    ledger(tmp_path, 1, [event("target"), event("selector", role="patient_selector"),
                         event("judge", role="rubric_judge"), event("calibration"),
                         event("embedding", role="embedding"), event("unknown", role="unrecognized"),
                         event("google", basis="estimated_standard_list_price_not_invoice")])
    audit = build_audit(tmp_path)
    assert set(audit["groups"]) == {"target", "selector", "judge", "calibration", "embedding", "unknown"}
    assert audit["summary"]["amounts_by_basis_usd"]["google_estimated"] == 0.25
    assert audit["summary"]["amounts_by_basis_usd"]["provider_reported"] == 1.5
    assert "DO_NOT_EXPORT" not in json.dumps(audit)
    assert audit["mem0_cost_included"] is False and audit["is_invoice_total"] is False


def test_trailing_partial_append_is_reported_without_discarding_complete_records(tmp_path):
    path = ledger(tmp_path, 1, [event("one")])
    with path.open("a", encoding="utf-8") as stream:
        stream.write('{"request_id":')
    audit = build_audit(tmp_path)
    assert audit["summary"]["requests"] == 1
    assert audit["parse_warnings"][0]["kind"] == "partial_trailing_record"
    assert audit["ledger_coverage_complete"] is False
    assert audit["summary"]["api_cost_total_usd"] is None


def test_malformed_complete_ledger_line_fails_with_location(tmp_path):
    path = ledger(tmp_path, 1, [event("one")])
    with path.open("a", encoding="utf-8") as stream:
        stream.write("invalid\n")
    with pytest.raises(ValueError, match="Invalid ledger JSON"):
        build_audit(tmp_path)
