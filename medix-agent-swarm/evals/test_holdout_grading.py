"""Independent toy calibration: never inspect the sealed holdout or its runs."""

import asyncio
import importlib.util
import json
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location("holdout_grade", Path(__file__).with_name("holdout_grade.py"))
GRADE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GRADE)


def fixture():
    case = {"case_id": "TOY", "family_id": "TOY-FAMILY", "domain": "history", "difficulty": "routine",
            "interaction": {"scripted_followups": [{"text": "UNOBSERVED_SCRIPT"}]},
            "private": {"patient_facts": [{"text": "UNREVEALED_FACT"}],
                        "seed_history": [{"text": "UNEXECUTED_SEED"}],
                        "environment": {"documents": [{"content": "UNRETRIEVED_DOCUMENT"}]},
                        "source_refs": [{"note": "PRIVATE_REVIEW_NOTE"}],
                        "clinical_review_required": False,
                        "checkpoints": [{"id": "subject", "after": "each_turn", "target": "profile",
                                         "requirement": "Record the stated subject accurately.", "critical": True},
                                        {"id": "final", "after": "final", "target": "answer",
                                         "requirement": "Address the latest request.", "critical": False},
                                        {"id": "future", "after": "turn:3", "target": "answer",
                                         "requirement": "Address the third turn.", "critical": False}]}}
    result = {"run_id": "TOY__qwen_qwen3.5-27b__r1", "case_id": "TOY", "model": "qwen/qwen3.5-27b",
              "repetition": 1, "status": "completed", "domain": "history", "difficulty": "routine",
              "seed_turns": [], "metrics": {"prompt_tokens": 12, "completion_tokens": 4},
              "turns": [{"turn": 1, "user": "That appointment belongs to my cousin.",
                         "answer": "I recorded it as yours.", "user_id": "TOY__qwen_qwen3.5-27b__r1_user",
                         "session_id": "TOY__qwen_qwen3.5-27b__r1_session", "profile_before": [],
                         "profile_after": [{"subject": "self"}], "profile_record_after": {"source": "actual-store"},
                         "event_memory_after": [], "latency_seconds": 1.0, "system_result": {"call_trace": []}},
                        {"turn": 2, "user": "LATER_DISCLOSED_CORRECTION", "answer": "I corrected it.",
                         "user_id": "TOY__qwen_qwen3.5-27b__r1_user", "session_id": "new_session",
                         "profile_after": [], "event_memory_after": [], "latency_seconds": 2.0,
                         "system_result": {"call_trace": []}}]}
    trace = [{"event": "turn_start", "turn": 1, "phase": "target"},
             {"event": "api_request", "role": "patient_selector", "payload": {"messages": ["UNREVEALED_FACT"]}},
             {"event": "api_request", "role": "supervisor", "payload": {
                 "model": "qwen/qwen3.5-27b", "messages": [{"role": "system", "content": "PRODUCT_SYSTEM_PROMPT"}]},
              "response": {"model": "qwen/qwen3.5-27b", "choices": [{"message": {"content": "Internal thinking"}}]}},
             {"event": "retrieval", "documents": [{"id": "actual-doc", "content": "ACTUALLY_RETRIEVED",
                "metadata": {"source": "https://example.org", "published_at": "2020-01-01"}}]},
             {"event": "memory_write", "result": {"status": "success", "records": []}},
             {"event": "turn_start", "turn": 2, "phase": "target"},
             {"event": "tool_observation", "result": {"content": "LATER_TOOL_OUTCOME"}}]
    return case, result, trace


def test_checkpoint_timing_and_missing_turn_remain_explicit():
    case, result, _ = fixture()
    checkpoints = GRADE.expand_checkpoints(case["private"]["checkpoints"], result["turns"])
    assert [(c["id"], c["horizon"], c["executed"]) for c in checkpoints] == [
        ("subject@1", 1, True), ("subject@2", 2, True), ("final@2", 2, True), ("future@3", 3, False)]
    assert GRADE.expand_checkpoints(case["private"]["checkpoints"], [])[0]["executed"] is False


def test_first_horizon_is_blind_to_identity_and_future_facts():
    case, result, trace = fixture()
    checkpoint = GRADE.expand_checkpoints(case["private"]["checkpoints"], result["turns"])[0]
    payload = GRADE.build_judge_payload(case, result, trace, [checkpoint], 1)
    serialized = json.dumps(payload)
    for prohibited in ("qwen", "Qwen", "UNREVEALED_FACT", "UNOBSERVED_SCRIPT", "UNEXECUTED_SEED",
                       "UNRETRIEVED_DOCUMENT", "PRIVATE_REVIEW_NOTE", "PRODUCT_SYSTEM_PROMPT",
                       "LATER_DISCLOSED_CORRECTION", "LATER_TOOL_OUTCOME", "I corrected it."):
        assert prohibited not in serialized
    assert "ACTUALLY_RETRIEVED" in serialized
    assert "https://example.org" in serialized
    assert "2020-01-01" in serialized
    assert "actual-store" in serialized
    assert payload["observations"]["turns"][0]["user_id"] == "subject_1"


def test_unassigned_and_patient_selector_trace_never_leaks():
    case, result, trace = fixture()
    trace.insert(0, {"event": "memory_read", "content": "UNKNOWN_TIMING"})
    payload = GRADE.build_judge_payload(case, result, trace, [], 1)
    assert "UNKNOWN_TIMING" not in json.dumps(payload)


def valid_check(checkpoint_id, verdict="pass"):
    return {"id": checkpoint_id, "verdict": verdict, "reason": "Toy reason", "missing_action": False,
            "prerequisite_untriggered": verdict == "not_applicable",
            "requires_durable_write": False,
            "evidence": [{"id": "t1.answer"}]}


@pytest.mark.parametrize("change", [
    {"id": "made-up"}, {"verdict": "not_observed"},
    {"evidence": [{"pointer": "/observations/turns/0/answer", "quote": "I corrected it."}]},
    {"evidence": [{"pointer": "/observations/turns/1/answer", "quote": "I corrected it."}]},
    {"evidence": [{"pointer": "/checkpoints/0/requirement", "quote": "Record"}]},
    {"evidence": [{"pointer": "/observations/turns/0/profile_after", "value": []}]},
    {"evidence": []}, {"verdict": "not_applicable", "prerequisite_untriggered": False},
    {"requires_durable_write": None},
    {"evidence": [{"id": "invented-record"}]},
    {"evidence": [{"id": "t2.answer"}]},
])
def test_invalid_evidence_or_ids_are_rejected(change):
    case, result, trace = fixture()
    checkpoint = GRADE.expand_checkpoints(case["private"]["checkpoints"], result["turns"])[0]
    payload = GRADE.build_judge_payload(case, result, trace, [checkpoint], 1)
    check = {**valid_check(checkpoint["id"]), **change}
    with pytest.raises(ValueError):
        GRADE.validate_verdict({"checks": [check]}, [checkpoint], payload)


def test_missing_action_and_empty_state_evidence_are_valid():
    case, result, trace = fixture()
    checkpoint = GRADE.expand_checkpoints(case["private"]["checkpoints"], result["turns"])[0]
    payload = GRADE.build_judge_payload(case, result, trace, [checkpoint], 1)
    missing = {**valid_check(checkpoint["id"], "fail"), "evidence": [], "missing_action": True}
    assert GRADE.validate_verdict({"checks": [missing]}, [checkpoint], payload) == [missing]
    empty_state = {**valid_check(checkpoint["id"]), "evidence": [{"id": "t1.event_memory_after"}]}
    validated = GRADE.validate_verdict({"checks": [empty_state]}, [checkpoint], payload)
    assert validated[0]["evidence"] == [{"id": "t1.event_memory_after",
                                        "pointer": "/observations/turns/0/event_memory_after", "value": []}]


def test_catalogue_covers_observed_items_and_ids_remain_stable_across_horizons():
    case, result, trace = fixture()
    first = GRADE.build_judge_payload(case, result, trace, [], 1)
    second = GRADE.build_judge_payload(case, result, trace, [], 2)
    ids_first = {entry["id"]: entry["pointer"] for entry in first["evidence_catalogue"]}
    ids_second = {entry["id"]: entry["pointer"] for entry in second["evidence_catalogue"]}
    assert {key: ids_second[key] for key in ids_first} == ids_first
    assert "t1.user" in ids_first
    assert "t1.profile_record_after" in ids_first
    assert "t1.event_memory_after" in ids_first
    assert "t2.answer" not in ids_first and "t2.answer" in ids_second
    for entry in first["evidence_catalogue"]:
        GRADE.resolve_pointer(first, entry["pointer"])
    trace_entries = [entry for entry in first["evidence_catalogue"] if entry["id"].startswith("trace")]
    assert len(trace_entries) == len(first["observations"]["trace"])
    assert all("checkpoints" not in entry["pointer"] for entry in first["evidence_catalogue"])


def test_selected_id_resolves_actual_value_without_model_written_quotes():
    case, result, trace = fixture()
    checkpoint = GRADE.expand_checkpoints(case["private"]["checkpoints"], result["turns"])[0]
    payload = GRADE.build_judge_payload(case, result, trace, [checkpoint], 1)
    check = valid_check(checkpoint["id"])
    check["evidence"] = [{"id": "t1.answer", "pointer": "/invented", "value": "fabricated", "quote": "fabricated"}]
    validated = GRADE.validate_verdict({"checks": [check]}, [checkpoint], payload)
    assert validated[0]["evidence"] == [{"id": "t1.answer", "pointer": "/observations/turns/0/answer",
                                        "value": result["turns"][0]["answer"]}]


class ToyGateway:
    def __init__(self, root, *, failure=False, verdicts=None):
        self.root, self.failure, self.verdicts = root, failure, verdicts
        self.calls = []

    async def chat(self, model, messages, trace, role, max_tokens):
        payload = json.loads(messages[1]["content"])
        self.calls.append((model, payload, role))
        trace.append({"event": "api_request", "role": role, "status": "toy"})
        if self.failure:
            raise RuntimeError("TOY_SERVICE_UNAVAILABLE")
        checks = []
        for checkpoint in payload["checkpoints"]:
            verdict = (self.verdicts or {}).get(model, {}).get(checkpoint["id"], "pass")
            checks.append(valid_check(checkpoint["id"], verdict))
        return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps({"checks": checks})}}]}


def test_two_independent_judges_per_horizon_and_no_final_only_correction(tmp_path):
    case, result, trace = fixture()
    gateway = ToyGateway(tmp_path, verdicts={judge: {"subject@1": "fail"} for judge in GRADE.DEFAULT_JUDGES})
    record = asyncio.run(GRADE.grade_run(gateway, case, result, trace))
    assert len(gateway.calls) == 4
    assert [(call[0], call[1]["horizon"]) for call in gateway.calls] == [
        (GRADE.DEFAULT_JUDGES[0], 1), (GRADE.DEFAULT_JUDGES[0], 2),
        (GRADE.DEFAULT_JUDGES[1], 1), (GRADE.DEFAULT_JUDGES[1], 2)]
    assert all(call[2] == "rubric_judge" for call in gateway.calls)
    assert all("judge_results" not in json.dumps(call[1]) for call in gateway.calls)
    assert record["critical_failed"] is True
    assert record["all_applicable_passed"] is False
    assert [c["verdict"] for c in record["checks"]] == ["fail", "pass", "pass", "not_executed"]
    assert len(list((tmp_path / "judge_traces").glob("*.json"))) == 4
    assert asyncio.run(GRADE.grade_run(gateway, case, result, trace)) == record
    assert len(gateway.calls) == 4


@pytest.mark.parametrize("verdicts,expected", [(("pass", "fail"), "uncertain"),
                                                    (("uncertain", "uncertain"), "uncertain"),
                                                    (("not_applicable", "not_applicable"), "not_applicable")])
def test_uncertainty_disagreement_and_all_na_do_not_pass(verdicts, expected):
    checkpoint = {"id": "toy@1", "critical": True, "executed": True}
    judges = [{"checks": [valid_check("toy@1", verdict)]} for verdict in verdicts]
    record = GRADE.aggregate_checks([checkpoint], judges)
    assert record["checks"][0]["verdict"] == expected
    assert record["all_applicable_passed"] is False
    assert record["disagreement_count"] == (verdicts[0] != verdicts[1])


def test_infrastructure_and_judge_errors_are_not_clinical_failure(tmp_path):
    case, result, trace = fixture()
    gateway = ToyGateway(tmp_path / "execution")
    result["status"] = "unsupported"
    record = asyncio.run(GRADE.grade_run(gateway, case, result, trace))
    assert record["status"] == "not_graded"
    assert not gateway.calls
    assert record["critical_failed"] is False
    assert all(check["verdict"] == "not_executed" for check in record["checks"])
    result["status"] = "completed"
    gateway = ToyGateway(tmp_path / "judge", failure=True)
    record = asyncio.run(GRADE.grade_run(gateway, case, result, trace))
    assert record["status"] == "judge_error"
    assert record["critical_failed"] is False
    assert record["all_applicable_passed"] is False
    assert record["judge_results"][0]["errors"][0]["message"] == "TOY_SERVICE_UNAVAILABLE"


def test_mutated_result_cannot_reuse_grade(tmp_path):
    case, result, trace = fixture()
    gateway = ToyGateway(tmp_path)
    asyncio.run(GRADE.grade_run(gateway, case, result, trace))
    result["turns"][0]["answer"] = "changed"
    with pytest.raises(ValueError, match="Cached grade inputs changed"):
        asyncio.run(GRADE.grade_run(gateway, case, result, trace))


def test_report_retains_missing_denominators_and_separates_repeats(tmp_path):
    case, result, trace = fixture()
    case["private"]["checkpoints"] = case["private"]["checkpoints"][:1]
    gateway = ToyGateway(tmp_path)
    asyncio.run(GRADE.grade_run(gateway, case, result, trace))
    GRADE.dump(tmp_path / "runs" / f"{result['run_id']}.json", result)
    plan = [{key: result[key] for key in ("run_id", "case_id", "model", "repetition", "domain", "difficulty")},
            {"run_id": "missing-primary", "case_id": "OTHER", "model": result["model"], "repetition": 1,
             "domain": "history", "difficulty": "boundary"},
            {"run_id": "missing-repeat", "case_id": "TOY", "model": result["model"], "repetition": 2,
             "domain": "history", "difficulty": "routine"}]
    report = GRADE.build_report(tmp_path, plan)
    primary = next(g for g in report["strata"]["primary"] if g["domain"] == g["difficulty"] == "all")
    repeat = next(g for g in report["strata"]["repeat"] if g["domain"] == g["difficulty"] == "all")
    assert primary["expected_runs"] == 2
    assert primary["completed_runs"] == 1
    assert primary["all_applicable_passed"] == {"numerator": 1, "denominator": 1, "expected_denominator": 2}
    assert repeat["expected_runs"] == 1
    assert repeat["completed_runs"] == 0
    assert report["repeat_reliability"][0]["all_repeats_observed"] is False
    assert report["clinical_certification"] is False
    assert "UNREVEALED_FACT" not in (tmp_path / "holdout_report.md").read_text(encoding="utf-8")


def test_protocol_observations_do_not_decide_semantic_truth():
    trace = [{"event": "api_request", "role": "supervisor", "payload": {"messages": [
        {"role": "tool", "tool_call_id": "unknown", "content": "success"}]}}]
    protocol = GRADE.protocol_observations(trace)
    assert protocol["semantic_verdict"] is None
    assert protocol["tool_pairing_failures"][0]["type"] == "orphan_tool_result"


def test_mem0_receipts_and_seed_baseline_are_visible_with_provenance():
    case, result, trace = fixture()
    result["seed_event_memory_after"] = {"self": {"backend": "live_mem0_platform", "memories": [
        {"memory": "OBSERVED_SEED_MEMORY", "metadata": {"source_role": "user"}}]}}
    trace.insert(0, {"event": "mem0_operation", "phase": "seed_fixture", "operation": "seed_add",
                     "parameters": {"messages": [{"role": "user", "content": "ACTUALLY_SEEDED"}]},
                     "result": {"event_id": "receipt", "status": "SUCCEEDED"}})
    trace.insert(5, {"event": "mem0_operation", "phase": "target", "turn": 1, "operation": "search",
                     "result": {"results": [{"memory": "OBSERVED_MEMORY", "metadata": {"source_role": "user"}}]}})
    payload = GRADE.build_judge_payload(case, result, trace, [], 1)
    serialized = json.dumps(payload)
    assert "OBSERVED_SEED_MEMORY" in serialized
    assert "ACTUALLY_SEEDED" in serialized
    assert "SUCCEEDED" in serialized
    assert "OBSERVED_MEMORY" in serialized
    assert "source_role" in serialized


def test_controlled_pool_cannot_prove_new_durable_write():
    case, result, trace = fixture()
    result["turns"][0]["event_memory_after"] = {"backend": "controlled_read_pool", "write_observation_available": False,
                                                "memories": [{"memory": "PRELOADED"}]}
    checkpoint = GRADE.expand_checkpoints(case["private"]["checkpoints"], result["turns"])[0]
    payload = GRADE.build_judge_payload(case, result, trace, [checkpoint], 1)
    check = {**valid_check(checkpoint["id"]), "requires_durable_write": True}
    with pytest.raises(ValueError, match="cannot prove durable writes"):
        GRADE.validate_verdict({"checks": [check]}, [checkpoint], payload)


def test_failed_attempts_plan_matrix_and_actual_ledger_cost_are_preserved(tmp_path):
    case, result, _ = fixture()
    plan = {"expected_runs": [{key: result[key] for key in ("run_id", "case_id", "model", "repetition", "domain", "difficulty")}],
            "models": [result["model"]], "case_ids": [result["case_id"]], "repetition": 1}
    GRADE.dump(tmp_path / "plans/toy.json", plan)
    result.update(status="error", attempt_id="failed-attempt", error_type="ToyConnectionFailure")
    GRADE.dump(tmp_path / "attempts/attempt.json", result)
    GRADE.dump(tmp_path / "errors/attempt.json", result)
    GRADE.dump(tmp_path / "process_errors/child.json", {"run_id": result["run_id"], "status": "error"})
    entries = [{"request_id": "r1", "role": "supervisor", "cost_usd": None, "status": "started"},
               {"request_id": "r1", "role": "supervisor", "cost_usd": 0.12, "cost_status": "provider_reported"},
               {"request_id": "r2", "role": "rubric_judge", "cost_usd": 0.03, "cost_status": "estimated"},
               {"request_id": "r3", "role": "supervisor", "cost_usd": None, "status": "error"}]
    (tmp_path / "ledgers").mkdir()
    (tmp_path / "ledgers/api-process1.jsonl").write_text("\n".join(json.dumps(entry) for entry in entries), encoding="utf-8")
    (tmp_path / "ledgers/api-process2.jsonl").write_text(json.dumps(entries[0]), encoding="utf-8")
    report = GRADE.build_report(tmp_path)
    group = next(g for g in report["strata"]["primary"] if g["domain"] == g["difficulty"] == "all")
    assert report["expected_matrix_available"] is True
    assert group["execution_status_counts"] == {"error": 1}
    assert group["expected_runs"] == group["executed_runs"] == 1
    assert group["all_applicable_passed"]["numerator"] == 0
    assert len(report["attempt_history"][result["run_id"]]["attempts"]) == 2
    assert report["attempt_history"][result["run_id"]]["attempt_count"] == 1
    assert report["ledger"]["unique_requests"] == 3
    assert report["ledger"]["cost_by_role_usd"] == {"supervisor": 0.12, "rubric_judge": 0.03}
    assert report["ledger"]["requests_without_known_cost"] == 1
    assert len(report["ledger"]["sources"]) == 2


def test_authored_history_is_distinct_from_generated_seed_output():
    case, result, trace = fixture()
    result["seed_turns"] = [{"turn": 1, "user": "Yes", "answer": "BOOTSTRAP_OUTPUT",
                             "authored_history_before": [{"role": "assistant", "content": "ORIGINAL_HISTORY_QUESTION"}]}]
    payload = GRADE.build_judge_payload(case, result, trace, [], 1)
    seed = payload["observations"]["seed_turns"][0]
    assert seed["answer"] == "BOOTSTRAP_OUTPUT"
    assert seed["authored_history_before"][0]["content"] == "ORIGINAL_HISTORY_QUESTION"


def test_recovered_canonical_success_does_not_hide_first_attempt_failure(tmp_path):
    _, result, _ = fixture()
    first = {**result, "status": "incomplete", "attempt_id": "first"}
    GRADE.dump(tmp_path / "attempts/first.json", first)
    GRADE.dump(tmp_path / "process_errors/first.json", {"run_id": result["run_id"], "status": "incomplete"})
    successful = {**result, "status": "completed", "attempt_id": "recovery"}
    GRADE.dump(tmp_path / "attempts/recovery.json", successful)
    GRADE.dump(tmp_path / "runs" / f"{result['run_id']}.json", successful)
    report = GRADE.build_report(tmp_path, [result])
    group = next(g for g in report["strata"]["primary"] if g["domain"] == g["difficulty"] == "all")
    assert group["completed_runs"] == 1
    assert group["first_attempt_completion"] == {"numerator": 0, "denominator": 1, "expected_denominator": 1}
    assert group["attempt_count"] == 2
    assert group["retried_runs"] == group["canonical_completed_after_recovery"] == 1


def test_trace_hash_is_checked_before_judge_calls(tmp_path):
    case, result, trace = fixture()
    result["trace_sha256"] = "wrong-hash"
    GRADE.dump(tmp_path / "traces" / f"{result['run_id']}.json", trace)
    gateway = ToyGateway(tmp_path)
    with pytest.raises(ValueError, match="Recorded trace hash changed"):
        asyncio.run(GRADE.grade_run(gateway, case, result, trace))
    assert gateway.calls == []
