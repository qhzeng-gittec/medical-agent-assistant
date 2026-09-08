"""Synthetic transport/state fixtures only; no sealed data or frozen imports."""

import copy
import hashlib
import json

import pytest

import holdout_telemetry as telemetry


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def api(phase="target", answer="PRIVATE_ANSWER_SENTINEL", usage=None, **kwargs):
    event = {"event": "api_request", "phase": phase, "turn": 1, "role": "supervisor",
             "endpoint": "chat/completions", "status": "ok", "cost_status": "provider_reported",
             "cost_usd": .02, "elapsed_seconds": 1.0,
             "payload": {"model": "model-a", "messages": [{"role": "user", "content": "PRIVATE_USER_SENTINEL"}]},
             "response": {"usage": usage if usage is not None else {"prompt_tokens": 10, "completion_tokens": 5},
                          "choices": [{"finish_reason": "stop", "message": {"content": answer}}]}}
    event.update(kwargs)
    return event


def run_record(run_id="TOY-1", attempt_id="attempt-1", status="completed"):
    return {"run_id": run_id, "attempt_id": attempt_id, "case_id": "CASE-1", "model": "model-a",
            "repetition": 1, "status": status, "seed_turns": [],
            "state_path": "state/" + attempt_id,
            "turns": [{"turn": 1, "user": "PRIVATE_USER_SENTINEL", "answer": "PRIVATE_ANSWER_SENTINEL",
                       "user_id": "user-1", "session_id": "session-1", "latency_seconds": 2,
                       "system_result": {"call_trace": []}}]}


def operation(op="add", rpc="completed", receipt="PENDING", **extra):
    return {"event": "mem0_operation", "operation": op, "parameters": {"messages": "PRIVATE_MEMORY_SENTINEL"},
            "status": rpc, "backend": "live_mem0_platform", "result": {"status": receipt}, "seconds": .25, **extra}


def install_run(root, record, trace, failed=False):
    record = copy.deepcopy(record)
    trace_relative = f"traces/attempts/{record['run_id']}__{record['attempt_id']}.json" if failed else f"traces/{record['run_id']}.json"
    trace_path = root / trace_relative
    save(trace_path, trace)
    record["trace_sha256"] = hashlib.sha256(trace_path.read_bytes()).hexdigest()
    record["attempt_trace_path"] = trace_relative
    name = f"{record['run_id']}__{record['attempt_id']}.json" if failed else f"{record['run_id']}.json"
    save(root / ("errors" if failed else "runs") / name, record)
    return record


def test_unknown_measurements_are_not_zero_or_estimated():
    assert telemetry.measured([0])["total"] == 0
    mixed = telemetry.measured([0, None, 3])
    assert mixed["total"] is None and mixed["known_subtotal"] == 3
    assert mixed["unknown_observations"] == 1
    assert telemetry.measured([])["total"] is None
    assert telemetry.measured([float("nan"), True])["unknown_observations"] == 2


def test_latency_quantiles_use_observed_samples_and_keep_missing_count():
    result = telemetry.distribution([1, 3, None])
    assert result["p50"] == 2 and result["p95"] == 2.9
    assert result["observed"] == 2 and result["unknown"] == 1
    assert telemetry.distribution([None])["p50"] is None


def test_metadata_limit_is_not_promoted_to_unobserved_http400():
    event = {"error_type": "ValidationError", "error": "Metadata size (4704 chars) exceeds the limit of 2000 chars."}
    result = telemetry.metadata_diagnostic(event)
    assert result["metadata_limit_observed"] and result["metadata_chars"] == 4704
    assert not result["http400_observed"]
    event["http_status"] = 400
    assert telemetry.metadata_diagnostic(event)["http400_observed"]
    assert telemetry.metadata_diagnostic({"error": "Client error '400 Bad Request'"})["http400_observed"]
    assert not telemetry.metadata_diagnostic({"error": "Metadata size (400 chars) exceeds the limit of 200 chars."})["http400_observed"]


def test_transport_contract_checks_empty_truncation_and_pairing_without_text_output():
    empty = telemetry.protocol_anomalies(api(answer="  "))
    assert empty["empty_final_response"] == 1
    event = api()
    event["response"]["choices"][0]["finish_reason"] = "length"
    event["payload"]["messages"] += [{"role": "tool", "tool_call_id": "unpaired", "content": "PRIVATE_TOOL_SENTINEL"}]
    result = telemetry.protocol_anomalies(event)
    assert result["nonterminal_finish_reason"] == 1 and result["orphan_tool_result"] == 1
    assert "PRIVATE" not in json.dumps(result)
    event["response"]["choices"][0]["finish_reason"] = "tool_calls"
    assert telemetry.protocol_anomalies(event)["tool_calls_finish_without_calls"] == 1


def test_request_failure_has_unobserved_response_not_empty_model_answer():
    result = telemetry.protocol_anomalies(api(status="error", response=None))
    assert result["response_choices_unobserved"] == 1
    assert result["empty_final_response"] == 0


def test_mem0_state_and_trace_are_one_observation_and_wait_is_separate():
    op = operation(phase="target", turn=1)
    state = {key: value for key, value in op.items() if key not in {"phase", "turn"}}
    merged, coverage = telemetry.merge_memory_observations([op], [state])
    assert len(merged) == 1 and coverage["state_mem0_matched_to_trace"] == 1
    population = telemetry.Population()
    population.analyze(run_record(), [op, operation("event", receipt="SUCCEEDED", phase="target", turn=1),
                                      {"event": "memory_settle", "status": "settled", "seconds": 2, "phase": "target"}], [state])
    bucket = population.export()["by_model_stage_phase"][0]["mem0"]["counts"]
    assert bucket["rpc/add/completed"] == 1
    assert bucket["write_receipt/PENDING"] == 1 and bucket["wait_event/SUCCEEDED"] == 1
    assert bucket["settle/settled"] == 1


def test_final_state_can_complete_started_trace_without_counting_twice():
    old = operation(rpc="started", receipt=None, phase="target", turn=1)
    new = operation(rpc="error", error_type="ValidationError", error="provider-only diagnostic")
    merged, coverage = telemetry.merge_memory_observations([old], [new])
    assert len(merged) == 1 and merged[0]["status"] == "error" and merged[0]["phase"] == "target"
    assert coverage["state_completed_started_trace_observation"] == 1


def test_conflicting_final_state_is_unknown_not_duplicate_or_success():
    merged, coverage = telemetry.merge_memory_observations([operation()], [operation(rpc="error")])
    assert len(merged) == 1 and merged[0]["status"] == "unknown"
    assert coverage["state_trace_final_snapshot_conflicts"] == 1


def test_exact_query_repeat_is_literal_and_scoped_to_same_user_session():
    record = run_record()
    record["turns"].append({**record["turns"][0], "turn": 2, "user_id": "user-2", "session_id": "session-2"})
    base = {"event": "retrieval", "phase": "target", "turn": 1, "status": "live_rag",
            "query": "PRIVATE_QUERY_SENTINEL", "documents": [], "top_k": 3, "filter_type": None}
    population = telemetry.Population()
    population.analyze(record, [base, copy.deepcopy(base), {**base, "query": "PRIVATE_QUERY_SENTINEL "},
                                {**base, "turn": 2}, {**base, "top_k": 5}], [])
    group = population.export()["by_model_stage_phase"][0]
    assert group["counts"]["exact_query_repeat_lower_bound/rag"] == 1
    assert group["rag"]["calls/live_rag"] == 5
    assert "PRIVATE_QUERY" not in json.dumps(group)


def test_tool_return_without_status_is_not_automatically_success():
    assert telemetry.tool_outcome({"answer": "PRIVATE"}) == "returned_without_explicit_status"
    assert telemetry.tool_outcome({"success": True}) == "explicit_success"
    assert telemetry.tool_outcome({"success": False}) == "failure"
    assert telemetry.tool_outcome(None) == "unknown"


def test_permission_denial_and_upstream_unadvertised_request_are_distinct():
    record = run_record()
    record["turns"][0]["system_result"]["call_trace"] = [{"tool_name": "call_research_agent", "success": False,
                                                         "result": {"error_type": "PolicyDenied"}}]
    event = api()
    event["response"]["choices"][0] = {"finish_reason": "tool_calls", "message": {"tool_calls": [
        {"id": "call", "function": {"name": "private_unadvertised_name", "arguments": "{}"}}]}}
    population = telemetry.Population()
    population.analyze(record, [event], [])
    group = population.export()["by_model_stage_phase"][0]
    assert group["counts"]["requested_unadvertised_tools"] == 1
    assert group["counts"]["tool_permission_denials_observed"] == 1
    assert "private_unadvertised_name" not in json.dumps(group)


def test_complete_and_failed_populations_unknown_cost_and_no_text_leak(tmp_path):
    plan = {"expected_runs": [{"run_id": "TOY-1", "case_id": "CASE-1", "model": "model-a", "repetition": 1},
                              {"run_id": "TOY-2", "case_id": "CASE-2", "model": "model-b", "repetition": 2}]}
    save(tmp_path / "execution_plan.json", plan)
    good = install_run(tmp_path, run_record(), [api()])
    failed = run_record(attempt_id="attempt-failed", status="error")
    failed.update(error_type="ValidationError", error="PRIVATE_FAILURE_SENTINEL", turns=[])
    event = api(phase="seed", status="error", error_type="HTTPStatusError", cost_status="http_error_cost_not_reported", cost_usd=0,
                response=None)
    install_run(tmp_path, failed, [event], failed=True)
    save(tmp_path / "sealed/do_not_read.json", {"private": "SEALED_SENTINEL"})
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*.json")}
    result = telemetry.analyze_root(tmp_path)
    assert result["coverage"]["canonical_completed_records"] == 1
    assert result["coverage"]["without_canonical_completed_record"] == 1
    assert len(result["completed"]["records"]) == len(result["failed_attempts"]["records"]) == 1
    failure = result["failed_attempts"]["records"][0]
    assert failure["visible_turns"] == 0 and failure["api_cost_usd"]["total"] is None
    assert failure["last_observed_phase"] == "seed"
    assert failure["api_cost_usd"]["unknown_observations"] == 1
    assert result["completed"]["records"][0]["api_cost_usd"]["total"] == .02
    assert all(s not in json.dumps(result) for s in ("PRIVATE_", "SEALED_SENTINEL"))
    assert before == {str(p): p.read_bytes() for p in tmp_path.rglob("*.json")}


def test_missing_and_modified_trace_remain_unknown(tmp_path):
    record = run_record()
    save(tmp_path / "execution_plan.json", {"expected_runs": [record]})
    installed = install_run(tmp_path, record, [api()])
    save(tmp_path / "traces/TOY-1.json", [api(cost_usd=.99)])
    result = telemetry.analyze_root(tmp_path)
    assert result["completed"]["coverage"]["trace_unknown"] == 1
    assert result["completed"]["records"][0]["api_cost_usd"]["total"] is None
    assert any(issue["status"] == "trace_hash_mismatch" for issue in result["observation_issues"])


def test_embeddings_have_no_output_tokens_and_selector_stays_separate():
    record = run_record()
    population = telemetry.Population()
    embedding = api(endpoint="embeddings", role="embedding", response={"usage": {"prompt_tokens": 7}})
    selector = api(phase="patient_selector", role="patient_selector", usage={"prompt_tokens": 3, "completion_tokens": 1})
    population.analyze(record, [embedding, selector], [])
    groups = {row["phase"]: row for row in population.export()["by_model_stage_phase"]}
    assert groups["target"]["api"]["output_tokens"]["not_applicable_observations"] == 1
    assert groups["target"]["api"]["output_tokens"]["unknown_observations"] == 0
    assert groups["patient_selector"]["api"]["input_tokens"]["total"] == 3
