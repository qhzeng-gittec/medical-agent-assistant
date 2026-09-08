"""Read-only operational telemetry for recorded holdout executions; no model calls.

This intentionally imports no frozen harness and reads no sealed case or rubric.
Completed runs and finalized failed attempts are separate populations. Only
aggregate measurements, categorical codes, and execution identifiers are emitted.
"""

import argparse
import hashlib
import json
import math
import re
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_ROOT = Path(__file__).parent / "results" / "holdout_v1_live_20260908"
PHASES = {"seed", "seed_fixture", "target", "patient_selector"}
KNOWN_COST = {"provider_reported", "estimated_standard_list_price_not_invoice",
              "estimated_from_usage_and_catalog"}
METADATA_LIMIT = re.compile(r"Metadata size \((\d+) chars\) exceeds the limit of (\d+) chars", re.I)
HTTP400 = re.compile(r"\b400\s+(?:Bad Request|Client Error)\b|\bHTTP(?:/[\d.]+)?\s*400\b", re.I)
TOOL_NAMES = {"call_diagnostic_agent", "call_consultation_agent", "call_research_agent",
              "update_patient_profile", "search_patient_history", "analyze_symptoms", "assess_risk",
              "disease_code", "recommend_lifestyle", "clinical_guideline", "deep_research",
              "search_knowledge", "search_history", "search_similar_cases"}


def numeric(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def measured(values, not_applicable=0):
    known = [value for value in values if numeric(value)]
    unknown = len(values) - len(known)
    return {"known_subtotal": round(sum(known), 8) if known else None,
            "total": round(sum(known), 8) if known and not unknown else None,
            "known_observations": len(known), "unknown_observations": unknown,
            "not_applicable_observations": not_applicable}


def distribution(values):
    known = sorted(value for value in values if numeric(value))

    def percentile(fraction):
        if not known:
            return None
        position = (len(known) - 1) * fraction
        lower, upper = math.floor(position), math.ceil(position)
        return round(known[lower] + (known[upper] - known[lower]) * (position - lower), 6)

    return {"observed": len(known), "unknown": len(values) - len(known),
            "min": min(known) if known else None, "max": max(known) if known else None,
            "mean": round(sum(known) / len(known), 6) if known else None,
            "p50": percentile(.5), "p90": percentile(.9), "p95": percentile(.95), "p99": percentile(.99)}


def code(value):
    return value if isinstance(value, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]{0,100}", value) else "unknown"


def status(value, allowed):
    return value if isinstance(value, str) and value in allowed else "unknown"


def stage(record):
    return "primary" if record.get("repetition") == 1 else "repeat" if record.get("repetition") in (2, 3) else "unknown"


def phase(event):
    return event.get("phase") if event.get("phase") in PHASES else "unknown"


def metadata_diagnostic(event):
    error = event.get("error")
    error = error if isinstance(error, str) else ""
    match = METADATA_LIMIT.search(error)
    confirmed400 = event.get("http_status") == 400 or event.get("status_code") == 400 or bool(HTTP400.search(error))
    return {"metadata_limit_observed": bool(match), "http400_observed": confirmed400,
            "metadata_chars": int(match[1]) if match else None,
            "limit_chars": int(match[2]) if match else None}


def protocol_anomalies(event):
    """Check transport shape and tool pairing, never clinical wording."""
    issues = Counter()
    if event.get("endpoint") != "chat/completions":
        return issues
    response = event.get("response")
    choices = response.get("choices") if isinstance(response, dict) else None
    if not isinstance(choices, list) or not choices:
        issues["response_choices_unobserved"] += 1
    else:
        for choice in choices:
            if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
                issues["malformed_response_choice"] += 1
                continue
            message = choice["message"]
            finish = choice.get("finish_reason") if isinstance(choice.get("finish_reason"), str) else None
            calls = message.get("tool_calls")
            if finish == "tool_calls" and not calls:
                issues["tool_calls_finish_without_calls"] += 1
            elif finish == "stop" and not calls:
                content = message.get("content")
                if content is None or content == "" or (isinstance(content, str) and not content.strip()):
                    issues["empty_final_response"] += 1
                elif not isinstance(content, str):
                    issues["unknown_content_shape"] += 1
            elif finish not in {"stop", "tool_calls"}:
                issues["finish_reason_missing" if finish is None else "nonterminal_finish_reason"] += 1
            if calls is not None and not isinstance(calls, list):
                issues["malformed_tool_calls"] += 1
    payload = event.get("payload")
    messages = payload.get("messages") if isinstance(payload, dict) else None
    if not isinstance(messages, list):
        issues["request_messages_unobserved"] += 1
        return issues
    pending = set()
    for message in messages:
        if not isinstance(message, dict):
            issues["malformed_context_message"] += 1
            continue
        role = message.get("role")
        if role == "assistant":
            if pending:
                issues["unanswered_calls_before_next_assistant"] += 1
            calls = message.get("tool_calls", [])
            if not isinstance(calls, list):
                issues["malformed_context_tool_calls"] += 1
                continue
            for call in calls:
                call_id = call.get("id") if isinstance(call, dict) else None
                if not isinstance(call_id, str) or not call_id:
                    issues["tool_call_id_missing"] += 1
                elif call_id in pending:
                    issues["duplicate_pending_tool_id"] += 1
                else:
                    pending.add(call_id)
        elif role == "tool":
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or call_id not in pending:
                issues["orphan_tool_result"] += 1
            else:
                pending.remove(call_id)
        elif pending:
            issues["interrupted_tool_pair"] += 1
    if pending:
        issues["pending_tools_at_inference"] += 1
    return issues


def tool_outcome(result):
    if not isinstance(result, dict):
        return "unknown"
    if result.get("success") is False or result.get("status") in {"error", "failed"} or result.get("error") or result.get("error_type"):
        return "failure"
    if result.get("success") is True or result.get("status") in {"success", "completed", "ok"}:
        return "explicit_success"
    return "returned_without_explicit_status"


def event_fingerprint(event):
    value = {key: value for key, value in event.items() if key not in {"phase", "turn", "seed_turn"}}
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def merge_memory_observations(trace, state_events):
    """A persisted SDK observation is not an additional remote operation."""
    merged, counts = list(trace), Counter()
    unmatched = {i for i, event in enumerate(trace) if event.get("event") == "mem0_operation"}
    for state_event in state_events:
        exact = next((i for i in sorted(unmatched) if event_fingerprint(trace[i]) == event_fingerprint(state_event)), None)
        if exact is not None:
            unmatched.remove(exact)
            counts["state_mem0_matched_to_trace"] += 1
            continue
        same_call = next((i for i in sorted(unmatched)
                          if trace[i].get("operation") == state_event.get("operation")
                          and trace[i].get("parameters") == state_event.get("parameters")), None)
        if same_call is not None:
            unmatched.remove(same_call)
            old = trace[same_call]
            replacement = dict(state_event)
            replacement.update({key: old[key] for key in ("phase", "turn", "seed_turn") if key in old})
            if old.get("status") == "started" and state_event.get("status") in {"completed", "error"}:
                counts["state_completed_started_trace_observation"] += 1
            else:
                replacement.update(status="unknown", result=None, seconds=None)
                counts["state_trace_final_snapshot_conflicts"] += 1
            merged[same_call] = replacement
        else:
            merged.append(state_event)
            counts["state_only_mem0_observations"] += 1
    counts["trace_mem0_without_matching_state_record"] = len(unmatched)
    return merged, counts


class Bucket:
    def __init__(self):
        self.counts = Counter()
        self.api_status, self.http_status, self.cost_status = Counter(), Counter(), Counter()
        self.api_error_types = Counter()
        self.api_models, self.api_roles = Counter(), Counter()
        self.tokens_in, self.tokens_out, self.costs, self.latencies = [], [], [], []
        self.api_latencies, self.mem0_seconds, self.settle_seconds = [], [], []
        self.protocol, self.tools, self.mem0, self.rag = Counter(), Counter(), Counter(), Counter()
        self.metadata_chars = []
        self.run_ids, self.anomaly_ids = set(), set()

    def export(self):
        return {"run_ids": sorted(self.run_ids), "counts": dict(self.counts),
                "api": {"status": dict(self.api_status), "http_status": dict(self.http_status),
                        "error_types": dict(self.api_error_types),
                        "actual_models": dict(self.api_models), "roles": dict(self.api_roles),
                        "input_tokens": measured(self.tokens_in),
                        "output_tokens": measured(self.tokens_out, self.counts["embedding_requests"]),
                        "cost_usd": measured(self.costs), "cost_status": dict(self.cost_status),
                        "latency_seconds": distribution(self.api_latencies)},
                "visible_answer_latency_seconds": distribution(self.latencies),
                "protocol_shape_observations": dict(self.protocol), "tool_outcomes": dict(self.tools),
                "mem0": {"counts": dict(self.mem0), "operation_seconds": distribution(self.mem0_seconds),
                         "settle_seconds": distribution(self.settle_seconds),
                         "metadata_size_chars": distribution(self.metadata_chars), "cost_usd": None},
                "rag": dict(self.rag), "anomaly_run_ids": sorted(self.anomaly_ids)}


class Population:
    def __init__(self):
        self.groups = defaultdict(Bucket)
        self.records = []
        self.error_types, self.error_phases = Counter(), Counter()
        self.coverage = Counter()

    def bucket(self, record, event_phase):
        key = (record["model"], stage(record), event_phase)
        result = self.groups[key]
        result.run_ids.add(record["run_id"])
        return result

    def analyze(self, record, trace, state_events):
        self.coverage["finalized_records"] += 1
        trace_known = isinstance(trace, list)
        self.coverage["trace_observed" if trace_known else "trace_unknown"] += 1
        trace = trace if trace_known else []
        seen_queries = set()
        api_events = [event for event in trace if event.get("event") == "api_request"]
        costs = [event.get("cost_usd") if event.get("cost_status") in KNOWN_COST else None for event in api_events]
        last_phase = phase(trace[-1]) if trace else "unknown"
        last_status = status(record.get("status"), {"completed", "error", "incomplete"})
        self.error_types[code(record.get("error_type"))] += int(last_status != "completed")
        if last_status != "completed":
            self.error_phases[last_phase] += 1
        visible_turns = record.get("turns")
        self.records.append({"run_id": record["run_id"], "attempt_id": record.get("attempt_id"),
                             "model": record["model"], "stage": stage(record), "status": last_status,
                             "error_type": code(record.get("error_type")) if last_status != "completed" else None,
                             "last_observed_phase": last_phase,
                             "visible_turns": len(visible_turns) if isinstance(visible_turns, list) else None,
                             "api_cost_usd": measured(costs), "trace_observed": trace_known})
        for field, event_phase in (("seed_turns", "seed"), ("turns", "target")):
            turns = record.get(field)
            if not isinstance(turns, list):
                self.bucket(record, event_phase).counts["turn_list_unknown"] += 1
                continue
            for turn in turns:
                bucket = self.bucket(record, event_phase)
                bucket.counts["visible_turn_records"] += 1
                answer = turn.get("answer")
                bucket.counts["answer_empty" if isinstance(answer, str) and not answer.strip()
                              else "answer_nonempty" if isinstance(answer, str) else "answer_unknown"] += 1
                bucket.latencies.append(turn.get("latency_seconds"))
                system = turn.get("system_result")
                calls = system.get("call_trace") if isinstance(system, dict) else None
                if not isinstance(calls, list):
                    bucket.counts["supervisor_tool_outcomes_unknown"] += 1
                else:
                    for call in calls:
                        outcome = "explicit_success" if call.get("success") is True else "failure" if call.get("success") is False else "unknown"
                        name = call.get("tool_name") if call.get("tool_name") in TOOL_NAMES else "unknown_tool"
                        bucket.tools[f"supervisor/{name}/{outcome}"] += 1
                        result = call.get("result")
                        if isinstance(result, dict) and result.get("error_type") == "PolicyDenied":
                            bucket.counts["tool_permission_denials_observed"] += 1
                        if isinstance(result, dict) and result.get("error_type") == "DuplicateCall":
                            bucket.counts["duplicate_tool_calls_denied"] += 1
        scopes = {(phase(event), event.get("seed_turn") if phase(event) == "seed" else event.get("turn")):
                  [event.get("user_id"), event.get("session_id")]
                  for event in trace if event.get("event") in {"turn_start", "seed_turn_start"}}
        for field, event_phase in (("turns", "target"), ("seed_turns", "seed")):
            stored_turns = record.get(field)
            for turn in stored_turns if isinstance(stored_turns, list) else []:
                scopes.setdefault((event_phase, turn.get("turn")), [turn.get("user_id"), turn.get("session_id")])
        observations, memory_coverage = merge_memory_observations(trace, state_events)
        self.coverage.update(memory_coverage)
        for event in observations:
            event_phase = phase(event)
            scope = scopes.get((event_phase, event.get("seed_turn") if event_phase == "seed" else event.get("turn")))
            scope = scope if scope and all(isinstance(item, str) and item for item in scope) else None
            bucket = self.bucket(record, event_phase)
            event_name = event.get("event")
            if event_name == "api_request":
                self.api(bucket, event, record, seen_queries, scope)
            elif event_name == "tool_observation":
                name = event.get("tool_name") if event.get("tool_name") in TOOL_NAMES else "unknown_tool"
                bucket.tools[f"worker_skill/{name}/{tool_outcome(event.get('result'))}"] += 1
            elif event_name == "retrieval":
                mode = status(event.get("status"), {"live_rag", "controlled", "retrieval_timeout", "empty_retrieval", "not_needed"})
                bucket.rag[f"calls/{mode}"] += 1
                docs = event.get("documents")
                if mode == "retrieval_timeout":
                    bucket.rag["document_count_not_applicable_injected_timeout"] += 1
                elif isinstance(docs, list):
                    bucket.rag["documents_returned_observed"] += len(docs)
                    bucket.rag["document_count_observed"] += 1
                else:
                    bucket.rag["document_count_unknown"] += 1
                if scope:
                    self.query(bucket, seen_queries, record, event_phase, "rag", event.get("query"),
                               [scope, mode, event.get("filter_type"), event.get("top_k")])
                else:
                    bucket.counts["exact_query_scope_unknown/rag"] += 1
            elif event_name == "mem0_operation":
                self.memory(bucket, event, record, seen_queries)
            elif event_name == "memory_settle":
                bucket.mem0["settle/" + status(event.get("status"), {"settled", "error"})] += 1
                bucket.settle_seconds.append(event.get("seconds"))
            elif event_name == "memory_search":
                bucket.mem0["controlled_search_observed"] += 1
            elif event_name == "memory_write_suppressed":
                bucket.mem0["writes_suppressed/" + status(event.get("backend"), {"live_mem0_platform", "controlled_read_pool", "disabled"})] += 1

    @staticmethod
    def query(bucket, seen, record, event_phase, source, query, scope):
        if not isinstance(query, str):
            bucket.counts[f"exact_query_unknown/{source}"] += 1
            return
        # Full literal query + same scope/options only. Text and hashes never leave the analyzer.
        key = (record["run_id"], record.get("attempt_id"), event_phase, source, query,
               json.dumps(scope, sort_keys=True, ensure_ascii=False))
        bucket.counts[f"exact_query_observed/{source}"] += 1
        if key in seen:
            bucket.counts[f"exact_query_repeat_lower_bound/{source}"] += 1
        seen.add(key)

    def api(self, bucket, event, record, seen, scope):
        bucket.counts["api_requests"] += 1
        bucket.api_status[status(event.get("status"), {"ok", "error", "started"})] += 1
        if event.get("status") == "error":
            bucket.api_error_types[code(event.get("error_type"))] += 1
        bucket.http_status[str(event["http_status"]) if type(event.get("http_status")) is int else "unknown"] += 1
        bucket.api_latencies.append(event.get("elapsed_seconds"))
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        actual_model = payload.get("model")
        bucket.api_models[actual_model if isinstance(actual_model, str) and re.fullmatch(r"[A-Za-z0-9_./-]{1,100}", actual_model) else "unknown"] += 1
        bucket.api_roles[code(event.get("role"))] += 1
        response = event.get("response") if isinstance(event.get("response"), dict) else {}
        usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
        bucket.tokens_in.append(usage.get("prompt_tokens"))
        if event.get("endpoint") == "embeddings":
            bucket.counts["embedding_requests"] += 1
        else:
            bucket.tokens_out.append(usage.get("completion_tokens"))
        bucket.cost_status[code(event.get("cost_status"))] += 1
        bucket.costs.append(event.get("cost_usd") if event.get("cost_status") in KNOWN_COST else None)
        issues = protocol_anomalies(event)
        bucket.protocol.update(issues)
        if issues:
            bucket.counts["requests_with_protocol_shape_observations"] += 1
            bucket.anomaly_ids.add(record["run_id"])
        advertised = {t.get("function", {}).get("name") for t in payload.get("tools", []) if isinstance(t, dict)}
        choices = response.get("choices")
        for choice in choices if isinstance(choices, list) else []:
            message = choice.get("message") if isinstance(choice, dict) else None
            calls = message.get("tool_calls", []) if isinstance(message, dict) else []
            for call in calls if isinstance(calls, list) else []:
                function = call.get("function") if isinstance(call, dict) else None
                if not isinstance(function, dict):
                    bucket.counts["requested_tool_shape_unknown"] += 1
                    continue
                bucket.counts["requested_tool_calls"] += 1
                if function.get("name") not in advertised:
                    bucket.counts["requested_unadvertised_tools"] += 1
                try:
                    args = json.loads(function["arguments"])
                except (KeyError, TypeError, json.JSONDecodeError):
                    bucket.counts["requested_tool_arguments_invalid"] += 1
                    continue
                if isinstance(args, dict) and "query" in args:
                    if scope:
                        self.query(bucket, seen, record, phase(event), "requested_tool", args["query"],
                                   [scope, event.get("role"), function.get("name"), {k: v for k, v in args.items() if k != "query"}])
                    else:
                        bucket.counts["exact_query_scope_unknown/requested_tool"] += 1

    def memory(self, bucket, event, record, seen):
        operation = status(event.get("operation"), {"add", "seed_add", "search", "event", "list"})
        rpc = status(event.get("status"), {"started", "completed", "error"})
        bucket.mem0[f"rpc/{operation}/{rpc}"] += 1
        bucket.mem0_seconds.append(event.get("seconds"))
        result = event.get("result") if isinstance(event.get("result"), dict) else {}
        if operation in {"add", "seed_add", "event"}:
            receipt = status(result.get("status"), {"SUCCEEDED", "FAILED", "PENDING", "RUNNING"})
            bucket.mem0[f"{'wait_event' if operation == 'event' else 'write_receipt'}/{receipt}"] += 1
        diagnostic = metadata_diagnostic(event)
        if diagnostic["metadata_limit_observed"]:
            bucket.mem0["metadata_limit_validation_observed"] += 1
            bucket.metadata_chars.append(diagnostic["metadata_chars"])
            bucket.mem0["metadata_limit_http400_confirmed" if diagnostic["http400_observed"]
                        else "metadata_limit_http_status_unknown"] += 1
        elif diagnostic["http400_observed"]:
            bucket.mem0["other_http400_observed"] += 1
        if operation == "search":
            params = event.get("parameters") if isinstance(event.get("parameters"), dict) else {}
            self.query(bucket, seen, record, phase(event), "mem0_search", params.get("query"),
                       {key: value for key, value in params.items() if key != "query"})

    def export(self):
        return {"coverage": dict(self.coverage), "error_types": dict(+self.error_types),
                "error_phases": dict(self.error_phases), "records": self.records,
                "by_model_stage_phase": [{"model": model, "stage": run_stage, "phase": event_phase, **bucket.export()}
                                          for (model, run_stage, event_phase), bucket in sorted(self.groups.items())]}


def analyze_root(root):
    root = Path(root).resolve()
    issues = []

    def read(path):
        try:
            return json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            issues.append({"path": str(path.relative_to(root)), "status": "unreadable_or_incomplete_snapshot"})
            return None

    def contained(relative):
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            issues.append({"status": "rejected_path_outside_run_root"})
            return None
        return path

    plan = read(root / "execution_plan.json")
    if not isinstance(plan, dict) or not isinstance(plan.get("expected_runs"), list):
        raise ValueError("A readable execution_plan.json with expected_runs is required")
    expected = {row["run_id"]: row for row in plan["expected_runs"]}
    completed, failed = Population(), Population()
    completed_ids = set()
    for directory, population in (("runs", completed), ("errors", failed)):
        for path in sorted((root / directory).glob("*.json")):
            record = read(path)
            if not isinstance(record, dict) or record.get("run_id") not in expected:
                issues.append({"path": str(path.relative_to(root)), "status": "record_not_in_execution_plan"})
                continue
            expected_record = expected[record["run_id"]]
            if any(record.get(k) != expected_record.get(k) for k in ("model", "repetition", "case_id")):
                issues.append({"run_id": record["run_id"], "status": "planned_metadata_mismatch"})
                continue
            if (directory == "runs" and record.get("status") != "completed") or (directory == "errors" and record.get("status") not in {"error", "incomplete"}):
                issues.append({"run_id": record["run_id"], "status": "unexpected_finalized_record_status"})
                continue
            if directory == "runs":
                completed_ids.add(record["run_id"])
            trace_relative = f"traces/{record['run_id']}.json" if directory == "runs" else record.get("attempt_trace_path")
            trace_path = contained(trace_relative) if isinstance(trace_relative, str) else None
            trace = read(trace_path) if trace_path else None
            if trace_path and trace is not None:
                if not isinstance(trace, list):
                    issues.append({"run_id": record["run_id"], "status": "trace_not_an_event_list"})
                    trace = None
                elif record.get("trace_sha256"):
                    if hashlib.sha256(trace_path.read_bytes()).hexdigest() != record["trace_sha256"]:
                        issues.append({"run_id": record["run_id"], "status": "trace_hash_mismatch"})
                        trace = None
                else:
                    issues.append({"run_id": record["run_id"], "status": "trace_integrity_unknown"})
            state_events = []
            state_path = contained(record["state_path"]) if isinstance(record.get("state_path"), str) else None
            if state_path and (state_path / "mem0_operations").exists():
                for operation_path in sorted((state_path / "mem0_operations").glob("*.json")):
                    event = read(operation_path)
                    if isinstance(event, dict) and event.get("event") == "mem0_operation":
                        state_events.append(event)
            population.analyze(record, trace, state_events)
    planned_groups = defaultdict(list)
    for row in expected.values():
        planned_groups[(row["model"], stage(row))].append(row["run_id"])
    by_model_stage = [{"model": model, "stage": run_stage, "expected": len(ids),
                       "canonical_completed": len(set(ids) & completed_ids),
                       "without_canonical_completed": len(set(ids) - completed_ids)}
                      for (model, run_stage), ids in sorted(planned_groups.items())]
    return {"schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
            "execution_plan_sha256": hashlib.sha256((root / "execution_plan.json").read_bytes()).hexdigest(),
            "coverage": {"expected_runs": len(expected), "canonical_completed_records": len(completed_ids),
                         "without_canonical_completed_record": len(set(expected) - completed_ids),
                         "without_canonical_completed_record_ids": sorted(set(expected) - completed_ids),
                         "by_model_stage": by_model_stage},
            "completed": completed.export(), "failed_attempts": failed.export(), "observation_issues": issues,
            "limitations": [
                "Completed runs and finalized failed attempts are disjoint populations; the same planned run can occur in both after retry.",
                "Missing or in-flight runs, process-only crashes without a finalized error record, and absent fields remain unobserved.",
                "Numeric totals are null if any observation is unknown; known subtotals do not estimate missing token usage or charges. Mem0 monetary cost is not exposed.",
                "Recorded answer latency is elapsed nonstreaming product.process time, including initial memory submission; later settlement is separate. First visible safety-warning time is not instrumented.",
                "Tool explicit_success is an execution flag, not medical correctness; returned_without_explicit_status is not promoted to success.",
                "Protocol counts are affected API requests and observed shape occurrences; the same malformed historical tool pair can recur in several requests, so these are not independent incident counts.",
                "RAG counts are recorded backend events, not recall/coverage/citation correctness. A failure before a retrieval event can be absent from these counts.",
                "Exact-query repeats count extra literal matches within one attempt/phase and identical recorded scope/options; semantic duplication and whether repetition was justified are not evaluated.",
                "Mem0 RPC completion is distinct from SUCCEEDED asynchronous event receipts and observed settlement. Metadata limit validation does not imply HTTP 400 without recorded status evidence.",
                "Mem0 state files are deduplicated against traces by exact content excluding turn/phase; matching operation/parameters can complete a started trace observation. Conflicting final snapshots count once with unknown status; unpaired state observations have unknown phase.",
                "This is an in-flight filesystem snapshot, not an atomic batch snapshot; rerun after all target records are finalized. No clinical criteria, sealed questions, or rubrics are read."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze_root(args.root)
    output = args.output or args.root / "telemetry_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"status": "telemetry_written", "expected_runs": result["coverage"]["expected_runs"],
                      "completed_records": result["coverage"]["canonical_completed_records"],
                      "failed_attempts": len(result["failed_attempts"]["records"]),
                      "observation_issues": len(result["observation_issues"])}))


if __name__ == "__main__":
    main()
