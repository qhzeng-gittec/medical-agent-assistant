"""Separate v3 orchestration: identical frozen rubric with a 16384-token judge budget."""

import argparse
import asyncio
import json
import time
from pathlib import Path

from campaign_gateway import sha
from campaign_run import parse_json_answer
from holdout_grade import DEFAULT_JUDGES, build_report, grade_run, validate_verdict
from holdout_grading_pipeline import JudgeWireGateway
from holdout_run import HARNESS_FILES, PROJECT, immutable_json, load_cases, read_json, verify_dataset, verify_execution
from holdout_services import HoldoutGateway


VERSION = "holdout-grading-v3-16384"
JUDGE_MAX_TOKENS = 16384
PUBLIC_SOURCE = "calibration_grades/holdout-horizon-v2-evidence-ids/judge_traces"


class BudgetJudgeGateway(JudgeWireGateway):
    async def chat(self, model, messages, trace, role, **kwargs):
        kwargs["max_tokens"] = JUDGE_MAX_TOKENS
        return await super().chat(model, messages, trace, role, **kwargs)


def dependency_hashes():
    names = set(HARNESS_FILES) | {
        "evals/holdout_grading_pipeline.py", "evals/holdout_grading_pipeline_v3.py",
        "evals/run_holdout_grading_v3.mjs",
    }
    return {name: sha(PROJECT / name) for name in sorted(names)}


def freeze_json(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError("Frozen v3 configuration or input changed")
    else:
        immutable_json(path, value)


def public_sources(output):
    sources = []
    for path in sorted((output / PUBLIC_SOURCE).glob("PUBLIC-CALIBRATION*.json")):
        events = read_json(path)
        requests = [event for event in events if event.get("event") == "api_request"]
        if len(requests) != 1 or requests[0].get("role") != "rubric_judge":
            raise ValueError("Public calibration source must contain one rubric request")
        request = requests[0]["payload"]
        payload = json.loads(request["messages"][1]["content"])
        sources.append({"path": path, "sha256": sha(path), "model": request["model"],
                        "horizon": payload["horizon"], "messages": request["messages"], "payload": payload})
    if len(sources) != 6 or {(source["model"], source["horizon"]) for source in sources} != {
            (judge, horizon) for judge in DEFAULT_JUDGES for horizon in (1, 2, 3)}:
        raise ValueError("Require the six original public calibration judge/horizon requests")
    return sources


async def calibrate_budget(output, gateway_factory=HoldoutGateway):
    """Six sequential public requests; preserve failures and never silently reattempt them."""
    output = Path(output)
    sources = public_sources(output)
    folder = output / "calibration_grades" / VERSION
    protocol = {"version": VERSION, "dependencies": dependency_hashes(), "judges": list(DEFAULT_JUDGES),
                "max_tokens": JUDGE_MAX_TOKENS, "reasoning_effort": "low",
                "normalization": "Frozen v2 string evidence ID to id object",
                "sources": [{"path": str(source["path"].relative_to(output)), "sha256": source["sha256"],
                             "judge": source["model"], "horizon": source["horizon"]} for source in sources],
                "only_request_change": "max_tokens=16384; original public messages unchanged"}
    freeze_json(folder / "protocol.json", protocol)
    gateway = BudgetJudgeGateway(gateway_factory(output))
    records = []
    for source in sources:
        record_path = folder / "records" / source["path"].name
        trace_path = folder / "judge_traces" / source["path"].name
        started_path = folder / "started" / source["path"].name
        if record_path.exists():
            record = read_json(record_path)
            if record["trace_sha256"] != sha(trace_path):
                raise ValueError("Public v3 calibration trace changed")
            records.append(record)
            continue
        if trace_path.exists() or started_path.exists():
            raise ValueError("Unfinished public calibration trace exists; preserve it without a paid retry")
        if sha(source["path"]) != source["sha256"]:
            raise ValueError("Original public calibration source changed")
        trace = []
        record = {"source": str(source["path"].relative_to(output)), "source_sha256": source["sha256"],
                  "judge": source["model"], "horizon": source["horizon"], "status": "started"}
        immutable_json(started_path, {**record, "protocol_sha256": sha(folder / "protocol.json")})
        try:
            response = await gateway.chat(source["model"], source["messages"], trace, "rubric_judge")
            choice = response["choices"][0]
            record["finish_reason"] = choice.get("finish_reason")
            if choice.get("finish_reason") != "stop":
                raise ValueError("Incomplete public calibration judge response")
            parsed = parse_json_answer(choice["message"]["content"])
            record["checks"] = validate_verdict(parsed, source["payload"]["checkpoints"], source["payload"])
            record["status"] = "validated"
        except Exception as error:
            record.update(status="judge_error", error_type=type(error).__name__, error=gateway.redact(error))
        immutable_json(trace_path, trace)
        record["trace_sha256"] = sha(trace_path)
        immutable_json(record_path, record)
        records.append(record)
        print(json.dumps({"public_calibration": source["path"].name, "status": record["status"]}), flush=True)
    summary = {"version": VERSION, "protocol_sha256": sha(folder / "protocol.json"),
               "passed": sum(record["status"] == "validated" for record in records), "requests": len(records),
               "records": {path.name: sha(path) for path in sorted((folder / "records").glob("*.json"))}}
    freeze_json(folder / "summary.json", summary)
    if summary["passed"] != 6:
        raise ValueError("Public 16384-token calibration did not validate all six calls")
    return summary


def terminal_runs(output):
    terminal = {path.stem for path in (output / "runs").glob("*.json")}
    for folder in ("errors", "process_errors"):
        terminal.update(read_json(path)["run_id"] for path in (output / folder).glob("*.json"))
    return terminal


async def stream_grades(output, expected, completed, grade, report, *, concurrency=8,
                        report_every=4, watch=False, watch_seconds=86400, poll_seconds=5):
    """Refill available slots after each completion and publish progress between completions."""
    active = {}
    started, last_report = time.monotonic(), time.monotonic()
    since_report = 0
    report(output)
    while True:
        candidates = sorted(path for path in (output / "runs").glob("*.json")
                            if path.stem in expected and path.stem not in completed and path.stem not in active)
        for path in candidates[:max(0, concurrency - len(active))]:
            active[path.stem] = asyncio.create_task(grade(path))
        if active:
            finished, _ = await asyncio.wait(active.values(), timeout=poll_seconds, return_when=asyncio.FIRST_COMPLETED)
            for run_id, task in list(active.items()):
                if task in finished:
                    record = task.result()
                    completed.add(run_id)
                    del active[run_id]
                    since_report += 1
                    print(json.dumps({"run_id": run_id, "grade_status": record["status"]}), flush=True)
        if since_report and (since_report >= report_every or time.monotonic() - last_report >= 60):
            report(output)
            since_report, last_report = 0, time.monotonic()
        pending = any(path.stem in expected and path.stem not in completed for path in (output / "runs").glob("*.json"))
        terminal = terminal_runs(output)
        if not active and not pending and (not watch or expected <= terminal):
            report(output)
            return {"terminal_runs": len(expected & terminal), "expected_runs": len(expected),
                    "graded_runs": len(completed), "version": VERSION}
        if time.monotonic() - started > watch_seconds:
            raise TimeoutError("Bounded v3 grading watch expired; immutable results remain resumable")
        if not active:
            await asyncio.sleep(poll_seconds)


def freeze_pipeline(output, plan, concurrency, report_every):
    folder = output / "calibration_grades" / VERSION
    summary = read_json(folder / "summary.json")
    calibration_protocol = read_json(folder / "protocol.json")
    dependencies = dependency_hashes()
    if summary["passed"] != 6 or summary["requests"] != 6:
        raise ValueError("Six successful public budget calibrations required before holdout judging")
    if summary["protocol_sha256"] != sha(folder / "protocol.json") or calibration_protocol["dependencies"] != dependencies:
        raise ValueError("Budget calibration dependency freeze changed")
    for name, expected_hash in summary["records"].items():
        path = folder / "records" / name
        record = read_json(path)
        if sha(path) != expected_hash or sha(folder / "judge_traces" / name) != record["trace_sha256"]:
            raise ValueError("Budget calibration evidence changed")
    protocol = {"version": VERSION, "dependencies": dependencies, "judges": list(DEFAULT_JUDGES),
                "max_tokens": JUDGE_MAX_TOKENS, "reasoning_effort": "low",
                "execution_plan_sha256": sha(output / "execution_plan.json"),
                "runtime_freeze_sha256": sha(output / "runtime_freeze.json"),
                "calibration_summary_sha256": sha(folder / "summary.json"),
                "concurrency": concurrency, "report_every": report_every,
                "expected_runs": len(plan["expected_runs"]),
                "only_judge_request_change": "All rubric_judge max_tokens uniformly 16384",
                "frozen_criteria_and_observations_unchanged": True, "archive_reuse": False}
    destination = output / "grading_pipeline_v3_protocol.json"
    if not destination.exists() and (any((output / "grades").glob("*.json")) or any((output / "judge_traces").glob("*.json"))):
        raise ValueError("Root grades and judge_traces must be empty before the first v3 protocol freeze")
    freeze_json(destination, protocol)
    return sha(destination)


async def main(args):
    output, dataset = args.output.resolve(), args.dataset.resolve()
    manifest = verify_dataset(dataset)
    plan = read_json(output / "execution_plan.json")
    protocol_hash = freeze_pipeline(output, plan, args.concurrency, args.report_every)
    expected = {row["run_id"] for row in plan["expected_runs"]}
    # Only the runtime, after freezing v3, reads the sealed cases for the unchanged grader.
    cases = {case["case_id"]: case for case in load_cases(dataset, manifest, {row["case_id"] for row in plan["expected_runs"]})}
    completed = set()
    for path in (output / "grading_v3/completed").glob("*.json"):
        marker = read_json(path)
        if marker["protocol_sha256"] != protocol_hash or marker["grade_sha256"] != sha(output / "grades" / path.name):
            raise ValueError("Immutable v3 grade or protocol changed")
        completed.add(path.stem)
    for path in (output / "grades").glob("*.json"):
        if not (output / "grading_v3/inputs" / path.name).exists():
            raise ValueError("Grade without v3 input provenance cannot be reused")
    gateway = BudgetJudgeGateway(HoldoutGateway(output))

    async def grade(path):
        result = read_json(path)
        verify_execution(result["protocol"])
        trace_path = output / "traces" / path.name
        if sha(trace_path) != result["trace_sha256"]:
            raise ValueError("Completed target trace changed")
        inputs = {"protocol_sha256": protocol_hash, "result_sha256": sha(path), "trace_sha256": sha(trace_path)}
        freeze_json(output / "grading_v3/inputs" / path.name, inputs)
        record = await grade_run(gateway, cases[result["case_id"]], result, read_json(trace_path), judges=DEFAULT_JUDGES)
        freeze_json(output / "grading_v3/completed" / path.name,
                    {"protocol_sha256": protocol_hash, "grade_sha256": sha(output / "grades" / path.name), "status": record["status"]})
        return record

    summary = await stream_grades(output, expected, completed, grade, build_report,
                                  concurrency=args.concurrency, report_every=args.report_every,
                                  watch=args.watch, watch_seconds=args.watch_seconds)
    print(json.dumps({"grading_pipeline_finished": True, **summary}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path(__file__).parent / "holdout_v1_20260908")
    parser.add_argument("--concurrency", type=int, choices=range(1, 9), default=8)
    parser.add_argument("--report-every", type=int, default=4)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--watch-seconds", type=int, default=86400)
    parser.add_argument("--calibrate-budget", action="store_true")
    args = parser.parse_args()
    if args.report_every < 1 or args.watch_seconds < 1:
        parser.error("Report interval and watch bound must be positive")
    if args.calibrate_budget:
        asyncio.run(calibrate_budget(args.output.resolve()))
    else:
        asyncio.run(main(args))
