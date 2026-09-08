"""Read-only grading pipeline over immutable completed target runs.

The frozen rubric grader is unchanged. Its wire adapter accepts evidence IDs as
strings or {id: ...}; this does not change selected IDs, verdicts, or criteria.
The original provider response remains in its trace, followed by a normalization
event. This adapter is calibrated before any sealed-case judging.
"""

import argparse
import asyncio
import copy
import json
import time
from pathlib import Path

from campaign_gateway import dump, sha
from campaign_run import parse_json_answer
from holdout_grade import build_report, grade_run, validate_verdict
from holdout_run import load_cases, read_json, verify_dataset, verify_execution
from holdout_services import HoldoutGateway


def normalize_evidence_wire(verdict):
    normalized = copy.deepcopy(verdict)
    converted = 0
    if isinstance(normalized, dict) and isinstance(normalized.get("checks"), list):
        for check in normalized["checks"]:
            if isinstance(check, dict) and isinstance(check.get("evidence"), list):
                converted += sum(isinstance(item, str) for item in check["evidence"])
                check["evidence"] = [{"id": item} if isinstance(item, str) else item for item in check["evidence"]]
    return normalized, converted


class JudgeWireGateway:
    def __init__(self, gateway):
        self.gateway, self.root, self.key = gateway, gateway.root, gateway.key
        self.redact = gateway.redact

    async def chat(self, model, messages, trace, role, **kwargs):
        if role != "rubric_judge":
            raise ValueError("Judge adapter cannot handle target or patient calls")
        data = await self.gateway.chat(model, messages, trace, role, **kwargs)
        choice = data["choices"][0]
        if choice.get("finish_reason") != "stop":
            return data
        try:
            parsed = parse_json_answer(choice["message"]["content"])
        except (ValueError, TypeError, AttributeError):
            return data  # Core grader will record the original malformed response.
        normalized, count = normalize_evidence_wire(parsed)
        if not count:
            return data
        result = copy.deepcopy(data)
        result["choices"][0]["message"]["content"] = json.dumps(normalized, ensure_ascii=False)
        trace.append({"event": "judge_wire_normalization", "converted_string_ids": count,
                      "rule": "string evidence ID -> object with same id; no verdict or source selection changes"})
        return result


def calibrate_wire(output):
    folder = output / "calibration_grades/holdout-horizon-v2-evidence-ids/judge_traces"
    rows = []
    for path in sorted(folder.glob("*.json")):
        trace = read_json(path)
        request = next(event for event in trace if event["event"] == "api_request")
        payload = json.loads(request["payload"]["messages"][1]["content"])
        choice = request["response"]["choices"][0]
        parsed = parse_json_answer(choice["message"]["content"])
        normalized, count = normalize_evidence_wire(parsed)
        validate_verdict(normalized, payload["checkpoints"], payload)
        rows.append({"source": str(path.relative_to(output)), "source_sha256": sha(path),
                     "validated": True, "normalized_ids": count})
    if len(rows) != 6:
        raise ValueError("Expected all six independent public calibration judge responses")
    result = {"passed": len(rows), "cases": "One public calibration dialogue, three horizons, two judges",
              "paid_calls": 0, "verdicts_and_selected_evidence_unchanged": True, "responses": rows}
    dump(output / "calibration_grades/wire_calibration.json", result)
    print(json.dumps({"wire_calibration_passed": len(rows), "new_paid_calls": 0}), flush=True)


async def main(args):
    output, dataset = args.output.resolve(), args.dataset.resolve()
    manifest = verify_dataset(dataset)
    plan = read_json(output / "execution_plan.json")
    expected = {row["run_id"] for row in plan["expected_runs"]}
    calibration = read_json(output / "calibration_grades/wire_calibration.json")
    if calibration["passed"] != 6:
        raise ValueError("Public wire calibration did not pass")
    protocol = {"pipeline_sha256": sha(Path(__file__)), "frozen_grader_sha256": sha(Path(__file__).with_name("holdout_grade.py")),
                "launcher_sha256": sha(Path(__file__).with_name("run_holdout_grading.mjs")),
                "execution_plan_sha256": sha(output / "execution_plan.json"),
                "runtime_freeze_sha256": sha(output / "runtime_freeze.json"),
                "wire_calibration_sha256": sha(output / "calibration_grades/wire_calibration.json"),
                "concurrency": args.concurrency, "normalization": "Only string evidence IDs to id objects",
                "product_and_target_inputs_unchanged": True, "target_output_frozen_before_each_judgment": True}
    protocol_path = output / "grading_pipeline_protocol.json"
    if protocol_path.exists() and read_json(protocol_path) != protocol:
        raise ValueError("Frozen grading pipeline changed")
    if not protocol_path.exists():
        dump(protocol_path, protocol)
    cases = {case["case_id"]: case for case in load_cases(dataset, manifest, {row["case_id"] for row in plan["expected_runs"]})}
    gateway = JudgeWireGateway(HoldoutGateway(output))
    semaphore = asyncio.Semaphore(args.concurrency)

    async def grade(path):
        async with semaphore:
            result = read_json(path)
            verify_execution(result["protocol"])
            trace_path = output / "traces" / f"{result['run_id']}.json"
            if sha(trace_path) != result["trace_sha256"]:
                raise ValueError("Completed trace changed")
            record = await grade_run(gateway, cases[result["case_id"]], result, read_json(trace_path))
            print(json.dumps({"run_id": result["run_id"], "grade_status": record["status"]}), flush=True)

    started = time.monotonic()
    while True:
        paths = [path for path in (output / "runs").glob("*.json") if path.stem in expected
                 and not (output / "grades" / path.name).exists()]
        await asyncio.gather(*(grade(path) for path in paths))
        terminal = {path.stem for path in (output / "runs").glob("*.json")}
        for folder in ("errors", "process_errors"):
            terminal.update(read_json(path)["run_id"] for path in (output / folder).glob("*.json"))
        remaining = [path for path in (output / "runs").glob("*.json") if path.stem in expected
                     and not (output / "grades" / path.name).exists()]
        build_report(output)
        if not args.watch or (expected <= terminal and not remaining):
            print(json.dumps({"grading_pipeline_finished": True, "terminal_runs": len(expected & terminal),
                              "expected_runs": len(expected), "pending_completed_grades": len(remaining)}), flush=True)
            return
        if time.monotonic() - started > 14400:
            raise TimeoutError("Bounded grading watch expired; completed results remain resumable")
        await asyncio.sleep(10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path(__file__).parent / "holdout_v1_20260908")
    parser.add_argument("--concurrency", type=int, choices=range(1, 9), default=4)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--calibrate-wire", action="store_true")
    args = parser.parse_args()
    if args.calibrate_wire:
        calibrate_wire(args.output)
    else:
        asyncio.run(main(args))
