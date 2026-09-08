"""One bounded recovery for invalid grades, preserving every earlier attempt."""

import argparse
import json

from campaign_gateway import Gateway, dump
from campaign_regrade import numbered_dialogue, validate_ids
from campaign_run import OUTPUT


def complete_grade(original, checks, recovery):
    grade = {**original, "status": "graded", "checks": checks, "recovery": recovery}
    grade.pop("error", None)
    grade["critical_failed"] = any(c["critical"] and check["verdict"] == "fail"
                                   for c, check in zip(grade["criteria"], checks))
    grade["all_criteria_passed"] = all(check["verdict"] == "pass" for check in checks)
    return grade


def main(live=False):
    gateway = None
    for path in sorted((OUTPUT / "grades_v2_retries").glob("*.json")):
        original = json.loads(path.read_text(encoding="utf-8"))
        if original["status"] != "judge_error":
            continue
        destination = OUTPUT / "grades_v2_repaired" / path.name
        if destination.exists():
            continue
        result = json.loads((OUTPUT / "runs" / f"{original['run_id']}.json").read_text(encoding="utf-8"))
        _, lines = numbered_dialogue(result["turns"])
        events = json.loads((OUTPUT / "judge_traces_v2_retries" / path.name).read_text(encoding="utf-8"))
        if not events:
            continue
        event = events[-1]
        choice = event.get("response", {}).get("choices", [{}])[0]
        if choice.get("finish_reason") == "stop":
            try:
                checks = validate_ids(choice["message"]["content"], original["criteria"], lines)
            except (ValueError, KeyError, TypeError):
                pass  # Explicitly try the one optional provider retry below; no default grade.
            else:
                grade = complete_grade(original, checks, {"kind": "offline_fence_parser_fix", "api_calls": 0,
                                                         "source": str(path), "new_judgement": False})
                dump(destination, grade)
                print(json.dumps({"run": original["run_id"], "status": "graded", "recovery": "offline_parser"}), flush=True)
                continue
        if not live:
            continue
        if gateway is None:
            gateway = Gateway(OUTPUT)
        payload = {**event["payload"], "max_tokens": 12288}
        trace = []
        recovery = {"kind": "single_invalid_grade_retry", "source": str(path), "max_tokens": 12288,
                    "reasoning": payload.get("reasoning"), "same_messages_and_model": True}
        grade = {**original, "recovery": recovery}
        try:
            response = gateway.request("chat/completions", payload, trace, "rubric_judge_v2_repair")
            choice = response["choices"][0]
            if choice["finish_reason"] != "stop":
                raise ValueError("Repair judge response incomplete")
            checks = validate_ids(choice["message"]["content"], original["criteria"], lines)
            grade = complete_grade(original, checks, recovery)
        except Exception as error:
            grade.update(status="judge_error", error=f"{type(error).__name__}: {error}".replace(gateway.key, "[REDACTED]"))
        dump(destination, grade)
        dump(OUTPUT / "judge_traces_v2_repaired" / path.name, trace)
        print(json.dumps({"run": original["run_id"], "judge": original["judge"], "status": grade["status"]}), flush=True)
        if gateway.halt_reason:
            raise SystemExit(gateway.halt_reason)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Allow one provider retry per invalid grade, max output 12288")
    main(parser.parse_args().live)
