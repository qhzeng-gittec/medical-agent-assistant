"""Public live integration calibration, explicitly outside the sealed 72 cases."""

import argparse
import asyncio
import json
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

from loguru import logger

from campaign_gateway import dump
from campaign_run import build_system, trace_metrics
from holdout_services import HoldoutGateway, MODELS, build_environment


async def grade_calibration(output, models):
    from holdout_grade import GRADER_VERSION, grade_run
    gateway = HoldoutGateway(output)
    grading_gateway = SimpleNamespace(root=output / "calibration_grades" / GRADER_VERSION,
                                      chat=gateway.chat, redact=gateway.redact, key=gateway.key)
    case = {"case_id": "PUBLIC-CALIBRATION", "domain": "calibration", "difficulty": "routine",
            "private": {"clinical_review_required": False, "checkpoints": [
                {"id": "public1", "after": "turn:1", "target": "profile", "critical": False,
                 "requirement": "实际档案保存用户明确要求记住的运动习惯，不虚构其他健康事实。"},
                {"id": "public2", "after": "turn:2", "target": "answer", "critical": False,
                 "requirement": "根据实际可用的本人历史准确回答此前报告的每周运动次数。"},
                {"id": "public3", "after": "turn:3", "target": "trace", "critical": False,
                 "requirement": "准确表达实际检索的来源和局限，不把失败工具称为已成功完成研究。"}]}}
    paths = sorted((output / "calibration").glob("PUBLIC-CALIBRATION__*.json"))
    paths = [path for path in paths if not path.stem.endswith("_trace")]
    if not paths:
        raise RuntimeError("No completed public calibration exists")
    async def evaluate(path):
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["model"] not in models:
            return
        trace = json.loads(path.with_name(path.stem + "_trace.json").read_text(encoding="utf-8"))
        grade = await grade_run(grading_gateway, case, result, trace)
        print(json.dumps({"calibration_grade": result["model"], "status": grade["status"],
                          "judge_errors": sum(len(j["errors"]) for j in grade["judge_results"])}), flush=True)
    await asyncio.gather(*(evaluate(path) for path in paths))


async def main(output, models):
    logger.remove()
    logger.add(sys.stderr, level="ERROR", diagnose=False, backtrace=False)
    gateway = HoldoutGateway(output)
    for model in models:
        run_id = "PUBLIC-CALIBRATION__" + model.replace("/", "_")
        destination = output / "calibration" / f"{run_id}.json"
        if destination.exists():
            raise RuntimeError("Calibration record already exists; inspect it instead of repeating paid work")
        trace, turns = [], []
        prefix = "medix-public-calibration-" + uuid.uuid4().hex
        identities = {"self": prefix + "-self", "other": prefix + "-other"}
        environment = {"rag_mode": "live_rag", "memory_mode": "live_mem0", "documents": [], "memory_records": []}
        state = output / "calibration_state" / uuid.uuid4().hex[:12]
        result = {"run_id": run_id, "case_id": "PUBLIC-CALIBRATION", "model": model,
                  "repetition": 0, "status": "running", "domain": "calibration", "difficulty": "routine",
                  "seed_turns": [], "turns": turns, "excluded_from_holdout_scores": True}
        try:
            kb, memory = await asyncio.to_thread(build_environment, gateway, environment, identities, trace, state)
            supervisor, profile = build_system(gateway, model, trace, kb, state / "profiles", memory)
            questions = ["我通常每周打两次羽毛球，请记住这个生活习惯。",
                         "我以前告诉过你，我通常每周打几次羽毛球？",
                         "请检索资料，解释在家测量血压有哪些注意事项，并说明实际查到的来源。"]
            for number, question in enumerate(questions, 1):
                start = len(trace)
                trace.append({"event": "turn_start", "phase": "target", "turn": number,
                              "user_id": identities["self"], "session_id": prefix + f"-session-{number}", "text": question})
                before, raw_before = profile.get_context(identities["self"]), profile._load(identities["self"])
                began = time.monotonic()
                response = await asyncio.wait_for(supervisor.process(question, user_id=identities["self"],
                                                     session_id=prefix + f"-session-{number}"), 600)
                latency = time.monotonic() - began
                await asyncio.to_thread(memory.settle)
                snapshot = await asyncio.to_thread(memory.snapshot, identities["self"])
                turns.append({"turn": number, "user": question, "answer": response["answer"],
                              "user_id": identities["self"], "session_id": prefix + f"-session-{number}",
                              "profile_before": before, "profile_record_before": raw_before,
                              "profile_after": profile.get_context(identities["self"]),
                              "profile_record_after": profile._load(identities["self"]),
                              "event_memory_after": snapshot, "system_result": response,
                              "latency_seconds": latency})
                for event in trace[start:]:
                    event.setdefault("phase", "target")
                    event.setdefault("turn", number)
                trace.append({"event": "turn_end", "phase": "target", "turn": number})
            result["status"] = "completed"
        except Exception as error:
            result.update(status="error", error_type=type(error).__name__, error=gateway.redact(error))
            raise
        finally:
            result["metrics"] = trace_metrics(trace)
            dump(output / "calibration" / f"{run_id}_trace.json", trace)
            dump(destination, result)
            print(json.dumps({"calibration": model, "status": result["status"], "turns": len(turns),
                              "error_type": result.get("error_type"), "requests": result["metrics"]["model_requests"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--grade-only", action="store_true")
    args = parser.parse_args()
    asyncio.run(grade_calibration(args.output, args.models) if args.grade_only else main(args.output, args.models))
