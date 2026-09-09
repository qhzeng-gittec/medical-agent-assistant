"""Existing two-judge rubric with explicit timing and lossless ID normalization."""
import asyncio
import json
from campaign_gateway import dump, sha
from campaign_run import parse_json_answer
from holdout_grade import (DEFAULT_JUDGES, JUDGE_SYSTEM, aggregate_checks, build_judge_payload,
                           expand_checkpoints, fingerprint, protocol_observations, validate_verdict)

VERSION = "iteration-v1-explicit-timing-and-evidence-shape"
TIMING = """\n检查点 after 指定评估范围：final 检查截至对话结束整个已观察过程是否满足要求，
不能仅因最后一轮换了话题就把之前已完成的要求判 not_applicable；但后来的纠正不能抹去早期已发生的关键错误。
each_turn 或 turn:N 检查对应时点的要求，不能借用未来回答。患者状态和当前档案以该时点实际记录为准。
返回 evidence 时仅使用目录中真实存在的 ID；资料未支持的操作细节不能因表述详细就视为有来源支持。"""


def validated_response(content, current, payload):
    data = parse_json_answer(content)
    wrapped = 0
    for check in data["checks"]:
        for index, item in enumerate(check["evidence"]):
            if isinstance(item, str):
                check["evidence"][index] = {"id": item}
                wrapped += 1
    return validate_verdict(data, current, payload), wrapped


async def grade_run(gateway, case, result, trace):
    root, run_id = gateway.root, result["run_id"]
    if sha(root / "traces" / f"{run_id}.json") != result["trace_sha256"]:
        raise ValueError("Trace changed before judging")
    destination = root / "grades" / f"{run_id}.json"
    inputs = {"case": fingerprint(case), "result": fingerprint(result), "trace": fingerprint(trace), "version": VERSION}
    if destination.exists():
        cached = json.loads(destination.read_text(encoding="utf-8"))
        if cached["inputs"] != inputs:
            raise ValueError("Preserve prior grade; use a new output")
        return cached
    checkpoints = expand_checkpoints(case["private"]["checkpoints"], result["turns"])
    record = {"run_id": run_id, "case_id": case["case_id"], "status": "not_graded", "inputs": inputs,
              "judge_models": list(DEFAULT_JUDGES), "judge_results": [],
              "protocol_observations": protocol_observations(trace)}
    if result["status"] == "completed":
        async def judge(model, index):
            row = {"judge_model": model, "checks": [], "calls": [], "errors": []}
            for horizon in sorted({c["horizon"] for c in checkpoints if c["executed"]}):
                current = [c for c in checkpoints if c["horizon"] == horizon and c["executed"]]
                payload = build_judge_payload(case, result, trace, current, horizon)
                timing = {c["id"]: c["after"] for c in current}
                for check in payload["checkpoints"]:
                    check["after"] = timing[check["id"]]
                messages = [{"role": "system", "content": JUDGE_SYSTEM + TIMING},
                            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
                judge_trace = []
                relative = f"judge_traces/{run_id}.judge{index}.turn{horizon}.json"
                call = {"horizon": horizon, "trace": relative}
                try:
                    response = await gateway.chat(model, messages, judge_trace, "rubric_judge")
                    choice = response["choices"][0]
                    if choice["finish_reason"] != "stop":
                        raise ValueError("Incomplete judge response")
                    checks, wrapped = validated_response(choice["message"]["content"], current, payload)
                    row["checks"].extend(checks)
                    call.update(status="validated", evidence_strings_wrapped=wrapped)
                except Exception as error:
                    call["status"] = "judge_error"
                    row["errors"].append({"horizon": horizon, "error_type": type(error).__name__, "message": gateway.redact(error)})
                dump(root / relative, judge_trace)
                row["calls"].append(call)
            return row
        record["judge_results"] = await asyncio.gather(*(judge(model, i) for i, model in enumerate(DEFAULT_JUDGES, 1)))
        record["status"] = "judge_error" if any(j["errors"] for j in record["judge_results"]) else "graded"
    record.update(aggregate_checks(checkpoints, record["judge_results"]))
    dump(destination, record)
    return record
