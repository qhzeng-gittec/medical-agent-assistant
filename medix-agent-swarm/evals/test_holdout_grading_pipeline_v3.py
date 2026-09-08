"""Independent public toy transport/queue tests; no sealed inputs or paid calls."""

import asyncio
import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).parent))
SPEC = importlib.util.spec_from_file_location("grading_v3", Path(__file__).with_name("holdout_grading_pipeline_v3.py"))
V3 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V3)


class FakeGateway:
    def __init__(self, root):
        self.root, self.key, self.calls = root, "toy-key", []

    def redact(self, value):
        return str(value)

    async def chat(self, model, messages, trace, role, **kwargs):
        self.calls.append({"model": model, "messages": copy.deepcopy(messages), "role": role, **kwargs})
        payload = json.loads(messages[1]["content"])
        checks = [{"id": checkpoint["id"], "verdict": "pass", "reason": "Observed toy answer",
                   "requires_durable_write": False, "missing_action": False, "prerequisite_untriggered": False,
                   "evidence": ["t1.answer"]} for checkpoint in payload["checkpoints"]]
        response = {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps({"checks": checks})}}]}
        trace.append({"event": "api_request", "role": role, "payload": {"model": model, "messages": messages, **kwargs},
                      "response": copy.deepcopy(response), "status": "ok"})
        return response


def public_toys(output):
    for index, model in enumerate(V3.DEFAULT_JUDGES, 1):
        for horizon in (1, 2, 3):
            payload = {"horizon": horizon, "checkpoints": [{"id": f"toy@{horizon}"}],
                       "observations": {"turns": [{"turn": 1, "answer": "OBSERVED_TOY", "event_memory_after": []}]},
                       "evidence_catalogue": [{"id": "t1.answer", "pointer": "/observations/turns/0/answer"}]}
            messages = [{"role": "system", "content": "Original public system message"},
                        {"role": "user", "content": json.dumps(payload)}]
            V3.immutable_json(output / V3.PUBLIC_SOURCE / f"PUBLIC-CALIBRATION.judge{index}.turn{horizon}.json",
                              [{"event": "api_request", "role": "rubric_judge",
                                "payload": {"model": model, "messages": messages, "max_tokens": 4096}}])


def test_budget_override_preserves_messages_models_and_wire_selection(tmp_path):
    public_toys(tmp_path)
    source = V3.public_sources(tmp_path)[0]
    fake = FakeGateway(tmp_path)
    trace = []
    response = asyncio.run(V3.BudgetJudgeGateway(fake).chat(source["model"], source["messages"], trace,
                                                           "rubric_judge", max_tokens=4096))
    assert fake.calls == [{"model": source["model"], "messages": source["messages"],
                           "role": "rubric_judge", "max_tokens": 16384}]
    assert json.loads(response["choices"][0]["message"]["content"])["checks"][0]["evidence"] == [{"id": "t1.answer"}]
    assert json.loads(trace[0]["response"]["choices"][0]["message"]["content"])["checks"][0]["evidence"] == ["t1.answer"]
    with pytest.raises(ValueError, match="cannot handle target"):
        asyncio.run(V3.BudgetJudgeGateway(fake).chat(source["model"], source["messages"], [], "supervisor"))
    assert len(fake.calls) == 1


def test_public_calibration_freezes_then_executes_six_calls_and_resumes_without_cost(tmp_path, monkeypatch):
    public_toys(tmp_path)
    monkeypatch.setattr(V3, "dependency_hashes", lambda: {"toy-dependency": "frozen"})
    fake = FakeGateway(tmp_path)

    def factory(output):
        assert (output / "calibration_grades" / V3.VERSION / "protocol.json").exists()
        return fake

    summary = asyncio.run(V3.calibrate_budget(tmp_path, factory))
    assert summary["passed"] == summary["requests"] == len(fake.calls) == 6
    assert all(call["max_tokens"] == 16384 for call in fake.calls)
    assert asyncio.run(V3.calibrate_budget(tmp_path, factory)) == summary
    assert len(fake.calls) == 6
    assert len(list((tmp_path / "calibration_grades" / V3.VERSION / "judge_traces").glob("*.json"))) == 6


def test_budget_calibration_rejects_changed_dependency_without_paid_call(tmp_path, monkeypatch):
    public_toys(tmp_path)
    monkeypatch.setattr(V3, "dependency_hashes", lambda: {"toy-dependency": "before"})
    fake = FakeGateway(tmp_path)
    asyncio.run(V3.calibrate_budget(tmp_path, lambda _: fake))
    monkeypatch.setattr(V3, "dependency_hashes", lambda: {"toy-dependency": "after"})
    with pytest.raises(ValueError, match="Frozen v3 configuration"):
        asyncio.run(V3.calibrate_budget(tmp_path, lambda _: fake))
    assert len(fake.calls) == 6


def test_old_root_grade_cannot_enter_v3_protocol(tmp_path, monkeypatch):
    public_toys(tmp_path)
    monkeypatch.setattr(V3, "dependency_hashes", lambda: {"toy-dependency": "frozen"})
    asyncio.run(V3.calibrate_budget(tmp_path, lambda output: FakeGateway(output)))
    plan = {"expected_runs": [{"run_id": "toy"}]}
    V3.immutable_json(tmp_path / "execution_plan.json", plan)
    V3.immutable_json(tmp_path / "runtime_freeze.json", {"toy": True})
    V3.immutable_json(tmp_path / "grades/old.json", {"old": True})
    with pytest.raises(ValueError, match="must be empty"):
        V3.freeze_pipeline(tmp_path, plan, 8, 4)


def test_freeze_pipeline_is_immutable_and_keeps_archive_separate(tmp_path, monkeypatch):
    public_toys(tmp_path)
    monkeypatch.setattr(V3, "dependency_hashes", lambda: {"toy-dependency": "frozen"})
    asyncio.run(V3.calibrate_budget(tmp_path, lambda output: FakeGateway(output)))
    plan = {"expected_runs": [{"run_id": "toy"}]}
    V3.immutable_json(tmp_path / "execution_plan.json", plan)
    V3.immutable_json(tmp_path / "runtime_freeze.json", {"toy": True})
    V3.immutable_json(tmp_path / "grading_archive/v2_4096/grades/old.json", {"old": True})
    first = V3.freeze_pipeline(tmp_path, plan, 8, 4)
    assert V3.freeze_pipeline(tmp_path, plan, 8, 4) == first
    with pytest.raises(ValueError, match="Frozen v3 configuration"):
        V3.freeze_pipeline(tmp_path, plan, 4, 4)


def test_streaming_reports_before_slow_run_finishes_and_refills_slots(tmp_path):
    for name in ("a_fast", "b_slow", "c_next"):
        V3.immutable_json(tmp_path / "runs" / f"{name}.json", {})

    async def scenario():
        release_slow = asyncio.Event()
        events, active = [], set()

        async def grade(path):
            active.add(path.stem)
            assert len(active) <= 2
            if path.stem == "b_slow":
                await release_slow.wait()
            else:
                await asyncio.sleep(0)
            active.remove(path.stem)
            events.append(path.stem)
            return {"status": "graded"}

        def report(output):
            if "a_fast" in events and "b_slow" not in events:
                events.append("report_before_slow")
                release_slow.set()

        result = await V3.stream_grades(tmp_path, {"a_fast", "b_slow", "c_next"}, set(), grade, report,
                                        concurrency=2, report_every=1, poll_seconds=.01)
        assert result["graded_runs"] == 3
        assert events.index("report_before_slow") < events.index("b_slow")

    asyncio.run(scenario())
