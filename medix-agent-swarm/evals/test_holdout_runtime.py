"""Public toy calibration only; no test reads the sealed holdout or its results."""

import asyncio
import copy
import json
import sys
import types

import pytest

import holdout_run as runner


def toy_case():
    return {"case_id": "TOY-001", "family_id": "TOY-F001", "domain": "history", "difficulty": "routine",
            "interaction": {"opening": "Please help with my appointment note.", "max_patient_replies": 0,
                            "scripted_followups": []},
            "private": {"patient_facts": [], "seed_history": [],
                        "environment": {"rag_mode": "not_needed", "memory_mode": "controlled",
                                        "documents": [], "memory_records": []},
                        "checkpoints": [{"id": "c1", "requirement": "RUBRIC_CANARY_NEVER_TARGET"}],
                        "source_refs": [{"note": "GOLD_CANARY_NEVER_TARGET"}],
                        "clinical_review_required": False}}


def completion(text="The note is acknowledged.", finish="stop", calls=None):
    message = {"role": "assistant", "content": text}
    if calls:
        message["tool_calls"] = calls
    return {"choices": [{"finish_reason": finish, "message": message}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}}


class FakeGateway:
    def __init__(self, root, target=None, selector=None):
        self.root, self.key, self.spent = root, "", 0
        self.target = list(target or [])
        self.selector = list(selector or [])
        self.calls = []

    def redact(self, error):
        return str(error)

    async def chat(self, model, messages, trace, role, tools=None, tool_choice="auto", max_tokens=2048):
        payload = {"model": model, "messages": copy.deepcopy(messages), "tools": copy.deepcopy(tools or [])}
        self.calls.append({"role": role, "payload": payload})
        queue = self.selector if role == "patient_selector" else self.target
        response = queue.pop(0) if queue else completion()
        trace.append({"event": "api_request", "request_id": str(len(self.calls)), "role": role,
                      "payload": payload, "response": copy.deepcopy(response), "status": "ok", "cost_usd": 0})
        return response


class FakeKB:
    backend_label = "controlled_toy"

    def search(self, **kwargs):
        return []


class FakeMemory:
    backend_label = "controlled_toy"
    enabled = True

    def __init__(self, users):
        self.users, self.records, self.prepared = dict(users), {}, None
        self.seed_replay = False
        self.settled = False
        self.fail_settle = False

    def search_similar_sessions(self, query, user_id, limit=3):
        return []

    def add_session_summary(self, user_id, session_id, question, answer, metadata=None):
        if self.seed_replay:
            return None
        self.records.setdefault(user_id, []).append({"content": question, "source_role": "user",
                                                    "session_id": session_id})
        self.settled = False
        return "toy-operation"

    def prepare_fixture(self, history):
        self.prepared = copy.deepcopy(history)
        for seed in history:
            self.records.setdefault(self.users[seed["user_key"]], []).extend(copy.deepcopy(seed["messages"]))
        self.settled = False

    def settle(self):
        if self.fail_settle:
            raise RuntimeError("Toy storage not visible")
        self.settled = True

    def snapshot(self, user_id):
        assert self.settled
        return copy.deepcopy(self.records.get(user_id, []))


@pytest.fixture
def setup_runtime(monkeypatch):
    memories = []

    def environment(gateway, env, user_ids, trace, run_dir):
        memory = FakeMemory(user_ids)
        memories.append(memory)
        return FakeKB(), memory

    monkeypatch.setitem(sys.modules, "holdout_services", types.SimpleNamespace(build_environment=environment))
    monkeypatch.setattr(runner, "verify_execution", lambda protocol: None)
    return memories


def execute(gateway, case, repetition=1):
    return asyncio.run(runner.run_case(gateway, runner.MODELS[0], case, repetition,
                                      {"calibration": "public-toy-fixture"}))


def target_payloads(gateway):
    return [json.dumps(row["payload"], ensure_ascii=False) for row in gateway.calls
            if row["role"] in runner.TARGET_ROLES]


def test_actual_target_transport_excludes_gold_unrevealed_facts_and_future_followups(tmp_path, setup_runtime):
    case = toy_case()
    case["private"]["patient_facts"] = [
        {"id": "color", "text": "SELECTED_FACT_CANARY", "reveal_when": "ASK_COLOR_CANARY"},
        {"id": "shape", "text": "UNREVEALED_FACT_CANARY", "reveal_when": "ASK_SHAPE_CANARY"}]
    case["interaction"].update(max_patient_replies=1, scripted_followups=[
        {"text": "FUTURE_FOLLOWUP_CANARY", "new_user": False, "new_session": False}])
    gateway = FakeGateway(tmp_path, target=[completion("What color is the appointment card?"), completion(), completion()],
                          selector=[completion('{"needs_reply":true,"fact_ids":["color"]}')])
    result = execute(gateway, case)
    assert result["status"] == "completed"
    payloads = target_payloads(gateway)
    assert len(payloads) == 3
    assert "SELECTED_FACT_CANARY" not in payloads[0]
    assert "SELECTED_FACT_CANARY" in payloads[1]
    assert all("FUTURE_FOLLOWUP_CANARY" not in p for p in payloads[:2])
    assert "FUTURE_FOLLOWUP_CANARY" in payloads[2]
    for canary in ("RUBRIC_CANARY", "GOLD_CANARY", "UNREVEALED_FACT_CANARY", "ASK_COLOR_CANARY", "ASK_SHAPE_CANARY"):
        assert all(canary not in p for p in payloads)
    selector = json.dumps(next(c["payload"] for c in gateway.calls if c["role"] == "patient_selector"))
    assert "UNREVEALED_FACT_CANARY" in selector
    assert all(c not in selector for c in ("RUBRIC_CANARY", "GOLD_CANARY", "FUTURE_FOLLOWUP_CANARY"))
    assert result["patient_fact_ids_revealed"] == ["color"]
    assert all(t["event_memory_after"] for t in result["turns"])


@pytest.mark.parametrize("selected", [
    {"needs_reply": True, "fact_ids": ["invented"]},
    {"needs_reply": True, "fact_ids": [0]},
    {"needs_reply": True, "fact_ids": ["one", "one"]},
    {"needs_reply": False, "fact_ids": ["one"]},
    {"needs_reply": "yes", "fact_ids": []},
    {"needs_reply": True, "fact_ids": [], "invented_text": "unsafe"},
])
def test_patient_selector_rejects_invalid_schema_or_invented_ids(tmp_path, selected):
    gateway = FakeGateway(tmp_path, selector=[completion(json.dumps(selected))])
    facts = [{"id": "one", "text": "only supplied fact", "reveal_when": "appropriate question"}]
    with pytest.raises(ValueError):
        asyncio.run(runner.patient_reply(gateway, facts, "Question?", [], []))


def test_unknown_patient_answer_adds_no_examination_or_medical_fact(tmp_path):
    gateway = FakeGateway(tmp_path, selector=[completion('{"needs_reply":true,"fact_ids":[]}')])
    facts = [{"id": "one", "text": "one fact", "reveal_when": "different question"}]
    revealed = []
    assert asyncio.run(runner.patient_reply(gateway, facts, "Unknown question?", revealed, [])) == "这点我不清楚。"
    assert revealed == []


def test_no_question_ends_adaptive_dialogue_before_due_script(tmp_path, setup_runtime):
    case = toy_case()
    case["private"]["patient_facts"] = [{"id": "one", "text": "PRIVATE", "reveal_when": "asked"}]
    case["interaction"].update(max_patient_replies=3, scripted_followups=[
        {"text": "Now the scheduled follow-up.", "new_user": False, "new_session": False}])
    gateway = FakeGateway(tmp_path, selector=[completion('{"needs_reply":false,"fact_ids":[]}')])
    result = execute(gateway, case)
    assert result["status"] == "completed"
    assert len(result["turns"]) == 2
    assert result["turns"][1]["user"] == "Now the scheduled follow-up."
    assert sum(c["role"] == "patient_selector" for c in gateway.calls) == 1


def test_identity_mapping_sessions_repetitions_and_attempts_are_isolated(tmp_path, setup_runtime):
    case = toy_case()
    case["interaction"]["scripted_followups"] = [
        {"text": "same session", "new_user": False, "new_session": False},
        {"text": "new session", "new_user": False, "new_session": True},
        {"text": "second user", "new_user": True, "new_session": False},
        {"text": "third user", "new_user": True, "new_session": False}]
    first = execute(FakeGateway(tmp_path), case)
    second = execute(FakeGateway(tmp_path), case, 2)
    a, b, c, d, e = first["turns"]
    assert a["session_id"] == b["session_id"] != c["session_id"]
    assert a["user_id"] == c["user_id"] != d["user_id"] != e["user_id"]
    assert d["user_id"] == first["user_ids"]["other"]
    assert len({c["session_id"], d["session_id"], e["session_id"]}) == 3
    assert set(t["session_id"] for t in first["turns"]).isdisjoint(t["session_id"] for t in second["turns"])
    assert d["profile_before"] == {}


def test_seed_uses_real_profile_extraction_and_preserves_original_memory_roles(tmp_path, setup_runtime):
    case = toy_case()
    original = [{"user_key": "self", "session": "shared", "messages": [
        {"role": "user", "content": "My age is 29."},
        {"role": "assistant", "content": "AUTHORED_ASSISTANT_HISTORY"}]},
        {"user_key": "other", "session": "shared", "messages": [{"role": "user", "content": "A different user's note."}]}]
    case["private"]["seed_history"] = original
    update = {"id": "profile-update", "type": "function", "function": {
        "name": "update_patient_profile", "arguments": json.dumps({"updates": [
            {"category": "demographics", "field": "age", "value": "29", "status": "active", "evidence": "My age is 29."}]})}}
    gateway = FakeGateway(tmp_path, target=[completion(None, "tool_calls", [update]), completion("GENERATED_SEED_ANSWER"),
                                           completion(), completion()])
    result = execute(gateway, case)
    assert result["status"] == "completed"
    assert len(result["seed_turns"]) == 2 and len(result["turns"]) == 1
    assert setup_runtime[0].prepared == original
    assert result["turns"][0]["profile_before"]["demographics"]["age"]["value"] == 29
    seed = result["seed_turns"][0]
    raw = seed["profile_record_after"]["demographics"]["age"]
    assert raw["source_session_id"] == seed["session_id"]
    assert raw["updated_at"]
    stored = json.dumps(result["seed_event_memory_after"])
    assert "AUTHORED_ASSISTANT_HISTORY" in stored and "GENERATED_SEED_ANSWER" not in stored
    assert result["seed_turns"][0]["session_id"] != result["seed_turns"][1]["session_id"]
    assert result["turns"][0]["session_id"] not in {t["session_id"] for t in result["seed_turns"]}
    trace = runner.read_json(tmp_path / "traces" / f"{result['run_id']}.json")
    assert any(e["event"] == "seed_turn_start" and e["seed_turn"] == 1 for e in trace)
    assert [e["turn"] for e in trace if e["event"] == "turn_start"] == [1]


def test_seed_followup_receives_authored_prefix_and_no_future_authored_message(tmp_path, setup_runtime):
    case = toy_case()
    case["private"]["seed_history"] = [{"user_key": "self", "session": "history", "messages": [
        {"role": "user", "content": "Please clarify the appointment."},
        {"role": "assistant", "content": "AUTHORED_PAST_QUESTION"},
        {"role": "user", "content": "Yes."},
        {"role": "assistant", "content": "AUTHORED_FUTURE_REPLY"}]}]
    gateway = FakeGateway(tmp_path, target=[completion("GENERATED_DIFFERENT_QUESTION"), completion(), completion()])
    result = execute(gateway, case)
    assert result["status"] == "completed"
    first, followup, opening = target_payloads(gateway)
    assert "AUTHORED_PAST_QUESTION" not in first
    assert "AUTHORED_PAST_QUESTION" in followup
    assert "GENERATED_DIFFERENT_QUESTION" not in followup
    assert "AUTHORED_FUTURE_REPLY" not in followup
    assert "AUTHORED_PAST_QUESTION" not in opening  # New main session, empty fake retrieval.
    assert result["seed_turns"][0]["authored_history_before"] == []
    assert result["seed_turns"][1]["authored_history_before"] == case["private"]["seed_history"][0]["messages"][:2]


def test_storage_settlement_failure_preserves_actual_visible_answer_as_error(tmp_path, setup_runtime, monkeypatch):
    original = FakeMemory.add_session_summary

    def fail_after_write(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        self.fail_settle = True
        return result

    monkeypatch.setattr(FakeMemory, "add_session_summary", fail_after_write)
    result = execute(FakeGateway(tmp_path), toy_case())
    assert result["status"] == "error"
    assert len(result["turns"]) == 1
    assert result["turns"][0]["answer"] == "The note is acknowledged."
    assert result["turns"][0]["event_memory_after"] is None
    assert not (tmp_path / "runs" / f"{result['run_id']}.json").exists()


@pytest.mark.parametrize("response", [completion(None), completion("partial", "length")])
def test_missing_or_truncated_target_response_is_never_completed(tmp_path, setup_runtime, response):
    result = execute(FakeGateway(tmp_path, target=[response]), toy_case())
    assert result["status"] in {"error", "incomplete"}
    assert result["turns"] == []
    assert not (tmp_path / "runs" / f"{result['run_id']}.json").exists()
    assert len(list((tmp_path / "errors").glob("*.json"))) == 1
    assert len(list((tmp_path / "traces" / "attempts").glob("*.json"))) == 1


def test_resume_completed_run_without_rewriting_or_calls(tmp_path, setup_runtime):
    first = execute(FakeGateway(tmp_path), toy_case())
    path = tmp_path / "runs" / f"{first['run_id']}.json"
    before = path.read_bytes(), path.stat().st_mtime_ns
    gateway = FakeGateway(tmp_path)
    assert execute(gateway, toy_case()) == first
    assert gateway.calls == []
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before


def test_failed_attempts_survive_retry_and_use_fresh_state(tmp_path, setup_runtime):
    first = execute(FakeGateway(tmp_path, target=[completion(None)]), toy_case())
    error_path = next((tmp_path / "errors").glob("*.json"))
    before = error_path.read_bytes()
    second = execute(FakeGateway(tmp_path), toy_case())
    assert first["status"] != "completed" and second["status"] == "completed"
    assert first["attempt_id"] != second["attempt_id"]
    assert first["user_ids"]["self"] != second["user_ids"]["self"]
    assert error_path.read_bytes() == before
    assert len(list((tmp_path / "attempts").glob("*.json"))) == 2


def test_frozen_guard_rejects_before_any_target_payload(tmp_path, setup_runtime, monkeypatch):
    def reject(protocol):
        raise ValueError("Runtime sources changed")

    monkeypatch.setattr(runner, "verify_execution", reject)
    gateway = FakeGateway(tmp_path)
    with pytest.raises(ValueError, match="sources changed"):
        execute(gateway, toy_case())
    assert gateway.calls == []


def frozen_toy_dataset(tmp_path):
    dataset, project = tmp_path / "public-toy-dataset", tmp_path / "public-toy-project"
    (project / "agents").mkdir(parents=True)
    source = project / "agents" / "toy.py"
    source.write_text("TOY_VALUE = 1\n", encoding="utf-8")
    required = {"contract_sha256": "author_contract.md", "review_sha256": "sealed/review.json",
                "validator_sha256": "prepare_holdout.py", "execution_protocol_sha256": "评测隔离与执行协议.md",
                "construction_summary_sha256": "construction_summary.json"}
    for relative in list(required.values()) + ["sealed/authors/toy.jsonl", "sealed/authors/toy_attestation.json"]:
        path = dataset / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("PUBLIC_TOY_ONLY", encoding="utf-8")
    runner.immutable_json(dataset / "product_lock.json", {
        "sources": {str(source.relative_to(project)): runner.sha(source)},
        "contract_sha256": runner.sha(dataset / "author_contract.md")})
    manifest = {key: runner.sha(dataset / relative) for key, relative in required.items()}
    manifest.update(product_lock_sha256=runner.sha(dataset / "product_lock.json"),
                    case_files={"sealed/authors/toy.jsonl": runner.sha(dataset / "sealed/authors/toy.jsonl")},
                    author_attestations={"toy": runner.sha(dataset / "sealed/authors/toy_attestation.json")},
                    cases=[{"case_id": f"TOY-{i:03d}"} for i in range(72)])
    runner.immutable_json(dataset / "freeze_manifest.json", manifest)
    return dataset, project


@pytest.mark.parametrize("kind", ["source", "dataset", "protocol"])
def test_physical_hash_guard_detects_changed_frozen_inputs(tmp_path, kind):
    dataset, project = frozen_toy_dataset(tmp_path)
    assert len(runner.verify_dataset(dataset, project)["cases"]) == 72
    path = {"source": project / "agents/toy.py", "dataset": dataset / "sealed/authors/toy.jsonl",
            "protocol": dataset / "评测隔离与执行协议.md"}[kind]
    path.write_text("CHANGED_PUBLIC_TOY", encoding="utf-8")
    with pytest.raises(ValueError, match="mismatch|differs"):
        runner.verify_dataset(dataset, project)


def test_immutable_record_never_overwrites_existing_result(tmp_path):
    path = tmp_path / "record.json"
    runner.immutable_json(path, {"status": "error", "attempt": 1})
    with pytest.raises(FileExistsError):
        runner.immutable_json(path, {"status": "completed", "attempt": 2})
    assert runner.read_json(path) == {"status": "error", "attempt": 1}


def test_full_execution_plan_fixes_stratified_repeats_before_any_answers(tmp_path, monkeypatch):
    cases = [{"case_id": f"TOY-{domain}-{difficulty}-{i}", "family_id": f"F-{domain}-{difficulty}-{i}",
              "domain": domain, "difficulty": difficulty}
             for domain in ("consultation", "history", "evidence")
             for difficulty in ("routine", "complex", "boundary") for i in range(8)]
    monkeypatch.setattr(runner, "load_cases", lambda *args: cases)
    plan = runner.register_execution_plan(tmp_path, tmp_path, {"cases": cases}, {"runtime_freeze_sha256": "toy"})
    assert len(plan["expected_runs"]) == 324
    assert len(plan["repeat_case_ids"]) == 18
    assert sum(row["repetition"] == 1 for row in plan["expected_runs"]) == 216
    assert all({"domain", "difficulty", "family_id"} <= set(row) for row in plan["expected_runs"])
    assert runner.register_execution_plan(tmp_path, tmp_path, {"cases": cases}, {"runtime_freeze_sha256": "toy"}) == plan


def test_fresh_worker_skill_globals_keep_their_own_kb(tmp_path):
    first_kb, second_kb = FakeKB(), FakeKB()
    first, _ = runner.build_system(FakeGateway(tmp_path), runner.MODELS[0], [], first_kb,
                                   tmp_path / "first", FakeMemory({}))
    second, _ = runner.build_system(FakeGateway(tmp_path), runner.MODELS[0], [], second_kb,
                                    tmp_path / "second", FakeMemory({}))
    for supervisor, expected in ((first, first_kb), (second, second_kb)):
        checked = 0
        for worker in supervisor.workers.values():
            for spec in worker.skill_registry.skills.values():
                namespace = spec["function"].__globals__
                if "_kb_instance" in namespace:
                    checked += 1
                    assert namespace["_kb_instance"] is expected
        assert checked > 0
