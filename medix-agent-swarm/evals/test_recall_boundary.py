import json

import pytest

import campaign_mem0_boundary as mem0_eval
from campaign_rag_boundary import CASES, DATA, coverage, summarize


def test_rag_boundary_labels_are_frozen_valid_documents():
    cases = [json.loads(s) for s in CASES.read_text(encoding="utf-8").splitlines()]
    ids = {json.loads(s)["id"] for s in (DATA / "corpus.jsonl").read_text(encoding="utf-8").splitlines()}
    assert len(cases) == len({c["id"] for c in cases}) == 60
    assert all(set(g) <= ids for c in cases for g in c["gold_groups"])
    assert all(c["gold_groups"] for c in cases if c["answerability"] == "answerable")
    assert not any(c["gold_groups"] for c in cases if c["answerability"] == "unanswerable")


def test_multiple_evidence_groups_cannot_pass_on_one_hit():
    check = coverage([["a", "equivalent-a"], ["b"]], [{"id": "a", "score": .8}])
    assert check["any_hit"] is True
    assert check["all_hit"] is False
    assert check["group_recall"] == .5
    assert coverage([["a"]], [{"id": "a", "score": .2}], threshold=.3)["any_hit"] is False


def test_rag_summary_separates_missing_history_and_execution_errors():
    rows = [{"status": "completed", "case": {"group": "x", "gold_groups": [["a"]], "answerability": kind},
             "documents": [{"id": "a", "score": .5}]} for kind in ("answerable", "needs_history", "filtered_out")]
    rows += [{"status": "error", "case": {"id": "timeout"}, "error": "ReadTimeout"}]
    result = summarize(rows)
    assert result["answerable_n"] == 1
    assert result["completed"] == 3
    assert result["threshold_sweep"][0]["positive_n"] == 1
    assert result["errors"] == [{"id": "timeout", "error": "ReadTimeout"}]


def test_mem0_anchor_checks_accept_language_variants_but_not_missing_facts():
    required = [["青霉素", "penicillin"], ["皮疹", "rash"]]
    assert mem0_eval.anchor_checks(required, [{"memory": "Penicillin allergy causing rash"}]) == [True, True]
    assert mem0_eval.anchor_checks(required, [{"content": "青霉素过敏"}]) == [True, False]
    assert mem0_eval.anchor_checks(required, []) == [False, False]


def test_mem0_recorded_write_is_never_blindly_repeated(tmp_path, monkeypatch):
    monkeypatch.setattr(mem0_eval, "ROOT", tmp_path)
    recorder = mem0_eval.Recorder("fake-secret")
    calls = []
    fn = lambda: calls.append(1) or {"event_id": "id1", "status": "PENDING"}
    assert recorder.call("write", {"user": "synthetic"}, fn)["event_id"] == "id1"
    recorder.call("write", {"user": "synthetic"}, fn)
    assert calls == [1]
    with pytest.raises(ValueError, match="Parameters changed"):
        recorder.call("write", {"user": "changed"}, fn)


def test_mem0_incomplete_write_is_not_a_success_and_secret_is_redacted(tmp_path, monkeypatch):
    monkeypatch.setattr(mem0_eval, "ROOT", tmp_path)
    recorder = mem0_eval.Recorder("fake-secret")

    def fail():
        raise TimeoutError("failure with fake-secret")

    with pytest.raises(TimeoutError):
        recorder.call("write", {}, fail)
    saved = mem0_eval.read(tmp_path / "operations" / "write.json")
    assert saved["status"] == "error"
    assert "fake-secret" not in saved["error"]
    with pytest.raises(RuntimeError, match="Unresolved prior"):
        recorder.call("write", {}, fail)
