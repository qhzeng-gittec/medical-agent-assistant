"""Wire shape tests with public toy evidence, never semantic score rewriting."""

import copy

import pytest

from holdout_grading_pipeline import normalize_evidence_wire
from holdout_grade import validate_verdict


def test_wire_normalization_preserves_verdict_reason_and_ids():
    original = {"checks": [{"id": "toy", "verdict": "fail", "reason": "Observed omission",
                            "evidence": ["t1.answer", {"id": "t1.profile_after"}]}]}
    before = copy.deepcopy(original)
    converted, count = normalize_evidence_wire(original)
    assert original == before
    assert count == 1
    assert converted["checks"][0] == {**original["checks"][0],
                                     "evidence": [{"id": "t1.answer"}, {"id": "t1.profile_after"}]}


def test_unknown_id_is_not_invented_or_repaired():
    verdict = {"checks": [{"id": "toy", "verdict": "pass", "reason": "test",
                            "requires_durable_write": False, "evidence": ["future.answer"]}]}
    converted, _ = normalize_evidence_wire(verdict)
    with pytest.raises(ValueError, match="not in the observed catalogue"):
        validate_verdict(converted, [{"id": "toy"}], {"evidence_catalogue": [], "observations": {"turns": []}})


def test_no_semantic_inference_or_missing_evidence_invention():
    verdict = {"checks": [{"id": "toy", "verdict": "uncertain", "evidence": []}]}
    assert normalize_evidence_wire(verdict) == (verdict, 0)
