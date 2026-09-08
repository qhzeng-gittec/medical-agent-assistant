"""Validator tests use public toy records, never the sealed consultation cases."""

import importlib.util
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location(
    "holdout_preparation", Path(__file__).parent / "holdout_v1_20260908/prepare_holdout.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def toy_case():
    return {
        "case_id": "TOY-001", "family_id": "TOY-F001", "domain": "history",
        "tags": ["toy", "format"], "difficulty": "routine",
        "interaction": {"opening": "Please summarize my appointment note.",
                        "max_patient_replies": 0, "scripted_followups": []},
        "private": {
            "patient_facts": [], "seed_history": [],
            "environment": {"rag_mode": "not_needed", "memory_mode": "off",
                            "documents": [], "memory_records": []},
            "checkpoints": [{"id": f"c{i}", "after": "final", "target": "answer",
                             "requirement": "PRIVATE_TEST_CANARY", "critical": False} for i in range(3)],
            "source_refs": [], "clinical_review_required": False,
        },
        "provenance": {"author_bucket": "history", "synthetic": True,
                       "parent_history_inherited": False, "implementation_seen": False,
                       "existing_results_seen": False},
    }


def test_valid_toy_structure():
    MODULE.validate_case(toy_case(), "history")


def test_private_field_cannot_be_added_to_interaction():
    case = toy_case()
    case["interaction"]["checkpoints"] = case["private"]["checkpoints"]
    with pytest.raises(ValueError, match="interaction fields"):
        MODULE.validate_case(case, "history")


def test_author_exposure_is_rejected():
    case = toy_case()
    case["provenance"]["existing_results_seen"] = True
    with pytest.raises(ValueError, match="isolation attestation"):
        MODULE.validate_case(case, "history")


def test_source_must_reference_existing_checkpoint():
    case = toy_case()
    case["private"]["source_refs"] = [{"url": "https://example.org", "title": "Toy",
                                       "supports": ["missing"], "note": "PRIVATE_TEST_CANARY"}]
    with pytest.raises(ValueError) as error:
        MODULE.validate_case(case, "history")
    assert "PRIVATE_TEST_CANARY" not in str(error.value)
    assert "source checkpoint references" in str(error.value)


def test_adaptive_patient_requires_supplied_facts():
    case = toy_case()
    case["interaction"]["max_patient_replies"] = 1
    with pytest.raises(ValueError, match="adaptive patient facts"):
        MODULE.validate_case(case, "history")


def test_history_identity_is_restricted():
    case = toy_case()
    case["private"]["seed_history"] = [{"user_key": "unbounded-user", "session": "s", "messages": []}]
    with pytest.raises(ValueError, match="history scope"):
        MODULE.validate_case(case, "history")


def test_exact_duplicate_normalization_is_not_semantic_grading():
    assert MODULE.normalized("ＡＢＣ， 信息！") == MODULE.normalized("abc信息")
    assert MODULE.normalized("I am not ill") != MODULE.normalized("I am ill")
