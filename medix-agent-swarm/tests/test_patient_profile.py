import sys
from pathlib import Path
import pytest


PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from memory.patient_profile import PatientProfileStore


def test_explicit_user_facts_are_validated_and_persisted(tmp_path):
    store = PatientProfileStore(tmp_path)

    updates = store.apply_updates(
        "user-1",
        "我52岁，我是女性，我被医生确诊为2型糖尿病，目前服用二甲双胍，我对青霉素过敏。",
        "session-1",
        [
            {"category": "demographics", "field": "age", "value": "52", "status": "active", "evidence": "我52岁"},
            {"category": "demographics", "field": "sex", "value": "female", "status": "active", "evidence": "我是女性"},
            {"category": "conditions", "value": "2型糖尿病", "status": "active", "evidence": "我被医生确诊为2型糖尿病"},
            {"category": "medications", "value": "二甲双胍", "status": "active", "evidence": "目前服用二甲双胍"},
            {"category": "allergies", "value": "青霉素", "status": "active", "evidence": "我对青霉素过敏"},
        ],
    )

    profile = store.get_context("user-1")
    assert len(updates) == 5
    assert profile["demographics"]["age"]["value"] == 52
    assert profile["demographics"]["sex"]["value"] == "female"
    assert profile["conditions"][0]["value"] == "2型糖尿病"
    assert profile["medications"][0]["value"] == "二甲双胍"
    assert profile["allergies"][0]["value"] == "青霉素"
    assert profile["allergies"][0]["source"] == "user_reported"


def test_fact_updates_rewrite_status_instead_of_appending_duplicates(tmp_path):
    store = PatientProfileStore(tmp_path)
    store.apply_updates("user-2", "我目前服用二甲双胍。", "session-1", [
        {"category": "medications", "value": "二甲双胍", "status": "active", "evidence": "服用二甲双胍"}])

    store.apply_updates("user-2", "医生让我停了它。", "session-2", [
        {"category": "medications", "value": "二甲双胍", "status": "stopped", "evidence": "医生让我停了它"}])

    medications = store.get_context("user-2")["medications"]
    assert len(medications) == 1
    assert medications[0]["value"] == "二甲双胍"
    assert medications[0]["status"] == "stopped"
    assert medications[0]["source_text"] == "医生让我停了它"
    assert "source_session_id" not in medications[0]


def test_profiles_are_user_scoped_and_unquoted_evidence_is_rejected(tmp_path):
    store = PatientProfileStore(tmp_path)
    store.apply_updates("user-a", "我对青霉素过敏。", "session-a", [
        {"category": "allergies", "value": "青霉素", "status": "active", "evidence": "我对青霉素过敏"}])

    with pytest.raises(ValueError, match="verbatim"):
        store.apply_updates("user-b", "这种情况可能是高血压吗？", "session-b", [
            {"category": "conditions", "value": "高血压", "status": "active", "evidence": "我确诊了高血压"}])
    assert store.get_context("user-b") == {}
    assert store.get_context("user-a")["allergies"][0]["value"] == "青霉素"


@pytest.mark.parametrize("change", [{"user_id": "someone-else"}, {"category": "unknown"},
                                   {"status": "maybe"}, {"value": ""}, {"evidence": ""}])
def test_invalid_batch_is_atomic(tmp_path, change):
    store = PatientProfileStore(tmp_path)
    valid = {"category": "medications", "value": "某药", "status": "active", "evidence": "用户原话"}
    with pytest.raises(ValueError):
        store.apply_updates("u", "用户原话", "s", [valid, {**valid, **change}])
    assert store.get_context("u") == {}
    assert not list(tmp_path.glob("*.json"))


def test_lifestyle_restrictions_and_reaction_details_keep_sources(tmp_path):
    store = PatientProfileStore(tmp_path)
    text = "我轮班工作，腰伤后医生要求避免搬重物，我对青霉素过敏，服用后全身皮疹。"
    store.apply_updates("u", text, "s", [
        {"category": "lifestyle", "value": "轮班工作", "status": "active", "evidence": "我轮班工作"},
        {"category": "limitations", "value": "避免搬重物", "status": "active", "evidence": "腰伤后医生要求避免搬重物"},
        {"category": "allergies", "value": "青霉素", "status": "active", "evidence": "我对青霉素过敏，服用后全身皮疹"},
    ])
    profile = PatientProfileStore(tmp_path).get_context("u")
    assert profile["limitations"][0]["value"] == "避免搬重物"
    assert profile["allergies"][0]["source_text"].endswith("全身皮疹")
