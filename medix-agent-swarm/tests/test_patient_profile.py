import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from memory.patient_profile import PatientProfileStore


def test_explicit_user_facts_are_extracted_and_persisted(tmp_path):
    store = PatientProfileStore(tmp_path)

    updates = store.update_from_user_message(
        "user-1",
        "我52岁，我是女性，我被医生确诊为2型糖尿病，目前服用二甲双胍，我对青霉素过敏。",
        "session-1",
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
    store.update_from_user_message("user-2", "我目前服用二甲双胍。", "session-1")

    store.update_from_user_message("user-2", "我已经停用二甲双胍了。", "session-2")

    medications = store.get_context("user-2")["medications"]
    assert len(medications) == 1
    assert medications[0]["value"] == "二甲双胍"
    assert medications[0]["status"] == "stopped"
    assert medications[0]["source_session_id"] == "session-2"


def test_profiles_are_user_scoped_and_inferred_diagnoses_are_ignored(tmp_path):
    store = PatientProfileStore(tmp_path)
    store.update_from_user_message("user-a", "我对青霉素过敏。", "session-a")

    updates = store.update_from_user_message(
        "user-b",
        "这种情况可能是高血压吗？",
        "session-b",
    )

    assert updates == []
    assert store.get_context("user-b") == {}
    assert store.get_context("user-a")["allergies"][0]["value"] == "青霉素"
