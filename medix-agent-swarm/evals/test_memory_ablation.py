import asyncio

import pytest

from campaign_memory_ablation import ARMS, FrozenRecall, NoProfile
from campaign_mem0_repeat_report import summarize
from memory import PatientProfileStore
from swarm.supervisor_agent import MedicalSupervisorAgent, SUBAGENT_TOOLS


def test_frozen_recall_is_scope_bound_and_does_not_leak_mutation():
    trace = []
    source = [{"content": "user fact", "score": .5}]
    memory = FrozenRecall(True, "query", "user-a", source, trace)
    source[0]["content"] = "changed"
    result = memory.search_similar_sessions("query", "user-a")
    result[0]["content"] = "changed again"
    assert memory.search_similar_sessions("query", "user-a")[0]["content"] == "user fact"
    with pytest.raises(ValueError, match="scope/query"):
        memory.search_similar_sessions("query", "user-b")
    assert memory.add_session_summary() is None
    assert trace[-1]["event"] == "memory_write_suppressed"


@pytest.mark.parametrize("arm", ARMS)
def test_four_arms_provide_only_the_selected_memory_channels(tmp_path, arm):
    question, user = "我的过敏史是什么？", "synthetic-user"
    profile = PatientProfileStore(tmp_path)
    if arm.startswith("profile"):
        profile.apply_updates(user, "我对青霉素过敏。", "old-session", [
            {"category": "allergies", "value": "青霉素", "status": "active", "evidence": "我对青霉素过敏。"}])
    else:
        profile = NoProfile()
    memory = FrozenRecall("mem0_replay" in arm, question, user,
                          [{"content": "Prior penicillin allergy", "score": .5}], [])
    supervisor = MedicalSupervisorAgent(
        llm_client=object(), workers={t["function"]["name"]: object() for t in SUBAGENT_TOOLS},
        patient_profiles=profile, long_term_memory=memory)
    context, warnings, _ = asyncio.run(supervisor._build_context(question, None, "new-session", user))
    assert ("patient_profile" in context) == arm.startswith("profile")
    assert ("historical_memories" in context) == ("mem0_replay" in arm)
    assert "recent_history" not in context
    assert not warnings


def test_repeated_report_does_not_turn_empty_anchor_lists_or_errors_into_passes():
    query = {"query": {"id": "contamination", "query": "history?", "required_terms": []},
             "storage_anchors": [], "baseline_anchors": [], "top10_anchors": [],
             "threshold0_anchors": [], "baseline": [{"content": "assistant's invented claim"}]}
    result = summarize([("r1", [{"scenario": {"id": "M07"}, "status": "completed", "stored": [], "queries": [query]},
                                 {"scenario": {"id": "M08"}, "status": "error"}])])
    row = result["repetitions"][0]
    assert row["completed_scenarios"] == 1
    assert row["positive_queries"] == row["baseline_anchor_hits"] == 0
    assert row["statuses"]["error"] == 1
    assert result["queries"]["M07/contamination"]["observations"][0]["baseline_text"]
