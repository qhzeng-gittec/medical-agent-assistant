import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from memory import RecentHistoryBudget


def test_recent_history_keeps_complete_latest_turns():
    messages = []
    for index in range(8):
        messages.extend([
            {"role": "user", "content": f"question-{index}"},
            {"role": "assistant", "content": f"answer-{index}"},
        ])

    selected = RecentHistoryBudget(max_turns=3, max_chars=1000).select(messages)

    assert [item["content"] for item in selected] == [
        "question-5", "answer-5",
        "question-6", "answer-6",
        "question-7", "answer-7",
    ]


def test_recent_history_drops_whole_old_turns_to_meet_character_budget():
    messages = [
        {"role": "user", "content": "a" * 10},
        {"role": "assistant", "content": "b" * 10},
        {"role": "user", "content": "c" * 10},
        {"role": "assistant", "content": "d" * 10},
    ]

    selected = RecentHistoryBudget(max_turns=5, max_chars=25).select(messages)

    assert selected == messages[-2:]


def test_latest_oversized_turn_is_truncated_but_not_dropped():
    messages = [
        {"role": "user", "content": "u" * 100},
        {"role": "assistant", "content": "a" * 100},
    ]

    selected = RecentHistoryBudget(max_turns=5, max_chars=40).select(messages)

    assert [len(item["content"]) for item in selected] == [20, 20]
