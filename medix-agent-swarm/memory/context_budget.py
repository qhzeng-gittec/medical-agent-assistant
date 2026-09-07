"""Bound the recent transcript without mixing it with durable user facts."""

from typing import Any, Dict, List


class RecentHistoryBudget:
    """Keep complete recent turns under both message and character budgets."""

    def __init__(self, max_turns: int = 5, max_chars: int = 8000):
        if max_turns < 1:
            raise ValueError("max_turns must be positive")
        if max_chars < 1:
            raise ValueError("max_chars must be positive")
        self.max_turns = max_turns
        self.max_chars = max_chars

    def select(self, messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        turns: List[List[Dict[str, str]]] = []
        current: List[Dict[str, str]] = []
        for message in messages:
            role = str(message.get("role", ""))
            if role not in {"user", "assistant"}:
                continue
            normalized = {"role": role, "content": str(message.get("content", ""))}
            if role == "user" and current:
                turns.append(current)
                current = []
            current.append(normalized)
        if current:
            turns.append(current)

        selected: List[List[Dict[str, str]]] = []
        used_chars = 0
        for turn in reversed(turns[-self.max_turns:]):
            turn_chars = sum(len(message["content"]) for message in turn)
            if selected and used_chars + turn_chars > self.max_chars:
                break
            if not selected and turn_chars > self.max_chars:
                selected.append(self._truncate_latest_turn(turn))
                break
            selected.append(turn)
            used_chars += turn_chars

        return [message for turn in reversed(selected) for message in turn]

    def _truncate_latest_turn(self, turn: List[Dict[str, str]]) -> List[Dict[str, str]]:
        per_message = max(1, self.max_chars // len(turn))
        return [
            {"role": message["role"], "content": message["content"][:per_message]}
            for message in turn
        ]
