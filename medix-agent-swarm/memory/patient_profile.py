"""User-scoped persistence for model-proposed, source-linked profile updates."""

import hashlib
import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class PatientProfileError(RuntimeError):
    """Raised when a patient profile cannot be loaded or persisted."""


PROFILE_UPDATE_TOOL = {
    "type": "function",
    "function": {
        "name": "update_patient_profile",
        "description": "保存本轮用户明确提供或更正的本人健康信息。结合已有档案理解更新，保留否定和停用状态；证据必须摘自本轮用户原话。提问、假设、引用练习和他人信息不构成本人事实。",
        "parameters": {
            "type": "object", "additionalProperties": False,
            "properties": {"updates": {"type": "array", "minItems": 1, "maxItems": 20, "items": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "category": {"type": "string", "enum": ["demographics", "conditions", "medications", "allergies", "lifestyle", "limitations"]},
                    "field": {"type": "string", "enum": ["age", "sex"], "description": "仅 demographics 使用"},
                    "value": {"type": "string", "description": "事实或已有条目的名称；年龄填数字文本，性别填 male 或 female", "maxLength": 160},
                    "status": {"type": "string", "enum": ["active", "stopped", "denied"]},
                    "evidence": {"type": "string", "description": "支持此次更新的本轮用户原话片段", "maxLength": 1000},
                }, "required": ["category", "value", "status", "evidence"],
            }}}, "required": ["updates"],
        },
    },
}


class PatientProfileStore:
    """Persist stable user-reported facts separately from conversation history."""

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = Path(base_dir or Path(__file__).parent / "data" / "profiles")
        self._lock = threading.RLock()

    def apply_updates(
        self,
        user_id: str,
        message: str,
        session_id: str,
        updates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Validate the entire proposal before writing; semantic selection belongs to the model."""
        self._validate_scope(user_id, session_id)
        if not isinstance(message, str) or not message.strip():
            raise ValueError("message must be a non-empty string")

        if not isinstance(updates, list) or not 1 <= len(updates) <= 20:
            raise ValueError("updates must contain 1 to 20 facts")
        validated = []
        for update in updates:
            if not isinstance(update, dict) or set(update) - {"category", "field", "value", "status", "evidence"}:
                raise ValueError("Unexpected profile update fields")
            category, value = update.get("category"), update.get("value")
            evidence, status = update.get("evidence"), update.get("status")
            if category not in {"demographics", "conditions", "medications", "allergies", "lifestyle", "limitations"}:
                raise ValueError("Invalid profile category")
            if not isinstance(value, str) or not value.strip() or len(value) > 160:
                raise ValueError("Profile value must be non-empty text of at most 160 characters")
            if status not in {"active", "stopped", "denied"}:
                raise ValueError("Invalid profile status")
            if not isinstance(evidence, str) or not evidence.strip() or len(evidence) > 1000 or evidence not in message:
                raise ValueError("Evidence must be a verbatim excerpt of the current user message")
            fact = {"category": category, "value": value.strip(), "status": status,
                    "source": "user_reported", "source_text": evidence,
                    "source_session_id": session_id, "updated_at": self._now()}
            if category == "demographics":
                field = update.get("field")
                if status != "active" or field not in {"age", "sex"}:
                    raise ValueError("Demographics require age/sex and active status")
                if field == "age":
                    if not value.isdecimal() or not 0 < int(value) <= 120:
                        raise ValueError("Invalid age")
                    fact["value"] = int(value)
                elif value not in {"male", "female"}:
                    raise ValueError("Invalid sex value")
                fact["field"] = field
            elif "field" in update:
                raise ValueError("field is only valid for demographics")
            validated.append(fact)

        with self._lock:
            profile = self._load(user_id)
            for update in validated:
                self._merge_update(profile, update)
            profile["updated_at"] = self._now()
            self._write(user_id, profile)
        return validated

    def get_context(self, user_id: str) -> Dict[str, Any]:
        """Return the bounded profile view supplied to the Supervisor."""
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id must be a non-empty string")
        with self._lock:
            profile = self._load(user_id)

        context: Dict[str, Any] = {}
        if profile["demographics"]:
            context["demographics"] = {
                field: self._context_fact(fact)
                for field, fact in profile["demographics"].items()
            }
        for category in ("conditions", "medications", "allergies", "lifestyle", "limitations"):
            if profile.get(category):
                context[category] = [
                    self._context_fact(fact)
                    for fact in profile[category][-20:]
                ]
        return context

    def _merge_update(self, profile: Dict[str, Any], update: Dict[str, Any]) -> None:
        category = update["category"]
        if category == "demographics":
            profile[category][update["field"]] = {
                key: value for key, value in update.items()
                if key not in {"category", "field"}
            }
            return

        normalized = self._normalize(update["value"])
        facts = profile.setdefault(category, [])
        existing = next(
            (fact for fact in facts if self._normalize(fact["value"]) == normalized),
            None,
        )
        payload = {key: value for key, value in update.items() if key != "category"}
        if existing is None:
            facts.append(payload)
        else:
            existing.update(payload)

    def _load(self, user_id: str) -> Dict[str, Any]:
        path = self._path(user_id)
        if not path.exists():
            return {
                "schema_version": 1,
                "user_id": user_id,
                "demographics": {},
                "conditions": [],
                "medications": [],
                "allergies": [],
                "lifestyle": [],
                "limitations": [],
                "updated_at": None,
            }
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise PatientProfileError(f"Patient profile load failed: {error}") from error
        if data.get("user_id") != user_id:
            raise PatientProfileError("Patient profile user scope mismatch")
        return data

    def _write(self, user_id: str, profile: Dict[str, Any]) -> None:
        path = self._path(user_id)
        temp_path = path.with_suffix(".tmp")
        try:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            temp_path.write_text(
                json.dumps(profile, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temp_path.replace(path)
        except OSError as error:
            raise PatientProfileError(f"Patient profile write failed: {error}") from error

    def _path(self, user_id: str) -> Path:
        digest = hashlib.sha256(user_id.encode("utf-8")).hexdigest()
        return self.base_dir / f"{digest}.json"

    @staticmethod
    def _context_fact(fact: Dict[str, Any]) -> Dict[str, Any]:
        return {key: value for key, value in fact.items() if key in {"value", "status", "source", "source_text"}}

    @staticmethod
    def _normalize(value: str) -> str:
        return re.sub(r"\s+", "", value).lower()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _validate_scope(user_id: str, session_id: str) -> None:
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id must be a non-empty string")
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be a non-empty string")
