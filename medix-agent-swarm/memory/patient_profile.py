"""User-scoped, structured medical facts with conservative extraction."""

import hashlib
import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class PatientProfileError(RuntimeError):
    """Raised when a patient profile cannot be loaded or persisted."""


class PatientProfileStore:
    """Persist stable user-reported facts separately from conversation history."""

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = Path(base_dir or Path(__file__).parent / "data" / "profiles")
        self._lock = threading.RLock()

    def update_from_user_message(
        self,
        user_id: str,
        message: str,
        session_id: str,
    ) -> List[Dict[str, Any]]:
        """Extract only explicit first-person facts and merge them into the profile."""
        self._validate_scope(user_id, session_id)
        if not isinstance(message, str) or not message.strip():
            raise ValueError("message must be a non-empty string")

        updates = self._extract_explicit_facts(message, session_id)
        if not updates:
            return []

        with self._lock:
            profile = self._load(user_id)
            for update in updates:
                self._merge_update(profile, update)
            profile["updated_at"] = self._now()
            self._write(user_id, profile)
        return updates

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
        for category in ("conditions", "medications", "allergies"):
            if profile[category]:
                context[category] = [
                    self._context_fact(fact)
                    for fact in profile[category][-20:]
                ]
        return context

    def _extract_explicit_facts(self, message: str, session_id: str) -> List[Dict[str, Any]]:
        text = message.strip()
        timestamp = self._now()
        source = {
            "source": "user_reported",
            "source_session_id": session_id,
            "source_text": text[:200],
            "updated_at": timestamp,
        }
        updates: List[Dict[str, Any]] = []

        age_match = re.search(r"(?:我(?:今年)?是?|本人)?\s*(\d{1,3})\s*岁", text)
        if age_match and 0 < int(age_match.group(1)) <= 120:
            updates.append({"category": "demographics", "field": "age", "value": int(age_match.group(1)), **source})

        if re.search(r"(?:我是|本人为|性别[:：]?\s*)(?:一名)?(?:女性|女)", text):
            updates.append({"category": "demographics", "field": "sex", "value": "female", **source})
        elif re.search(r"(?:我是|本人为|性别[:：]?\s*)(?:一名)?(?:男性|男)", text):
            updates.append({"category": "demographics", "field": "sex", "value": "male", **source})

        for match in re.finditer(r"(?:我|本人)(?:被医生)?(?:已经)?(?:确诊(?:为|了)?|患有|得了)([^，。；;！？!?]{1,24})", text):
            value = self._clean_value(match.group(1))
            if value:
                updates.append({"category": "conditions", "value": value, "status": "active", **source})

        for match in re.finditer(r"(?:我|本人)?(?:已经)?(?:停用|停服|不再服用|不再吃)([^，。；;、！？!?]{1,24})", text):
            value = self._clean_value(match.group(1))
            if value:
                updates.append({"category": "medications", "value": value, "status": "stopped", **source})

        for match in re.finditer(r"(?:我|本人)?(?:目前|现在|平时|正在)?(?:在)?(?:服用|吃|使用)([^，。；;、！？!?]{1,24})", text):
            value = self._clean_value(match.group(1))
            if value and not re.search(rf"(?:停用|停服|不再服用|不再吃){re.escape(match.group(1))}", text):
                updates.append({"category": "medications", "value": value, "status": "active", **source})

        for match in re.finditer(r"(?:我|本人)?对([^，。；;、！？!?]{1,20})(?:不过敏|没有过敏)", text):
            value = self._clean_value(match.group(1))
            if value:
                updates.append({"category": "allergies", "value": value, "status": "denied", **source})

        for match in re.finditer(r"(?:我|本人)?对([^，。；;、！？!?]{1,20})过敏", text):
            value = self._clean_value(match.group(1))
            if value and "不过敏" not in match.group(0) and "没有过敏" not in match.group(0):
                updates.append({"category": "allergies", "value": value, "status": "active", **source})

        return updates

    def _merge_update(self, profile: Dict[str, Any], update: Dict[str, Any]) -> None:
        category = update["category"]
        if category == "demographics":
            profile[category][update["field"]] = {
                key: value for key, value in update.items()
                if key not in {"category", "field"}
            }
            return

        normalized = self._normalize(update["value"])
        facts = profile[category]
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
        return {key: value for key, value in fact.items() if key != "source_text"}

    @staticmethod
    def _clean_value(value: str) -> str:
        return value.strip(" \t\r\n的了")

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
