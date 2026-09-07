"""Mem0-backed, user-scoped long-term conversation memory."""

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE_ROOT))

try:
    from config import MEM0_CONFIG
except ImportError:
    MEM0_CONFIG = {}

try:
    from mem0 import MemoryClient
except ImportError:
    MemoryClient = None


class LongTermMemoryError(RuntimeError):
    """Raised when an enabled Mem0 operation fails."""


class LongTermMemory:
    """Store and retrieve cross-session memories through Mem0 Platform."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        client: Optional[Any] = None,
    ):
        settings = dict(MEM0_CONFIG or {}) if config is None else dict(config)
        self.app_id = settings.get("app_id", "medix-agent-swarm")
        self.threshold = float(settings.get("threshold", 0.3))
        self.disabled_reason: Optional[str] = None

        if client is not None:
            self.client = client
            self.enabled = True
            return

        api_key = settings.get("api_key") or os.getenv("MEM0_API_KEY")
        if not api_key:
            self.client = None
            self.enabled = False
            self.disabled_reason = "MEM0_API_KEY is not configured"
            logger.info("Long-term memory disabled: MEM0_API_KEY is not configured")
            return
        if MemoryClient is None:
            self.client = None
            self.enabled = False
            self.disabled_reason = "mem0ai is not installed"
            logger.warning("Long-term memory disabled: install mem0ai")
            return

        self.client = MemoryClient(api_key=api_key)
        self.enabled = True
        logger.info("Long-term memory initialized with Mem0 Platform")

    def add_session_summary(
        self,
        user_id: str,
        session_id: str,
        question: str,
        answer: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Let Mem0 extract durable user memories from one completed turn."""
        if not self.enabled:
            return None
        self._validate_scope(user_id, session_id)

        memory_metadata = {
            "type": "conversation_turn",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **(metadata or {}),
        }
        try:
            result = self.client.add(
                messages=[
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ],
                user_id=user_id,
                app_id=self.app_id,
                run_id=session_id,
                metadata=memory_metadata,
            )
        except Exception as error:
            raise LongTermMemoryError(f"Mem0 add failed: {error}") from error

        operation_id = self._operation_id(result)
        logger.info(f"Submitted conversation turn to Mem0: {operation_id or 'accepted'}")
        return operation_id

    def search_similar_sessions(
        self,
        query: str,
        user_id: str,
        limit: int = 3,
    ) -> List[Dict[str, Any]]:
        """Search relevant memories for one user across previous sessions."""
        if not self.enabled:
            return []
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id must be a non-empty string")
        if limit < 1:
            raise ValueError("limit must be positive")

        filters = {
            "AND": [
                {"user_id": user_id},
                {"app_id": self.app_id},
            ]
        }
        try:
            result = self.client.search(
                query=query,
                filters=filters,
                top_k=limit,
                threshold=self.threshold,
            )
        except Exception as error:
            raise LongTermMemoryError(f"Mem0 search failed: {error}") from error

        items = result.get("results", []) if isinstance(result, dict) else result
        if not isinstance(items, list):
            return []

        memories = []
        seen_content = set()
        for item in items:
            if not isinstance(item, dict):
                continue
            content = str(item.get("memory") or item.get("text") or "").strip()
            if not content or content in seen_content:
                continue
            seen_content.add(content)
            memories.append({
                "memory_id": str(item.get("id", "")),
                "content": content,
                "score": item.get("score"),
                "metadata": item.get("metadata") or {},
                "timestamp": item.get("created_at"),
            })

        logger.info(f"Retrieved {len(memories)} long-term memories")
        return memories[:limit]

    @staticmethod
    def _validate_scope(user_id: str, session_id: str) -> None:
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id must be a non-empty string")
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be a non-empty string")

    @staticmethod
    def _operation_id(result: Any) -> Optional[str]:
        if not isinstance(result, dict):
            return None
        operation_id = result.get("event_id") or result.get("id")
        if operation_id:
            return str(operation_id)
        results = result.get("results")
        if isinstance(results, list) and results and isinstance(results[0], dict):
            memory_id = results[0].get("id")
            return str(memory_id) if memory_id else None
        return None
