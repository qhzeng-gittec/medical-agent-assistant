"""Local Mem0 OSS storage with API-based extraction and dense embeddings."""

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

from loguru import logger


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE_ROOT))

try:
    from config import MEM0_CONFIG
except ImportError:
    MEM0_CONFIG = {}

MEMORY_INSTRUCTIONS = (
    "Remember useful consultation events and reported changes, with the speaker, event time when stated, "
    "and unresolved follow-up. Preserve negation and uncertainty, and identify who each event concerns. "
    "Distinguish the user's reports from the assistant's hypotheses and recommendations; "
    "the latter are not confirmed patient facts. Omit generic explanations and routine acknowledgements."
    " Write in the user's language. Do not add dates, frequencies, or other details not stated by the user."
)


class LongTermMemoryError(RuntimeError):
    """Raised when an enabled Mem0 operation fails."""


class LongTermMemory:
    """Store events on disk; retrieve by API embeddings within one user/app scope.

    Share one instance per local storage directory while it is open. Call close()
    before reopening that directory or shutting down its owning application.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        client: Optional[Any] = None,
    ):
        settings = dict(MEM0_CONFIG or {}) if config is None else dict(config)
        self.app_id = settings.get("app_id", "medix-agent-swarm")
        self.threshold = float(settings.get("threshold", 0.3))
        self.disabled_reason: Optional[str] = None
        self._lock = Lock()
        self._owns_client = client is None

        if client is not None:
            self.client = client
            self.enabled = True
            return

        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            self.client = None
            self.enabled = False
            self.disabled_reason = "OPENROUTER_API_KEY is not configured"
            logger.info("Long-term memory disabled: OPENROUTER_API_KEY is not configured")
            return

        storage_path = Path(os.getenv("MEM0_LOCAL_PATH") or settings.get("storage_path", ".mem0"))
        if not storage_path.is_absolute():
            storage_path = Path(__file__).resolve().parents[1] / storage_path
        self.storage_path = storage_path.resolve()
        self.storage_path.mkdir(parents=True, exist_ok=True)
        os.environ["MEM0_TELEMETRY"] = "false"
        os.environ.setdefault("MEM0_DIR", str(self.storage_path))

        from mem0 import Memory
        from qdrant_client import QdrantClient

        # The supervisor uses asyncio.to_thread. Serialize access to this local
        # client with _lock instead of binding its SQLite connection to one thread.
        vector_client = QdrantClient(
            path=str(self.storage_path / "qdrant"),
            force_disable_check_same_thread=True,
        )
        dimensions = int(settings.get("embedding_dims", 4096))
        base_url = "https://openrouter.ai/api/v1"
        try:
            self.client = Memory.from_config({
                "vector_store": {"provider": "qdrant", "config": {
                    "client": vector_client,
                    "path": str(self.storage_path / "qdrant"),
                    "collection_name": "consultation_events",
                    "embedding_model_dims": dimensions,
                    "on_disk": True,
                }},
                "embedder": {"provider": "openai", "config": {
                    "api_key": api_key,
                    "openai_base_url": base_url,
                    "model": settings.get("embedding_model", "qwen/qwen3-embedding-8b"),
                    "embedding_dims": dimensions,
                }},
                "llm": {"provider": "openai", "config": {
                    "api_key": api_key,
                    "openai_base_url": base_url,
                    "openrouter_base_url": base_url,
                    "model": settings.get("llm_model", "qwen/qwen3.5-27b"),
                    "temperature": 0.1,
                    "max_tokens": 8192,
                }},
                "history_db_path": str(self.storage_path / "history.sqlite3"),
                "custom_instructions": MEMORY_INSTRUCTIONS,
            })
        except Exception:
            vector_client.close()
            raise
        self.enabled = True
        logger.info(f"Long-term memory initialized with local Mem0 OSS: {self.storage_path}")

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
            **(metadata or {}),
            "type": "consultation_event",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source_session_id": session_id,
        }
        try:
            with self._lock:
                result = self.client.add(
                    messages=[
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer},
                    ],
                    user_id=user_id,
                    agent_id=self.app_id,
                    # run_id would isolate extraction to one session. Keep the
                    # session as provenance, not part of the cross-session scope.
                    metadata=memory_metadata,
                )
        except Exception as error:
            raise LongTermMemoryError(f"Mem0 add failed: {error}") from error

        operation_id = self._operation_id(result)
        logger.info(f"Processed conversation turn with local Mem0: {operation_id or 'no new memory'}")
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

        try:
            with self._lock:
                vector = self.client.embedding_model.embed(query, "search")
                # Mem0 2.0.20 Memory.search adds BM25/entity scoring. This
                # application explicitly uses its dense vector-store interface.
                items = self.client.vector_store.search(
                    query=query,
                    vectors=vector,
                    filters={"user_id": user_id, "agent_id": self.app_id},
                    top_k=limit,
                )
        except Exception as error:
            raise LongTermMemoryError(f"Mem0 search failed: {error}") from error

        memories = []
        seen_content = set()
        for item in items:
            if item.score < self.threshold:
                continue
            payload = item.payload
            content = payload.get("data", "").strip()
            if not content or content in seen_content:
                continue
            seen_content.add(content)
            memories.append({
                "memory_id": str(item.id),
                "content": content,
                "score": item.score,
                "metadata": {
                    key: value for key, value in payload.items()
                    if key not in {"data", "hash", "text_lemmatized", "created_at", "updated_at"}
                },
                "timestamp": payload.get("created_at"),
            })

        logger.info(f"Retrieved {len(memories)} long-term memories")
        return memories[:limit]

    def close(self) -> None:
        """Release owned SQLite, Qdrant and HTTP clients, including the disk lock."""
        with self._lock:
            if self.enabled and self._owns_client:
                try:
                    self.client.close()
                finally:
                    self.client.vector_store.client.close()
                    self.client.embedding_model.client.close()
                    self.client.llm.client.close()
            self.enabled = False

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
        results = result.get("results")
        if isinstance(results, list) and results and isinstance(results[0], dict):
            memory_id = results[0].get("id")
            return str(memory_id) if memory_id else None
        return None
