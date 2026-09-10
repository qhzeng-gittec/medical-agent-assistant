"""Exact-call cache scoped to one worker Agent invocation."""

import asyncio
import copy
import json
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Awaitable, Callable, Dict, Iterator, Optional


EVIDENCE_SKILLS = frozenset({
    "search_knowledge",
    "clinical_guideline",
    "deep_research",
    "recommend_lifestyle",
    "analyze_symptoms",
    "disease_code",
})


class EvidenceStore:
    """Deduplicate identical evidence-producing calls within one Agent loop."""

    def __init__(self, namespace: str = "medical-evidence-v1"):
        self.namespace = namespace
        self._results: Dict[str, Any] = {}
        self._inflight: Dict[str, asyncio.Task] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def execute(
        self,
        skill_name: str,
        arguments: Dict[str, Any],
        consumer: str,
        executor: Callable[[], Awaitable[Any]],
    ) -> Any:
        key = self._key(skill_name, arguments)
        async with self._lock:
            if key in self._results:
                self._record_consumer(key, consumer, reused=True)
                return copy.deepcopy(self._results[key])

            task = self._inflight.get(key)
            creator = task is None
            if creator:
                task = asyncio.create_task(executor())
                self._inflight[key] = task
                self._metadata[key] = {
                    "evidence_id": f"evidence-{len(self._metadata) + 1}",
                    "skill": skill_name,
                    "arguments": copy.deepcopy(arguments),
                    "first_consumer": consumer,
                    "consumers": [consumer],
                    "cache_hits": 0,
                }

        try:
            result = await task
        except Exception:
            if creator:
                async with self._lock:
                    self._inflight.pop(key, None)
                    self._metadata.pop(key, None)
            raise

        async with self._lock:
            if creator:
                self._results[key] = copy.deepcopy(result)
                self._inflight.pop(key, None)
            else:
                self._record_consumer(key, consumer, reused=True)
        return copy.deepcopy(result)

    def summary(self) -> Dict[str, Any]:
        entries = [copy.deepcopy(item) for item in self._metadata.values()]
        return {
            "executed_calls": len(self._results),
            "cache_hits": sum(item["cache_hits"] for item in entries),
            "entries": entries,
        }

    def _record_consumer(self, key: str, consumer: str, reused: bool) -> None:
        metadata = self._metadata[key]
        if consumer not in metadata["consumers"]:
            metadata["consumers"].append(consumer)
        if reused:
            metadata["cache_hits"] += 1

    def _key(self, skill_name: str, arguments: Dict[str, Any]) -> str:
        payload = json.dumps(arguments, ensure_ascii=False, sort_keys=True, default=str)
        return f"{self.namespace}:{skill_name}:{payload}"


_CURRENT_EVIDENCE_STORE: ContextVar[Optional[EvidenceStore]] = ContextVar(
    "current_evidence_store",
    default=None,
)


def current_evidence_store() -> Optional[EvidenceStore]:
    return _CURRENT_EVIDENCE_STORE.get()


@contextmanager
def use_evidence_store(store: EvidenceStore) -> Iterator[None]:
    token = _CURRENT_EVIDENCE_STORE.set(store)
    try:
        yield
    finally:
        _CURRENT_EVIDENCE_STORE.reset(token)
