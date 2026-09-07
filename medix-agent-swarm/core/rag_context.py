"""Keep retrieved bodies once in the currently visible, worker-local messages."""

import copy
import hashlib
import json
from typing import Any


def document_block(document: dict[str, Any], max_chars: int | None = None) -> dict[str, Any]:
    """Identify the exact excerpt, including its source metadata and revision."""
    block = {
        "document_id": str(document["id"]),
        "content": document["content"][:max_chars],
        "metadata": copy.deepcopy(document["metadata"]),
    }
    block["block_id"] = _block_id(block)
    block["score"] = document["score"]
    return block


def _block_id(block: dict[str, Any]) -> str:
    identity = {key: block[key] for key in ("document_id", "content", "metadata")}
    serialized = json.dumps(identity, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def compact_rag_result(
    result: dict[str, Any], messages: list[dict[str, Any]], tool_call_id: str,
) -> dict[str, Any]:
    """Only reference bodies still present in tool messages; never mutate raw results."""
    visible: dict[str, str] = {}
    for message in messages:
        if message.get("role") != "tool":
            continue
        try:
            payload = json.loads(message["content"])
        except json.JSONDecodeError:
            continue  # Plain-text tool results contain no structured RAG blocks.
        if not isinstance(payload, dict):
            continue
        for block in payload.get("documents", []):
            if "content" in block and _block_id(block) == block["block_id"]:
                visible[block["block_id"]] = message["tool_call_id"]

    compacted = copy.deepcopy(result)
    documents = []
    for block in compacted.get("documents", []):
        block_id = block["block_id"]
        if block_id in visible:
            reference = {key: value for key, value in block.items()
                         if key not in {"content", "metadata", "score"}}
            reference["reference"] = {"tool_call_id": visible[block_id]}
            documents.append(reference)
        else:
            documents.append(block)
            visible[block_id] = tool_call_id
    if "documents" in compacted:
        compacted["documents"] = documents
    return compacted
