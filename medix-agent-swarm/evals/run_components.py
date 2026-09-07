"""Execute dataset-driven local RAG contracts, with no LLM or external services."""

import argparse
import asyncio
import importlib.util
import json
from pathlib import Path

from dataset_tools import DEFAULT_DATASET, indexed, read_jsonl, validate


def load_component(filename: str):
    # Load the pure module only; core/__init__.py imports clients unrelated to this test.
    path = Path(__file__).resolve().parents[1] / "core" / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def run(root: Path) -> dict:
    validate(root)
    rag = load_component("rag_context.py")
    cache = load_component("evidence_store.py")
    cases = read_jsonl(root / "cases.jsonl")
    fixtures = indexed(read_jsonl(root / "private/fixtures.jsonl"), "case_id")
    rubrics = indexed(read_jsonl(root / "private/rubrics.jsonl"), "case_id")
    documents = indexed(read_jsonl(root / "documents.jsonl"), "key")
    results = []
    for case in cases:
        if case["suite"] != "component":
            continue
        stores, histories = {}, {}
        observed = {"backend_calls": 0, "cache_hits": 0,
                    "new_bodies_per_call": [], "references_per_call": []}
        for index, action in enumerate(fixtures[case["case_id"]]["component_actions"]):
            invocation = action["invocation"]
            if action["op"] == "clear_visible_messages":
                histories[invocation] = []
                continue
            store = stores.setdefault(invocation, cache.EvidenceStore())
            history = histories.setdefault(invocation, [])

            async def execute():
                observed["backend_calls"] += 1
                return {"query": action["arguments"]["query"], "documents": [
                    rag.document_block(documents[key]) for key in action["document_keys"]]}

            raw = await store.execute(action["tool"], action["arguments"], action["worker"], execute)
            call_id = f"{case['case_id']}-call-{index + 1}"
            compacted = rag.compact_rag_result(raw, history, call_id)
            blocks = compacted["documents"]
            observed["new_bodies_per_call"].append(sum("content" in d for d in blocks))
            observed["references_per_call"].append(sum("reference" in d for d in blocks))
            history.append({"role": "tool", "tool_call_id": call_id,
                            "content": json.dumps(compacted, ensure_ascii=False)})
        observed["cache_hits"] = sum(store.summary()["cache_hits"] for store in stores.values())
        expected = rubrics[case["case_id"]]["checks"][0]["expectation"]
        results.append({"case_id": case["case_id"], "passed": observed == expected,
                        "observed": observed, "expected": expected})
    return {"suite": "component", "scope": "pure cache and RAG functions; not Agent integration",
            "model_calls": 0, "passed": sum(r["passed"] for r in results),
            "total": len(results), "results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args()
    result = asyncio.run(run(args.dataset))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["passed"] == result["total"] else 1)
