"""Re-run the old 72 retrieval probes with fresh remote embeddings; not new holdout cases."""

import argparse
import json
from pathlib import Path

import numpy as np

from campaign_gateway import EMBED_MODEL, dump, sha
from campaign_rag import QUERY_INSTRUCTION
from campaign_rag_boundary import summarize
from holdout_services import DATA, HoldoutGateway, read


def main(output):
    destination = output / "rag_regression/summary.json"
    if destination.exists():
        raise RuntimeError("Completed retrieval regression preserved; no paid repeat")
    gateway = HoldoutGateway(output)
    suites = {name: [json.loads(line) for line in (DATA / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()]
              for name in ("rag_boundary", "rag_near_miss")}
    cases = [(name, case) for name, entries in suites.items() for case in entries]
    trace = []
    inputs = [f"Instruct: {QUERY_INSTRUCTION}\nQuery: {case['query']}" for _, case in cases]
    data = gateway.request("embeddings", {"model": EMBED_MODEL, "input": inputs,
                                          "encoding_format": "float"}, trace, "embedding")
    ordered = sorted(data["data"], key=lambda row: row["index"])
    if [row["index"] for row in ordered] != list(range(len(cases))):
        raise ValueError("Batch embedding response does not match queries")
    vectors = np.asarray([row["embedding"] for row in ordered], dtype=float)
    stored = read(output / "corpus_vectors.json")
    if stored["corpus_sha256"] != sha(DATA / "corpus.jsonl") or stored["model"] != EMBED_MODEL:
        raise ValueError("Frozen corpus/vector identity mismatch")
    docs = [json.loads(line) for line in (DATA / "corpus.jsonl").read_text(encoding="utf-8").splitlines()]
    corpus = np.asarray(stored["vectors"], dtype=float)
    if not np.isfinite(vectors).all() or np.any(np.linalg.norm(vectors, axis=1) == 0):
        raise ValueError("Invalid remote query vectors")
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
    results = {name: [] for name in suites}
    for (name, case), vector in zip(cases, vectors):
        eligible = [i for i, doc in enumerate(docs) if case.get("filter_type") is None
                    or doc["metadata"]["type"] == case["filter_type"]]
        ranked = sorted(eligible, key=lambda i: float(corpus[i] @ vector), reverse=True)[:10]
        results[name].append({"case": case, "status": "completed",
                              "documents": [{**docs[i], "score": float(corpus[i] @ vector)} for i in ranked]})
    summary = {"suite_results": {name: summarize(rows) for name, rows in results.items()},
               "query_count": len(cases), "embedding_api_batches": 1,
               "corpus_sha256": sha(DATA / "corpus.jsonl"), "corpus_vector_sha256": sha(output / "corpus_vectors.json"),
               "source_case_hashes": {name: sha(DATA / f"{name}.jsonl") for name in suites},
               "scope": "Known development regression, not independent holdout; original labels retained including documented equivalence dispute",
               "latency_scope": "Batch embedding latency is not per-query online retrieval latency",
               "no_threshold_selected_from_results": True}
    dump(output / "rag_regression/query_vectors.json", {"model": EMBED_MODEL, "vectors": [row["embedding"] for row in ordered]})
    dump(output / "rag_regression/trace.json", trace)
    dump(output / "rag_regression/results.json", results)
    dump(destination, summary)
    print(json.dumps({"rag_regression_completed": len(cases), "independent_new_cases": 0,
                      "all_vectors_from": "OpenRouter Qwen API"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
