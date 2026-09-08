"""Verify all 72 frozen vector rankings and threshold metrics without an API key."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def read(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def coverage(groups, documents, k, threshold):
    found = {doc["id"] for doc in documents[:k] if doc["score"] >= threshold}
    hits = [bool(found.intersection(group)) for group in groups]
    return found, hits


def main():
    manifest = read("manifest.json")
    for entry in manifest["files"]:
        path = (ROOT / entry["path"]).resolve()
        assert path.is_relative_to(ROOT), entry["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"], entry["path"]
    corpus = [json.loads(line) for line in (ROOT / "corpus.jsonl").read_text(encoding="utf-8").splitlines()]
    order = read("vector_order.json")
    assert order["doc_ids"] == [doc["id"] for doc in corpus]
    assert len(order["doc_ids"]) == len(set(order["doc_ids"])) == 15
    frozen = read("results.json")
    cases = {suite: [json.loads(line) for line in (ROOT / f"{suite}.jsonl").read_text(encoding="utf-8").splitlines()]
             for suite in ("rag_boundary", "rag_near_miss")}
    assert order["queries"] == [{"suite": suite, "id": case["id"]} for suite, rows in cases.items() for case in rows]
    with np.load(ROOT / "vectors.npz", allow_pickle=False) as arrays:
        documents, queries = arrays["corpus"], arrays["queries"]
    assert documents.shape == (15, 4096) and queries.shape == (72, 4096)
    for matrix in (documents, queries):
        assert np.isfinite(matrix).all() and (np.linalg.norm(matrix, axis=1) > 0).all()
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)

    replay = {suite: [] for suite in cases}
    max_error = 0.0
    query_index = 0
    for suite, suite_cases in cases.items():
        assert len(suite_cases) == len(frozen[suite])
        for case, saved in zip(suite_cases, frozen[suite]):
            assert case == saved["case"] and saved["status"] == "completed"
            assert all(set(group) <= set(order["doc_ids"]) for group in case["gold_groups"])
            scores = documents @ queries[query_index]
            query_index += 1
            eligible = [i for i, doc in enumerate(corpus) if not case.get("filter_type")
                        or doc["metadata"]["type"] == case["filter_type"]]
            ranked = sorted(eligible, key=lambda i: float(scores[i]), reverse=True)[:10]
            assert [corpus[i]["id"] for i in ranked] == [doc["id"] for doc in saved["documents"]], case["id"]
            found = []
            for index, original in zip(ranked, saved["documents"]):
                assert {key: value for key, value in original.items() if key != "score"} == corpus[index]
                difference = abs(float(scores[index]) - original["score"])
                max_error = max(max_error, difference)
                assert difference < 1e-10, case["id"]
                found.append({"id": corpus[index]["id"], "score": float(scores[index])})
            replay[suite].append({"case": case, "documents": found})

    summary = read("summary.json")
    for suite, rows in replay.items():
        expected = summary["suite_results"][suite]
        assert expected["total_attempted"] == expected["completed"] == len(rows)
        assert not expected["errors"]
        positives = [row for row in rows if row["case"]["answerability"] == "answerable"]
        negatives = [row for row in rows if row["case"]["answerability"] == "unanswerable"]
        assert (len(positives), len(negatives)) == (expected["answerable_n"], expected["unanswerable_n"])
        for check in expected["threshold_sweep"]:
            covered = [coverage(row["case"]["gold_groups"], row["documents"], check["k"], check["threshold"])[1]
                       for row in positives]
            assert all(covered) or not covered
            assert sum(any(hits) for hits in covered) == check["any_hit_n"]
            assert sum(all(hits) for hits in covered) == check["all_hit_n"]
            assert sum(bool(coverage([], row["documents"], check["k"], check["threshold"])[0]) for row in negatives) == check["unanswerable_returned_n"]
            if covered:
                recall = sum(sum(hits) / len(hits) for hits in covered) / len(covered)
                assert math.isclose(recall, check["mean_group_recall"], abs_tol=1e-12)
        for name, expected_group in expected["groups"].items():
            group = [row for row in rows if row["case"]["group"] == name]
            gold = [row for row in group if row["case"]["gold_groups"]]
            hits = [coverage(row["case"]["gold_groups"], row["documents"], 3, 0)[1] for row in gold]
            assert expected_group == {"cases": len(group), "with_gold": len(gold),
                                      "top3_any_hits": sum(any(hit) for hit in hits),
                                      "top3_all_hits": sum(all(hit) for hit in hits)}
    assert query_index == summary["query_count"] == 72
    print(json.dumps({"verified_queries": query_index, "corpus_chunks": len(corpus),
                      "threshold_settings_verified": 56, "max_score_error": max_error,
                      "model_api_calls": 0, "result": "PASS"}, indent=2))


if __name__ == "__main__":
    main()
