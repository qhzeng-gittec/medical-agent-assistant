"""A small, labeled retrieval smoke set, distinct from clinical or Agent quality."""

from campaign_gateway import EMBED_MODEL, Gateway, dump
from campaign_rag import LocalRAG, QUERY_INSTRUCTION
from campaign_run import DATA, OUTPUT


QUERIES = [
    ("咳嗽多久不好要看医生，有哪些危险表现", ["NHS-COUGH"]),
    ("咳出黄色的痰就是细菌感染必须吃抗生素吗", ["CDC-BRONCHITIS"]),
    ("胸口一直压着还出汗喘不过气，要不要等一会", ["NHS-CHEST"]),
    ("突然嘴歪一边手抬不起来但后来好了", ["NHS-STROKE"]),
    ("头痛突然很剧烈，什么情况下需要急救", ["NHS-HEADACHE"]),
    ("拉肚子后尿很少口干，怎样识别脱水", ["NHS-DEHYDRATION"]),
    ("家里测血压高了一次，能确定是高血压吗", ["NHS-BP"]),
    ("吃药后舌头肿大呼吸不顺畅", ["NHS-ALLERGY"]),
    ("2个月宝宝发烧38度要及时就医吗", ["NHS-CHILD-FEVER"]),
    ("能吃别人剩下的抗生素治疗感冒吗", ["NHS-ANTIBIOTIC"]),
    ("ZX-947疗法治咳嗽的随机对照试验结果", []),
    ("汽车发动机无法启动的修理步骤", []),
]


def main():
    path = OUTPUT / "retrieval_smoke.json"
    if path.exists():
        raise SystemExit("Existing retrieval results preserved; do not overwrite a completed run")
    gateway = Gateway(OUTPUT)
    trace, results = [], []
    rag = LocalRAG(gateway, DATA / "corpus.jsonl", trace)
    try:
        for query, expected in QUERIES:
            docs = rag.search(query, top_k=3)
            ids = [d["id"] for d in docs]
            results.append({"query": query, "expected_ids": expected,
                            "hits": [{"id": d["id"], "cosine": d["score"]} for d in docs],
                            "top1_hit": bool(set(expected) & set(ids[:1])) if expected else None,
                            "top3_hit": bool(set(expected) & set(ids)) if expected else None,
                            "reciprocal_rank": next((1 / (i + 1) for i, doc in enumerate(ids) if doc in expected), 0) if expected else None})
        positives = [r for r in results if r["expected_ids"]]
        dump(path, {"model": EMBED_MODEL, "embedding_location": "OpenRouter API only",
                    "index_location": "local normalized vector matrix", "query_instruction": QUERY_INSTRUCTION,
                    "corpus_sha256": rag.fingerprint, "total": len(results), "positive_queries": len(positives),
                    "top1_hit_rate": sum(r["top1_hit"] for r in positives) / len(positives),
                    "top3_hit_rate": sum(r["top3_hit"] for r in positives) / len(positives),
                    "mrr_at_3": sum(r["reciprocal_rank"] for r in positives) / len(positives),
                    "scope": "15 curated chunks, 10 positive paraphrases and 2 OOD probes; NOT broad retrieval benchmark",
                    "results": results})
        print({"retrieval_queries_completed": len(results), "top3_hits": sum(r["top3_hit"] for r in positives)})
    finally:
        dump(OUTPUT / "retrieval_smoke_trace.json", trace)


if __name__ == "__main__":
    main()
