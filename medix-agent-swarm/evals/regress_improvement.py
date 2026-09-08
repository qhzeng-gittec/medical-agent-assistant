"""Targeted current-product regressions, separate from the frozen paired dataset."""

import asyncio
import json
import sys
import time

from loguru import logger

from campaign_gateway import Gateway, dump, sha
from campaign_grade import protocol_checks
from campaign_improvement import ComparisonGoogle, ControlledMemory, ROOT, read
from campaign_gemini_memory import MODEL
from campaign_rag import LocalRAG
from campaign_run import DATA, OUTPUT, PROJECT, build_system, trace_metrics


async def main():
    logger.remove()
    logger.add(sys.stderr, level="ERROR", backtrace=False, diagnose=False)
    frozen = read(ROOT / "protocol_candidate.json")
    if any(sha(PROJECT / name) != digest for name, digest in frozen["sources"].items()):
        raise ValueError("Candidate differs from the paired comparison")
    old_path = OUTPUT / "mem0_boundary_v1/cases/M07.json"
    old = read(old_path)["queries"][0]
    new_path = ROOT / "mem0_live/cases/source.json"
    new = read(new_path)
    cases = [
        {"id": "old_M07", "question": old["query"]["query"], "memory": old["baseline"],
         "source": str(old_path), "source_sha256": sha(old_path)},
        {"id": "new_mem0_source", "question": new["scenario"]["query"], "memory": new["queries"]["3"],
         "source": str(new_path), "source_sha256": sha(new_path)},
    ]
    source = DATA / "cases.private.jsonl"
    for line in source.read_text(encoding="utf-8").splitlines():
        case = json.loads(line)
        if case["case_id"] in {"CONSULT-07", "CONSULT-12", "CONSULT-22", "CONSULT-23"}:
            cases.append({"id": case["case_id"], "question": case["opening"],
                          "environment": case["environment"], "source": str(source),
                          "source_sha256": sha(source)})
    dump(ROOT / "regressions/protocol.json", {"cases": cases, "model": MODEL, "product": frozen,
         "scope": "Single-turn development regressions; real model/RAG, frozen cloud recall, no cloud writes"})
    embeddings = Gateway(OUTPUT, limit=4)
    gateway = ComparisonGoogle(embeddings)
    for case in cases:
        path = ROOT / "regressions/runs" / f"{case['id']}.json"
        if path.exists():
            if read(path)["status"] != "completed":
                raise RuntimeError("Inspect the previous failed regression before retrying")
            continue
        trace = []
        user = "regression-v2-" + case["id"]
        kb = LocalRAG(gateway, DATA / "corpus.jsonl", trace, case.get("environment", "live_rag"))
        memory = ControlledMemory(case, user, trace)
        supervisor, profile = build_system(gateway, MODEL, trace, kb, ROOT / "regressions/profiles" / user, memory)
        record = {"case_id": case["id"], "question": case["question"], "status": "running", "model": MODEL}
        dump(path, record)
        started = time.monotonic()
        try:
            result = await asyncio.wait_for(supervisor.process(case["question"], user_id=user, session_id=user), 180)
            record.update(status="completed", answer=result["answer"], result=result, profile=profile.get_context(user))
        except Exception as error:
            record.update(status="error", error=f"{type(error).__name__}: {error}".replace(gateway.key, "[REDACTED]"))
            raise
        finally:
            record.update(seconds=time.monotonic() - started, metrics=trace_metrics(trace), protocol=protocol_checks(trace))
            dump(ROOT / "regressions/traces" / f"{case['id']}.json", trace)
            dump(path, record)
            print(json.dumps({"case": case["id"], "status": record["status"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
