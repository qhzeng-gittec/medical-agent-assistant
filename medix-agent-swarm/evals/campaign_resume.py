"""Resume only the actual one-turn patient-selector failures; never reroll prior answers."""

import asyncio
import copy
import json
import shutil
import time

from loguru import logger

from campaign_gateway import Gateway, dump
from campaign_rag import LocalRAG
from campaign_run import DATA, OUTPUT, build_system, patient_reply, trace_metrics


async def resume(gateway, original, case):
    if len(original["turns"]) != 1 or case["max_patient_replies"] != 1 or case["scripted_followups"]:
        raise ValueError("This recovery is only for the observed single selector boundary failure")
    source_id = original["run_id"]
    run_id = source_id + "_resumed"
    destination = OUTPUT / "runs" / f"{run_id}.json"
    if destination.exists():
        raise ValueError("A recovery attempt already exists; preserving it")
    trace = json.loads((OUTPUT / "traces" / f"{source_id}.json").read_text(encoding="utf-8"))
    result = copy.deepcopy(original)
    result.update(run_id=run_id, resumed_from=source_id,
                  recovery="Qwen patient selector reasoning disabled; original doctor reply and context retained")
    started = time.monotonic()
    try:
        profile_source = OUTPUT / "profiles" / source_id
        profile_copy = OUTPUT / "profiles" / run_id
        if profile_source.exists():
            shutil.copytree(profile_source, profile_copy)
        kb = LocalRAG(gateway, DATA / "corpus.jsonl", trace, case["environment"])
        supervisor, profile = build_system(gateway, result["model"], trace, kb, profile_copy)
        previous = original["turns"][0]
        session, user = previous["session_id"], previous["user_id"]
        # This is exactly what product _save_memory stores; worker transcripts were not persisted.
        supervisor.short_term_memory.add_message(session, "user", previous["user"])
        supervisor.short_term_memory.add_message(session, "assistant", previous["answer"])
        trace.append({"event": "harness_checkpoint_resume", "source_run": source_id,
                      "changed_setting": "patient_selector reasoning.enabled=false; target model config unchanged"})
        revealed = []
        reply = await patient_reply(gateway, case, previous["answer"], revealed, trace)
        if reply is not None:
            trace.append({"event": "turn_start", "turn": 2, "user_id": user, "session_id": session, "text": reply})
            before = profile.get_context(user)
            turn_started = time.monotonic()
            response = await supervisor.process(reply, user_id=user, session_id=session)
            result["turns"].append({"user": reply, "answer": response["answer"], "system_result": response,
                "user_id": user, "session_id": session, "profile_before": before, "profile_after": profile.get_context(user),
                "latency_seconds": round(time.monotonic() - turn_started, 3)})
            trace.append({"event": "turn_end", "turn": 2})
        result.update(status="completed", patient_fact_ids_revealed=revealed)
        result.pop("error", None)
        result.pop("error_type", None)
    except Exception as error:
        result.update(status="error", error=str(error).replace(gateway.key, "[REDACTED]"), error_type=type(error).__name__)
    finally:
        result["elapsed_seconds"] = original["elapsed_seconds"] + round(time.monotonic() - started, 3)
        result["elapsed_note"] = "Active execution time includes original failed selector; idle checkpoint gap excluded"
        result["metrics"] = trace_metrics(trace)
        dump(OUTPUT / "traces" / f"{run_id}.json", trace)
        dump(destination, result)
    print(json.dumps({"resumed": run_id, "status": result["status"], "turns": len(result["turns"])}, ensure_ascii=False), flush=True)


async def main():
    logger.remove()
    gateway = Gateway(OUTPUT)
    cases = {c["case_id"]: c for c in map(json.loads, (DATA / "cases.private.jsonl").read_text(encoding="utf-8").splitlines())}
    for path in sorted((OUTPUT / "runs").glob("*__r4.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("error") == "Patient selector response incomplete":
            await resume(gateway, result, cases[result["case_id"]])


if __name__ == "__main__":
    asyncio.run(main())
