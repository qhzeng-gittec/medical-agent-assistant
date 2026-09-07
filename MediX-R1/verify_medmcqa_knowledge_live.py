"""Second-pass dataset inference as completed first-pass batches become available."""
import json
import hashlib
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from data_io import load_jsonl, save_jsonl
from review_medmcqa_knowledge import SOURCE, OUTPUT, MODEL, REVIEW_PROMPT, REVIEW_SCHEMA, run_batch


def main():
    sources = {r["id"]: r for s in ("train", "validation") for r in load_jsonl(SOURCE / f"{s}_candidates.jsonl")}
    expected_batches = {}
    rows = list(sources.values())
    fields = ("id", "question", "options", "answer_label", "answer", "source_explanation", "subject")
    for offset in range(0, len(rows), 20):
        batch = rows[offset:offset + 20]
        payload = [{k: r[k] for k in fields} for r in batch]
        prompt = REVIEW_PROMPT + json.dumps(payload, ensure_ascii=False)
        fingerprint = hashlib.sha256((MODEL + prompt + json.dumps(REVIEW_SCHEMA, sort_keys=True)).encode()).hexdigest()
        expected_batches[fingerprint + ".json"] = {r["id"] for r in batch}
    scanned, reviewed, submitted = set(), set(), set()
    results, waiting, pending = [], [], set()
    last_progress = time.monotonic()
    with ThreadPoolExecutor(max_workers=6) as executor:
        while len(reviewed) < len(sources) or waiting or pending:
            for path in sorted((OUTPUT / "review").glob("*.json")):
                if path.name not in expected_batches or path in scanned or not path.with_suffix(".log").exists():
                    continue
                batch = json.loads(path.read_text(encoding="utf-8"))["annotations"]
                if len(batch) != len(expected_batches[path.name]) or {a["id"] for a in batch} != expected_batches[path.name]:
                    continue  # The first pass must repair incomplete model output before consumption.
                scanned.add(path)
                reviewed.update(a["id"] for a in batch)
                candidates = [dict(sources[a["id"]], annotation=a) for a in batch if a["decision"] != "quarantine" and a["id"] not in submitted]
                if candidates:
                    submitted.update(a["id"] for a in candidates)
                    waiting.append(candidates)
                last_progress = time.monotonic()
            while waiting and len(pending) < 6:
                pending.add(executor.submit(run_batch, waiting.pop(0), "verify"))
            if pending:
                done, pending = wait(pending, timeout=20, return_when=FIRST_COMPLETED)
                for future in done:
                    results.extend(future.result())
                    last_progress = time.monotonic()
                    save_jsonl(OUTPUT / "verify_annotations.jsonl", sorted(results, key=lambda r: r["id"]))
                    print(json.dumps({"first_pass_seen": len(reviewed), "second_pass_done": len(results),
                                      "decisions": dict(Counter(r["decision"] for r in results)), "queued_batches": len(waiting)}), flush=True)
            elif len(reviewed) < len(sources):
                time.sleep(5)
            if time.monotonic() - last_progress > 1200:
                raise TimeoutError("No dataset review progress for 20 minutes; inspect first-pass failures")
    assert len(results) == len(submitted) and {r["id"] for r in results} == submitted
    print(f"verification_complete={len(results)} original_inputs={len(reviewed)}", flush=True)


if __name__ == "__main__":
    main()
