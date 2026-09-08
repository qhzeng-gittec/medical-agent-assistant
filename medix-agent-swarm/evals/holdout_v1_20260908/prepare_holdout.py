"""Validate and freeze unseen case files without printing patient text or rubrics.

No model calls, no medical grading and no product modifications. This is a
construction check, not an executable end-to-end evaluation harness.
"""

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
AUTHOR_NAMES = ("consultation", "history", "evidence")
TOP_FIELDS = {"case_id", "family_id", "domain", "tags", "difficulty", "interaction", "private", "provenance"}
PRIVATE_FIELDS = {"patient_facts", "seed_history", "environment", "checkpoints", "source_refs", "clinical_review_required"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def product_hashes():
    files = [p for folder in ("agents", "core", "swarm", "memory", "constraints", "validation", "knowledge", "research", ".claude")
             for p in (PROJECT / folder).rglob("*")
             if p.suffix in {".py", ".yaml", ".yml", ".md"} and "__pycache__" not in p.parts and "data" not in p.parts]
    return {str(p.relative_to(PROJECT)): sha(p) for p in sorted(files)}


def check(condition, case_id, label):
    if not condition:
        raise ValueError(f"{case_id}: invalid {label}; inspect in isolated review, not developer context")


def validate_case(case, author):
    cid = case.get("case_id", "unknown")
    check(set(case) == TOP_FIELDS, cid, "top-level fields")
    check(case["domain"] == author and isinstance(case["family_id"], str), cid, "domain/family")
    check(case["difficulty"] in {"routine", "complex", "boundary"}, cid, "difficulty")
    check(isinstance(case["tags"], list) and 2 <= len(case["tags"]) <= 6, cid, "tags")
    interaction, private = case["interaction"], case["private"]
    check(set(interaction) == {"opening", "max_patient_replies", "scripted_followups"}, cid, "interaction fields")
    check(isinstance(interaction["opening"], str) and bool(interaction["opening"].strip()), cid, "opening")
    check(type(interaction["max_patient_replies"]) is int and 0 <= interaction["max_patient_replies"] <= 3, cid, "reply cap")
    for followup in interaction["scripted_followups"]:
        check(set(followup) == {"text", "new_session", "new_user"}, cid, "follow-up fields")
        check(isinstance(followup["text"], str) and all(type(followup[k]) is bool for k in ("new_session", "new_user")), cid, "follow-up types")
    check(set(private) == PRIVATE_FIELDS, cid, "private fields")
    for fact in private["patient_facts"]:
        check(set(fact) == {"id", "text", "reveal_when"} and all(isinstance(v, str) and v.strip() for v in fact.values()), cid, "patient fact")
    check(private["patient_facts"] or interaction["max_patient_replies"] == 0, cid, "adaptive patient facts")
    for history in private["seed_history"]:
        check(set(history) == {"user_key", "session", "messages"} and history["user_key"] in {"self", "other"}, cid, "history scope")
        for message in history["messages"]:
            check(set(message) == {"role", "content"} and message["role"] in {"user", "assistant"} and isinstance(message["content"], str), cid, "history message")
    env = private["environment"]
    check(set(env) == {"rag_mode", "memory_mode", "documents", "memory_records"}, cid, "environment fields")
    check(env["rag_mode"] in {"live_rag", "controlled", "empty_retrieval", "retrieval_timeout", "not_needed"}, cid, "RAG mode")
    check(env["memory_mode"] in {"live_mem0", "controlled", "off", "not_needed"}, cid, "memory mode")
    for document in env["documents"]:
        check({"id", "content", "metadata"} <= set(document), cid, "document fields")
        check({"source", "title", "evidence_form", "subject", "published_at"} <= set(document["metadata"]), cid, "document metadata")
    for memory in env["memory_records"]:
        check(set(memory) == {"id", "user_key", "content", "source_role"} and memory["user_key"] in {"self", "other"}, cid, "memory record")
    checkpoints = private["checkpoints"]
    check(3 <= len(checkpoints) <= 6, cid, "checkpoint count")
    check(len({c["id"] for c in checkpoints}) == len(checkpoints), cid, "checkpoint IDs")
    for checkpoint in checkpoints:
        check(set(checkpoint) == {"id", "after", "target", "requirement", "critical"}, cid, "checkpoint fields")
        check(checkpoint["after"] in {"each_turn", "final"} or re.fullmatch(r"turn:[1-9]\d*", checkpoint["after"]), cid, "checkpoint timing")
        check(checkpoint["target"] in {"answer", "profile", "event_memory", "trace", "retrieval"}, cid, "checkpoint target")
        check(type(checkpoint["critical"]) is bool and isinstance(checkpoint["requirement"], str), cid, "checkpoint type")
    for source in private["source_refs"]:
        check(set(source) == {"url", "title", "supports", "note"} and source["url"].startswith("https://"), cid, "source")
        check(set(source["supports"]) <= {c["id"] for c in checkpoints}, cid, "source checkpoint references")
    check(type(private["clinical_review_required"]) is bool, cid, "clinical review flag")
    provenance = case["provenance"]
    check(set(provenance) == {"author_bucket", "synthetic", "parent_history_inherited", "implementation_seen", "existing_results_seen"}, cid, "provenance fields")
    check(provenance["author_bucket"] == author and provenance["synthetic"] is True, cid, "provenance identity")
    check(all(provenance[k] is False for k in ("parent_history_inherited", "implementation_seen", "existing_results_seen")), cid, "author isolation attestation")


def normalized(text):
    return "".join(c for c in unicodedata.normalize("NFKC", text).casefold() if c.isalnum())


def main(lock_product, freeze):
    lock_path = ROOT / "product_lock.json"
    if lock_product:
        if lock_path.exists():
            check(read(lock_path)["sources"] == product_hashes(), "product", "unchanged frozen source")
        else:
            save(lock_path, {"created_at": datetime.now(timezone.utc).isoformat(), "sources": product_hashes(),
                             "contract_sha256": sha(ROOT / "author_contract.md"), "case_text_seen_by_parent": False})
        print(json.dumps({"product_locked": True, "source_files": len(read(lock_path)["sources"])}))
        return
    check(read(lock_path)["sources"] == product_hashes(), "product", "source lock")
    check(read(lock_path)["contract_sha256"] == sha(ROOT / "author_contract.md"), "contract", "unchanged contract")
    all_cases, summaries, files = [], [], {}
    for author in AUTHOR_NAMES:
        path = ROOT / "sealed/authors" / f"{author}.jsonl"
        cases = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        check(len(cases) == 24, author, "case count")
        for case in cases:
            validate_case(case, author)
        balance = Counter(c["difficulty"] for c in cases)
        check(balance == {"routine": 8, "complex": 8, "boundary": 8}, author, "difficulty balance")
        multi = sum(bool(c["interaction"]["max_patient_replies"] or c["interaction"]["scripted_followups"] or c["private"]["seed_history"]) for c in cases)
        check(multi >= 12, author, "longitudinal coverage")
        summaries.append({"domain": author, "cases": len(cases), "difficulty": dict(balance), "longitudinal_cases": multi,
                          "clinical_review_required": sum(c["private"]["clinical_review_required"] for c in cases),
                          "checkpoint_targets": dict(Counter(t["target"] for c in cases for t in c["private"]["checkpoints"]))})
        all_cases.extend(cases)
        files[path.relative_to(ROOT).as_posix()] = sha(path)
    check(len({c["case_id"] for c in all_cases}) == 72, "all", "unique case IDs")
    check(len({c["family_id"] for c in all_cases}) == 72, "all", "unique declared family IDs")
    old = [json.loads(line) for line in (PROJECT / "evals/campaign_v1/cases.private.jsonl").read_text(encoding="utf-8").splitlines()]
    seen = {normalized(c["opening"]): c["case_id"] for c in old}
    seen.update({normalized(c["turns"][0]): c["id"] for c in read(PROJECT / "evals/improvement_cases.json")["cases"]})
    duplicates = []
    for case in all_cases:
        opening = normalized(case["interaction"]["opening"])
        if opening in seen:
            duplicates.append({"case_id": case["case_id"], "matches": seen[opening]})
        seen[opening] = case["case_id"]
    summary = {"constructed_cases": len(all_cases), "declared_families": 72, "domain_summaries": summaries,
               "exact_opening_duplicates": duplicates, "clinical_validation": False,
               "live_evaluation_executed": False, "developer_has_not_read_case_text": True,
               "isolation_is_instruction_and_context_based_not_os_sandbox": True,
               "semantic_nonoverlap_not_proven_by_unique_ids_or_exact_matching": True}
    save(ROOT / "construction_summary.json", summary)
    check(not duplicates, "all", "no exact opening duplicates")
    if freeze:
        review_path = ROOT / "sealed/review.json"
        review = read(review_path)
        check(review["approved_for_engineering_evaluation"] is True, "review", "independent construction review")
        check(review["reviewed_file_hashes"] == files, "review", "reviewed source hashes")
        freeze_path = ROOT / "freeze_manifest.json"
        manifest = {"frozen_at": datetime.now(timezone.utc).isoformat(), "case_files": files,
                    "author_attestations": {name: sha(ROOT / "sealed/authors" / f"{name}_attestation.json") for name in AUTHOR_NAMES},
                    "contract_sha256": sha(ROOT / "author_contract.md"), "product_lock_sha256": sha(lock_path),
                    "review_sha256": sha(review_path), "validator_sha256": sha(Path(__file__)),
                    "execution_protocol_sha256": sha(ROOT / "评测隔离与执行协议.md"),
                    "construction_summary_sha256": sha(ROOT / "construction_summary.json"),
                    "executed": False, "cases": [{"case_id": c["case_id"], "family_id": c["family_id"], "domain": c["domain"]} for c in all_cases]}
        if freeze_path.exists():
            previous = read(freeze_path)
            manifest["frozen_at"] = previous["frozen_at"]
            check(previous == manifest, "freeze", "immutable manifest")
        else:
            save(freeze_path, manifest)
        summary["frozen"] = True
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--lock-product", action="store_true")
    group.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    main(args.lock_product, args.freeze)
