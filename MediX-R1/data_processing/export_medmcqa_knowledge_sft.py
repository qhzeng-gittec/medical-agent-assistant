"""Export only fully audited knowledge records; retain every rejection and original."""
import hashlib
import html
import json
import re
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from common.io import load_jsonl, save_jsonl
from common.native_reasoning import SYSTEM_PROMPT
from data_processing.review_medmcqa_knowledge import LAB, SOURCE, OUTPUT, MODEL
from data_processing.select_medmcqa_knowledge import QuestionIndex, normalized


def main():
    sources = {s: load_jsonl(SOURCE / f"{s}_candidates.jsonl") for s in ("train", "validation", "test")}
    selection = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
    for split in sources:
        assert hashlib.sha256((SOURCE / f"{split}_candidates.jsonl").read_bytes()).hexdigest() == selection["splits"][split]["sha256"]
    reviews = {r["id"]: r for r in load_jsonl(OUTPUT / "review_annotations.jsonl")}
    checks = {r["id"]: r for r in load_jsonl(OUTPUT / "verify_annotations.jsonl")}
    expected = sources["train"] + sources["validation"]
    assert len(reviews) == len(expected) == 2200 and set(reviews) == {r["id"] for r in expected}
    assert set(checks) == {r["id"] for r in reviews.values() if r["decision"] != "quarantine"}
    resolution_path = OUTPUT / "resolutions.json"
    resolutions = json.loads(resolution_path.read_text(encoding="utf-8"))
    assert set(resolutions) <= set(reviews)
    manual_flags = json.loads((OUTPUT / "manual_flags.json").read_text(encoding="utf-8"))

    protected = QuestionIndex()
    for r in pq.read_table(LAB / "data/medmcqa_raw/validation.parquet").to_pylist():
        protected.add(r["question"], "official-validation:" + r["id"])
    case_ids = set()
    for split in ("train", "validation", "test"):
        for r in load_jsonl(LAB / "data/medmcqa_cases" / f"{split}.jsonl"):
            protected.add(r["case_question"], "existing-case:" + r["source_id"])
            case_ids.add(r["source_id"])
    open_annotations = LAB / "data/open_cases/annotations.jsonl"
    if open_annotations.exists():
        for r in load_jsonl(open_annotations):
            if r.get("open_question"):
                protected.add(r["open_question"], "existing-open-case:" + r["id"])
    assert not case_ids.intersection(r["source_id"] for r in expected)
    accepted = {"train": [], "validation": []}
    audit = []
    # Validation is reserved before train so rewritten questions cannot cross splits.
    for source in sources["validation"] + sources["train"]:
        first = reviews[source["id"]]
        second = checks.get(source["id"])
        resolution = resolutions.get(source["id"])
        proposal = dict(first)
        reasons = []
        if source["id"] in manual_flags:
            reasons.append("manual_review:" + manual_flags[source["id"]])
        if first["decision"] == "quarantine":
            reasons.append("first_review_quarantine")
        if second and second["decision"] == "quarantine":
            reasons.append("second_review_quarantine")
        if first["label_judgment"] != "correct" or first["correct_label"] != source["answer_label"]:
            reasons.append("label_change_or_uncertainty_requires_evidence")
        if first["evidence_required"] or (second and second["evidence_required"]):
            reasons.append("unresolved_external_evidence_request")
        if "subject_mismatch" in first["changes"]:
            reasons.append("subject_mismatch_outside_foundational_pilot")
        if resolution:
            if resolution["decision"] == "accept":
                assert resolution["sources"] and resolution["note"]
                proposal.update(resolution.get("replacement", {}))
                reasons = []
            else:
                reasons.append("final_review_quarantine:" + resolution["note"])
        if not reasons:
            for key in ("question", "reasoning_content", "final_answer"):
                assert proposal[key].strip(), (source["id"], key)
                assert not re.search(r"<\|[^>]+\|>|</?(?:think|answer|evidence)>", proposal[key]), source["id"]
            if re.fullmatch(r"[A-Da-d][.)]?|all of the above|both|none of the above", proposal["final_answer"].strip(), re.I):
                reasons.append("nonsemantic_answer")
            match = protected.match(proposal["question"])
            if match:
                reasons.append(f"rewritten_question_lexical_overlap:{match[0]}:{match[1]:.3f}")
        if not reasons:
            protected.add(proposal["question"], source["id"])
            target = proposal["final_answer"].strip()
            row = {
                "source_id": source["id"], "source_group": source["source_id"], "task": "knowledge",
                "image_path": None,
                "user_text": "Medical knowledge question:\n" + proposal["question"].strip() + "\n\nGive the supported answer in natural language.",
                "reasoning_content": proposal["reasoning_content"].strip(), "target": target, "reference": target,
                "reasoning_source": "model_audit_with_recorded_source_checked_resolutions_not_clinician_validated",
                "source": source["source"], "source_split": source["source_split"], "split": source["split"],
                "subject": source["subject"], "training_ready": source["split"] == "train",
                "review_status": "accepted_for_research_sft", "original_answer_label": source["answer_label"],
                "label_corrected": bool(resolution and resolution.get("label_corrected", False)),
            }
            accepted[source["split"]].append(row)
        audit.append({"id": source["id"], "split": source["split"], "decision": "quarantine" if reasons else "accept",
                      "exclusion_reasons": reasons, "source": source, "first_review": first,
                      "second_review": second, "final_resolution": resolution,
                      "manual_review_note": manual_flags.get(source["id"]),
                      "final_proposal": {k: proposal[k] for k in ("question", "reasoning_content", "final_answer")}})
    audit.sort(key=lambda r: (r["split"], r["id"]))
    save_jsonl(OUTPUT / "audit.jsonl", audit)
    save_jsonl(OUTPUT / "quarantine.jsonl", [r for r in audit if r["decision"] == "quarantine"])
    recipe = OUTPUT / "recipes/knowledge"
    recipe.mkdir(parents=True, exist_ok=True)
    for split, rows in accepted.items():
        rows.sort(key=lambda r: r["source_id"])
        assert rows
        save_jsonl(OUTPUT / f"{split}.jsonl", rows)
        save_jsonl(recipe / f"{split}.jsonl", rows)
        messages = [{"id": r["source_id"], "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": r["user_text"]},
            {"role": "assistant", "reasoning_content": r["reasoning_content"], "content": r["target"]}
        ], "split": split} for r in rows]
        save_jsonl(OUTPUT / f"{split}_messages.jsonl", messages)
    train_ids = {r["source_id"] for r in accepted["train"]}
    val_ids = {r["source_id"] for r in accepted["validation"]}
    test_ids = {r["id"] for r in sources["test"]}
    assert not (train_ids & val_ids or train_ids & test_ids or val_ids & test_ids)
    assert len({normalized(r["user_text"]) for rows in accepted.values() for r in rows}) == sum(map(len, accepted.values()))
    manifest = {
        "status": "model_audited_research_sft_not_clinician_validated", "model": MODEL,
        "protocol": "qwen_native_open_qa_llm_judge", "enable_thinking": True, "system_prompt": SYSTEM_PROMPT,
        "train_sha256": hashlib.sha256((recipe / "train.jsonl").read_bytes()).hexdigest(),
        "validation_sha256": hashlib.sha256((recipe / "validation.jsonl").read_bytes()).hexdigest(),
        "input_train": 2000, "input_validation": 200, "reviewed": len(audit),
        "second_pass_reviewed": len(checks),
        "accepted": {s: len(r) for s, r in accepted.items()},
        "quarantined": dict(Counter(r["split"] for r in audit if r["decision"] == "quarantine")),
        "subjects": {s: dict(Counter(r["subject"] for r in rows)) for s, rows in accepted.items()},
        "label_corrections": {s: sum(r["label_corrected"] for r in rows) for s, rows in accepted.items()},
        "initial_issue_flags": dict(Counter(c for r in reviews.values() for c in set(r["changes"]))),
        "source_selection_manifest_sha256": hashlib.sha256((SOURCE / "manifest.json").read_bytes()).hexdigest(),
        "test_policy": "200 original test candidates remain unchanged, not exported as training or newly certified evaluation labels",
        "dedup_policy": "source IDs plus exact/near lexical matches of rewritten questions against official validation, existing cases and reserved validation; not semantic deduplication",
        "review_limitations": ["Two passes use the same teacher model, not independent medical experts.",
            "Second pass sees original labels and proposed answers; it is not a blinded answer-agreement experiment.",
            "Only explicitly linked final resolutions received external-source checks; remaining accepted facts are model-screened.",
            "Questions that need alternatives preserve candidate contents inline; natural-language output does not mean all questions are option-free.",
            "No model training or proof of downstream improvement has been performed."],
    }
    for path in (OUTPUT / "manifest.json", recipe / "manifest.json"):
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    cards = []
    for r in audit:
        p = r["final_proposal"]
        contents = {"原题": r["source"]["question"], "原答案": r["source"]["answer"],
                    "新题": p["question"], "新答案": p["final_answer"], "解释": p["reasoning_content"],
                    "初审": r["first_review"]["review_note"], "复审": (r["second_review"] or {}).get("review_note", "初审已隔离"),
                    "最终处理": (r["final_resolution"] or {}).get("note", "; ".join(r["exclusion_reasons"]) or "两轮模型审核通过")}
        body = "".join(f"<p><b>{html.escape(k)}</b>：{html.escape(v)}</p>" for k, v in contents.items())
        cards.append(f'<details data-status="{r["decision"]}"><summary>{r["split"]} · {r["decision"]} · {html.escape(r["source"]["question"])}</summary><small>{r["id"]}</small>{body}</details>')
    preview = '<!doctype html><html lang="zh"><meta charset="utf-8"><title>基础医学 QA 全量审核</title><style>body{max-width:1100px;margin:32px auto;font:16px/1.6 system-ui;background:#f7f8fa;color:#18202c}details{background:white;padding:14px;margin:12px 0;border:1px solid #dde2e8;border-radius:8px}summary{cursor:pointer}input{padding:10px;width:90%}</style><h1>基础医学 QA 全量审核</h1><p>全量模型审核，非临床专家认证。原始 2000 训练题与 200 验证题全部保留决策。</p><input id="q" placeholder="搜索题目、答案、审核原因"><div id="rows">' + "".join(cards) + '</div><script>document.getElementById("q").addEventListener("input",e=>{const s=e.target.value.toLowerCase();document.querySelectorAll("details").forEach(d=>d.hidden=!d.textContent.toLowerCase().includes(s));});</script></html>'
    (OUTPUT / "preview.html").write_text(preview, encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
