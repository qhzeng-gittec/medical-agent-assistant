"""Select source-preserving knowledge QA candidates; no medical relabeling."""
import hashlib
import html
import json
import random
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import pyarrow.parquet as pq

from common.io import load_jsonl, save_jsonl
from common.native_reasoning import missing_case_image_reason


LAB = Path(__file__).resolve().parents[1]
DATA = LAB / "data"
OUTPUT = DATA / "medmcqa_knowledge_candidates_v1"
QUOTAS = {"Anatomy": 350, "Physiology": 350, "Biochemistry": 300,
          "Pathology": 350, "Pharmacology": 350, "Microbiology": 300}
CASE = re.compile(r"year[- ]old|\bpatient\b|\bpresents?\b|\bcomplains?\b|\bhistory of\b|"
                  r"\bon examination\b|\bwoman\b|\bman\b|\bchild\b|\binfant\b|\bpregnant\b", re.I)
OCR = re.compile(r"\b(?:aery|aeries|cailage|impoant|transpo|transpos|conveed|hyperspineism|IG1-1|IGI-1GH)\b", re.I)
ANSWER_MARKER = re.compile(r"\b(?:ans(?:wer)?\s*[.:-]?\s*|correct\s+(?:answer|option)\s+(?:is\s*)?[:.-]?\s*)\(?([a-d])\b", re.I)


def normalized(text):
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


class QuestionIndex:
    """Conservative lexical exclusion, not a claim of semantic deduplication."""
    def __init__(self):
        self.questions = {}
        self.bigrams = defaultdict(set)

    def add(self, question, source_id):
        question = normalized(question)
        if question in self.questions:
            return
        self.questions[question] = source_id
        words = question.split()
        for gram in set(zip(words, words[1:])):
            self.bigrams[gram].add(question)

    def match(self, question):
        question = normalized(question)
        if question in self.questions:
            return self.questions[question], 1.0
        words = question.split()
        grams = set(zip(words, words[1:]))
        overlaps = Counter(q for gram in grams for q in self.bigrams.get(gram, ()))
        for other, count in overlaps.items():
            if count < 2 or min(len(question), len(other)) / max(len(question), len(other)) < 0.8:
                continue
            score = SequenceMatcher(None, question, other, autojunk=False).ratio()
            if score >= 0.9:
                return self.questions[other], score
        return None


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Selection is immutable; use a new version directory: {OUTPUT}")
    source_paths = [DATA / "medmcqa_raw" / f"{s}.parquet" for s in ("train", "validation")]
    case_paths = [DATA / "medmcqa_cases" / f"{s}.jsonl" for s in ("train", "validation", "test")]
    hashes = {str(p.relative_to(LAB)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths + case_paths}
    raw = {s: pq.read_table(p).to_pylist() for s, p in zip(("train", "validation"), source_paths)}
    cases = [r for p in case_paths for r in load_jsonl(p)]
    existing_ids = {r["source_id"] for r in cases}
    pool = {s: defaultdict(list) for s in raw}
    rejected = []
    input_counts = {s: len(rows) for s, rows in raw.items()}

    for split, rows in raw.items():
        seen = set()
        for row in rows:
            if row["subject_name"] not in QUOTAS:
                continue
            question = (row["question"] or "").strip()
            explanation = (row["exp"] or "").strip()
            options = {label: (row[key] or "").strip() for label, key in zip("ABCD", ("opa", "opb", "opc", "opd"))}
            key = normalized(question)
            reason = None
            if row["id"] in existing_ids:
                reason = "existing_case_source"
            elif row["choice_type"] != "single":
                reason = "outside_single_choice_pilot"
            elif not question or not all(options.values()) or row["cop"] not in range(4):
                reason = "invalid_question_options_or_label"
            elif CASE.search(question):
                reason = "case_pattern_not_knowledge_pilot"
            elif missing_case_image_reason({"case_question": question, "options": options}):
                reason = "missing_image"
            elif len(set(normalized(x) for x in options.values())) != 4:
                reason = "duplicate_options"
            elif not explanation:
                reason = "missing_explanation"
            else:
                body = re.split(r"\b(?:ref(?:erence)?\s*[:.]|references\s*:)", explanation, maxsplit=1, flags=re.I)[0]
                body = ANSWER_MARKER.sub(" ", body)
                if len(body.split()) < 15:
                    reason = "insufficient_explanation_before_reference"
                elif any(label.upper() != "ABCD"[row["cop"]] for label in ANSWER_MARKER.findall(explanation)):
                    reason = "explicit_answer_marker_conflict"
            if reason is None and key in seen:
                reason = "exact_question_duplicate"
            if reason:
                rejected.append({"source_split": split, "source_id": row["id"], "reason": reason})
                continue
            seen.add(key)
            flags = []
            if OCR.search(question + " " + " ".join(options.values()) + " " + explanation):
                flags.append("suspected_ocr_tokens")
            if re.search(r"\b(?:figure|fig\.|image|diagram|shown below)\b", explanation, re.I):
                flags.append("explanation_mentions_image")
            if re.search(r"<[^>]+>|\ufffd", question + explanation):
                flags.append("markup_or_encoding")
            if re.search(r"\b(?:all|none|both)\s+(?:of\s+)?(?:the\s+)?(?:above|these)|\b(?:a and b|a & b)\b", " ".join(options.values()), re.I):
                flags.append("options_required_for_answer")
            pool[split][row["subject_name"]].append({
                "id": "medmcqa-" + row["id"], "source_id": row["id"],
                "source": "openlifescienceai/medmcqa", "source_split": split,
                "task": "medical_knowledge_candidate", "language": "en",
                "question": question, "options": options,
                "answer_label": "ABCD"[row["cop"]], "answer": options["ABCD"[row["cop"]]],
                "source_explanation": explanation, "subject": row["subject_name"], "topic": row["topic_name"],
                "review_flags": flags, "review_status": "pending_medical_review", "training_ready": False,
                "label_provenance": "original_medmcqa_label_unchanged",
                "raw": row,
            })

    index = QuestionIndex()
    for row in cases:
        index.add(row["case_question"], row["source_id"])
    rng = random.Random(42)
    selected = {}
    for split, origin, multiplier in (("test", "validation", 0.1), ("validation", "train", 0.1), ("train", "train", 1)):
        if split == "validation":
            for row in raw["validation"]:
                index.add(row["question"] or "", row["id"])
        result = []
        for subject, train_size in QUOTAS.items():
            target = round(train_size * multiplier)
            candidates = pool[origin][subject].copy()
            rng.shuffle(candidates)
            candidates.sort(key=lambda r: len(r["review_flags"]))
            accepted = []
            for row in candidates:
                if len(accepted) == target:
                    break
                match = index.match(row["question"])
                if match:
                    rejected.append({"source_split": origin, "source_id": row["source_id"],
                                     "reason": "protected_or_selected_lexical_duplicate", "match_id": match[0], "similarity": match[1]})
                    continue
                accepted.append(dict(row, split=split))
                index.add(row["question"], row["source_id"])
            if len(accepted) != target:
                raise RuntimeError(f"Insufficient candidates for {split}/{subject}: {len(accepted)}/{target}")
            result.extend(accepted)
        rng.shuffle(result)
        selected[split] = result

    all_rows = [r for rows in selected.values() for r in rows]
    assert len(all_rows) == len({r["source_id"] for r in all_rows}) == 2400
    assert len({normalized(r["question"]) for r in all_rows}) == 2400
    assert not existing_ids & {r["source_id"] for r in all_rows}
    heldout_questions = {normalized(r["question"] or "") for r in raw["validation"]}
    assert not heldout_questions & {normalized(r["question"]) for s in ("train", "validation") for r in selected[s]}
    for row in all_rows:
        assert row["answer"] == row["raw"][("opa", "opb", "opc", "opd")[row["raw"]["cop"]]].strip()

    OUTPUT.mkdir()
    for split, rows in selected.items():
        save_jsonl(OUTPUT / f"{split}_candidates.jsonl", rows)
    save_jsonl(OUTPUT / "selection_exclusions.jsonl", rejected)
    review = []
    for subject in QUOTAS:
        rows = [r for r in selected["train"] if r["subject"] == subject]
        review.extend(rows[:5])
    save_jsonl(OUTPUT / "review_sample_30.jsonl", review)
    manifest = {
        "status": "selected_not_reannotated_not_training_ready", "seed": 42,
        "source_sha256": hashes, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "raw_counts": input_counts, "train_subject_quotas": QUOTAS,
        "split_policy": "train and validation from official train; test from official validation; all existing case sources excluded",
        "selection_policy": "six foundational subjects; single-choice; no case regex match; valid distinct options; no missing input image; >=15 explanation words before first reference marker; no explicit answer-letter conflict; fewer review flags preferred",
        "near_duplicate_policy": "normalized question exact match, or >=2 shared word bigrams + character length ratio >=0.8 + SequenceMatcher ratio >=0.9; excludes all official validation questions from train/local validation and all case/selected questions; not semantic deduplication",
        "candidate_pools": {s: {k: len(v) for k,v in groups.items()} for s,groups in pool.items()},
        "splits": {s: {"rows": len(rows), "subjects": dict(Counter(r["subject"] for r in rows)),
                       "answers": dict(Counter(r["answer_label"] for r in rows)),
                       "flagged_rows": sum(bool(r["review_flags"]) for r in rows),
                       "sha256": hashlib.sha256((OUTPUT / f"{s}_candidates.jsonl").read_bytes()).hexdigest()} for s,rows in selected.items()},
        "exclusion_counts": dict(Counter(r["reason"] for r in rejected)),
        "limitations": ["No medical correctness or source recency certification", "No model-generated explanation or answer replacement", "Subject labels and single-choice metadata can be noisy", "Word-count and lexical thresholds are heuristics; short valid explanations may be excluded", "Existing test sets may share knowledge concepts despite lexical isolation", "Review sample is unreviewed and is not an error-rate estimate"],
        "verification": {"unique_sources": 2400, "unique_normalized_questions": 2400,
                         "existing_case_id_overlap": 0, "train_local_validation_exact_official_validation_overlap": 0,
                         "original_answers_preserved": True},
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    cards = []
    for row in review:
        options = "\n".join(f"{k}. {v}" for k,v in row["options"].items())
        cards.append(f'<article><small>{html.escape(row["subject"])} · {html.escape(row["source_id"])}</small><h2>{html.escape(row["question"])}</h2><pre>{html.escape(options)}</pre><p><b>原始答案：</b>{row["answer_label"]}. {html.escape(row["answer"])}</p><p><b>原始解析：</b>{html.escape(row["source_explanation"])}</p><p class="status">待医学复核 · 未重新标注 · 非可直接训练的最终集</p></article>')
    preview = '<!doctype html><html lang="zh"><meta charset="utf-8"><title>MedMCQA 基础知识候选预览</title><style>body{font:16px/1.65 system-ui;max-width:960px;margin:40px auto;padding:0 20px;background:#f5f7fa;color:#17243a}article{background:white;padding:24px;margin:20px 0;border-radius:12px;border:1px solid #dce3eb}h2{font-size:19px}pre{white-space:pre-wrap;font:inherit}small,.status{color:#65748a}</style><h1>基础医学知识 QA · 候选预览</h1><p>训练候选 2,000 条；验证 200 条；测试 200 条。以下每科抽取 5 条，共 30 条，保留原始英文和原始标注。页面不展示测试答案。</p>' + ''.join(cards) + '</html>'
    (OUTPUT / "preview.html").write_text(preview, encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "splits": manifest["splits"], "exclusions": manifest["exclusion_counts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
