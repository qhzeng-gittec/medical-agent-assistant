import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

from datasets import load_dataset

from native_reasoning import missing_case_image_reason
from data_io import save_jsonl


DATASET_ID = "openlifescienceai/medmcqa"
CASE_PATTERN = re.compile(
    r"(?i)(year[- ]old|\bpatient\b|\bpresents?\b|\bcomplains?\b|\bhistory of\b|"
    r"\bon examination\b|\bwoman\b|\bman\b|\bchild\b|\binfant\b|\bpregnant\b)"
)
OPTION_KEYS = ("opa", "opb", "opc", "opd")
OPTION_LABELS = ("A", "B", "C", "D")


def normalize_question(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def normalize_row(sample: dict, source_split: str) -> dict | None:
    question = str(sample.get("question") or "").strip()
    explanation = str(sample.get("exp") or "").strip()
    options = [str(sample.get(key) or "").strip() for key in OPTION_KEYS]
    answer_index = int(sample.get("cop", -1))
    if sample.get("choice_type") != "single":
        return None
    if len(question.split()) < 20 or not CASE_PATTERN.search(question):
        return None
    if len(explanation.split()) < 15 or not all(options):
        return None
    if answer_index not in range(4):
        return None
    if missing_case_image_reason({"case_question": question, "options": dict(zip(OPTION_LABELS, options))}):
        return None

    rendered_options = "\n".join(f"{label}. {option}" for label, option in zip(OPTION_LABELS, options))
    return {
        "id": f"medmcqa-{sample['id']}",
        "source": DATASET_ID,
        "source_split": source_split,
        "source_id": sample["id"],
        "task": "clinical_case_qa",
        "language": "en",
        "case_question": question,
        "options": dict(zip(OPTION_LABELS, options)),
        "prompt": f"{question}\n\n{rendered_options}",
        "answer_label": OPTION_LABELS[answer_index],
        "answer": options[answer_index],
        "evidence": explanation,
        "subject": str(sample.get("subject_name") or "").strip(),
        "topic": str(sample.get("topic_name") or "").strip(),
    }


def load_candidates(path: Path, source_split: str) -> list[dict]:
    dataset = load_dataset("parquet", data_files=str(path.resolve()), split="train")
    rows = []
    seen_questions = set()
    for sample in dataset:
        row = normalize_row(sample, source_split)
        if row is None:
            continue
        normalized = normalize_question(row["case_question"])
        if normalized in seen_questions:
            continue
        seen_questions.add(normalized)
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare explanation-backed clinical vignettes from MedMCQA.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/medmcqa_raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/medmcqa_cases"))
    parser.add_argument("--train-size", type=int, default=2000)
    parser.add_argument("--validation-size", type=int, default=200)
    parser.add_argument("--test-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_dir = args.raw_dir.resolve()
    train_candidates = load_candidates(raw_dir / "train.parquet", "train")
    test_candidates = load_candidates(raw_dir / "validation.parquet", "validation")
    rng = random.Random(args.seed)
    rng.shuffle(train_candidates)
    rng.shuffle(test_candidates)

    required_train = args.train_size + args.validation_size
    if len(train_candidates) < required_train or len(test_candidates) < args.test_size:
        raise RuntimeError(
            f"Not enough filtered cases: train={len(train_candidates)}, test={len(test_candidates)}."
        )

    splits = {
        "train": train_candidates[: args.train_size],
        "validation": train_candidates[args.train_size : required_train],
        "test": test_candidates[: args.test_size],
    }
    normalized_by_split = {
        name: {normalize_question(row["case_question"]) for row in rows}
        for name, rows in splits.items()
    }
    if normalized_by_split["train"] & normalized_by_split["validation"]:
        raise RuntimeError("Clinical case leakage detected between train and validation.")
    if (normalized_by_split["train"] | normalized_by_split["validation"]) & normalized_by_split["test"]:
        raise RuntimeError("Clinical case leakage detected against the held-out official validation split.")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, rows in splits.items():
        save_jsonl(output_dir / f"{split_name}.jsonl", rows)

    manifest = {
        "dataset_id": DATASET_ID,
        "seed": args.seed,
        "selection": "single-choice clinical vignettes with non-empty expert explanations",
        "filtered_candidates": {
            "official_train": len(train_candidates),
            "official_validation": len(test_candidates),
        },
        "split_policy": "train/local validation from official train; test from official validation",
        "splits": {
            name: {
                "samples": len(rows),
                "subjects": dict(Counter(row["subject"] for row in rows)),
            }
            for name, rows in splits.items()
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
