import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset

from data_io import save_jsonl


DATASET_ID = "qiaojin/PubMedQA"
DATASET_CONFIG = "pqa_labeled"


def normalize_row(sample: dict) -> dict:
    contexts = [text.strip() for text in sample["context"]["contexts"]]
    labels = sample["context"]["labels"]
    if not contexts or len(contexts) != len(labels):
        raise ValueError(f"Invalid context sections for PubMed ID {sample['pubid']}.")

    sections = [
        {"label": label.strip().lower(), "text": text}
        for label, text in zip(labels, contexts, strict=True)
    ]
    answer = sample["final_decision"].strip().lower()
    if answer not in {"yes", "no", "maybe"}:
        raise ValueError(f"Unexpected answer {answer!r} for PubMed ID {sample['pubid']}.")

    question = sample["question"].strip()
    explanation = sample["long_answer"].strip()
    if not question or not explanation:
        raise ValueError(f"Missing question or explanation for PubMed ID {sample['pubid']}.")

    pubmed_id = str(sample["pubid"])
    return {
        "id": f"pubmedqa-{pubmed_id}",
        "source": DATASET_ID,
        "source_config": DATASET_CONFIG,
        "pubmed_id": pubmed_id,
        "task": "context_medical_qa",
        "language": "en",
        "context_sections": sections,
        "context": "\n\n".join(f"[{section['label'].upper()}] {section['text']}" for section in sections),
        "question": question,
        "answer": answer,
        "explanation": explanation,
    }


def stratified_split(rows: list[dict], seed: int) -> dict[str, list[dict]]:
    by_answer: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_answer[row["answer"]].append(row)

    rng = random.Random(seed)
    splits = {"train": [], "validation": [], "test": []}
    for answer_rows in by_answer.values():
        rng.shuffle(answer_rows)
        validation_count = round(len(answer_rows) * 0.1)
        test_count = round(len(answer_rows) * 0.1)
        splits["validation"].extend(answer_rows[:validation_count])
        splits["test"].extend(answer_rows[validation_count : validation_count + test_count])
        splits["train"].extend(answer_rows[validation_count + test_count :])

    for split_rows in splits.values():
        rng.shuffle(split_rows)
    return splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare expert-labeled PubMedQA data.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/pubmedqa"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source = load_dataset(DATASET_ID, DATASET_CONFIG, split="train")
    rows = [normalize_row(sample) for sample in source]
    if len({row["id"] for row in rows}) != len(rows):
        raise RuntimeError("Duplicate PubMed IDs found in PubMedQA.")

    splits = stratified_split(rows, args.seed)
    split_ids = {name: {row["id"] for row in split_rows} for name, split_rows in splits.items()}
    if split_ids["train"] & split_ids["validation"] or split_ids["train"] & split_ids["test"]:
        raise RuntimeError("PubMedQA split leakage detected.")
    if split_ids["validation"] & split_ids["test"]:
        raise RuntimeError("PubMedQA validation/test leakage detected.")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_rows in splits.items():
        save_jsonl(output_dir / f"{split_name}.jsonl", split_rows)

    manifest = {
        "dataset_id": DATASET_ID,
        "dataset_config": DATASET_CONFIG,
        "source_type": "expert_labeled",
        "seed": args.seed,
        "split_policy": "stratified by yes/no/maybe answer with unique PubMed IDs",
        "splits": {
            name: {
                "samples": len(split_rows),
                "answers": dict(Counter(row["answer"] for row in split_rows)),
            }
            for name, split_rows in splits.items()
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
