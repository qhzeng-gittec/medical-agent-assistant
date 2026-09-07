import argparse
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path

from PIL import Image
from transformers import AutoTokenizer

from common.io import load_jsonl, save_jsonl
from data_processing.complete_reasoning import source_items
from common.native_reasoning import SYSTEM_PROMPT, make_example, missing_case_image_reason
from data_processing.prepare_medmcqa_cases import normalize_question


LAB_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = LAB_DIR / "data"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build image-disjoint native Qwen reasoning recipes.")
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR / "native_reasoning")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--annotations-dir", type=Path, default=DATA_DIR / "reasoning_full")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(LAB_DIR / "models" / "Qwen3.5-2B")
    for token in ("<think>", "</think>"):
        if tokenizer.encode(token, add_special_tokens=False) != [tokenizer.convert_tokens_to_ids(token)]:
            raise RuntimeError(f"Missing native single token: {token}")

    pools = {split: {task: [] for task in ("vqa", "context", "case")} for split in ("train", "validation", "test")}
    excluded = []
    source_hashes = {}

    def read_source(path: Path) -> list[dict]:
        source_hashes[str(path.relative_to(DATA_DIR))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return load_jsonl(path)

    items = source_items()
    for directory in ("vqa_rad_original", "pubmedqa"):
        for split in pools:
            path = DATA_DIR / directory / f"{split}.jsonl"
            source_hashes[str(path.relative_to(DATA_DIR))] = hashlib.sha256(path.read_bytes()).hexdigest()
    annotations = read_source(args.annotations_dir / "annotations.jsonl")
    annotation_map = {r["id"]: r for r in annotations}
    if len(annotation_map) != len(annotations) or set(annotation_map) != {r["id"] for r in items}:
        raise RuntimeError("Reasoning coverage is incomplete or duplicated; finish complete_reasoning.py before building.")
    open_cases = read_source(DATA_DIR / "open_cases/annotations.jsonl")
    open_case_map = {r["id"]: r for r in open_cases}
    expected_open_ids = {r["id"] for r in items if r["task"] == "case" and annotation_map[r["id"]]["status"] == "keep"}
    if len(open_case_map) != len(open_cases) or set(open_case_map) != expected_open_ids:
        raise RuntimeError("Open-question conversion is incomplete; finish prepare_open_cases.py before building.")
    checked_images = set()
    for item in items:
        annotation = annotation_map[item["id"]]
        split, task, row = item["split"], item["task"], item["source"]
        if annotation["status"] != "keep":
            excluded.append({"split": split, "task": task, "id": item["id"], "reason": "teacher_quarantine", "review_note": annotation["review_note"]})
            continue
        if task == "case":
            rewrite = open_case_map[item["id"]]
            if rewrite["status"] != "keep":
                excluded.append({"split": split, "task": task, "id": item["id"], "reason": "open_question_quarantine", "review_note": rewrite["review_note"]})
                continue
            row = dict(row, open_question=rewrite["question"])
            annotation = dict(annotation, reasoning_content=rewrite["reasoning_content"], final_answer=rewrite["final_answer"], teacher_model=rewrite["teacher_model"], rationale_provenance="synthetic_open_question_reasoning_not_expert_reviewed")
        if not annotation["reasoning_content"].strip() or not annotation["final_answer"].strip():
            raise ValueError(f"Empty reasoning or final answer: {item['id']}")
        if item["image_path"]:
            image_path = Path(item["image_path"])
            if image_path not in checked_images:
                with Image.open(image_path) as img:
                    img.verify()
                checked_images.add(image_path)
        example = make_example(row, task, item["image_path"])
        example.update(reasoning_content=annotation["reasoning_content"], target=annotation["final_answer"], reasoning_source=annotation["rationale_provenance"], teacher_model=annotation["teacher_model"], reference_explanation=item.get("source_explanation", ""))
        if task == "case":
            example["normalized_question"] = normalize_question(row["open_question"])
            example["original_normalized_question"] = normalize_question(row["case_question"])
            example["reference"] = annotation["final_answer"]
            example["reference_explanation"] = annotation["reasoning_content"]
        pools[split][task].append(example)
    for split in pools:
        for row in read_source(DATA_DIR / "medmcqa_cases" / f"{split}.jsonl"):
            reason = missing_case_image_reason(row)
            if reason:
                excluded.append({"split": split, "task": "case", "id": row["id"], "reason": reason, "row": row})

    seen_questions = {}
    for split in ("test", "validation", "train"):
        retained = []
        for row in pools[split]["case"]:
            question = row["normalized_question"]
            if question in seen_questions:
                excluded.append({"split": split, "task": "case", "id": row["source_id"], "reason": "duplicate_open_question", "duplicate_of": seen_questions[question]})
            else:
                seen_questions[question] = row["source_id"]
                retained.append(row)
        pools[split]["case"] = retained

    stats = {}
    budgets = {}
    final_reserves = {}
    for task in ("vqa", "context", "case"):
        for field in ("source_id", "source_group") + (("normalized_question", "original_normalized_question") if task == "case" else ()):
            sets = [{r[field] for r in pools[s][task]} for s in pools]
            if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
                raise RuntimeError(f"Cross-split leakage: {task}/{field}")
        for split in pools:
            rows = pools[split][task]
            for row in rows:
                row["final_tokens"] = len(tokenizer.encode(row["target"], add_special_tokens=False)) + 1
                row["has_reasoning_reference"] = bool(row["reasoning_content"])
                if not row["has_reasoning_reference"]:
                    row["supervised_tokens"] = None
                    continue
                prompt = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": row["user_text"]}]
                prefix = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, enable_thinking=True)
                full = tokenizer.apply_chat_template(
                    prompt + [{"role": "assistant", "content": row["target"], "reasoning_content": row["reasoning_content"]}],
                    tokenize=False, add_generation_prompt=False, enable_thinking=True,
                )
                if not full.startswith(prefix):
                    raise RuntimeError(f"Template prefix mismatch: {row['source_id']}")
                row["supervised_tokens"] = len(tokenizer.encode(full[len(prefix):], add_special_tokens=False))
            lengths = sorted(r["supervised_tokens"] for r in rows if r["supervised_tokens"] is not None)
            stats[f"{split}/{task}"] = {
                "samples": len(rows), "unique_sources": len({r["source_id"] for r in rows}),
                "unique_source_groups": len({r["source_group"] for r in rows}),
                "reasoning_references": len(lengths), "supervised_tokens": sum(lengths),
                "p95": lengths[math.floor(0.95 * (len(lengths) - 1))] if lengths else None,
                "max": max(lengths) if lengths else None,
            }
        # Freeze budgets from train/validation only, without inspecting test target lengths.
        max_target = max(stats[f"{s}/{task}"]["max"] or 0 for s in ("train", "validation"))
        max_final = max(r["final_tokens"] for s in ("train", "validation") for r in pools[s][task])
        final_reserves[task] = max(256, math.ceil((max_final + 64) / 128) * 128)
        budgets[task] = max(2048, math.ceil((max_target + final_reserves[task] + 256) / 256) * 256)

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    save_jsonl(output / "excluded.jsonl", excluded)
    for split in pools:
        save_jsonl(output / f"{split}.jsonl", [row for rows in pools[split].values() for row in rows])
    manifest = {
        "protocol": "qwen_native_open_qa_llm_judge", "enable_thinking": True, "system_prompt": SYSTEM_PROMPT,
        "open_question_coverage": {"eligible": len(expected_open_ids), "processed": len(open_cases), "complete": True},
        "open_question_deduplication": "normalized question; preserve test, then validation, then train",
        "image_preprocessing": "original pixels; zero-pad to 32; do_resize=False",
        "vqa_split_manifest": json.loads((DATA_DIR / "vqa_rad_original/manifest.json").read_text(encoding="utf-8")),
        "annotation_coverage": {"eligible": len(items), "processed": len(annotations), "complete": True},
        "seed": args.seed, "source_sha256": source_hashes, "split_stats": stats,
        "generation_max_new_tokens": budgets,
        "generation_final_reserve_tokens": final_reserves,
        "budget_policy": "train/validation maximum supervised tokens + final reserve + 256, rounded up to 256; minimum 2048 after native-thinking validation smoke; reserve >= maximum final answer + 64",
        "exclusions": dict(Counter(f"{r['split']}/{r['task']}/{r['reason']}" for r in excluded)),
        "sampling_policy": "each accepted reasoning source once; reasoning only",
        "cross_split_overlap": 0,
        "limitations": ["synthetic reasoning is not expert medical review", "custom VQA split, not an official benchmark", "judge scores are model assessments, not clinical validation"],
    }
    for name, tasks in {"all_reasoning": ("vqa", "context", "case")}.items():
        recipe_dir = output / "recipes" / name
        recipe_dir.mkdir(parents=True, exist_ok=True)
        train = [r for task in tasks for r in pools["train"][task]]
        validation = [r for task in tasks for r in pools["validation"][task] if r["has_reasoning_reference"]]
        random.Random(args.seed).shuffle(train)
        save_jsonl(recipe_dir / "train.jsonl", train)
        save_jsonl(recipe_dir / "validation.jsonl", validation)
        recipe = dict(manifest, name=name, train_samples=len(train), validation_samples=len(validation))
        recipe["train_task_counts"] = dict(Counter(r["task"] for r in train))
        recipe["train_task_supervised_tokens"] = {t: sum(r["supervised_tokens"] for r in train if r["task"] == t) for t in tasks}
        recipe["train_sha256"] = hashlib.sha256((recipe_dir / "train.jsonl").read_bytes()).hexdigest()
        (recipe_dir / "manifest.json").write_text(json.dumps(recipe, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "stats": stats, "budgets": budgets, "exclusions": manifest["exclusions"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
