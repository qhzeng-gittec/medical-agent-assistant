import argparse
import hashlib
import json
import random
from pathlib import Path

from datasets import Dataset

from data_io import save_jsonl


LAB_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Preserve original VQA-RAD images and build a 90/5/5 project split by image.")
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache/huggingface/datasets/flaviagiammarino___vqa-rad/default/0.0.0/bcf91e7654fb9d51c8ab6a5b82cacf3fafd2fae9")
    parser.add_argument("--output-dir", type=Path, default=LAB_DIR / "data/vqa_rad_original")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    image_dir = args.output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    rows, duplicates = [], []
    images, seen = {}, {}
    for split in ("train", "test"):
        dataset = Dataset.from_file(str(args.cache_dir / f"vqa-rad-{split}.arrow"))
        for index, sample in enumerate(dataset):
            image = sample["image"].convert("RGB")
            pixel_digest = hashlib.sha256(image.tobytes()).hexdigest()
            image_id = pixel_digest[:16]
            if image_id not in images:
                path = image_dir / f"{image_id}.png"
                image.save(path, format="PNG", optimize=True)
                images[image_id] = {"width": image.width, "height": image.height, "pixel_sha256": pixel_digest}
            row = {
                "id": f"{split}-{index:05d}", "source_split": split, "source": "flaviagiammarino/vqa-rad",
                "image_id": image_id, "image_file": f"images/{image_id}.png",
                "question": sample["question"].strip(), "answer": sample["answer"].strip(),
            }
            key = (image_id, " ".join(row["question"].lower().split()), " ".join(row["answer"].lower().split()))
            if key in seen:
                duplicates.append({"id": row["id"], "duplicate_of": seen[key]})
            else:
                seen[key] = row["id"]
                rows.append(row)
    image_ids = sorted(images)
    random.Random(args.seed).shuffle(image_ids)
    holdout_count = round(len(image_ids) * 0.05)
    image_splits = {"validation": set(image_ids[:holdout_count]), "test": set(image_ids[holdout_count:2*holdout_count]), "train": set(image_ids[2*holdout_count:])}
    splits = {name: [r for r in rows if r["image_id"] in ids] for name, ids in image_splits.items()}
    for name, items in splits.items():
        save_jsonl(args.output_dir / f"{name}.jsonl", items)
    save_jsonl(args.output_dir / "duplicates.jsonl", duplicates)
    manifest = {
        "source": "flaviagiammarino/vqa-rad", "seed": args.seed,
        "split_policy": "official train+test pooled; exact duplicate QA removed; 90/5/5 by original-pixel image identity",
        "benchmark_status": "custom project split, not official VQA-RAD benchmark; historical test images now enter training",
        "image_storage": "original RGB pixels, lossless PNG; no resize, crop or upscaling",
        "images": images, "source_qa": sum(len(s) for s in splits.values()) + len(duplicates),
        "duplicate_qa_removed": len(duplicates),
        "splits": {name: {"qa": len(items), "images": len(image_splits[name])} for name, items in splits.items()},
        "original_images_over_384": sum(max(v["width"], v["height"]) > 384 for v in images.values()),
        "cross_split_image_overlap": 0,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "images"}, indent=2))


if __name__ == "__main__":
    main()
