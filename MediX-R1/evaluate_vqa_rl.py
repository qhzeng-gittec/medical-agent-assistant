"""Fixed validation protocol for the SFT adapter and its RL continuation."""

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from data_io import load_jsonl
from evaluate_native_reasoning import generate
from llm_judge import judge_responses, summarize_judgments
from train_vqa_gspo_probe import format_score, save


LAB = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    data = LAB / "data/knowledge_experiments_v1"
    rows = [r for r in load_jsonl(data / "validation.jsonl") if r["task"] == "vqa"]
    train_groups = {r["source_group"] for r in load_jsonl(data / "train.jsonl") if r["task"] == "vqa"}
    if train_groups & {r["source_group"] for r in rows}:
        raise ValueError("Validation images overlap training images.")
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    adapter_hash = hashlib.sha256((args.adapter_dir / "adapter_model.safetensors").read_bytes()).hexdigest()
    config = dict(adapter_dir=str(args.adapter_dir.resolve()), adapter_sha256=adapter_hash,
                  split="validation", data_sha256=hashlib.sha256((data / "validation.jsonl").read_bytes()).hexdigest(),
                  source_ids=[r["source_id"] for r in rows], distinct_images=len({r["source_group"] for r in rows}),
                  decoding="greedy", max_new_tokens=512, image_max_edge=768,
                  enforce_thinking_budget=False, judge_model="gpt-5.5", judge_workers=2, judge_batch_size=16)
    save(out / "config.json", config)
    torch.cuda.set_per_process_memory_fraction(0.8)
    torch.manual_seed(42)
    processor = AutoProcessor.from_pretrained(LAB / "models/Qwen3.5-2B", do_resize=False)
    base = Qwen3_5ForConditionalGeneration.from_pretrained(
        LAB / "models/Qwen3.5-2B", dtype=torch.bfloat16, attn_implementation="sdpa")
    model = PeftModel.from_pretrained(base, args.adapter_dir).to("cuda").eval()
    responses = []
    begin = time.monotonic()
    with (out / "generations.jsonl").open("w", encoding="utf-8") as file:
        for index, row in enumerate(rows):
            response = generate(model, processor, row, manifest["system_prompt"], 512,
                                decoding="greedy", image_max_edge=768, enforce_thinking_budget=False)
            response.update(source_id=row["source_id"],
                            format_score=format_score(response["raw_prediction"], response["stop_reason"] == "eos"))
            responses.append(response)
            file.write(json.dumps(response, ensure_ascii=False) + "\n")
            file.flush()
            print(f"validation {index + 1}/{len(rows)} tokens={response['generated_tokens']}", flush=True)
    generation_seconds = time.monotonic() - begin
    del model, base, processor
    torch.cuda.empty_cache()

    def grade(start):
        return judge_responses(rows[start:start + 16], responses[start:start + 16], out / "judge_calls", "gpt-5.5")

    begin = time.monotonic()
    with ThreadPoolExecutor(max_workers=2) as executor:
        judgments = [j for group in executor.map(grade, range(0, len(rows), 16)) for j in group]
    judge_seconds = time.monotonic() - begin
    with (out / "scored.jsonl").open("w", encoding="utf-8") as file:
        for row, response, judgment in zip(rows, responses, judgments, strict=True):
            file.write(json.dumps(dict(response, source_group=row["source_group"], judge=judgment), ensure_ascii=False) + "\n")
    if hashlib.sha256((args.adapter_dir / "adapter_model.safetensors").read_bytes()).hexdigest() != adapter_hash:
        raise RuntimeError("Evaluation adapter changed on disk.")
    summary = dict(status="completed", config=config, content=summarize_judgments(judgments),
                   format_rate=sum(r["format_score"] for r in responses) / len(responses),
                   truncated_responses=sum(r["stop_reason"] == "length" for r in responses),
                   generation_seconds=generation_seconds, judge_seconds=judge_seconds,
                   total_seconds=time.monotonic() - started)
    save(out / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
