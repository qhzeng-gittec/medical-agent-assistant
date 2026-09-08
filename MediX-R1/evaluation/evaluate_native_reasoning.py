import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoProcessor, LogitsProcessor, LogitsProcessorList, Qwen3_5ForConditionalGeneration

from common.native_reasoning import parse_completion
from common.original_images import load_model_image
from evaluation.llm_judge import judge_responses, summarize_judgments


LAB_DIR = Path(__file__).resolve().parents[1]


class ThinkingBudget(LogitsProcessor):
    def __init__(self, prompt_length: int, reasoning_budget: int, close_token_id: int):
        self.prompt_length = prompt_length
        self.reasoning_budget = reasoning_budget
        self.close_token_id = close_token_id
        self.forced = False

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        generated = input_ids[:, self.prompt_length:]
        if generated.shape[1] >= self.reasoning_budget and not (generated == self.close_token_id).any():
            scores.fill_(-float("inf"))
            scores[:, self.close_token_id] = 0
            self.forced = True
        return scores


class GeneratedPresencePenalty(LogitsProcessor):
    def __init__(self, prompt_length: int, penalty: float):
        self.prompt_length = prompt_length
        self.penalty = penalty

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        seen = torch.zeros_like(scores, dtype=torch.bool)
        seen.scatter_(1, input_ids[:, self.prompt_length:], True)
        return scores - seen.to(scores.dtype) * self.penalty


def qwen_sampling_parameters(enable_thinking: bool, has_image: bool) -> dict:
    if enable_thinking:
        return dict(temperature=0.6 if has_image else 1.0, top_p=0.95, top_k=20,
                    presence_penalty=0.0 if has_image else 1.5)
    return dict(temperature=0.7 if has_image else 1.0, top_p=0.8 if has_image else 1.0,
                top_k=20, presence_penalty=1.5 if has_image else 2.0)


def generate(model, processor, row: dict, system_prompt: str, budget: int, decoding: str = "sample", reasoning_budget: int | None = None, image_max_edge: int | None = None, *, sampling_profile: str = "legacy", enable_thinking: bool = True, enforce_thinking_budget: bool = True, stop_repetition: bool = False) -> dict:
    content = []
    if row["image_path"]:
        content.append({"type": "image", "image": load_model_image(row["image_path"], image_max_edge)})
    content.append({"type": "text", "text": row["user_text"]})
    inputs = processor.apply_chat_template(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": content}],
        tokenize=True, add_generation_prompt=True, enable_thinking=enable_thinking,
        return_dict=True, return_tensors="pt",
    ).to(model.device)
    eos_id = processor.tokenizer.eos_token_id
    reasoning_budget = budget - 256 if reasoning_budget is None else reasoning_budget
    if not 0 < reasoning_budget < budget - 1:
        raise ValueError("The reasoning budget must leave room for </think> and the final response.")
    thinking_budget = ThinkingBudget(
        inputs["input_ids"].shape[1], reasoning_budget, processor.tokenizer.convert_tokens_to_ids("</think>"),
    )
    generation_args = {"do_sample": decoding == "sample"}
    if decoding == "sample":
        generation_args.update(temperature=0.6 if row["image_path"] else 1.0, top_p=0.95, top_k=20)
    processors = LogitsProcessorList()
    if enable_thinking and enforce_thinking_budget:
        processors.append(thinking_budget)
    presence_penalty = 0.0
    if sampling_profile == "qwen35":
        if decoding != "sample":
            raise ValueError("The qwen35 sampling profile requires sampling.")
        parameters = qwen_sampling_parameters(enable_thinking, bool(row["image_path"]))
        presence_penalty = parameters.pop("presence_penalty")
        generation_args.update(parameters, repetition_penalty=1.0)
        if presence_penalty:
            processors.append(GeneratedPresencePenalty(inputs["input_ids"].shape[1], presence_penalty))
    elif sampling_profile != "legacy":
        raise ValueError(f"Unknown sampling profile: {sampling_profile}")
    stopping_args = {}
    repetition_stop = None
    if stop_repetition:
        from common.calibration_stopping import RepetitionStop
        repetition_stop = RepetitionStop(processor.tokenizer, inputs['input_ids'].shape[1])
        stopping_args['stopping_criteria'] = [repetition_stop]
    with torch.inference_mode():
        output = model.generate(
            **inputs, max_new_tokens=budget, **generation_args,
            eos_token_id=eos_id, pad_token_id=processor.tokenizer.pad_token_id,
            logits_processor=processors, **stopping_args,
        )
    ids = output[0, inputs["input_ids"].shape[1]:].tolist()
    stopped_on_eos = bool(ids and ids[-1] == eos_id)
    text = processor.tokenizer.decode(ids[:-1] if stopped_on_eos else ids, skip_special_tokens=False).strip()
    parsed = parse_completion(text) if enable_thinking else dict(reasoning="", final_answer=text, response_complete=bool(text))
    return dict(
        parsed, raw_prediction=text, generated_tokens=len(ids), max_new_tokens=budget,
        stop_reason=("eos" if stopped_on_eos else "repetition" if repetition_stop and repetition_stop.triggered
                     else "overlong_suspected_repetition" if stop_repetition and len(ids) >= budget
                     else "length" if len(ids) >= budget else "other"),
        input_tokens=inputs["input_ids"].shape[1],
        generation_args=generation_args,
        reasoning_budget=reasoning_budget, forced_reasoning_end=thinking_budget.forced,
        enable_thinking=enable_thinking, enforce_thinking_budget=enable_thinking and enforce_thinking_budget,
        sampling_profile=sampling_profile, presence_penalty=presence_penalty,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate native thinking with natural-language final answers.")
    parser.add_argument("--name", required=True)
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--data-dir", type=Path, default=LAB_DIR / "data" / "native_reasoning")
    parser.add_argument("--output-root", type=Path, default=LAB_DIR / "outputs" / "native_evaluations")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--tasks", nargs="+", choices=("vqa", "context", "case", "knowledge"), default=("vqa", "context", "case"))
    parser.add_argument("--image-max-edge", type=int, help="Must match adapter training; for the base model use the same setting as the comparison runs.")
    parser.add_argument("--gpu-memory-fraction", type=float, default=1.0)
    parser.add_argument("--limit-per-task", type=int, default=20, help="Random samples per task; 0 evaluates the complete split.")
    parser.add_argument("--judge-model", default="gpt-5.5")
    parser.add_argument("--decoding", choices=("sample", "greedy"), default="sample")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reasoning-budget", type=int, help="Maximum thinking tokens; default reserves 256 tokens for the final response.")
    parser.add_argument("--max-new-tokens", type=int, help="Generation limit; default comes from train/validation lengths.")
    args = parser.parse_args()
    if not 0 < args.gpu_memory_fraction <= 1:
        parser.error("GPU memory fraction must be in (0, 1].")
    if args.max_new_tokens is not None and args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    if args.limit_per_task < 0:
        parser.error("--limit-per-task must be nonnegative")
    manifest = json.loads((args.data_dir / "manifest.json").read_text(encoding="utf-8"))
    data_path = args.data_dir / f"{args.split}.jsonl"
    with data_path.open(encoding="utf-8") as file:
        rows = [json.loads(line) for line in file if line.strip()]
    output_dir = args.output_root / args.name / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    for task in args.tasks:
        if any((output_dir / filename).exists() for filename in (f"{task}.jsonl", f"{task}_generations.jsonl")):
            raise FileExistsError(f"Use a new --name to preserve previous evaluation: {output_dir}")
    model_path = LAB_DIR / "models" / "Qwen3.5-2B"
    torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    processor = AutoProcessor.from_pretrained(model_path, do_resize=False)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(model_path, dtype=torch.bfloat16, attn_implementation="sdpa")
    if args.adapter_dir:
        run_config = json.loads((args.adapter_dir.parent / "run_config.json").read_text(encoding="utf-8"))
        if not run_config["enable_thinking"]:
            raise ValueError("This evaluator requires a native-thinking adapter trained with the current pipeline.")
        trained_edge = run_config.get("image_max_edge")
        if args.image_max_edge is not None and args.image_max_edge != trained_edge:
            raise ValueError("Evaluation image resolution must match adapter training.")
        args.image_max_edge = trained_edge
        model = PeftModel.from_pretrained(model, args.adapter_dir)
    model.to("cuda").eval()
    summary = {
        "name": args.name, "split": args.split, "protocol": manifest["protocol"],
        "verification_only": manifest.get("verification_only", False),
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "adapter_dir": str(args.adapter_dir) if args.adapter_dir else None,
        "budget_override": args.max_new_tokens, "generation_max_new_tokens": manifest["generation_max_new_tokens"],
        "generation_final_reserve_tokens": manifest["generation_final_reserve_tokens"],
        "evaluation": "LLM judge only; 0 incorrect, 1 partial, 2 correct; null unjudgeable",
        "score_formula": "task score = 50 * mean(non-null judge scores); overall = equal-weight mean of selected task scores; undefined if any task has no scorable samples",
        "tasks": list(args.tasks),
        "judge_model": args.judge_model,
        "decoding": args.decoding, "seed": args.seed,
        "reasoning_budget_override": args.reasoning_budget,
        "image_max_edge": args.image_max_edge,
        "gpu_memory_fraction": args.gpu_memory_fraction,
    }
    for task in args.tasks:
        task_rows = [r for r in rows if r["task"] == task]
        if args.limit_per_task:
            task_rows = random.Random(args.seed).sample(task_rows, min(args.limit_per_task, len(task_rows)))
        if not task_rows:
            raise ValueError(f"Empty evaluation task: {task}")
        results = []
        budget = args.max_new_tokens or manifest["generation_max_new_tokens"][task]
        reasoning_budget = args.reasoning_budget if args.reasoning_budget is not None else budget - manifest["generation_final_reserve_tokens"][task]
        if not 0 < reasoning_budget < budget - 1:
            raise ValueError("Increase --max-new-tokens or provide a smaller --reasoning-budget.")
        with (output_dir / f"{task}_generations.jsonl").open("w", encoding="utf-8") as file:
            for index, row in enumerate(task_rows, start=1):
                sample_seed = (args.seed + int(hashlib.sha256(row["source_id"].encode()).hexdigest()[:8], 16)) % 2**32
                torch.manual_seed(sample_seed)
                result = generate(model, processor, row, manifest["system_prompt"], budget, args.decoding, reasoning_budget, args.image_max_edge)
                result["sample_seed"] = sample_seed
                result.update(source_id=row["source_id"], reference=row["reference"])
                results.append(result)
                file.write(json.dumps(result, ensure_ascii=False) + "\n")
                file.flush()
                print(f"{task} {index}/{len(task_rows)} tokens={result['generated_tokens']} stop={result['stop_reason']}", flush=True)
        with (output_dir / f"{task}.jsonl").open("w", encoding="utf-8") as file:
            for start in range(0, len(results), 4):
                judgments = judge_responses(task_rows[start:start+4], results[start:start+4], output_dir / "judge_calls", args.judge_model)
                for response, judgment in zip(results[start:start+4], judgments, strict=True):
                    response["judge"] = judgment
                    file.write(json.dumps(response, ensure_ascii=False) + "\n")
                file.flush()
        summary[task] = summarize_judgments([r["judge"] for r in results])
    task_scores = [summary[task]["score_0_to_100"] for task in args.tasks]
    summary["overall_score_0_to_100"] = sum(task_scores) / len(task_scores) if all(s is not None for s in task_scores) else None
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
