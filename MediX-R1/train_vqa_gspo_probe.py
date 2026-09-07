"""Bounded single-GPU GSPO infrastructure experiment on the existing VQA split."""

import argparse
import hashlib
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import zip_longest
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import PeftModel, get_peft_model_state_dict
from safetensors.torch import load_file
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from data_io import load_jsonl
from llm_judge import judge_responses
from native_reasoning import parse_completion
from original_images import load_model_image


LAB = Path(__file__).resolve().parent
INITIAL = LAB / "outputs/knowledge_experiments_v1/seed42/vqa_knowledge_attention/final_adapter"


def save(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def policy_loss(new: torch.Tensor, old: torch.Tensor, advantage: float) -> tuple[torch.Tensor, torch.Tensor]:
    ratio = (new - old).mean().exp()
    clipped = ratio.clamp(1 - 0.0003, 1 + 0.0004)
    return -torch.minimum(ratio * advantage, clipped * advantage), ratio


def format_score(text: str, stopped_on_eos: bool) -> float:
    parsed = parse_completion(text)
    return float(stopped_on_eos and text.count("</think>") == 1 and parsed["response_complete"])


def select_training_rows(pool: list[dict], count: int) -> list[dict]:
    by_image = {}
    for row in pool:
        by_image.setdefault(row["source_group"], []).append(row)
    selected = [row for layer in zip_longest(*by_image.values()) for row in layer if row is not None][:count]
    if len(selected) != count:
        raise ValueError("Insufficient VQA training questions.")
    return selected


def prepare_prompt(processor, row: dict, system_prompt: str) -> dict:
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "image", "image": load_model_image(row["image_path"], 768)},
                    {"type": "text", "text": row["user_text"]}]}]
    return processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
        enable_thinking=True, return_dict=True, return_tensors="pt")


def trim_completion(ids: list[int], eos: int) -> list[int]:
    return ids[:ids.index(eos) + 1] if eos in ids else ids


def collate_responses(items: list[dict], pad_token_id: int) -> dict:
    sequences, modalities = [], []
    for item in items:
        reply = torch.tensor(item["completion_ids"], dtype=torch.long)
        sequences.append(torch.cat([item["inputs"]["input_ids"][0], reply]))
        modalities.append(torch.cat([item["inputs"]["mm_token_type_ids"][0], torch.zeros_like(reply)]))
    width = max(len(s) for s in sequences)
    return dict(
        input_ids=torch.stack([F.pad(s, (width - len(s), 0), value=pad_token_id) for s in sequences]),
        attention_mask=torch.stack([F.pad(torch.ones_like(s), (width - len(s), 0)) for s in sequences]),
        mm_token_type_ids=torch.stack([F.pad(s, (width - len(s), 0)) for s in modalities]),
        pixel_values=torch.cat([i["inputs"]["pixel_values"] for i in items]),
        image_grid_thw=torch.cat([i["inputs"]["image_grid_thw"] for i in items]),
    )


def response_logprobs(model, items: list[dict], pad_token_id: int) -> list[torch.Tensor]:
    inputs = {k: v.to(model.device) for k, v in collate_responses(items, pad_token_id).items()}
    length = max(len(i["completion_ids"]) for i in items)
    logits = model(**inputs, use_cache=False, logits_to_keep=length + 1).logits[:, :-1]
    return [-F.cross_entropy(logits[index, -len(i["completion_ids"]):].float(),
                torch.tensor(i["completion_ids"], device=model.device), reduction="none")
            for index, i in enumerate(items)]


def microbatches(items: list[dict], maximum: int):
    start = 0
    while start < len(items):
        size = min(maximum, len(items) - start)
        while size > 1 and size * max(len(i["completion_ids"]) for i in items[start:start + size]) > 1024:
            size -= 1
        yield items[start:start + size]
        start += size


def score_rollouts(batch: list[dict], group_size: int, judge_batch_size: int,
                   output_dir: Path, judge_model: str, judge_workers: int = 2) -> float:
    def score_chunk(chunk):
        begin = time.monotonic()
        judgments = judge_responses([i["row"] for i in chunk], [i["public"] for i in chunk],
                                   output_dir, judge_model)
        duration = time.monotonic() - begin
        if any(j["score"] is None for j in judgments):
            raise ValueError("Judge could not assess a training response; inspect the cached judgment.")
        print(f"judge responses={len(chunk)} seconds={duration:.2f}", flush=True)
        return chunk, judgments

    begin = time.monotonic()
    chunks = [batch[offset:offset + judge_batch_size] for offset in range(0, len(batch), judge_batch_size)]
    with ThreadPoolExecutor(max_workers=judge_workers) as executor:
        results = list(executor.map(score_chunk, chunks))
    elapsed = time.monotonic() - begin
    for chunk, judgments in results:
        for item, judgment in zip(chunk, judgments, strict=True):
            item["public"]["judge"] = judgment
    for offset in range(0, len(batch), group_size):
        group = batch[offset:offset + group_size]
        rewards = torch.tensor([0.9 * i["public"]["judge"]["score"] / 2
                                + 0.1 * i["public"]["format_score"] for i in group])
        advantages = (rewards - rewards.mean()) / (rewards.std(correction=0) + 1e-6)
        for item, reward, advantage in zip(group, rewards, advantages, strict=True):
            item["advantage"] = float(advantage)
            item["public"].update(reward=float(reward), advantage=float(advantage))
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompts", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--judge-model", default="gpt-5.5")
    parser.add_argument("--judge-batch-size", type=int, default=16)
    parser.add_argument("--judge-workers", type=int, default=2)
    parser.add_argument("--prompts-per-rollout", type=int, default=8)
    parser.add_argument("--generation-batch-size", type=int, default=4)
    parser.add_argument("--train-micro-batch-size", type=int, default=4)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args()
    if (args.prompts_per_rollout < 2 or args.prompts_per_rollout % 2
            or args.prompts < args.prompts_per_rollout or args.prompts % args.prompts_per_rollout
            or args.group_size < 2):
        parser.error("Use an even prompts-per-rollout, a positive multiple for prompts, and group-size >= 2.")
    if args.judge_batch_size < 1 or args.judge_workers < 1:
        parser.error("Judge batch size and worker count must be positive.")
    if (args.generation_batch_size < 1 or args.group_size % args.generation_batch_size
            or args.train_micro_batch_size < 1 or (2 * args.group_size) % args.train_micro_batch_size):
        parser.error("Generation batch must divide group size; train micro batch must divide the update batch.")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.set_per_process_memory_fraction(0.8)
    data = LAB / "data/knowledge_experiments_v1"
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    pool = [r for r in load_jsonl(data / "train.jsonl") if r["task"] == "vqa"]
    random.shuffle(pool)
    selected = select_training_rows(pool, args.prompts)
    image_groups = {row["source_group"] for row in selected}
    heldout = load_jsonl(data / "validation.jsonl") + load_jsonl(data / "test.jsonl")
    if image_groups & {r["source_group"] for r in heldout if r["task"] == "vqa"}:
        raise ValueError("Selected training images overlap held-out data.")
    config = dict(initial_adapter=str(INITIAL), initial_sha256=hashlib.sha256(
        (INITIAL / "adapter_model.safetensors").read_bytes()).hexdigest(),
        prompts=args.prompts, group_size=args.group_size, prompts_per_rollout=args.prompts_per_rollout,
        judge_batch_size=args.judge_batch_size, judge_workers=args.judge_workers,
        generation_batch_size=args.generation_batch_size, train_micro_batch_size=args.train_micro_batch_size,
        gradient_checkpointing=args.gradient_checkpointing,
        micro_batch_completion_token_budget=1024,
        distinct_training_images=len(image_groups), selection="Image-round-robin over seed42 shuffled training questions",
        responses_per_update=2 * args.group_size, max_new_tokens=args.max_new_tokens,
        learning_rate=1e-6, clip_low=0.0003, clip_high=0.0004, seed=42,
        temperature=1.0, top_p=1.0, top_k=0, dropout=0.0, kl_coefficient=0.0,
        image_max_edge=768, gpu_memory_fraction=0.8, judge_model=args.judge_model,
        reward="0.9 * (judge_score / 2) + 0.1 * native_format_score",
        format="One </think>, nonempty reasoning and final answer, normal EOS termination",
        truncation="Retain capped trajectories, score final answer as generated; format reward requires EOS",
        scope="Infrastructure probe, not a generalization or performance experiment",
        source_ids=[r["source_id"] for r in selected])
    start_prompt = 0
    resume_state = None
    if args.resume_from:
        args.resume_from = args.resume_from.resolve()
        previous_out = args.resume_from.parent
        previous_config = json.loads((previous_out / "config.json").read_text(encoding="utf-8"))
        for key, value in previous_config.items():
            if key not in {"generation_batch_size", "train_micro_batch_size", "gradient_checkpointing", "resume_from", "resume_next_prompt"} and config[key] != value:
                raise ValueError(f"Resume configuration mismatch: {key}")
        resume_state = torch.load(args.resume_from / "training_state.pt", map_location="cpu", weights_only=True)
        start_prompt = resume_state["next_prompt"]
        previous_progress = json.loads((previous_out / "progress.json").read_text(encoding="utf-8"))
        if start_prompt != len(previous_progress["cycles"]) * args.prompts_per_rollout:
            raise ValueError("Resume checkpoint must match the last committed progress record.")
        config.update(resume_from=str(args.resume_from), resume_next_prompt=start_prompt)
    save(out / "config.json", config)
    processor = AutoProcessor.from_pretrained(LAB / "models/Qwen3.5-2B", do_resize=False)
    base = Qwen3_5ForConditionalGeneration.from_pretrained(
        LAB / "models/Qwen3.5-2B", dtype=torch.bfloat16, attn_implementation="sdpa")
    model = PeftModel.from_pretrained(base, args.resume_from or INITIAL, is_trainable=True).to("cuda")
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if not trainable or any("lora_" not in n or ".visual." in n for n in trainable):
        raise RuntimeError("Only language LoRA parameters may be trainable.")
    initial_weights = load_file(INITIAL / "adapter_model.safetensors")
    if initial_weights.keys() != get_peft_model_state_dict(model).keys():
        raise RuntimeError("LoRA checkpoint parameter keys differ from the initial adapter.")
    optimizer = torch.optim.AdamW(trainable.values(), lr=1e-6, weight_decay=0.0)
    torch.cuda.reset_peak_memory_stats()
    steps, records, cycle_metrics = [], [], []
    counts = dict(generation_seconds=0.0, judge_seconds=0.0, old_logprob_seconds=0.0, update_seconds=0.0)
    eos = processor.tokenizer.eos_token_id
    pad = processor.tokenizer.pad_token_id
    if resume_state:
        optimizer.load_state_dict(resume_state["optimizer"])
        steps, cycle_metrics, counts = previous_progress["steps"], previous_progress["cycles"], previous_progress["timing"]
        disk_weights = load_file(args.resume_from / "adapter_model.safetensors")
        weight_error = max(float((p.detach().cpu() - disk_weights[n]).abs().max())
                           for n, p in get_peft_model_state_dict(model).items())
        optimizer_steps = [int(s["step"]) for s in optimizer.state.values()]
        if weight_error != 0 or not optimizer_steps or set(optimizer_steps) != {len(steps)}:
            raise RuntimeError("Resume weights or optimizer step counters differ from the checkpoint.")
        save(out / "resume_verification.json", dict(status="passed", next_prompt=start_prompt,
            weight_max_error=weight_error, optimizer_states=len(optimizer_steps), optimizer_step=len(steps)))
        del disk_weights
        for filename in ["rollouts.jsonl", "scored_rollouts.jsonl"]:
            committed = [r for r in load_jsonl(previous_out / filename)
                         if r["cycle"] < start_prompt // args.prompts_per_rollout]
            if len(committed) != start_prompt * args.group_size:
                raise ValueError(f"Incomplete committed trajectories: {filename}")
            with (out / filename).open("w", encoding="utf-8") as file:
                for r in committed:
                    file.write(json.dumps(r, ensure_ascii=False) + "\n")
            if filename == "scored_rollouts.jsonl":
                records = [{k: v for k, v in r.items() if k != "old_logprobs"} for r in committed]
        torch.set_rng_state(resume_state["rng"])
        torch.cuda.set_rng_state(resume_state["cuda_rng"])
        save(out / "progress.json", dict(steps=steps, cycles=cycle_metrics, timing=counts))
        print(f"resumed prompts={start_prompt} updates={len(steps)}", flush=True)
        del resume_state

    for start in range(start_prompt, len(selected), args.prompts_per_rollout):
        cycle = start // args.prompts_per_rollout
        batch = []
        model.eval()
        for row in selected[start:start + args.prompts_per_rollout]:
            inputs = prepare_prompt(processor, row, manifest["system_prompt"])
            for sample_start in range(0, args.group_size, args.generation_batch_size):
                torch.cuda.synchronize()
                begin = time.monotonic()
                with torch.inference_mode():
                    generated = model.generate(**{k: v.to("cuda") for k, v in inputs.items()},
                        do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                        repetition_penalty=1.0, max_new_tokens=args.max_new_tokens,
                        eos_token_id=eos, pad_token_id=pad, num_return_sequences=args.generation_batch_size,
                        use_cache=True)
                torch.cuda.synchronize()
                elapsed = time.monotonic() - begin
                counts["generation_seconds"] += elapsed
                for index, sequence in enumerate(generated):
                    ids = trim_completion(sequence[inputs["input_ids"].shape[1]:].tolist(), eos)
                    sample = sample_start + index
                    stopped = ids[-1] == eos
                    raw = processor.tokenizer.decode(ids[:-1] if stopped else ids, skip_special_tokens=False).strip()
                    candidate_id = f"{row['source_id']}::cycle{cycle}::sample{sample}"
                    public = dict(source_id=candidate_id, original_source_id=row["source_id"],
                        source_group=row["source_group"], cycle=cycle, sample=sample,
                        policy_version=len(steps), completion_ids=ids, raw_prediction=raw,
                        **parse_completion(raw), generated_tokens=len(ids), stopped_on_eos=stopped,
                        format_score=format_score(raw, stopped), generation_seconds=elapsed / args.generation_batch_size)
                    batch.append(dict(inputs=inputs, row=dict(row, source_id=candidate_id), public=public,
                                      completion_ids=ids))
                print(f"rollout {row['source_id']} responses={args.generation_batch_size} seconds={elapsed:.2f}", flush=True)
                del generated
        with (out / "rollouts.jsonl").open("a", encoding="utf-8") as file:
            for item in batch:
                file.write(json.dumps(item["public"], ensure_ascii=False) + "\n")
        counts["judge_seconds"] += score_rollouts(
            batch, args.group_size, args.judge_batch_size, out / "judge_calls", args.judge_model,
            args.judge_workers)
        model.train()
        begin = time.monotonic()
        with torch.no_grad():
            for offset in range(0, len(batch), 2 * args.group_size):
                for microbatch in microbatches(batch[offset:offset + 2 * args.group_size], args.train_micro_batch_size):
                    for item, old in zip(microbatch, response_logprobs(model, microbatch, pad), strict=True):
                        item["old"] = old.detach().cpu()
                        if not torch.isfinite(item["old"]).all():
                            raise RuntimeError("Nonfinite old log probabilities.")
        torch.cuda.synchronize()
        counts["old_logprob_seconds"] += time.monotonic() - begin
        with (out / "scored_rollouts.jsonl").open("a", encoding="utf-8") as file:
            for item in batch:
                file.write(json.dumps(dict(item["public"], old_logprobs=item["old"].tolist()), ensure_ascii=False) + "\n")
        begin = time.monotonic()
        update_size = 2 * args.group_size
        for offset in range(0, len(batch), update_size):
            minibatch = batch[offset:offset + update_size]
            optimizer.zero_grad(set_to_none=True)
            losses, ratios = [], []
            initial_logprob_max_error = None
            for microbatch in microbatches(minibatch, args.train_micro_batch_size):
                new_values = response_logprobs(model, microbatch, pad)
                micro_losses = []
                for item, new in zip(microbatch, new_values, strict=True):
                    old = item["old"].to("cuda")
                    if offset == 0:
                        error = float((new.detach() - old).abs().max())
                        initial_logprob_max_error = max(initial_logprob_max_error or 0.0, error)
                        if error > 0.01:
                            raise RuntimeError(f"Unchanged-policy logprob mismatch: {error}")
                    loss, ratio = policy_loss(new, old, item["advantage"])
                    if not torch.isfinite(loss):
                        raise RuntimeError("Nonfinite GSPO loss.")
                    micro_losses.append(loss)
                    losses.append(float(loss.detach()))
                    ratios.append(float(ratio.detach()))
                (torch.stack(micro_losses).sum() / len(minibatch)).backward()
                del new_values, micro_losses, new, loss, ratio
            grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable.values(), 1.0, error_if_nonfinite=True))
            if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
                raise RuntimeError("Frozen parameter received a gradient.")
            optimizer.step()
            step = dict(step=len(steps) + 1, cycle=cycle, loss=sum(losses) / len(losses),
                        ratios=ratios, grad_norm=grad_norm, initial_logprob_max_error=initial_logprob_max_error)
            steps.append(step)
            print(f"update {json.dumps(step)}", flush=True)
        torch.cuda.synchronize()
        counts["update_seconds"] += time.monotonic() - begin
        records.extend(i["public"] for i in batch)
        cycle_metrics.append(dict(cycle=cycle, policy_version_after=len(steps),
            reward_mean=sum(i["public"]["reward"] for i in batch) / len(batch),
            zero_advantage_groups=sum(all(i["advantage"] == 0 for i in batch[k:k + args.group_size])
                                      for k in range(0, len(batch), args.group_size))))
        checkpoint = out / f"checkpoint-cycle-{cycle + 1}"
        model.save_pretrained(checkpoint)
        torch.save(dict(optimizer=optimizer.state_dict(), rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state(), next_prompt=start + args.prompts_per_rollout), checkpoint / "training_state.pt")
        save(out / "progress.json", dict(steps=steps, cycles=cycle_metrics, timing=counts))
        del batch
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
    changes = {n: float((p.detach().cpu() - initial_weights[n]).abs().max())
               for n, p in get_peft_model_state_dict(model).items()}
    if not any(v > 0 for v in changes.values()) or not any(s["grad_norm"] > 0 for s in steps):
        raise RuntimeError("No nonzero policy update; inspect tied rewards and gradients.")
    model.save_pretrained(out / "final_adapter")
    processor.save_pretrained(out / "final_adapter")
    model.eval()
    anchor = dict(inputs=prepare_prompt(processor, selected[0], manifest["system_prompt"]), completion_ids=records[0]["completion_ids"])
    with torch.no_grad():
        expected = response_logprobs(model, [anchor], pad)[0].cpu()
    del model, base
    torch.cuda.empty_cache()
    reloaded_base = Qwen3_5ForConditionalGeneration.from_pretrained(
        LAB / "models/Qwen3.5-2B", dtype=torch.bfloat16, attn_implementation="sdpa")
    model = PeftModel.from_pretrained(reloaded_base, out / "final_adapter").to("cuda").eval()
    with torch.no_grad():
        reloaded = response_logprobs(model, [anchor], pad)[0].cpu()
    roundtrip_error = float((expected - reloaded).abs().max())
    if roundtrip_error > 1e-5:
        raise RuntimeError(f"Saved adapter changed log probabilities: {roundtrip_error}")
    save(out / "summary.json", dict(status="completed", config=config, steps=steps, cycles=cycle_metrics,
        timing=counts, total_seconds=time.monotonic() - started, total_seconds_scope="current process including resume/load/save; timing includes prior committed cycles",
        generated_tokens=sum(r["generated_tokens"] for r in records),
        format_rate=sum(r["format_score"] for r in records) / len(records),
        truncated_responses=sum(not r["stopped_on_eos"] for r in records),
        trainable_parameters=sum(p.numel() for p in trainable.values()),
        changed_parameter_tensors=sum(v > 0 for v in changes.values()),
        max_parameter_change=max(changes.values()),
        saved_adapter_roundtrip_max_logprob_error=roundtrip_error,
        peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
        initial_adapter_unchanged=hashlib.sha256((INITIAL / "adapter_model.safetensors").read_bytes()).hexdigest() == config["initial_sha256"],
        heldout_used_for_training=False, efficacy_evaluated=False))
    print(f"probe_complete={out}", flush=True)


if __name__ == "__main__":
    main()
