import argparse
import json
import hashlib
import re
import math
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration, set_seed
from trl import SFTConfig, SFTTrainer

from common.original_images import load_model_image
from common.io import load_jsonl


LAB_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = LAB_DIR / "models" / "Qwen3.5-2B"


class MultitaskCollator:
    def __init__(self, processor: AutoProcessor, system_prompt: str, image_max_edge: int | None = None) -> None:
        self.processor = processor
        self.system_prompt = system_prompt
        self.image_max_edge = image_max_edge

    def __call__(self, examples: list[dict]) -> dict[str, torch.Tensor]:
        prompt_texts, full_texts, images = [], [], []
        for example in examples:
            user_content = []
            if example.get("image_path"):
                images.append(load_model_image(example["image_path"], self.image_max_edge))
                user_content.append({"type": "image"})
            user_content.append({"type": "text", "text": example["user_text"]})
            prompt_messages = [
                {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
                {"role": "user", "content": user_content},
            ]
            if not example["reasoning_content"].strip():
                raise ValueError(f"Missing reasoning supervision: {example['source_id']}")
            assistant = {"role": "assistant", "content": example["target"], "reasoning_content": example["reasoning_content"]}
            prompt = self.processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
            full = self.processor.apply_chat_template(prompt_messages + [assistant], tokenize=False, add_generation_prompt=False, enable_thinking=True)
            if not full.startswith(prompt):
                raise ValueError(f"Training/inference template mismatch: {example['source_id']}")
            prompt_texts.append(prompt)
            full_texts.append(full)
        kwargs = {"return_tensors": "pt", "padding": True}
        if images:
            kwargs["images"] = images
        prompt_batch = self.processor(text=prompt_texts, **kwargs)
        batch = self.processor(text=full_texts, **kwargs)
        labels = batch["input_ids"].clone()
        for i, example in enumerate(examples):
            prompt_length = int(prompt_batch["attention_mask"][i].sum())
            if not torch.equal(batch["input_ids"][i, :prompt_length], prompt_batch["input_ids"][i, :prompt_length]):
                raise ValueError(f"Tokenized prompt is not a prefix: {example['source_id']}")
            labels[i, :prompt_length] = -100
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one Qwen3.5-2B LoRA medical experiment recipe.")
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--recipes-dir", type=Path, default=LAB_DIR / "data" / "native_reasoning" / "recipes")
    parser.add_argument("--output-root", type=Path, default=LAB_DIR / "outputs" / "native_experiments")
    parser.add_argument("--lora-scope", choices=("attention", "attention_ffn"), default="attention_ffn")
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gpu-memory-fraction", type=float, default=1.0)
    parser.add_argument("--image-max-edge", type=int, help="Downscale larger images with Lanczos; preserve aspect ratio and smaller originals.")
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--initial-adapter-dir", type=Path, help="Continue an existing adapter with a fresh optimizer and schedule.")
    parser.add_argument("--base-adapter-dir", type=Path, help="Merge a frozen CPT adapter into the base in memory, then train a NEW SFT adapter.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.base_adapter_dir and args.initial_adapter_dir:
        raise ValueError("CPT base initialization and continuing an SFT adapter are separate experiments.")
    if args.initial_adapter_dir and args.resume_from_checkpoint:
        raise ValueError("Choose adapter initialization or checkpoint resume, not both.")
    if args.initial_adapter_dir:
        initial_config = json.loads((args.initial_adapter_dir.parent / "run_config.json").read_text(encoding="utf-8"))
        if initial_config["lora_scope"] != args.lora_scope or not initial_config["enable_thinking"]:
            raise ValueError("Initial adapter must match the requested LoRA scope and thinking protocol.")
    if args.batch_size < 1 or args.gradient_accumulation_steps < 1:
        raise ValueError("Batch size and gradient accumulation must be positive.")
    if not 0 < args.gpu_memory_fraction <= 1:
        raise ValueError("GPU memory fraction must be in (0, 1].")
    set_seed(args.seed)
    recipe_dir = (args.recipes_dir / args.recipe).resolve()
    recipe_manifest = json.loads((recipe_dir / "manifest.json").read_text(encoding="utf-8"))
    if recipe_manifest.get("protocol") != "qwen_native_open_qa_llm_judge" or recipe_manifest.get("enable_thinking") is not True:
        raise ValueError("Use the current native open-QA recipe generated by build_native_recipes.py.")
    if hashlib.sha256((recipe_dir / "train.jsonl").read_bytes()).hexdigest() != recipe_manifest["train_sha256"]:
        raise ValueError("Recipe hash does not match its manifest; rebuild the recipe.")
    train_rows = load_jsonl(recipe_dir / "train.jsonl")
    validation_rows = load_jsonl(recipe_dir / "validation.jsonl")
    processor = AutoProcessor.from_pretrained(MODEL_PATH, do_resize=False)
    processor.tokenizer.padding_side = "right"
    collator = MultitaskCollator(processor, recipe_manifest["system_prompt"], args.image_max_edge)

    task_probes = {}
    for row in train_rows:
        if row["task"] in task_probes:
            continue
        probe = collator([row])
        task_probes[row["task"]] = {
            "input_shape": list(probe["input_ids"].shape),
            "supervised_tokens": int((probe["labels"] != -100).sum().item()),
        }
    if any(probe["supervised_tokens"] == 0 for probe in task_probes.values()):
        raise RuntimeError(f"A task has no supervised assistant tokens: {task_probes}")

    output_dir = (args.output_root / f"{args.recipe}_{args.lora_scope}").resolve()
    if (output_dir / "final_adapter").exists() and not args.dry_run:
        raise FileExistsError(f"Refusing to overwrite trained adapter: {output_dir}")
    estimated_steps = max(
        1,
        math.ceil(math.ceil(len(train_rows) / args.batch_size) / args.gradient_accumulation_steps * args.epochs),
    )
    if args.max_steps > 0:
        estimated_steps = args.max_steps
    warmup_steps = round(estimated_steps * 0.05)
    attention_modules = "q_proj|k_proj|v_proj|o_proj|in_proj_qkv|in_proj_z|out_proj"
    target_modules = rf"model\.language_model\.layers\.\d+\.(?:self_attn|linear_attn)\.(?:{attention_modules})"
    if args.lora_scope == "attention_ffn":
        target_modules += r"|model\.language_model\.layers\.\d+\.mlp\.(?:gate_proj|up_proj|down_proj)"
    run_config = {
        "recipe": args.recipe,
        "recipe_sha256": recipe_manifest["train_sha256"],
        "validation_sha256": hashlib.sha256((recipe_dir / "validation.jsonl").read_bytes()).hexdigest(),
        "model_path": str(MODEL_PATH),
        "enable_thinking": True,
        "image_preprocessing": "RGB; optional aspect-preserving Lanczos downscale; right/bottom zero-padding to 32; do_resize=False",
        "image_max_edge": args.image_max_edge,
        "batch_size": args.batch_size,
        "gpu_memory_fraction": args.gpu_memory_fraction,
        "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
        "lora_scope": args.lora_scope,
        "target_modules": target_modules,
        "max_steps": args.max_steps,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "eval_steps": args.eval_steps,
        "checkpoint_policy": "final training state; validation loss is diagnostic only",
        "estimated_steps": estimated_steps,
        "warmup_steps": warmup_steps,
        "seed": args.seed,
        "base_adapter_dir": str(args.base_adapter_dir.resolve()) if args.base_adapter_dir else None,
        "base_adapter_sha256": hashlib.sha256((args.base_adapter_dir / "adapter_model.safetensors").read_bytes()).hexdigest() if args.base_adapter_dir else None,
        "initial_adapter_dir": str(args.initial_adapter_dir.resolve()) if args.initial_adapter_dir else None,
        "initial_adapter_sha256": hashlib.sha256((args.initial_adapter_dir / "adapter_model.safetensors").read_bytes()).hexdigest() if args.initial_adapter_dir else None,
        "resume_from_checkpoint": (
            str(args.resume_from_checkpoint.resolve()) if args.resume_from_checkpoint else None
        ),
        "task_probes": task_probes,
    }
    print(json.dumps(run_config, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        print("dry_run_ok=true", flush=True)
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A CUDA GPU with BF16 support is required.")
    torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)

    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    if args.base_adapter_dir:
        # Loading the CPT adapter must not change the fresh SFT adapter's random initialization.
        sft_rng_state = torch.get_rng_state()
        model = PeftModel.from_pretrained(model, args.base_adapter_dir).merge_and_unload(safe_merge=True)
        del model.peft_config
        torch.set_rng_state(sft_rng_state)
    model.config.use_cache = False
    model.enable_input_require_grads()
    selected_modules = [name for name, module in model.named_modules() if re.fullmatch(target_modules, name)]
    if not selected_modules or (args.lora_scope == "attention_ffn" and not any(".mlp." in n for n in selected_modules)):
        raise RuntimeError("Requested language attention/FFN modules were not found.")
    run_config["selected_modules"] = selected_modules
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    if args.initial_adapter_dir:
        model = PeftModel.from_pretrained(model, args.initial_adapter_dir, is_trainable=True)
        peft_config = None
    training_args = SFTConfig(
        output_dir=str(output_dir),
        max_length=None,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=False,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=0,
        optim="adamw_torch",
        seed=args.seed,
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=Dataset.from_list(train_rows),
        eval_dataset=Dataset.from_list(validation_rows),
        processing_class=processor,
        peft_config=peft_config,
    )
    trainer.model.print_trainable_parameters()
    if any(p.requires_grad for n, p in trainer.model.named_parameters() if ".visual." in n):
        raise RuntimeError("Visual encoder must remain frozen for this experiment.")
    torch.cuda.reset_peak_memory_stats()
    resume_checkpoint = None
    if args.resume_from_checkpoint is not None:
        resume_checkpoint = str(args.resume_from_checkpoint.resolve())
    result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    train_metrics = dict(result.metrics)
    train_metrics["peak_gpu_memory_gb"] = torch.cuda.max_memory_allocated() / 1024**3
    train_metrics["peak_gpu_reserved_gb"] = torch.cuda.max_memory_reserved() / 1024**3
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)
    trainer.save_model(str(output_dir / "final_adapter"))
    processor.save_pretrained(output_dir / "final_adapter")
    if args.base_adapter_dir:
        (output_dir / "final_adapter" / "base_initialization.json").write_text(
            json.dumps({"base_model": str(MODEL_PATH),
                        "required_cpt_adapter": run_config["base_adapter_dir"],
                        "required_cpt_adapter_sha256": run_config["base_adapter_sha256"],
                        "loading_order": "Load original base; merge CPT adapter; then load this SFT adapter."}, indent=2),
            encoding="utf-8",
        )
    (output_dir / "training_complete.json").write_text(
        json.dumps({"global_step": trainer.state.global_step, "expected_steps": estimated_steps}),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
