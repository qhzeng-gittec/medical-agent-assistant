"""Raw-text medical CPT on language attention AND FFN; keep the original base intact."""
import argparse
import hashlib
import inspect
import json
import math
import os
import re
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
FAST_KERNELS = os.environ.get('MEDICAL_CPT_FAST_KERNELS') == '1'
if FAST_KERNELS:
    sys.path.insert(0, str(LAB/'.gpu-kernels'))
    os.environ.setdefault('CC', str(LAB/'.gpu-kernels/triton/runtime/tcc/tcc.exe'))
    os.environ.setdefault('CUDA_PATH', str(LAB/'.gpu-kernels/triton/backends/nvidia'))

import torch
if FAST_KERNELS:
    # Import FLA before PEFT/Transformers model discovery to avoid a circular-import fallback.
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration, Trainer, TrainingArguments, set_seed
from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule

from common.io import load_jsonl

TARGETS = (r'model\.language_model\.layers\.\d+\.(?:self_attn|linear_attn)\.'
           r'(?:q_proj|k_proj|v_proj|o_proj|in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj)'
           r'|model\.language_model\.layers\.\d+\.mlp\.(?:gate_proj|up_proj|down_proj)')


class ProseCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, rows):
        batch = self.tokenizer.pad([{'input_ids': r['input_ids']} for r in rows],
                                   padding=True, return_tensors='pt', pad_to_multiple_of=8)
        labels = batch['input_ids'].clone()
        labels[batch['attention_mask'] == 0] = -100
        batch['labels'] = labels
        return batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=LAB/'data/cpt_knowledge_coverage_v1')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--rank', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=5e-5)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--gradient-accumulation-steps', type=int, default=4)
    parser.add_argument('--max-steps', type=int, default=-1)
    parser.add_argument('--resume-from-checkpoint', type=Path)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--allow-unverified-coverage', action='store_true',
                        help='Explicit exploratory experiment only; does not certify semantic coverage.')
    args = parser.parse_args()
    attention_kernel = inspect.getclosurevars(torch_chunk_gated_delta_rule).nonlocals['implementation']
    if FAST_KERNELS:
        if attention_kernel is not chunk_gated_delta_rule:
            raise RuntimeError('FLA was requested but Transformers selected a different kernel.')
    if args.output_dir is None:
        args.output_dir = LAB/'outputs'/args.data_dir.name/f'cpt_seed{args.seed}'
    if (args.output_dir/'training_complete.json').exists():
        raise FileExistsError(args.output_dir)
    coverage_path=args.data_dir/'coverage_status.json'
    manifest = json.loads((args.data_dir/'manifest.json').read_text(encoding='utf-8'))
    if args.allow_unverified_coverage:
        if manifest.get('experiment_type') != 'expanded_candidate_not_full_coverage' or manifest.get('semantic_full_coverage_verified') is not False:
            raise ValueError('Exploratory override requires an explicitly unverified candidate corpus.')
        audit = json.loads((args.data_dir/'audit.json').read_text(encoding='utf-8'))
        if audit['status'] != 'passed':
            raise ValueError('Candidate token audit has not passed.')
    if coverage_path.exists():
        coverage=json.loads(coverage_path.read_text(encoding='utf-8'))
        if not coverage['ready_for_training'] and not args.allow_unverified_coverage:
            raise ValueError(f"Knowledge coverage is not ready: {coverage['accepted']}/{coverage['required']} items accepted.")
    manifest_sha = hashlib.sha256((args.data_dir/'manifest.json').read_bytes()).hexdigest()
    if coverage_path.exists() and coverage['corpus_manifest_sha256'] != manifest_sha:
        raise ValueError('Knowledge-coverage audit does not match the training corpus.')
    config_path = args.output_dir/'run_config.json'
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding='utf-8'))
        if not args.resume_from_checkpoint or previous['corpus_manifest_sha256'] != manifest_sha:
            raise ValueError('Existing output requires an explicit resume with the same corpus manifest.')
    for split in ['train','validation']:
        assert hashlib.sha256((args.data_dir/f'{split}.jsonl').read_bytes()).hexdigest() == manifest[f'{split}_sha256']
    train_rows = load_jsonl(args.data_dir/'train.jsonl')
    val_rows = load_jsonl(args.data_dir/'validation.jsonl')
    if args.max_steps > 0:
        # Exercise maximum-length rows first, so a probe tests the real memory requirement.
        train_rows = sorted(train_rows, key=lambda row: -len(row['input_ids']))[:args.batch_size * args.gradient_accumulation_steps * args.max_steps]
        val_rows = val_rows[:4]
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(.8)
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(LAB/'models/Qwen3.5-2B')
    tokenizer.padding_side = 'right'
    model = Qwen3_5ForConditionalGeneration.from_pretrained(LAB/'models/Qwen3.5-2B',
                dtype=torch.bfloat16,attn_implementation='sdpa')
    selected = [name for name, _ in model.named_modules() if re.fullmatch(TARGETS,name)]
    ffn = [name for name in selected if '.mlp.' in name]
    attention = [name for name in selected if '.mlp.' not in name]
    assert len(ffn) == 72 and len(attention) >= 96
    model.config.use_cache = False
    model.enable_input_require_grads()
    model = get_peft_model(model,LoraConfig(r=args.rank,lora_alpha=args.rank*2,lora_dropout=.05,
                           target_modules=TARGETS,bias='none',task_type='CAUSAL_LM'))
    assert not any(p.requires_grad for n,p in model.named_parameters() if '.visual.' in n)
    trainable = {n:p.numel() for n,p in model.named_parameters() if p.requires_grad}
    assert trainable and all('lora_' in n for n in trainable)
    steps = args.max_steps if args.max_steps > 0 else math.ceil(math.ceil(len(train_rows)/args.batch_size)/args.gradient_accumulation_steps)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    config = dict(stage='CPT',lora_scope='attention_ffn',rank=args.rank,alpha=args.rank*2,
        learning_rate=args.learning_rate,epochs=1,seed=args.seed,batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,expected_steps=steps,
        selected_attention_modules=attention,selected_ffn_modules=ffn,trainable_parameters=sum(trainable.values()),
        target_modules=TARGETS,base_model=str((LAB/'models/Qwen3.5-2B').resolve()),
        corpus_manifest_sha256=manifest_sha, max_length=manifest['max_length'],
        allow_unverified_coverage=args.allow_unverified_coverage,
        semantic_full_coverage_verified=manifest.get('semantic_full_coverage_verified'),
        data_dir=str(args.data_dir.resolve()), packing=manifest.get('packing'),
        train_sha256=manifest['train_sha256'],validation_sha256=manifest['validation_sha256'],
        objective=manifest['objective'],train_tokens=manifest['train_tokens'],
        linear_attention_kernel=attention_kernel.__module__,torch_version=torch.__version__,
        checkpoint_policy='one prespecified epoch; final checkpoint; no selection on task test results',
        probe_only=args.max_steps>0)
    (args.output_dir/'run_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in config.items() if not isinstance(v,list)},indent=2),flush=True)
    training_args = TrainingArguments(output_dir=str(args.output_dir),num_train_epochs=1,max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,learning_rate=args.learning_rate,
        warmup_steps=max(1,round(steps*.05)),weight_decay=.01,lr_scheduler_type='cosine',
        logging_steps=1 if args.max_steps>0 else 25,eval_strategy='steps',eval_steps=250,
        save_strategy='steps',save_steps=250,save_total_limit=2,bf16=True,
        gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},
        remove_unused_columns=False,report_to='none',dataloader_num_workers=0,optim='adamw_torch',
        seed=args.seed,prediction_loss_only=True,disable_tqdm=True)
    trainer = Trainer(model=model,args=training_args,data_collator=ProseCollator(tokenizer),
        train_dataset=Dataset.from_list(train_rows),eval_dataset=Dataset.from_list(val_rows),processing_class=tokenizer)
    baseline = trainer.evaluate(metric_key_prefix='before_cpt') if not args.resume_from_checkpoint else {}
    if baseline:
        (args.output_dir/'before_cpt_metrics.json').write_text(json.dumps(baseline,indent=2),encoding='utf-8')
    torch.cuda.reset_peak_memory_stats()
    result = trainer.train(resume_from_checkpoint=str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None)
    metrics = dict(result.metrics,peak_gpu_memory_gb=torch.cuda.max_memory_allocated()/1024**3,
                   **trainer.evaluate(metric_key_prefix='after_cpt'))
    trainer.save_metrics('train',metrics)
    trainer.save_model(str(args.output_dir/'final_adapter'))
    tokenizer.save_pretrained(args.output_dir/'final_adapter')
    assert trainer.state.global_step == steps
    (args.output_dir/'training_complete.json').write_text(json.dumps(dict(global_step=trainer.state.global_step,
                expected_steps=steps,probe_only=args.max_steps>0,metrics=metrics),indent=2),encoding='utf-8')
    print(json.dumps(metrics,indent=2),flush=True)


if __name__ == '__main__':
    main()
