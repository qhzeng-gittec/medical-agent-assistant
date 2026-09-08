"""Load the published CPT/SFT/RL chain in the same BF16 merge order as evaluation."""
import argparse
import hashlib
import json
import os
from pathlib import Path

# Set before importing Hub/Transformers, which cache loading configuration.
os.environ['HF_DEACTIVATE_ASYNC_LOAD'] = '1'
from huggingface_hub import snapshot_download


def verify_package(folder):
    folder = Path(folder)
    manifest = json.loads((folder / 'release_manifest.json').read_text(encoding='utf-8'))
    for name, details in manifest['stages'].items():
        for filename, expected in details['files'].items():
            with (folder / name / filename).open('rb') as stream:
                actual = hashlib.file_digest(stream, 'sha256').hexdigest()
            if actual != expected:
                raise ValueError(f'Checksum mismatch: {name}/{filename}')
    return manifest


def load_chain(folder, stage, device='cpu', base_model=None):
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    folder = Path(folder)
    manifest = verify_package(folder)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        base_model or manifest['base_model'],
        **({} if base_model else {'revision': manifest['base_revision']}),
        dtype=torch.bfloat16, attn_implementation='sdpa',
    )
    # Merge on the same dtype before moving to the inference device.
    for dependency in manifest['stages'][stage]['merge_before']:
        model = PeftModel.from_pretrained(model, folder / dependency).merge_and_unload(safe_merge=True)
        # PEFT 0.18.1 leaves this empty bookkeeping attribute after unloading.
        del model.peft_config
    model = PeftModel.from_pretrained(model, folder / stage).to(device).eval()
    processor = AutoProcessor.from_pretrained(folder, do_resize=False)
    return model, processor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', help='Local published package; otherwise use the pinned registry revision')
    parser.add_argument('--stage', choices=['cpt', 'sft', 'sft_only', 'rl'], default='rl')
    parser.add_argument('--question')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--base-model', help='Optional local base directory for offline use')
    parser.add_argument('--max-new-tokens', type=int, default=512)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if args.package:
        folder = Path(args.package)
    else:
        registry = json.loads((Path(__file__).resolve().parents[1] / 'configs/expanded_chain.json').read_text(encoding='utf-8'))
        folder = Path(snapshot_download(registry['repository'], revision=registry['revision']))
    verify_package(folder)
    if args.verify_only:
        print('All four adapter weights and configurations match the release manifest.')
        return
    if not args.question:
        parser.error('--question is required for generation')
    import torch
    from common.native_reasoning import SYSTEM_PROMPT

    model, processor = load_chain(folder, args.stage, args.device, args.base_model)
    messages = [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': args.question}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    inputs = processor(text=[prompt], return_tensors='pt').to(args.device)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    print(processor.decode(output[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True))


if __name__ == '__main__':
    main()
