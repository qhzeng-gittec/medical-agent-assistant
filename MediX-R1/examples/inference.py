"""Run text or image inference with a published PEFT adapter."""

import argparse
from pathlib import Path

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from common.native_reasoning import SYSTEM_PROMPT
from common.original_images import load_model_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adapter', required=True, help='Hugging Face model ID or local adapter directory')
    parser.add_argument('--question', required=True)
    parser.add_argument('--image', type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--max-new-tokens', type=int, default=256)
    args = parser.parse_args()
    config = PeftConfig.from_pretrained(args.adapter)
    processor = AutoProcessor.from_pretrained(args.adapter, do_resize=False)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        config.base_model_name_or_path, revision=config.revision,
        dtype=torch.bfloat16, attn_implementation='sdpa',
    ).to(args.device)
    model = PeftModel.from_pretrained(model, args.adapter).eval()
    content = ([{'type': 'image'}] if args.image else []) + [{'type': 'text', 'text': args.question}]
    messages = [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    images = [load_model_image(str(args.image), 768)] if args.image else None
    inputs = processor(text=[text], images=images, return_tensors='pt').to(args.device)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    print(processor.decode(output[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True))


if __name__ == '__main__':
    main()
