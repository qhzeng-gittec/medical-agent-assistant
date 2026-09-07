"""Batched text-only inference with the existing 2048/1792 thinking protocol."""
import torch
from transformers import LogitsProcessor

from native_reasoning import SYSTEM_PROMPT, parse_completion


class BatchThinkingBudget(LogitsProcessor):
    def __init__(self, prompt_length, close_id, batch_size):
        self.prompt_length = prompt_length
        self.close_id = close_id
        self.forced = [False] * batch_size

    def __call__(self, input_ids, scores):
        generated = input_ids[:, self.prompt_length:]
        if generated.shape[1] >= 1792:
            mask = ~(generated == self.close_id).any(dim=1)
            for i in mask.nonzero().flatten().tolist():
                scores[i].fill_(-float('inf'))
                scores[i, self.close_id] = 0
                self.forced[i] = True
        return scores


def generate_batch(model, processor, rows):
    assert all(not r['image_path'] for r in rows)
    processor.tokenizer.padding_side = 'left'
    messages = [[{'role': 'system', 'content': SYSTEM_PROMPT},
                 {'role': 'user', 'content': [{'type': 'text', 'text': r['user_text']}]}] for r in rows]
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                enable_thinking=True, return_dict=True, return_tensors='pt', padding=True).to(model.device)
    prompt_length = inputs['input_ids'].shape[1]
    eos = processor.tokenizer.eos_token_id
    budget = BatchThinkingBudget(prompt_length, processor.tokenizer.convert_tokens_to_ids('</think>'), len(rows))
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=2048, do_sample=False,
                    eos_token_id=eos, pad_token_id=processor.tokenizer.pad_token_id, logits_processor=[budget])
    results = []
    for i, output in enumerate(outputs):
        ids = output[prompt_length:].tolist()
        stopped = eos in ids
        if stopped:
            ids = ids[:ids.index(eos)+1]
        text = processor.tokenizer.decode(ids[:-1] if stopped else ids, skip_special_tokens=False).strip()
        # A finished row can be padded while another row continues past the thinking budget.
        forced = budget.forced[i] and len(ids) >= 1793
        results.append(dict(parse_completion(text), raw_prediction=text, generated_tokens=len(ids),
                    max_new_tokens=2048, stop_reason='eos' if stopped else 'length' if len(ids)>=2048 else 'other',
                    input_tokens=int(inputs['attention_mask'][i].sum()), generation_args={'do_sample': False},
                    reasoning_budget=1792, forced_reasoning_end=forced, enable_thinking=True,
                    enforce_thinking_budget=True, sampling_profile='legacy', presence_penalty=0.0,
                    generation_batch_size=len(rows)))
    return results
