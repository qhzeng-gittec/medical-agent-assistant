"""Batched text inference with configurable thinking limits and Qwen sampling."""
import torch
from transformers import LogitsProcessor

from common.native_reasoning import SYSTEM_PROMPT, parse_completion


class BatchThinkingBudget(LogitsProcessor):
    def __init__(self, prompt_length, close_id, batch_size, reasoning_budget=1792):
        self.prompt_length = prompt_length
        self.close_id = close_id
        self.forced = [False] * batch_size
        self.reasoning_budget = reasoning_budget

    def __call__(self, input_ids, scores):
        generated = input_ids[:, self.prompt_length:]
        if generated.shape[1] >= self.reasoning_budget:
            mask = ~(generated == self.close_id).any(dim=1)
            for i in mask.nonzero().flatten().tolist():
                scores[i].fill_(-float('inf'))
                scores[i, self.close_id] = 0
                self.forced[i] = True
        return scores


def generate_batch(model, processor, rows, *, max_new_tokens=2048, reasoning_budget=1792, sampling_profile='legacy'):
    assert 0 < reasoning_budget < max_new_tokens - 1
    assert all(not r['image_path'] for r in rows)
    processor.tokenizer.padding_side = 'left'
    messages = [[{'role': 'system', 'content': SYSTEM_PROMPT},
                 {'role': 'user', 'content': [{'type': 'text', 'text': r['user_text']}]}] for r in rows]
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                enable_thinking=True, return_dict=True, return_tensors='pt', padding=True).to(model.device)
    prompt_length = inputs['input_ids'].shape[1]
    eos = processor.tokenizer.eos_token_id
    budget = BatchThinkingBudget(prompt_length, processor.tokenizer.convert_tokens_to_ids('</think>'), len(rows), reasoning_budget)
    processors = [budget]
    generation_args = {'do_sample': False}
    presence_penalty = 0.0
    if sampling_profile == 'qwen35':
        from evaluation.evaluate_native_reasoning import GeneratedPresencePenalty, qwen_sampling_parameters
        parameters = qwen_sampling_parameters(True, False)
        presence_penalty = parameters.pop('presence_penalty')
        generation_args.update(do_sample=True, repetition_penalty=1.0, **parameters)
        processors.append(GeneratedPresencePenalty(prompt_length, presence_penalty))
    elif sampling_profile != 'legacy':
        raise ValueError(sampling_profile)
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, **generation_args,
                    eos_token_id=eos, pad_token_id=processor.tokenizer.pad_token_id, logits_processor=processors)
    results = []
    for i, output in enumerate(outputs):
        ids = output[prompt_length:].tolist()
        stopped = eos in ids
        if stopped:
            ids = ids[:ids.index(eos)+1]
        text = processor.tokenizer.decode(ids[:-1] if stopped else ids, skip_special_tokens=False).strip()
        # A finished row can be padded while another row continues past the thinking budget.
        forced = budget.forced[i] and len(ids) >= reasoning_budget + 1
        results.append(dict(parse_completion(text), raw_prediction=text, generated_tokens=len(ids),
                    max_new_tokens=max_new_tokens, stop_reason='eos' if stopped else 'length' if len(ids)>=max_new_tokens else 'other',
                    input_tokens=int(inputs['attention_mask'][i].sum()), generation_args=generation_args,
                    reasoning_budget=reasoning_budget, forced_reasoning_end=forced, enable_thinking=True,
                    enforce_thinking_budget=True, sampling_profile=sampling_profile, presence_penalty=presence_penalty,
                    generation_batch_size=len(rows)))
    return results
