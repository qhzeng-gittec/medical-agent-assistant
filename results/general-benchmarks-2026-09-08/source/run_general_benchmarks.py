"""Paired local zero-shot likelihood benchmarks and chat-format diagnostics."""
import argparse
from collections import Counter
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from prepare_general_benchmarks import DATA, ROOT, SEED, SOURCES, prepare
from run_causal_fact_experiment import LAB, available_commit_mib, load_base, read, save, sha

MODEL = LAB / 'models/Qwen3.5-2B'
CPT = LAB / 'outputs/cpt_medical_v1/cpt_seed42/final_adapter'
SFT = LAB / 'outputs/cpt_medical_v1/sft_seed42/vqa_knowledge_case_context_attention_ffn/final_adapter'
SYSTEM = 'Answer the multiple-choice question. Return only the single best option letter.'


def records():
    return [json.loads(line) for line in (DATA / 'evaluation.jsonl').read_text(encoding='utf-8').splitlines()]


def chat_prompt(row):
    instruction = 'Choose the most plausible continuation:\n' if row['benchmark'] == 'hellaswag' else ''
    return instruction + row['question'] + '\n\n' + '\n'.join(
        f'{label}. {option}' for label, option in zip('ABCDE', row['options']))


def parse_letter(text, count):
    match = re.match(r'^\s*(?:(?i:answer)\s*:\s*)?\*{0,2}([A-E])\*{0,2}(?:[.)]|\s*$)', text)
    return match.group(1) if match and match.group(1) in 'ABCDE'[:count] else None


def encode_pair(tok, context, candidate):
    prefix = tok.encode(context, add_special_tokens=False)
    full = tok.encode(context + ' ' + candidate, add_special_tokens=False)
    if full[:len(prefix)] != prefix or len(full) == len(prefix):
        raise ValueError('Unstable context/continuation token boundary')
    return full, full[len(prefix):]


def plan():
    if (ROOT / 'plan.json').exists():
        return read(ROOT / 'plan.json')
    from transformers import AutoTokenizer
    manifest = prepare()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    rows = records()
    pairs = [encode_pair(tokenizer, row['context'], candidate) for row in rows for candidate in row['candidates']]
    max_length = max(len(full) for full, _ in pairs)
    chats = [tokenizer.apply_chat_template([dict(role='system', content=SYSTEM), dict(role='user', content=chat_prompt(row))],
             tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False) for row in rows]
    assert max(max_length, max(map(len, chats))) <= 4096
    training = read(SFT.parent / 'run_config.json')
    assert training['base_adapter_sha256'] == sha(CPT / 'adapter_model.safetensors')
    frozen = [p for p in MODEL.iterdir() if p.suffix in ('.json', '.safetensors', '.jinja')]
    frozen += [adapter / name for adapter in (CPT, SFT) for name in ('adapter_config.json', 'adapter_model.safetensors')]
    frozen += [DATA / 'evaluation.jsonl', DATA / 'manifest.json', Path(__file__), LAB / 'prepare_general_benchmarks.py',
               LAB / 'gpu_runtime.py', LAB / 'run_causal_fact_experiment.py', SFT.parent / 'run_config.json']
    value = dict(seed=SEED, independent_questions=1500, per_benchmark=manifest['per_benchmark'],
                 arms={'baseline': [], 'cpt_sft': [str(CPT), str(SFT)]},
                 model='Local post-trained Qwen3.5-2B; baseline means original weights before project training',
                 adapter_loading='Load original BF16 model; merge CPT with safe_merge; attach fresh SFT adapter in inference mode.',
                 frozen_files={str(p): sha(p) for p in frozen},
                 primary={'mmlu': 'Zero-shot plain-text subject preamble, letter continuation summed log likelihood (acc). Nonmedical subset.',
                          'arc_challenge': 'Zero-shot Question/Answer prompt, answer continuation sum log likelihood divided by candidate character count (acc_norm).',
                          'hellaswag': 'Zero-shot cleaned activity/context, ending continuation sum log likelihood divided by candidate character count (acc_norm).'},
                 scoring='Teacher-forced continuation only, no EOS target; one separating space; save summed, character-normalized and token-normalized scores. Raw and character-normalized accuracy reported separately.',
                 likelihood_max_input_tokens=max_length, likelihood_max_target_tokens=max(len(target) for _, target in pairs),
                 likelihood_candidate_batch=1, chat_batch=4, chat_system=SYSTEM, chat_max_new_tokens=16,
                 chat_max_input_tokens=max(map(len, chats)), chat_thinking=False,
                 chat_decoding='Greedy; no sampling, no repetition penalty; free leading letter accuracy plus first-token option-letter ranking.',
                 alignment='One unpadded candidate per forward. First question of each benchmark: compare last-logit target slices against separate unpadded full forward target slices, tolerance 0.125 logprob; chat first-step logits against direct forward with matching Qwen positions. See protocol_amendment.json.',
                 uncertainty='Per-question paired bootstrap 10000 replicates, seed20260908; exact McNemar; Holm adjustment over three primary tests. Report repair/regression counts. No pooled general-capability score.',
                 limitations=[manifest['scope'], manifest['contamination_limit'],
                              'Single training seed/checkpoint; confidence intervals reflect sampled questions, not retraining variability.',
                              'Plain-text likelihood and nonthinking short chat do not measure native thinking performance; SFT was trained with thinking.',
                              'These are a fixed 500-question subset per benchmark, not full benchmark or few-shot leaderboard scores.',
                              'Any regression is behavioral interference under this protocol; it does not establish physical erasure of original frozen weights.'],
                 versions={name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'peft', 'safetensors')})
    save(ROOT / 'plan.json', value)
    return value


def validate():
    config = plan()
    for path, digest in config['frozen_files'].items():
        assert sha(Path(path)) == digest, path
    return config


def positions(mask):
    return (mask.long().cumsum(-1) - 1).masked_fill(mask == 0, 0).unsqueeze(0).expand(4, -1, -1)


def target_logprobs(logits, target):
    import torch
    # Last input token has no scored successor. Preceding n logits predict n target tokens.
    selected = logits[-len(target)-1:-1].float()
    return selected.log_softmax(-1).gather(1, torch.tensor(target, device=logits.device)[:, None]).squeeze(1)


def likelihood(model, tok, row, verify=False):
    import torch
    pairs = [encode_pair(tok, row['context'], choice) for choice in row['candidates']]
    scores, token_scores, errors = [], [], []
    with torch.inference_mode():
        for offset in range(len(pairs)):
            batch = pairs[offset:offset+1]
            inputs = tok.pad({'input_ids': [x[0] for x in batch]}, padding=True, return_tensors='pt').to(model.device)
            output = model(**inputs, position_ids=positions(inputs['attention_mask']), use_cache=False,
                           logits_to_keep=max(len(target) for _, target in batch) + 1).logits
            for index, (full, target) in enumerate(batch):
                values = target_logprobs(output[index], target)
                scores.append(float(values.sum()))
                token_scores.append(float(values.mean()))
                if verify:
                    # Independent full forward; select absolute causal positions, with no padding.
                    ids = torch.tensor([full], device=model.device)
                    mask = torch.ones_like(ids)
                    direct = model(input_ids=ids, attention_mask=mask, position_ids=positions(mask), use_cache=False).logits[0]
                    start = len(full) - len(target)
                    independent = direct[start-1:len(full)-1].float().log_softmax(-1)
                    expected = independent[torch.arange(len(target), device=model.device), torch.tensor(target, device=model.device)]
                    error = float((values - expected).abs().max())
                    assert error <= .125, (row['id'], error)
                    errors.append(error)
                    del direct, independent, expected
            del output, inputs
    normalized = [score / len(choice) for score, choice in zip(scores, row['candidates'])]
    selected = max(range(len(scores)), key=scores.__getitem__)
    selected_norm = max(range(len(scores)), key=normalized.__getitem__)
    primary = selected if row['benchmark'] == 'mmlu' else selected_norm
    return dict(id=row['id'], benchmark=row['benchmark'], gold=row['gold'], sum_logprobs=scores,
                char_normalized_logprobs=normalized, token_normalized_logprobs=token_scores,
                target_tokens=[len(target) for _, target in pairs], selected=selected, selected_norm=selected_norm,
                primary_correct=primary == row['gold'], raw_correct=selected == row['gold'], norm_correct=selected_norm == row['gold'],
                alignment_max_error=max(errors) if errors else None)


def run(arm, smoke=False):
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer
    config = validate()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(.65)
    model = load_base()
    if arm == 'cpt_sft':
        model = PeftModel.from_pretrained(model, CPT).merge_and_unload(safe_merge=True)
        del model.peft_config
        model = PeftModel.from_pretrained(model, SFT, is_trainable=False)
    model.to('cuda').eval()
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    tok.padding_side = 'left'
    rows = records()
    if smoke:
        rows = [next(r for r in rows if r['benchmark'] == name) for name in SOURCES]
    out = ROOT / ('smoke' if smoke else 'runs') / arm
    identity = dict(plan_sha256=sha(ROOT / 'plan.json'), evaluator_sha256=sha(Path(__file__)), arm=arm)
    if (out / 'identity.json').exists():
        assert read(out / 'identity.json') == identity
    else:
        save(out / 'identity.json', identity)
    checked = set()
    started = time.monotonic()
    for i, row in enumerate(rows):
        target = out / 'likelihood' / f"{row['id']}.json"
        if not target.exists():
            result = likelihood(model, tok, row, verify=row['benchmark'] not in checked)
            checked.add(row['benchmark'])
            save(target, result)
        if (i + 1) % 25 == 0 or smoke:
            save(ROOT / 'status.json', dict(state='likelihood', arm=arm, done=i+1, total=len(rows), seconds=time.monotonic()-started, pid=os.getpid()))
            print('Likelihood', arm, i+1, '/', len(rows), 'seconds', round(time.monotonic()-started), flush=True)
    labels = [tok.encode(x, add_special_tokens=False) for x in 'ABCDE']
    assert all(len(x) == 1 for x in labels)
    label_ids = [x[0] for x in labels]
    pending = [r for r in rows if not (out / 'chat' / f"{r['id']}.json").exists()]
    for offset in range(0, len(pending), config['chat_batch']):
        batch = pending[offset:offset+config['chat_batch']]
        texts = [tok.apply_chat_template([dict(role='system', content=SYSTEM), dict(role='user', content=chat_prompt(r))],
                 tokenize=False, add_generation_prompt=True, enable_thinking=False) for r in batch]
        inputs = tok(texts, padding=True, add_special_tokens=False, return_tensors='pt').to(model.device)
        with torch.inference_mode():
            generated = model.generate(**inputs, do_sample=False, num_beams=1, repetition_penalty=1.,
                        max_new_tokens=config['chat_max_new_tokens'], use_cache=True, return_dict_in_generate=True,
                        output_logits=True, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
            first = generated.logits[0][:, label_ids].float()
            if offset == 0:
                forward = getattr(model, '_original_forward', model)
                direct = forward(**inputs, position_ids=positions(inputs['attention_mask']), use_cache=True, logits_to_keep=1).logits[:, -1, label_ids].float()
                error = float((first-direct).abs().max())
                assert error <= .125, error
                save(out / 'chat_alignment.json', dict(max_logit_error=error))
        for index, row in enumerate(batch):
            tokens = generated.sequences[index, inputs['input_ids'].shape[1]:].tolist()
            stopped = tok.eos_token_id in tokens
            if stopped:
                tokens = tokens[:tokens.index(tok.eos_token_id)+1]
            text = tok.decode(tokens, skip_special_tokens=True).strip()
            letter = parse_letter(text, len(row['options']))
            logits = first[index, :len(row['options'])].tolist()
            ranked = max(range(len(logits)), key=logits.__getitem__)
            save(out / 'chat' / f"{row['id']}.json", dict(id=row['id'], prediction=text, gold=row['gold'],
                 letter=letter, free_correct=letter == 'ABCDE'[row['gold']], ranked_correct=ranked == row['gold'],
                 letter_logits=logits, generated_tokens=len(tokens), stopped_on_eos=stopped,
                 strict_format=bool(re.fullmatch(r'[A-E][.)]?', text))))
        done = len(rows)-len(pending)+offset+len(batch)
        save(ROOT / 'status.json', dict(state='chat', arm=arm, done=done, total=len(rows), pid=os.getpid()))
        if done % 40 == 0 or smoke:
            print('Chat', arm, done, '/', len(rows), flush=True)
        del generated, inputs, first
    save(out / 'complete.json', dict(n=len(rows), seconds=time.monotonic()-started,
         peak_cuda_allocated_mib=torch.cuda.max_memory_allocated()/1024**2, identity=identity))


def analyze():
    import numpy as np
    from scipy.stats import binomtest
    validate()
    rows = records()
    result = dict(independent_questions=len(rows), benchmarks={}, comparisons={})
    for name in SOURCES:
        subset = [r for r in rows if r['benchmark'] == name]
        vectors = {}
        result['benchmarks'][name] = {}
        for arm in ('baseline', 'cpt_sft'):
            out = ROOT / 'runs' / arm
            if not (out / 'complete.json').exists():
                continue
            likelihoods = [read(out / 'likelihood' / f"{r['id']}.json") for r in subset]
            chats = [read(out / 'chat' / f"{r['id']}.json") for r in subset]
            assert all(v['gold'] == r['gold'] and v['id'] == r['id'] for values in (likelihoods, chats) for v, r in zip(values, subset))
            values = {metric: np.array([r[metric] for r in likelihoods], dtype=int) for metric in ('primary_correct', 'raw_correct', 'norm_correct')}
            values.update({metric: np.array([r[metric] for r in chats], dtype=int) for metric in ('free_correct', 'ranked_correct', 'strict_format', 'stopped_on_eos')})
            vectors[arm] = values
            result['benchmarks'][name][arm] = dict(n=len(subset), counts={k:int(v.sum()) for k,v in values.items()},
                    percentages={k:float(v.mean()*100) for k,v in values.items()},
                    missing_letter=sum(r['letter'] is None for r in chats),
                    alignment_max_error=max(r['alignment_max_error'] or 0 for r in likelihoods))
        if len(vectors) != 2:
            continue
        result['comparisons'][name] = {}
        for metric in ('primary_correct', 'raw_correct', 'norm_correct', 'free_correct', 'ranked_correct'):
            a, b = vectors['baseline'][metric], vectors['cpt_sft'][metric]
            delta = b-a
            repaired, regressed = int((delta == 1).sum()), int((delta == -1).sum())
            rng = np.random.default_rng(SEED)
            means = rng.choice(delta, size=(10000,len(delta)), replace=True).mean(axis=1)*100
            result['comparisons'][name][metric] = dict(delta_pp=float(delta.mean()*100),
                paired_bootstrap95_pp=np.quantile(means, [.025,.975]).tolist(), repaired=repaired, regressed=regressed,
                mcnemar_exact_p=float(binomtest(repaired, repaired+regressed, .5).pvalue) if repaired+regressed else 1.,
                repaired_ids=[r['id'] for r,d in zip(subset,delta) if d == 1],
                regressed_ids=[r['id'] for r,d in zip(subset,delta) if d == -1])
    ordered = sorted(result['comparisons'], key=lambda name: result['comparisons'][name]['primary_correct']['mcnemar_exact_p'])
    previous = 0.
    for index, name in enumerate(ordered):
        item = result['comparisons'][name]['primary_correct']
        previous = max(previous, min(1., (len(ordered)-index)*item['mcnemar_exact_p']))
        item['holm_adjusted_p'] = previous
    save(ROOT / 'results.json', result)
    print(json.dumps({name: {arm: data['percentages']['primary_correct'] for arm,data in arms.items()}
                      for name,arms in result['benchmarks'].items()}), flush=True)
    return result


def pipeline():
    validate()
    for arm in ('baseline', 'cpt_sft'):
        if (ROOT / 'runs' / arm / 'complete.json').exists():
            continue
        while True:
            free = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
            commit = available_commit_mib()
            if free >= 11000 and commit >= 10000:
                break
            save(ROOT / 'status.json', dict(state='waiting_for_memory', gpu_free_mib=free, commit_free_mib=commit, pid=os.getpid()))
            time.sleep(30)
        subprocess.run([sys.executable, '-u', '-X', 'faulthandler', __file__, 'run', '--arm', arm], cwd=LAB, check=True)
    analyze()
    save(ROOT / 'status.json', dict(state='complete', independent_questions=1500, model_question_pairs=3000))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'smoke', 'run', 'pipeline', 'analyze'])
    parser.add_argument('--arm', choices=['baseline', 'cpt_sft'], default='baseline')
    args = parser.parse_args()
    try:
        if args.stage == 'prepare':
            plan()
        elif args.stage == 'analyze':
            analyze()
        elif args.stage == 'pipeline':
            pipeline()
        else:
            run(args.arm, smoke=args.stage == 'smoke')
    except Exception as error:
        save(ROOT / 'status.json', dict(state='failed', error=repr(error), pid=os.getpid()))
        raise
