"""Sample frozen RL validation failures to measure within-question reward diversity."""

import hashlib
import json
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from data_io import load_jsonl
from llm_judge import judge_responses
from native_reasoning import parse_completion
from train_vqa_gspo_probe import format_score, prepare_prompt, save, trim_completion


LAB = Path(__file__).resolve().parent
RUN = LAB / "outputs/gspo_vqa_rl_v1"
OUT = RUN / "failure_rollouts_v1"
ADAPTER = RUN / "training_resume_624/final_adapter"


def grade(rows, responses, name):
    started = time.monotonic()
    judgments = judge_responses(rows, responses, OUT / "judge_calls", "gpt-5.5")
    result = [dict(response, judge=judgment) for response, judgment in zip(responses, judgments, strict=True)]
    with (OUT / f"{name}.jsonl").open("x", encoding="utf-8") as file:
        for record in result:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"judged {name} responses={len(result)} seconds={time.monotonic() - started:.1f}", flush=True)
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    validation = {r["source_id"]: r for r in load_jsonl(LAB / "data/knowledge_experiments_v1/validation.jsonl")}
    failures = [r for r in load_jsonl(RUN / "evaluation/rl/scored.jsonl") if r["judge"]["score"] == 0]
    assert len(failures) == 27
    adapter_hash = hashlib.sha256((ADAPTER / "adapter_model.safetensors").read_bytes()).hexdigest()
    config = dict(adapter=str(ADAPTER), adapter_sha256=adapter_hash,
                  selection="RL greedy validation score 0; diagnostic only, no training",
                  source_ids=[r["source_id"] for r in failures], groups_per_question=4, group_size=4,
                  seed_rule="10000 + question_index * 4 + group_index",
                  temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0,
                  max_new_tokens=512, image_max_edge=768, judge_model="gpt-5.5",
                  judge_batch_size=16, judge_workers=2, rejudge_original_greedy=True)
    save(OUT / "config.json", config)
    futures = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        for start in range(0, len(failures), 16):
            responses, rows = [], []
            for index, original in enumerate(failures[start:start + 16], start):
                sid = f"audit-{index:03d}"
                rows.append(dict(validation[original["source_id"]], source_id=sid))
                responses.append(dict(original, source_id=sid, original_source_id=original["source_id"],
                                      kind="original_greedy"))
            futures.append(executor.submit(grade, rows, responses, f"audit_{start:03d}"))
        torch.cuda.set_per_process_memory_fraction(0.8)
        processor = AutoProcessor.from_pretrained(LAB / "models/Qwen3.5-2B", do_resize=False)
        base = Qwen3_5ForConditionalGeneration.from_pretrained(
            LAB / "models/Qwen3.5-2B", dtype=torch.bfloat16, attn_implementation="sdpa")
        model = PeftModel.from_pretrained(base, ADAPTER).to("cuda").eval()
        model.requires_grad_(False)
        assert not any(p.requires_grad for p in model.parameters())
        manifest = json.loads((LAB / "data/knowledge_experiments_v1/manifest.json").read_text(encoding="utf-8"))
        eos, pad = processor.tokenizer.eos_token_id, processor.tokenizer.pad_token_id
        generation_seconds = 0.0
        with (OUT / "generations.jsonl").open("x", encoding="utf-8") as file:
            for question_index, original in enumerate(failures):
                row = validation[original["source_id"]]
                inputs = prepare_prompt(processor, row, manifest["system_prompt"])
                responses, rows = [], []
                for group in range(4):
                    seed = 10000 + question_index * 4 + group
                    torch.manual_seed(seed)
                    torch.cuda.synchronize()
                    begin = time.monotonic()
                    with torch.inference_mode():
                        generated = model.generate(**{k: v.to("cuda") for k, v in inputs.items()},
                            do_sample=True, temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0,
                            max_new_tokens=512, eos_token_id=eos, pad_token_id=pad,
                            num_return_sequences=4, use_cache=True)
                    torch.cuda.synchronize()
                    generation_seconds += time.monotonic() - begin
                    for sample, sequence in enumerate(generated):
                        ids = trim_completion(sequence[inputs["input_ids"].shape[1]:].tolist(), eos)
                        stopped = ids[-1] == eos
                        raw = processor.tokenizer.decode(ids[:-1] if stopped else ids, skip_special_tokens=False).strip()
                        sid = f"candidate-{question_index:03d}-{group * 4 + sample:02d}"
                        response = dict(source_id=sid, original_source_id=row["source_id"],
                                        source_group=row["source_group"], kind="sampled", group=group, sample=sample,
                                        seed=seed, raw_prediction=raw, **parse_completion(raw),
                                        generated_tokens=len(ids), stopped_on_eos=stopped,
                                        format_score=format_score(raw, stopped))
                        responses.append(response)
                        rows.append(dict(row, source_id=sid))
                        file.write(json.dumps(response, ensure_ascii=False) + "\n")
                    file.flush()
                    del generated
                order = list(range(16))
                random.Random(question_index + 42).shuffle(order)
                futures.append(executor.submit(grade, [rows[i] for i in order],
                                               [responses[i] for i in order], f"question_{question_index:03d}"))
                for future in futures:
                    if future.done():
                        future.result()
                save(OUT / "status.json", dict(status="generating", questions_generated=question_index + 1,
                                               judge_batches_done=sum(f.done() for f in futures),
                                               elapsed_seconds=time.monotonic() - started))
                print(f"generated question={question_index + 1}/27 source={row['source_id']} responses=16", flush=True)
        del model, base, processor
        torch.cuda.empty_cache()
        save(OUT / "status.json", dict(status="judging", questions_generated=27,
                                       judge_batches_done=sum(f.done() for f in futures)))
        results = [record for future in as_completed(futures) for record in future.result()]
    if hashlib.sha256((ADAPTER / "adapter_model.safetensors").read_bytes()).hexdigest() != adapter_hash:
        raise RuntimeError("Adapter changed on disk during diagnosis")
    sampled = [r for r in results if r["kind"] == "sampled"]
    audit = [r for r in results if r["kind"] == "original_greedy"]
    assert len(sampled) == 432 and len(audit) == 27
    per_question = []
    for original in failures:
        sid = original["source_id"]
        records = sorted([r for r in sampled if r["original_source_id"] == sid], key=lambda r: (r["group"], r["sample"]))
        score_groups = [[r["judge"]["score"] for r in records if r["group"] == group] for group in range(4)]
        rewards = [[0.9 * r["judge"]["score"] / 2 + 0.1 * r["format_score"]
                    for r in records if r["group"] == group and r["judge"]["score"] is not None] for group in range(4)]
        scores = [r["judge"]["score"] for r in records]
        per_question.append(dict(source_id=sid, image=original["source_group"],
            question=validation[sid]["user_text"], reference=validation[sid]["reference"],
            original_answer=original["final_answer"],
            original_rejudge=next(r["judge"] for r in audit if r["original_source_id"] == sid),
            scores=score_groups, correct=sum(s == 2 for s in scores), partial=sum(s == 1 for s in scores),
            incorrect=sum(s == 0 for s in scores), unjudgeable=sum(s is None for s in scores),
            reward_variable_groups=sum(len(g) == 4 and len(set(g)) > 1 for g in rewards),
            most_common_final_answers=Counter(r["final_answer"] for r in records).most_common(5)))
    summary = dict(status="completed", questions=27, responses=432,
        any_correct_questions=sum(r["correct"] > 0 for r in per_question),
        majority_correct_questions=sum(r["correct"] > 8 for r in per_question),
        all_wrong_questions=sum(r["incorrect"] == 16 for r in per_question),
        no_full_correct_questions=sum(r["correct"] == 0 for r in per_question),
        variable_reward_groups=sum(r["reward_variable_groups"] for r in per_question), total_groups=108,
        sampled_score_counts=dict(Counter(str(r["judge"]["score"]) for r in sampled)),
        original_rejudge_score_counts=dict(Counter(str(r["judge"]["score"]) for r in audit)),
        format_failures=sum(r["format_score"] != 1 for r in sampled),
        truncated=sum(not r["stopped_on_eos"] for r in sampled),
        generation_seconds=generation_seconds, total_seconds=time.monotonic() - started,
        per_question=per_question)
    save(OUT / "summary.json", summary)
    save(OUT / "status.json", dict(status="completed", total_seconds=time.monotonic() - started))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_question"}), flush=True)


if __name__ == "__main__":
    main()
