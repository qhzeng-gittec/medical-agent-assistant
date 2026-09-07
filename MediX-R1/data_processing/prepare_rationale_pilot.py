"""Build the fixed 96-example paired continuation pilot; never use test answers."""
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from common.io import load_jsonl, save_jsonl


LAB = Path(__file__).resolve().parents[1]
ROOT = LAB / 'data/rationale_pilot_v1'
SOURCE = LAB / 'data/knowledge_experiments_v1'
INITIAL = LAB / 'outputs/knowledge_experiments_v1/seed42/vqa_knowledge_attention/final_adapter'


def main():
    originals = load_jsonl(ROOT / 'selected_original.jsonl')
    edits = json.loads((ROOT / 'rewrites.json').read_text(encoding='utf-8'))
    train = {r['source_id']: r for r in load_jsonl(SOURCE / 'train.jsonl')}
    validation = load_jsonl(SOURCE / 'validation.jsonl')
    assert len(originals) == 96 and set(edits) == {str(i) for i in range(96)}
    assert all(r == train[r['source_id']] for r in originals)
    assert set(r['source_id'] for r in originals).isdisjoint(r['source_id'] for r in validation)
    assert set(r['source_group'] for r in originals).isdisjoint(r['source_group'] for r in validation)
    assert set(Counter(r['subject'] for r in originals).values()) == {16}
    manifest = json.loads((SOURCE / 'manifest.json').read_text(encoding='utf-8'))
    tokenizer = AutoTokenizer.from_pretrained(LAB / 'models/Qwen3.5-2B')
    rewritten = []
    audit = []
    for i, row in enumerate(originals):
        rationale = edits[str(i)]
        assert rationale.strip() and rationale != row['reasoning_content']
        assert '<' not in rationale and len(rationale.split()) <= 100
        revised = dict(row, reasoning_content=rationale,
                       reference_explanation=rationale,
                       reasoning_source='main_assistant_pilot_rewrite_of_reviewed_training_data',
                       reasoning_provenance='main_assistant_pilot_rewrite_of_reviewed_training_data')
        messages = [{'role': 'system', 'content': manifest['system_prompt']},
                    {'role': 'user', 'content': row['user_text']}]
        prefix = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        full = tokenizer.apply_chat_template(messages + [{'role': 'assistant', 'reasoning_content': rationale,
                                                         'content': row['target']}], tokenize=False, enable_thinking=True)
        assert full.startswith(prefix)
        revised['supervised_tokens'] = len(tokenizer.encode(full[len(prefix):], add_special_tokens=False))
        rewritten.append(revised)
        audit.append(dict(source_id=row['source_id'], subject=row['subject'], question=row['user_text'],
                          answer=row['target'], original=row['reasoning_content'], rewritten=rationale,
                          original_supervised_tokens=row['supervised_tokens'],
                          rewritten_supervised_tokens=revised['supervised_tokens']))
    save_jsonl(ROOT / 'rewrite_audit.jsonl', audit)
    baseline = LAB / 'outputs/knowledge_experiments_v1/validation_screen/vqa_knowledge_attention/validation'
    old_ids = {r['source_id'] for r in load_jsonl(baseline / 'knowledge.jsonl')}
    eval_knowledge = [r for r in validation if r['source_id'] in old_ids]
    for subject in sorted({r['subject'] for r in originals}):
        selected_count = sum(r['subject'] == subject for r in eval_knowledge)
        candidates = sorted([r for r in validation if r['task'] == 'knowledge' and r['subject'] == subject
                             and r['source_id'] not in old_ids], key=lambda r: r['source_id'])
        eval_knowledge += random.Random('rationale-eval-20260906-' + subject).sample(candidates, 10 - selected_count)
    visual_ids = {r['source_id'] for r in load_jsonl(baseline / 'vqa.jsonl')}
    eval_visual = [r for r in validation if r['source_id'] in visual_ids]
    assert len(eval_knowledge) == 60 and len(eval_visual) == 20
    eval_rows = eval_knowledge + eval_visual
    save_jsonl(ROOT / 'validation.jsonl', eval_rows)
    # Both arms have the same held-out loss diagnostic; model selection uses final state.
    loss_validation = [next(r for r in eval_knowledge if r['subject'] == subject)
                       for subject in sorted({r['subject'] for r in originals})]
    for name, rows in [('original', originals), ('structured', rewritten)]:
        order = list(range(len(rows)))
        random.Random(42).shuffle(order)
        rows = [rows[i] for i in order]
        destination = ROOT / 'recipes' / name
        destination.mkdir(parents=True, exist_ok=True)
        save_jsonl(destination / 'train.jsonl', rows)
        save_jsonl(destination / 'validation.jsonl', loss_validation)
        recipe = dict(manifest, name=name, train_samples=96, validation_samples=len(loss_validation),
                      train_task_counts={'knowledge': 96},
                      train_task_supervised_tokens={'knowledge': sum(r['supervised_tokens'] for r in rows)},
                      train_sha256=hashlib.sha256((destination / 'train.jsonl').read_bytes()).hexdigest())
        (destination / 'manifest.json').write_text(json.dumps(recipe, ensure_ascii=False, indent=2), encoding='utf-8')
    (ROOT / 'manifest.json').write_text(json.dumps(dict(manifest, name='rationale_pilot_v1'), ensure_ascii=False, indent=2), encoding='utf-8')
    plan = dict(
        initial_adapter=str(INITIAL), initial_adapter_sha256=hashlib.sha256((INITIAL / 'adapter_model.safetensors').read_bytes()).hexdigest(),
        arms=['unchanged_C', 'original', 'structured'], seed=42, train_examples=96,
        train_subject_counts=dict(Counter(r['subject'] for r in originals)), epochs=3,
        batch_size=2, gradient_accumulation_steps=2, optimizer_steps=72, learning_rate=2e-5,
        gpu_memory_fraction=0.45, lora_scope='attention', knowledge_only_continuation=True,
        evaluation_counts={'knowledge': 60, 'vqa': 20}, evaluation_knowledge_subject_counts=dict(Counter(r['subject'] for r in eval_knowledge)),
        historical_knowledge_ids=sorted(old_ids), evaluation_ids=[r['source_id'] for r in eval_rows],
        evaluation_sha256=hashlib.sha256((ROOT / 'validation.jsonl').read_bytes()).hexdigest(),
        training_edits_sha256=hashlib.sha256((ROOT / 'rewrites.json').read_bytes()).hexdigest(),
        primary_metric='Existing 0/1/2 joint answer-and-rationale score on all 60 knowledge validation questions.',
        diagnostic_metrics=['final_answer_correct', 'reasoning_fact_error', 'reasoning_answer_contradiction'],
        promising_rule='Structured score above both original continuation and unchanged C, final-answer accuracy not below original, VQA no more than 5 points below original. Exploratory, not a significance claim.',
        limitations=['Single seed; 96 previously seen knowledge examples continued from C.',
                     'Structure and length change together; supervised tokens and compute are not matched.',
                     'No independent test evaluation; historical 20 validation questions have been inspected.',
                     'Rewrites derive from reviewed references, not a new independent clinical audit.'])
    (ROOT / 'experiment_plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(train=plan['train_subject_counts'], evaluation=plan['evaluation_counts'],
                          old_supervised_tokens=sum(r['supervised_tokens'] for r in originals),
                          new_supervised_tokens=sum(r['supervised_tokens'] for r in rewritten)), indent=2))


if __name__ == '__main__':
    main()
