"""Trace the trained knowledge targets to reviewed natural-language records."""
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import median

from transformers import AutoProcessor

from data_io import load_jsonl
from native_reasoning import SYSTEM_PROMPT
from train_multitask_lora import MultitaskCollator

LAB = Path(__file__).resolve().parent
OUT = LAB / 'outputs/independent_holdout_20260906/output_style'


def main():
    reviewed = {r['source_id']: r for r in load_jsonl(LAB / 'data/medmcqa_knowledge_reviewed_v1/train.jsonl')}
    decisions = {r['id']: r for r in load_jsonl(LAB / 'data/medmcqa_knowledge_reviewed_v1/direct_review_final.jsonl')}
    checks = {}
    for name in ['vqa_knowledge_attention', 'vqa_knowledge_attention_ffn',
                 'vqa_knowledge_case_attention_ffn', 'vqa_knowledge_case_context_attention_ffn']:
        config = json.loads((LAB / 'outputs/knowledge_experiments_v1/seed42' / name / 'run_config.json').read_text())
        path = LAB / 'data/knowledge_experiments_v1/recipes' / config['recipe'] / 'train.jsonl'
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == config['recipe_sha256']
        knowledge = [r for r in load_jsonl(path) if r['task'] == 'knowledge']
        assert len(knowledge) == 1193
        for row in knowledge:
            parent = reviewed[row['source_id']]
            decision = decisions[row['source_id']]
            assert decision['decision'] == 'accept'
            assert row['target'] == parent['target'] == decision['final']['final_answer']
            assert row['reasoning_content'] == parent['reasoning_content'] == decision['final']['reasoning_content']
            assert row['user_text'] == parent['user_text']
        checks[name] = dict(recipe_sha256_matches_training=True, knowledge_rows=1193,
                           reviewed_target_and_reasoning_exact_matches=1193)
    processor = AutoProcessor.from_pretrained(LAB / 'models/Qwen3.5-2B', do_resize=False)
    collator = MultitaskCollator(processor, SYSTEM_PROMPT)
    examples = []
    # One short final answer and one sentence answer, selected before inspecting labels.
    selected = [next(r for r in knowledge if len(r['target'].split()) == 1),
                next(r for r in knowledge if len(r['target'].split()) >= 10)]
    for row in selected:
        batch = collator([row])
        labels = batch['labels'][0]
        ids = labels[labels != -100].tolist()
        text = processor.tokenizer.decode(ids, skip_special_tokens=False)
        assert row['reasoning_content'] in text and row['target'] in text
        examples.append(dict(source_id=row['source_id'], question=row['user_text'],
            original_answer=decisions[row['source_id']]['original']['answer'],
            reviewed_reasoning=row['reasoning_content'], reviewed_final=row['target'],
            actual_supervised_text=text, supervised_tokens=len(ids),
            provenance=row['reasoning_provenance']))
    result = dict(checks=checks, provenance=dict(Counter(r['reasoning_provenance'] for r in knowledge)),
                  median_final_words=median(len(r['target'].split()) for r in knowledge),
                  median_reasoning_words=median(len(r['reasoning_content'].split()) for r in knowledge),
                  examples=examples,
                  conclusion='Training used reviewed natural-language explanations AND final answers. Short final fields do not imply raw-answer-only training.')
    (OUT / 'training_target_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
