"""Compare gold-answer likelihood after gold versus model-generated explanations."""
import gc
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from common.io import load_jsonl
from experiments.diagnose_reasoning_effects import ARMS, CHECKPOINTS
from common.native_reasoning import SYSTEM_PROMPT
from training.sft import MultitaskCollator

LAB = Path(__file__).resolve().parents[1]
ROOT = LAB / 'outputs/independent_holdout_20260906'
OUT = ROOT / 'output_style/prefix_probe'


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(.35)
    rows = [r for r in load_jsonl(ROOT / 'evaluation.jsonl') if r['task'] == 'knowledge']
    assert len(rows) == 150
    questions = {r['source_id']: json.loads((ROOT / 'diagnostics/questions' / f"{r['source_id']}.json").read_text(encoding='utf-8')) for r in rows}
    OUT.mkdir(exist_ok=True)
    frozen = json.loads((ROOT / 'plan.json').read_text(encoding='utf-8'))
    save(OUT / 'plan.json', dict(arms=['A','C','D'], conditions=['gold_reasoning','own_reasoning'],
        question_ids=[r['source_id'] for r in rows], expected_forwards=900, training=False,
        hypothesis='Measure dependence of gold answer likelihood on supplied explanation; not a generation accuracy experiment.',
        limitations=['Paired prefix interventions change both content and style.',
                     'Own explanations are existing free generations; not forced to be correct.',
                     'All answer tokens except the first still condition on previous gold answer tokens.']))
    processor = AutoProcessor.from_pretrained(LAB / 'models/Qwen3.5-2B', do_resize=False)
    collator = MultitaskCollator(processor, SYSTEM_PROMPT)
    results = {}
    for arm in ['A','C','D']:
        weights = CHECKPOINTS / ARMS[arm] / 'final_adapter/adapter_model.safetensors'
        assert hashlib.sha256(weights.read_bytes()).hexdigest() == frozen['checkpoint_sha256'][arm]
        previous = {r['source_id']: r for r in json.loads((ROOT / 'reference_loss_segments' / f'{arm}.json').read_text()) if r['cohort'] == 'test'}
        model = Qwen3_5ForConditionalGeneration.from_pretrained(LAB / 'models/Qwen3.5-2B', dtype=torch.bfloat16, attn_implementation='sdpa')
        model = PeftModel.from_pretrained(model, weights.parent).to('cuda').eval()
        model.config.use_cache = False
        records = []
        with torch.inference_mode():
            for index, row in enumerate(rows, 1):
                generated = questions[row['source_id']]['models'][arm]
                entry = dict(source_id=row['source_id'], answer_score=generated['diagnosis']['answer_score'])
                for condition, rationale in [('gold_reasoning',row['reasoning_content']), ('own_reasoning',generated['reasoning'])]:
                    batch = collator([dict(row, reasoning_content=rationale)])
                    batch.pop('labels')
                    ids = batch['input_ids'][0].tolist()
                    close = ids.index(processor.tokenizer.convert_tokens_to_ids('</think>'))
                    answer = processor.tokenizer.encode(row['target'], add_special_tokens=False)
                    starts = [i for i in range(close+1, len(ids)-len(answer)+1) if ids[i:i+len(answer)] == answer]
                    assert len(starts) == 1
                    keep = len(ids)-starts[0]+1
                    output = model(**{k:v.to('cuda') for k,v in batch.items()}, logits_to_keep=keep)
                    nll = torch.nn.functional.cross_entropy(output.logits[0,:len(answer)].float().cpu(), torch.tensor(answer), reduction='none')
                    entry[condition] = dict(answer_nll=float(nll.mean()), first_answer_token_nll=float(nll[0]))
                    assert torch.isfinite(nll).all()
                    del output, batch, nll
                assert math.isclose(entry['gold_reasoning']['answer_nll'], previous[row['source_id']]['answer_nll_mean'], abs_tol=1e-4)
                records.append(entry)
                if index % 30 == 0:
                    print(f'{arm}: {index}/150 paired prefix probes', flush=True)
        save(OUT / f'{arm}.json', records)
        results[arm] = {
            label: dict(n=len(group), **{
                condition: {metric: mean(r[condition][metric] for r in group)
                            for metric in ['answer_nll','first_answer_token_nll']}
                for condition in ['gold_reasoning','own_reasoning']})
            for label, group in [('all',records), ('wrong',[r for r in records if r['answer_score']==0]),
                                 ('full_credit',[r for r in records if r['answer_score']==2])]}
        del model
        gc.collect()
        torch.cuda.empty_cache()
    save(OUT / 'summary.json', results)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
