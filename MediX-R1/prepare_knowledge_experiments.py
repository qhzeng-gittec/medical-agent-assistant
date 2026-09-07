import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from PIL import Image
from transformers import AutoTokenizer

from data_io import load_jsonl, save_jsonl


LAB = Path(__file__).resolve().parent
OUTPUT = LAB / 'data/knowledge_experiments_v1'


def main():
    native = LAB / 'data/native_reasoning'
    knowledge = LAB / 'data/medmcqa_knowledge_reviewed_v1/train.jsonl'
    pools = {s: load_jsonl(native / f'{s}.jsonl') for s in ('train', 'validation', 'test')}
    manifest = json.loads((native / 'manifest.json').read_text(encoding='utf-8'))
    tokenizer = AutoTokenizer.from_pretrained(LAB / 'models/Qwen3.5-2B')
    rows = load_jsonl(knowledge)
    by_subject = {subject: [r for r in rows if r['subject'] == subject] for subject in sorted({r['subject'] for r in rows})}
    for subject, group in by_subject.items():
        random.Random('knowledge-42-' + subject).shuffle(group)
        for split, selected in (('validation', group[:25]), ('test', group[25:50]), ('train', group[50:])):
            for row in selected:
                row = dict(row, split=split, reasoning_source=row['reasoning_provenance'], reference_explanation=row['reasoning_content'])
                prompt = [{'role':'system','content':manifest['system_prompt']}, {'role':'user','content':row['user_text']}]
                prefix = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, enable_thinking=True)
                full = tokenizer.apply_chat_template(prompt + [{'role':'assistant','reasoning_content':row['reasoning_content'],'content':row['target']}], tokenize=False, enable_thinking=True)
                assert full.startswith(prefix)
                row['supervised_tokens'] = len(tokenizer.encode(full[len(prefix):], add_special_tokens=False))
                row['final_tokens'] = len(tokenizer.encode(row['target'], add_special_tokens=False)) + 1
                row['has_reasoning_reference'] = True
                pools[split].append(row)
    for field in ('source_id', 'source_group'):
        groups = [{r[field] for r in pools[s]} for s in pools]
        assert all(not groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)), field
    manifest.update(name='knowledge_experiments_v1', seed=42, image_max_edge=768,
                    image_preprocessing='Aspect-preserving Lanczos downscale to at most 768; no upscaling; pad to 32',
                    sampling_policy='Each accepted source once per epoch. Common sources have equal exposure across additive recipes; total compute is not matched.',
                    knowledge_split_policy='25 validation and 25 test per subject from reviewed pool; all other knowledge rows train. Fixed before training.',
                    knowledge_source_sha256=hashlib.sha256(knowledge.read_bytes()).hexdigest(),
                    split_stats={s:dict(Counter(r['task'] for r in rs)) for s,rs in pools.items()})
    manifest['generation_max_new_tokens']['knowledge'] = 2048
    manifest['generation_final_reserve_tokens']['knowledge'] = 256
    assert max(r['supervised_tokens'] for r in pools['train'] + pools['validation'] if r['task']=='knowledge') < 1792
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for split, group in pools.items():
        save_jsonl(OUTPUT / f'{split}.jsonl', group)
    recipes = {'vqa':('vqa',), 'vqa_knowledge':('vqa','knowledge'),
               'vqa_knowledge_case':('vqa','knowledge','case'),
               'vqa_knowledge_case_context':('vqa','knowledge','case','context')}
    for name, tasks in recipes.items():
        directory = OUTPUT / 'recipes' / name
        directory.mkdir(parents=True, exist_ok=True)
        train = [r for r in pools['train'] if r['task'] in tasks]
        validation = [r for r in pools['validation'] if r['task'] in tasks]
        random.Random(42).shuffle(train)
        save_jsonl(directory/'train.jsonl', train)
        save_jsonl(directory/'validation.jsonl', validation)
        recipe = dict(manifest, name=name, train_samples=len(train), validation_samples=len(validation),
                      train_task_counts=dict(Counter(r['task'] for r in train)),
                      train_task_supervised_tokens={t:sum(r['supervised_tokens'] for r in train if r['task']==t) for t in tasks},
                      train_sha256=hashlib.sha256((directory/'train.jsonl').read_bytes()).hexdigest())
        (directory/'manifest.json').write_text(json.dumps(recipe,ensure_ascii=False,indent=2),encoding='utf-8')
    visual = [r for r in pools['train'] if r['task']=='vqa']
    sizes = {}
    for r in visual:
        if r['image_path'] not in sizes:
            with Image.open(r['image_path']) as image:
                sizes[r['image_path']] = image.width * image.height
    stress = sorted(visual,key=lambda r:sizes[r['image_path']],reverse=True)[:4]
    for task in ('context','case'):
        stress += sorted([r for r in pools['train'] if r['task']==task],key=lambda r:len(tokenizer.encode(r['user_text']+r['reasoning_content']+r['target'])),reverse=True)[:2]
    directory = OUTPUT/'recipes/memory_probe'
    directory.mkdir(parents=True,exist_ok=True)
    save_jsonl(directory/'train.jsonl',stress)
    save_jsonl(directory/'validation.jsonl',pools['validation'][:4])
    (directory/'manifest.json').write_text(json.dumps(dict(manifest,verification_only=True,train_sha256=hashlib.sha256((directory/'train.jsonl').read_bytes()).hexdigest()),ensure_ascii=False,indent=2),encoding='utf-8')
    (OUTPUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(manifest['split_stats'],indent=2))


if __name__ == '__main__':
    main()
