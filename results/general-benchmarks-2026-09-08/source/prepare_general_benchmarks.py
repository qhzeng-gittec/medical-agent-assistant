"""Download and freeze independent general-capability benchmark questions."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re

from run_causal_fact_experiment import LAB, read, save, sha

DATA = LAB / 'data/general_benchmarks_v1'
ROOT = LAB / 'outputs/general_benchmarks_v1'
SEED = 20260908
SOURCES = {
    'mmlu': ('cais/mmlu', 'all/test-00000-of-00001.parquet', 'test'),
    'arc_challenge': ('allenai/ai2_arc', 'ARC-Challenge/test-00000-of-00001.parquet', 'test'),
    'hellaswag': ('Rowan/hellaswag', 'data/validation-00000-of-00001.parquet', 'validation'),
}
MEDICAL = {'anatomy', 'clinical_knowledge', 'college_medicine', 'medical_genetics',
           'professional_medicine', 'virology'}
TRAINING = [LAB / 'data/cpt_medical_v1/train.jsonl',
            LAB / 'data/knowledge_experiments_v1/recipes/vqa_knowledge_case_context/train.jsonl']


def normalize(text):
    return ' '.join(re.findall(r'\w+', text.lower()))


def clean_hella(text):
    return re.sub(r'\[.*?\]', '', text.strip().replace(' [title]', '. ')).replace('  ', ' ')


def convert(name, row, index):
    if name == 'mmlu':
        question, options, gold = row['question'], row['choices'], row['answer']
        subject = row['subject']
        context = ('The following are multiple choice questions (with answers) about '
                   + subject.replace('_', ' ') + '.\n\n' + question + '\n'
                   + '\n'.join(f'{label}. {value}' for label, value in zip('ABCD', options))
                   + '\nAnswer:')
        candidates = list('ABCD')
    elif name == 'arc_challenge':
        question, options = row['question'], row['choices']['text']
        gold = row['choices']['label'].index(row['answerKey'])
        subject = 'science'
        context, candidates = 'Question: ' + question + '\nAnswer:', options
    else:
        question = clean_hella(row['activity_label'] + ': ' + row['ctx_a'] + ' ' + row['ctx_b'].capitalize())
        options = [clean_hella(x) for x in row['endings']]
        gold, subject = int(row['label']), row['activity_label']
        context, candidates = question, options
    if not 0 <= gold < len(options) <= 5 or not all(options):
        raise ValueError(f'Invalid label/options: {name}:{index}')
    return dict(id=f'{name}-{index:05d}', benchmark=name, source_index=index,
                source_id=str(row.get('id', row.get('ind', index))), subject=subject,
                question=question, options=options, gold=int(gold), context=context,
                candidates=candidates)


def prepare():
    import pyarrow.parquet as pq
    import requests
    if (DATA / 'manifest.json').exists():
        manifest = read(DATA / 'manifest.json')
        assert sha(DATA / 'evaluation.jsonl') == manifest['evaluation_sha256']
        return manifest
    DATA.mkdir(parents=True, exist_ok=True)
    provenance = {}
    candidates = []
    for name, (repo, remote, split) in SOURCES.items():
        metadata_path = DATA / 'raw' / f'{name}.json'
        if metadata_path.exists():
            metadata = read(metadata_path)
        else:
            response = requests.get(f'https://huggingface.co/api/datasets/{repo}', timeout=60)
            response.raise_for_status()
            revision = response.json()['sha']
            response = requests.get(f'https://huggingface.co/api/datasets/{repo}/tree/{revision}/{remote.rsplit("/", 1)[0]}', timeout=60)
            response.raise_for_status()
            entry = next(x for x in response.json() if x['path'] == remote)
            metadata = dict(repo=repo, revision=revision, split=split, file=remote,
                            expected_sha256=entry['lfs']['oid'], bytes=entry['size'],
                            url=f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{remote}')
            save(metadata_path, metadata)
        target = DATA / 'raw' / f'{name}.parquet'
        if not target.exists():
            response = requests.get(metadata['url'], timeout=120)
            response.raise_for_status()
            temporary = target.with_suffix('.tmp')
            temporary.write_bytes(response.content)
            assert sha(temporary) == metadata['expected_sha256']
            temporary.replace(target)
        assert sha(target) == metadata['expected_sha256']
        raw = pq.read_table(target).to_pylist()
        provenance[name] = dict(metadata, downloaded_rows=len(raw))
        for index, row in enumerate(raw):
            if name == 'mmlu' and row['subject'] in MEDICAL:
                continue
            candidates.append(convert(name, row, index))
        print('Downloaded', name, len(raw), 'rows', flush=True)
    # Check the actual added training corpora before freezing any model results.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(LAB / 'models/Qwen3.5-2B', local_files_only=True)
    with TRAINING[0].open(encoding='utf-8') as handle:
        cpt_text = '\n'.join(tokenizer.decode(json.loads(line)['input_ids'], skip_special_tokens=True) for line in handle)
    with TRAINING[1].open(encoding='utf-8') as handle:
        sft_text = '\n'.join(str(value) for line in handle for key, value in json.loads(line).items()
                             if key in ('user_text', 'target', 'reasoning_content'))
    corpus = normalize(cpt_text + '\n' + sft_text)
    print('Decoded CPT/SFT training for overlap screening', len(corpus), 'characters', flush=True)
    pools = defaultdict(list)
    seen, excluded = set(), []
    for row in candidates:
        normalized = normalize(row['question'])
        digest = hashlib.sha256(normalized.encode()).hexdigest()
        if digest in seen:
            excluded.append(dict(id=row['id'], reason='duplicate normalized question'))
            continue
        seen.add(digest)
        pools[row['benchmark']].append(row)
    selected = []
    rng = random.Random(SEED)
    for name in SOURCES:
        groups = defaultdict(list)
        for row in pools[name]:
            groups[row['subject'] if name == 'mmlu' else name].append(row)
        # Proportional MMLU subject stratification with largest-remainder allocation.
        total = len(pools[name])
        allocations = {key: 500 * len(rows) // total for key, rows in groups.items()}
        remainder = sorted(groups, key=lambda key: (-(500 * len(groups[key]) % total), key))
        for key in remainder[:500 - sum(allocations.values())]:
            allocations[key] += 1
        for key in sorted(groups):
            order = rng.sample(groups[key], len(groups[key]))
            accepted = []
            for row in order:
                if len(accepted) == allocations[key]:
                    break
                normalized = normalize(row['question'])
                if len(normalized) >= 40 and normalized in corpus:
                    excluded.append(dict(id=row['id'], reason='full normalized question in decoded CPT/SFT training'))
                    continue
                accepted.append(row)
            assert len(accepted) == allocations[key], key
            selected.extend(accepted)
    selected.sort(key=lambda row: (row['benchmark'], row['id']))
    assert len(selected) == len({r['id'] for r in selected}) == 1500
    (DATA / 'evaluation.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in selected), encoding='utf-8')
    manifest = dict(seed=SEED, independent_questions=1500, per_benchmark=dict(Counter(r['benchmark'] for r in selected)),
                    sources=provenance, evaluation_sha256=sha(DATA / 'evaluation.jsonl'),
                    subject_counts=dict(Counter(r['subject'] for r in selected if r['benchmark'] == 'mmlu')),
                    mmlu_excluded_medical_subjects=sorted(MEDICAL),
                    training_files={str(path): sha(path) for path in TRAINING}, exclusions=excluded,
                    sampling='500 per benchmark; proportional subject-stratified MMLU; uniform ARC/HellaSwag; without replacement',
                    contamination_limit='Exact normalized question overlap with added CPT/SFT training only; paraphrases and original Qwen pretraining contamination cannot be ruled out.',
                    scope='English nonmedical MMLU subset, ARC-Challenge and HellaSwag; not full leaderboard scores or a test of Chinese, coding, instruction following, or vision.')
    save(DATA / 'manifest.json', manifest)
    print('Frozen', manifest['per_benchmark'], flush=True)
    return manifest


if __name__ == '__main__':
    prepare()
