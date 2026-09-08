"""Select a bounded, training-task-related Meditron guideline corpus for raw-text CPT."""
import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer

from common.io import load_jsonl, save_jsonl
from data_processing.cpt_data import pack_excerpts, sample_spans, token_fingerprint

LAB = Path(__file__).resolve().parents[1]
ROOT = LAB / 'data/cpt_medical_v2'
SOURCE = LAB / 'data/knowledge_experiments_v1'
SEED = 42
GUIDELINES_REVISION = '30f6d68ecdf19d27ed9c5c250b31ff3312ada189'
WIKI_REVISION = 'b08601e04326c79dfdd32d625aee71d232d685c3'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def words(text):
    return re.findall(r'[a-z0-9]+', text.lower())


def question_text(row):
    text = row['user_text']
    for suffix in ['\n\nAnswer the question', '\n\nExplain the clinical basis',
                   '\n\nReason from the visible', '\n\nExplain what the supplied']:
        text = text.split(suffix)[0]
    return text


def clean_text(text):
    # Keep the source prose, including qualifications; remove trailing bibliographies.
    text = re.split(r'(?im)^#{1,6}\s*(?:references|external links|see also)\s*$', text)[0]
    text = re.sub(r'\[([^\]\n]+)\]\(https?://[^)]+\)', r'\1', text)
    text = re.sub(r'(?im)^Please Take Over This Page[^\n]*\n?', '', text)
    text = re.sub(r'(?im)^There can be one or more than one Editor-In-Chief\.[^\n]*\n?', '', text)
    text = re.sub(r'(?im)^File:[^\n]*\n?', '', text)
    return re.sub(r'\n{3,}', '\n\n', text).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--medical-tokens', type=int, default=8_000_000)
    parser.add_argument('--output-dir', type=Path, default=ROOT)
    parser.add_argument('--raw-dir', type=Path, default=LAB / 'data/cpt_medical_v1/raw')
    parser.add_argument('--max-length', type=int, default=2048)
    parser.add_argument('--max-tokens-per-document', type=int, default=4096)
    args = parser.parse_args()
    if args.medical_tokens < 1 or args.max_length < 64 or args.max_tokens_per_document < args.max_length:
        parser.error('Use a positive medical budget and document budget >= max length >= 64.')
    root = args.output_dir
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f'Refusing to overwrite a corpus directory: {root}')
    splits = {s: load_jsonl(SOURCE / f'{s}.jsonl') for s in ['train', 'validation', 'test']}
    queries = [question_text(r) + '\n' + r['reference'] + '\n' + r['reasoning_content'] for r in splits['train']]
    buckets = [r.get('subject', r['task']) if r['task'] == 'knowledge' else r['task'] for r in splits['train']]
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), min_df=2,
                                 max_df=.35, max_features=30000, sublinear_tf=True, dtype=np.float32)
    query_matrix = vectorizer.fit_transform(queries)
    heldout_grams = set()
    for row in splits['validation'] + splits['test']:
        tokens = words(question_text(row))
        heldout_grams.update(tuple(tokens[i:i+13]) for i in range(len(tokens)-12))
    sources = json.loads((args.raw_dir / 'sources.json').read_text(encoding='utf-8'))
    assert sources['guidelines']['revision'] == GUIDELINES_REVISION
    counts = Counter()
    candidates = []
    seen = set()
    for path in sources['guidelines']['files']:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256, columns=['id','source','title','url','clean_text']):
            rows = []
            for row in batch.to_pylist():
                counts['scanned'] += 1
                text = clean_text(row['clean_text'] or '')
                tokens = words(text)
                if len(tokens) < 60:
                    counts['too_short'] += 1
                    continue
                normalized_sha = hashlib.sha256(' '.join(tokens).encode()).hexdigest()
                if normalized_sha in seen:
                    counts['duplicate_document'] += 1
                    continue
                seen.add(normalized_sha)
                if any(tuple(tokens[i:i+13]) in heldout_grams for i in range(len(tokens)-12)):
                    counts['heldout_13gram_overlap'] += 1
                    continue
                url = row['url'] if row['url'] not in (None, 'None') else ''
                title = row['title'] if row['title'] not in (None, 'None', '') else unquote(url.rsplit('/', 1)[-1]).replace('_', ' ')
                title = title or text.split('\n', 1)[0].lstrip('# ')[:160]
                rows.append(dict(id=row['id'], source=row['source'], title=title, url=url,
                                 text=text, normalized_sha256=normalized_sha))
            if not rows:
                continue
            document_matrix = vectorizer.transform([((r['title']+'\n')*3 + r['text']) for r in rows])
            similarities = (document_matrix @ query_matrix.T).toarray()
            for row, sims in zip(rows, similarities, strict=True):
                match = int(sims.argmax())
                if sims[match] < .08:
                    counts['low_relevance'] += 1
                    continue
                row.update(relevance=float(sims[match]), bucket=buckets[match],
                           matched_train_id=splits['train'][match]['source_id'])
                candidates.append(row)
            if counts['scanned'] % 4096 == 0:
                print('scanned', counts['scanned'], 'candidates', len(candidates), flush=True)
    tokenizer = AutoTokenizer.from_pretrained(LAB / 'models/Qwen3.5-2B')
    eos = tokenizer.convert_tokens_to_ids('<|endoftext|>')
    grouped = defaultdict(list)
    for row in candidates:
        grouped[row['bucket']].append(row)
    for group in grouped.values():
        group.sort(key=lambda r: (-r['relevance'], r['id']))
    selected, train_blocks, val_blocks = [], [], []
    chunk_hashes = set()
    token_counts = Counter()
    offsets = Counter()
    group_names = sorted(grouped)
    medical_tokens = 0
    # Round-robin across six knowledge subjects and three applied tasks.
    while medical_tokens < args.medical_tokens:
        progressed = False
        for name in group_names:
            if offsets[name] >= len(grouped[name]):
                continue
            progressed = True
            row = grouped[name][offsets[name]]
            offsets[name] += 1
            ids = tokenizer.encode(row['title'] + '\n\n' + row['text'], add_special_tokens=False)
            is_validation = int(hashlib.sha256((row['title'].casefold() or row['url'] or row['id']).encode()).hexdigest()[:8], 16) % 25 == 0
            blocks = []
            spans = sample_spans(ids, row['id'], args.max_tokens_per_document, args.max_length, SEED)
            for start, end in spans:
                block = ids[start:end] + [eos]
                fingerprint = token_fingerprint(block)
                if fingerprint in chunk_hashes:
                    counts['duplicate_token_chunk'] += 1
                    continue
                chunk_hashes.add(fingerprint)
                blocks.append(dict(input_ids=block, document_id=row['id'], source=row['source'], bucket=name,
                                   source_token_start=start, source_token_end=end, is_document_end=end == len(ids)))
            if not blocks:
                continue
            n_tokens = sum(len(b['input_ids']) for b in blocks)
            if is_validation:
                val_blocks.extend(blocks)
            else:
                train_blocks.extend(blocks)
                medical_tokens += n_tokens
                token_counts[name] += n_tokens
            selected.append(dict(row, split='validation' if is_validation else 'train', used_tokens=n_tokens,
                                 source_tokens=len(ids)+1,
                                 sampled_spans=[dict(start=b['source_token_start'], end=b['source_token_end']) for b in blocks]))
            if medical_tokens >= args.medical_tokens:
                break
        if not progressed:
            raise RuntimeError(f'Not enough relevant source material: {medical_tokens} tokens.')
    wiki_path = args.raw_dir / 'wikitext/wikitext-2-raw-v1/train-00000-of-00001.parquet'
    wiki = [r['text'].strip() for r in pq.read_table(wiki_path).to_pylist() if len(r['text'].split()) >= 80]
    random.Random(SEED).shuffle(wiki)
    replay_tokens = 0
    for index, text in enumerate(wiki):
        tok = words(text)
        if any(tuple(tok[i:i+13]) in heldout_grams for i in range(len(tok)-12)):
            continue
        ids = tokenizer.encode(text, add_special_tokens=False) + [eos]
        if len(ids) < 128:
            continue
        train_blocks.append(dict(input_ids=ids, document_id=f'wikitext-{index}', source='wikitext', bucket='replay',
                                 source_token_start=0, source_token_end=len(ids)-1, is_document_end=True))
        selected.append(dict(id=f'wikitext-{index}', source='Salesforce/wikitext', title='', url='', text=text,
                             split='train', used_tokens=len(ids), source_tokens=len(ids), bucket='replay',
                             sampled_spans=[dict(start=0, end=len(ids)-1)]))
        replay_tokens += len(ids)
        if replay_tokens >= medical_tokens / 99:
            break
    random.Random(SEED).shuffle(train_blocks)
    random.Random(SEED).shuffle(val_blocks)
    # Split documents before packing; retain the whole selected validation stream.
    train_blocks = pack_excerpts(train_blocks, args.max_length)
    val_blocks = pack_excerpts(val_blocks, args.max_length)
    train_ids = {s['document_id'] for b in train_blocks for s in b['spans']}
    val_ids = {s['document_id'] for b in val_blocks for s in b['spans']}
    assert not train_ids & val_ids
    assert sum(len(b['input_ids']) for b in train_blocks) == medical_tokens + replay_tokens
    root.mkdir(parents=True, exist_ok=True)
    save_jsonl(root / 'documents.jsonl', selected)
    save_jsonl(root / 'train.jsonl', train_blocks)
    save_jsonl(root / 'validation.jsonl', val_blocks)
    # A separate immutable copy of the existing project test set, before any new generation.
    save_jsonl(root / 'evaluation.jsonl', [dict(r, cohort='cpt_frozen_project_test') for r in sorted(splits['test'], key=lambda r:(r['task'],r['source_id']))])
    manifest = dict(seed=SEED, sources=sources,
        replay=dict(repository='Salesforce/wikitext',revision=WIKI_REVISION,subset='wikitext-2-raw-v1',split='train',
                    substitution='The original RedPajama sample is unavailable; use 1% Wikipedia prose replay, not an exact GAP-Replay reproduction.'),
        selection='Full-document TF-IDF cosine to SFT TRAIN questions, answers and explanations only; max match >=0.08; round-robin six subjects and three tasks; no test-driven retrieval.',
        decontamination=dict(normalization='lowercase alphanumeric words', heldout_question_and_context_ngram=13,
            excluded=counts['heldout_13gram_overlap'], limitations='Lexical decontamination only; does not prove absence of paraphrase overlap or base-pretraining contamination.'),
        filtering=dict(counts), candidate_buckets=dict(Counter(r['bucket'] for r in candidates)),
        selected_document_count=len(selected),train_blocks=len(train_blocks),validation_blocks=len(val_blocks),
        medical_train_tokens=medical_tokens,replay_train_tokens=replay_tokens,train_tokens=medical_tokens+replay_tokens,
        medical_tokens_by_bucket=dict(token_counts),source_document_counts=dict(Counter(r['source'] for r in selected)),
        max_length=args.max_length,max_tokens_per_document=args.max_tokens_per_document,
        objective='raw prose causal next-token prediction; no chat template; all nonpadding next tokens supervised',
        packing=dict(method='EOS-delimited concatenate-and-chunk', overlap=0, dropped_tail_tokens=0,
                     attention='ordinary causal attention and recurrent state across excerpts within each pack; no document isolation',
                     long_document_sampling='One seeded contiguous window per equal-sized source stratum; budget includes excerpt EOS; no fixed prefix truncation',
                     span_coordinates='token offsets into title + two newlines + cleaned body; replay offsets into body; ends are exclusive',
                     validation='All selected validation documents packed separately; no 128-block subsampling'),
        preparation_sources=[
            'https://arxiv.org/html/2402.10373v3#S3.SS2',
            'https://github.com/mosaicml/llm-foundry/blob/main/llmfoundry/data/data.py',
            'https://github.com/hiyouga/LlamaFactory/blob/main/src/llamafactory/data/processor/pretrain.py',
            'https://github.com/epfLLM/meditron/tree/main/gap-replay'],
        document_end_token='<|endoftext|>',document_end_token_id=eos,
        train_sha256=sha(root/'train.jsonl'),validation_sha256=sha(root/'validation.jsonl'),documents_sha256=sha(root/'documents.jsonl'),
        evaluation_sha256=sha(root/'evaluation.jsonl'),evaluation_counts=dict(Counter(r['task'] for r in splits['test'])),
        sft_source_sha256={s:sha(SOURCE/f'{s}.jsonl') for s in splits})
    (root/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in manifest.items() if k not in ['sources','sft_source_sha256']},ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    main()
