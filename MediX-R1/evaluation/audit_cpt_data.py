"""Reconstruct packed CPT rows from source documents and audit split/token accounting."""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoTokenizer

from common.io import load_jsonl

LAB = Path(__file__).resolve().parents[1]


def audit_corpus(data_dir: Path) -> dict:
    manifest = json.loads((data_dir/'manifest.json').read_text(encoding='utf-8'))
    for name in ['train', 'validation', 'documents', 'evaluation']:
        assert hashlib.sha256((data_dir/f'{name}.jsonl').read_bytes()).hexdigest() == manifest[f'{name}_sha256'], name
    tokenizer = AutoTokenizer.from_pretrained(LAB/'models/Qwen3.5-2B', local_files_only=True)
    eos = tokenizer.convert_tokens_to_ids(manifest['document_end_token'])
    assert eos == manifest['document_end_token_id']
    documents = load_jsonl(data_dir/'documents.jsonl')
    docs = {d['id']: d for d in documents}
    assert len(docs) == len(documents)
    encoded = {}
    selected_content = 0
    omitted_content = 0
    long_docs = 0
    later_window_docs = 0
    for doc in documents:
        text = doc['text'] if doc['bucket'] == 'replay' else doc['title'] + '\n\n' + doc['text']
        ids = tokenizer.encode(text, add_special_tokens=False)
        encoded[doc['id']] = ids
        assert len(ids) + 1 == doc['source_tokens'], doc['id']
        spans = doc['sampled_spans']
        previous_end = 0
        for span in spans:
            assert previous_end <= span['start'] < span['end'] <= len(ids), doc['id']
            previous_end = span['end']
        used_content = sum(s['end'] - s['start'] for s in spans)
        assert used_content + len(spans) == doc['used_tokens'], doc['id']
        if doc['bucket'] != 'replay':
            assert doc['used_tokens'] <= manifest['max_tokens_per_document']
        if doc['split'] == 'train' and doc['bucket'] != 'replay':
            selected_content += used_content
            omitted_content += len(ids) - used_content
            if len(ids) + 1 > manifest['max_tokens_per_document']:
                long_docs += 1
                later_window_docs += any(s['start'] >= len(ids)//2 for s in spans)
    split_ids = {}
    summaries = {}
    for split in ['train', 'validation']:
        blocks = load_jsonl(data_dir/f'{split}.jsonl')
        counts = Counter()
        by_document = Counter()
        intervals = defaultdict(list)
        split_ids[split] = set()
        for block in blocks:
            assert 2 <= len(block['input_ids']) <= manifest['max_length']
            recovered = []
            for span in block['spans']:
                doc = docs[span['document_id']]
                assert doc['split'] == split
                assert doc['bucket'] == span['bucket']
                assert span['block_start'] == len(recovered)
                start, end = span['source_token_start'], span['source_token_end']
                ids = encoded[doc['id']]
                assert 0 <= start <= end <= len(ids)
                recovered.extend(ids[start:end])
                if span['ends_with_eos']:
                    recovered.append(eos)
                    counts['eos_tokens'] += 1
                    assert span['is_document_end'] == (end == len(ids))
                else:
                    assert not span['is_document_end']
                assert span['block_end'] == len(recovered)
                by_document[doc['id']] += span['block_end'] - span['block_start']
                counts[span['bucket']] += span['block_end'] - span['block_start']
                split_ids[split].add(doc['id'])
                if end > start:
                    intervals[doc['id']].append((start, end))
            assert recovered == block['input_ids']
        for doc_id, used_tokens in by_document.items():
            assert used_tokens == docs[doc_id]['used_tokens'], doc_id
            ordered = sorted(intervals[doc_id])
            assert all(left[1] <= right[0] for left, right in zip(ordered, ordered[1:])), doc_id
            merged = []
            for start, end in ordered:
                if merged and merged[-1]['end'] == start:
                    merged[-1]['end'] = end
                else:
                    merged.append(dict(start=start, end=end))
            expected = []
            for span in docs[doc_id]['sampled_spans']:
                if expected and expected[-1]['end'] == span['start']:
                    expected[-1]['end'] = span['end']
                else:
                    expected.append(dict(span))
            assert merged == expected, doc_id
        assert split_ids[split] == {d['id'] for d in documents if d['split'] == split}
        lengths = [len(b['input_ids']) for b in blocks]
        summaries[split] = dict(blocks=len(blocks), tokens=sum(lengths), documents=len(split_ids[split]),
                                full_blocks=lengths.count(manifest['max_length']),
                                mean_length=sum(lengths)/len(lengths), minimum_length=min(lengths),
                                token_counts_by_bucket={k:v for k,v in counts.items() if k != 'eos_tokens'},
                                eos_tokens=counts['eos_tokens'])
    assert not split_ids['train'] & split_ids['validation']
    assert summaries['train']['tokens'] == manifest['train_tokens']
    assert summaries['train']['token_counts_by_bucket']['replay'] == manifest['replay_train_tokens']
    assert {k:v for k,v in summaries['train']['token_counts_by_bucket'].items() if k != 'replay'} == manifest['medical_tokens_by_bucket']
    return dict(status='passed', verification='All packed tokens reconstructed from original text; hashes, spans, budgets, EOS and split isolation checked',
                splits=summaries, medical_train_content_tokens=selected_content,
                medical_train_content_tokens_not_sampled=omitted_content,
                long_medical_train_documents=long_docs, long_documents_with_second_half_window=later_window_docs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=LAB/'data/cpt_medical_v2')
    args = parser.parse_args()
    result = audit_corpus(args.data_dir)
    (args.data_dir/'audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
