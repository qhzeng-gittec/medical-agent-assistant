"""Build an explicitly unverified expanded-prose candidate; never certify coverage."""
import bisect
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from evaluation.audit_cpt_data import audit_corpus
from data_processing.cpt_data import pack_excerpts
from common.io import load_jsonl, save_jsonl
from data_processing.download_cpt_sources import save

LAB = Path(__file__).resolve().parents[1]
OUT = LAB/'data/cpt_expanded_candidate_v1'
BACKGROUND = LAB/'data/cpt_medical_v2'
MEDICAL_TARGET = 8_000_000


def sha(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def normalized(text):
    return ' '.join(re.findall(r'[a-z0-9]+', text.lower()))


def rejection(title, text):
    if len(text.split()) < 25:
        return 'short_or_heading'
    if re.search(r'[A-Za-z]{35,}', text):
        return 'ocr_joined_words'
    if re.search(r'Answer\s*=|Answers to Case|Review Questions|The answer is|Medical (?:image|knowledge) question:', text, re.I):
        return 'question_answer_format'
    if re.search(r'Sandbox:|Image Insertion|^List of|Template:', title, re.I):
        return 'editorial_or_index_page'
    return None


def prose_spans(text, tokenized, width=2047):
    ids, offsets = tokenized['input_ids'], tokenized['offset_mapping']
    ends = [end for _, end in offsets]
    boundaries = sorted({bisect.bisect_right(ends, match.end())
                         for match in re.finditer(r'(?<=[.!?])\s+|\n\n+', text)})
    start, spans = 0, []
    while start < len(ids):
        end = min(start + width, len(ids))
        if end < len(ids):
            boundary_index = bisect.bisect_right(boundaries, end)-1
            if boundary_index >= 0 and boundaries[boundary_index] >= start+width//2:
                end = boundaries[boundary_index]
        spans.append((start, end))
        start = end
    return spans


def subdivide_spans(spans, width=2047):
    return [(start, min(start+width, end)) for begin, end in spans
            for start in range(begin, end, width)]


def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError('Candidate output already exists; keep frozen artifacts intact.')
    OUT.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(LAB/'models/Qwen3.5-2B', local_files_only=True)
    eos = tokenizer.convert_tokens_to_ids('<|endoftext|>')
    background_manifest = json.loads((BACKGROUND/'manifest.json').read_text(encoding='utf-8'))
    assert sha(BACKGROUND/'documents.jsonl') == background_manifest['documents_sha256']
    assert sha(BACKGROUND/'validation.jsonl') == background_manifest['validation_sha256']
    background = load_jsonl(BACKGROUND/'documents.jsonl')
    validation_docs = [d for d in background if d['split']=='validation']
    seen = {hashlib.sha256(normalized(d['text']).encode()).hexdigest() for d in validation_docs}
    documents, excerpts, associations = [], [], []
    reasons, source_counts, bucket_counts = Counter(), Counter(), Counter()
    medical_tokens = 0

    def add(title, body, source, bucket, linked_ids, provenance, existing=None):
        nonlocal medical_tokens
        reason = rejection(title, body)
        if reason:
            reasons[reason] += 1
            return
        fingerprint = hashlib.sha256(normalized(body).encode()).hexdigest()
        if fingerprint in seen:
            reasons['duplicate_body_or_validation_body'] += 1
            return
        seen.add(fingerprint)
        text = title+'\n\n'+body
        if existing is None:
            encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            ids = encoded['input_ids']
            spans = prose_spans(text, encoded)
            doc_id = 'expanded-'+fingerprint
        else:
            ids = tokenizer.encode(text, add_special_tokens=False)
            spans = subdivide_spans([(s['start'],s['end']) for s in existing['sampled_spans']])
            doc_id = existing['id']
        used = sum(b-a+1 for a,b in spans)
        documents.append(dict(id=doc_id, title=title, text=body, source=source, bucket=bucket, split='train',
                              source_tokens=len(ids)+1, used_tokens=used,
                              sampled_spans=[dict(start=a,end=b) for a,b in spans], provenance=provenance))
        for a,b in spans:
            excerpts.append(dict(input_ids=ids[a:b]+[eos], document_id=doc_id, source=source, bucket=bucket,
                                 source_token_start=a, source_token_end=b, is_document_end=b==len(ids)))
        for sid in linked_ids:
            associations.append(dict(source_id=sid, document_id=doc_id, relation='candidate_evidence_not_semantic_coverage'))
        source_counts[source] += used
        bucket_counts[bucket] += used
        medical_tokens += used

    for r in load_jsonl(LAB/'outputs/cpt_reuse_inventory_v1/reviewed_evidence.jsonl'):
        e = next((e for e in r['evidence'] if not e['format_review_needed']), None)
        if e:
            add('Medical knowledge', e['text'], 'existing_reviewed_prose', r['task'], [r['source_id']], dict(path=e['source']))
    for r in load_jsonl(LAB/'outputs/cpt_expanded_coverage_v1/reviewed_new_evidence.jsonl'):
        add('Medical knowledge', r['text'], 'reviewed_textbook_excerpt', r['task'], [r['source_id']],
            dict(dataset=r['dataset'], passage_id=r['passage_id'], source_path=r['source_path']))
    for r in load_jsonl(LAB/'outputs/cpt_reuse_inventory_v1/existing_study_contexts.jsonl'):
        add('Medical research study', r['text'], 'existing_study_context', 'context', [r['source_id']],
            dict(source_split=r['split'], scope='Study prose only; original task question and answer omitted'))
    priority = load_jsonl(LAB/'outputs/cpt_expanded_coverage_v1/priority_evidence.jsonl')
    for rank in range(5):
        for r in priority:
            if rank < len(r['candidates']):
                c = r['candidates'][rank]
                add(c['title'], c['text'], 'retrieved_'+c['dataset'], r['task'], [r['source_id']],
                    dict(dataset=c['dataset'], passage_id=c['passage_id'], path=c['path'], retrieval_rank=rank+1))
    targeted_tokens = medical_tokens
    assert targeted_tokens < MEDICAL_TARGET, 'Prioritized evidence exceeds the proposed budget.'
    # Fill remaining budget from the previously frozen train-only guideline excerpts.
    for d in background:
        if d['split']=='train' and d['bucket']!='replay':
            add(d['title'], d['text'], 'guideline_background', d['bucket'], [],
                dict(previous_corpus='cpt_medical_v2', original_source=d['source']), existing=d)
            if medical_tokens >= MEDICAL_TARGET:
                break
    assert medical_tokens >= MEDICAL_TARGET
    replay_tokens = 0
    for d in background:
        if d['split']=='train' and d['bucket']=='replay':
            ids = tokenizer.encode(d['text'], add_special_tokens=False)
            spans = subdivide_spans([(0, len(ids))])
            used = len(ids)+len(spans)
            documents.append(dict(d, used_tokens=used, sampled_spans=[dict(start=a,end=b) for a,b in spans]))
            for a,b in spans:
                excerpts.append(dict(input_ids=ids[a:b]+[eos], document_id=d['id'], source=d['source'], bucket='replay',
                                     source_token_start=a, source_token_end=b, is_document_end=b==len(ids)))
            replay_tokens += used
            if replay_tokens >= medical_tokens/99:
                break
    assert replay_tokens >= medical_tokens/99
    random.Random(42).shuffle(excerpts)
    blocks, group, length = [], [], 0
    for e in excerpts:
        assert len(e['input_ids']) <= 2048
        if group and length+len(e['input_ids']) > 2048:
            blocks.extend(pack_excerpts(group, 2048))
            group, length = [], 0
        group.append(e)
        length += len(e['input_ids'])
    if group:
        blocks.extend(pack_excerpts(group, 2048))
    documents.extend(validation_docs)
    for name, rows in [('train', blocks), ('validation', load_jsonl(BACKGROUND/'validation.jsonl')),
                       ('documents', documents), ('evaluation', load_jsonl(LAB/'data/knowledge_experiments_v1/test.jsonl')),
                       ('candidate_map', associations)]:
        save_jsonl(OUT/f'{name}.jsonl', rows)
    manifest = dict(experiment_type='expanded_candidate_not_full_coverage', semantic_full_coverage_verified=False,
        seed=42, max_length=2048, max_tokens_per_document=max(d['used_tokens'] for d in documents if d['bucket']!='replay'),
        objective='raw medical prose next-token prediction; original task question-answer pairs excluded',
        train_tokens=medical_tokens+replay_tokens, medical_train_tokens=medical_tokens, replay_train_tokens=replay_tokens,
        medical_tokens_by_bucket=dict(bucket_counts), medical_tokens_by_source=dict(source_counts),
        prioritized_evidence_tokens=targeted_tokens, train_blocks=len(blocks), validation_blocks=143,
        document_end_token='<|endoftext|>', document_end_token_id=eos,
        packing=dict(method='Greedy whole-excerpt packing with EOS; up to 2048 tokens per row',
                     long_new_prose='Lossless adjacent spans <=2047 content tokens; prefer sentence/paragraph boundary in the latter half of each window',
                     guideline_background='Reuse previously frozen stratified source windows', overlap=0, dropped_packing_tokens=0,
                     attention='ordinary causal attention across EOS; no document-isolated attention'),
        quality_exclusions=dict(reasons), selected_medical_documents=len(documents)-len(validation_docs)-sum(d['bucket']=='replay' and d['split']=='train' for d in documents),
        source_hashes=dict(background_manifest=sha(BACKGROUND/'manifest.json'),
                           priority_evidence=sha(LAB/'outputs/cpt_expanded_coverage_v1/priority_evidence.jsonl')),
        limitations=['Retrieval candidates have not all passed semantic review; this is not a full-knowledge-coverage corpus.',
                     'Train, validation and test answer knowledge was intentionally used for evidence selection.',
                     'PubMed full-source retrieval is still incomplete; no newly downloaded PubMed passages are included in this snapshot.',
                     'Existing guideline excerpts were sampled; new selected prose is retained without token truncation.'])
    manifest.update({name+'_sha256':sha(OUT/f'{name}.jsonl') for name in ['train','validation','documents','evaluation','candidate_map']})
    save(OUT/'manifest.json', manifest)
    save(OUT/'coverage_status.json', dict(ready_for_training=False, required=6435, accepted=0,
         reason='Semantic full coverage has not been verified. Candidate corpus only; requires an explicitly authorized exploratory run.',
         corpus_manifest_sha256=sha(OUT/'manifest.json')))
    audit = audit_corpus(OUT)
    save(OUT/'audit.json', audit)
    print(json.dumps(dict(train_tokens=manifest['train_tokens'], medical_tokens=medical_tokens,
        replay_tokens=replay_tokens, targeted_tokens=targeted_tokens, train_blocks=len(blocks),
        source_tokens=dict(source_counts), exclusions=dict(reasons), audit=audit['status']), indent=2), flush=True)


if __name__ == '__main__':
    main()
