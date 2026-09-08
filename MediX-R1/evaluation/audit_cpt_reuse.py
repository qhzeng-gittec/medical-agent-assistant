"""Local-only evidence inventory; retrieval scores are NOT semantic coverage labels."""
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from transformers import AutoTokenizer

from common.io import load_jsonl, save_jsonl
from data_processing.prepare_cpt_experiment import clean_text, question_text

LAB = Path(__file__).resolve().parents[1]
OUT = LAB / 'outputs/cpt_reuse_inventory_v1'
CORPUS = LAB / 'data/cpt_medical_v2'
GENERIC = {'yes', 'no', 'maybe', 'unknown', 'uncertain', 'none'}
STOP = set(ENGLISH_STOP_WORDS) - {'no', 'not', 'without', 'above', 'below'}


def normalized(text):
    return ' '.join(re.findall(r'[a-z0-9]+', text.lower()))


def anchors(row):
    answer = row['reference']
    if normalized(answer) in GENERIC:
        answer = row['target']
    return set(normalized(answer).split()) - STOP - GENERIC


def windows(length, size=384, stride=288):
    for start in range(0, length, stride):
        end = min(start + size, length)
        yield start, end
        if end == length:
            break


def reviewed_evidence(rows):
    evidence = defaultdict(list)
    skipped = Counter()
    requirements = {r['id']: dict(r, source_id=r['id']) for r in load_jsonl(
        LAB / 'data/cpt_knowledge_coverage_v1/requirements.jsonl')}

    def add(source, row, text, standalone_reviewed):
        sid = row['source_id']
        if sid not in rows:
            skipped['not_in_current_splits'] += 1
            return
        current = rows[sid]
        if any(key in row and normalized(row[key]) != normalized(current[key])
               for key in ['reference', 'target', 'user_text']):
            skipped['changed_question_or_answer'] += 1
            return
        if 'question' in row and normalized(row['question']) != normalized(current['user_text']):
            skipped['changed_question'] += 1
            return
        # This is a format flag, not an automatic acceptance/rejection decision.
        format_flag = bool(re.search(r'\b(this (?:patient|case|question|image)|the (?:answer|question)|correct (?:option|choice))\b', text, re.I))
        evidence[sid].append(dict(source=source, text=text,
                                 standalone_reviewed=standalone_reviewed,
                                 format_review_needed=format_flag))

    for path in sorted((LAB / 'data/cpt_knowledge_coverage_v1/items').glob('*.json')):
        r = json.loads(path.read_text(encoding='utf-8'))
        if r['accepted']:
            add(str(path.relative_to(LAB)), requirements[r['source_id']], r['training_text'], True)
    path = LAB / 'data/knowledge_retention_v1/families.jsonl'
    for r in load_jsonl(path):
        if r['probe']['ready'] and all(r['audit'][k] for k in ['source_valid', 'equivalent', 'no_added_cues', 'evidence_valid']):
            add(str(path.relative_to(LAB)), r, r['probe']['evidence'], False)
    path = LAB / 'data/rationale_pilot_v2/rewrite_audit.jsonl'
    for r in load_jsonl(path):
        if r['accepted']:
            add(str(path.relative_to(LAB)), r, r['rewritten'], False)
    return evidence, skipped


def actual_passages(tokenizer, docs):
    passages = []
    with (CORPUS / 'train.jsonl').open(encoding='utf-8') as file:
        for block_index, line in enumerate(file):
            block = json.loads(line)
            for span in block['spans']:
                length = span['source_token_end'] - span['source_token_start']
                assert span['block_end'] - span['block_start'] == length + int(span['ends_with_eos'])
                for start, end in windows(length):
                    bs, be = span['block_start'] + start, span['block_start'] + end
                    text = tokenizer.decode(block['input_ids'][bs:be], skip_special_tokens=False)
                    passages.append(dict(document_id=span['document_id'], block_index=block_index,
                                         block_start=bs, block_end=be,
                                         source_token_start=span['source_token_start'] + start,
                                         source_token_end=span['source_token_start'] + end,
                                         title=docs[span['document_id']]['title'], text=text))
    return passages


def query_parts(row):
    question = question_text(row)
    if row['task'] == 'context':
        question = question.split('\n\nQuestion: ', 1)[-1]
    return [question, row.get('reasoning_content', ''), row['target']]


def retrieve(rows, documents, count):
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), min_df=1,
                                 max_features=180000, sublinear_tf=True, dtype=np.float32)
    # Titles aid ranking, but never count as evidence unless in the stored text.
    matrix = vectorizer.fit_transform(d['title'] + '\n' + d['text'] for d in documents)
    results = []
    for offset in range(0, len(rows), 32):
        batch = rows[offset:offset + 32]
        parts = vectorizer.transform(p for r in batch for p in query_parts(r))
        queries = .35 * parts[0::3] + .45 * parts[1::3] + .2 * parts[2::3]
        scores = (queries @ matrix.T).toarray()
        for row, score in zip(batch, scores):
            top = np.argpartition(-score, min(80, len(score) - 1))[:80]
            top = top[np.argsort(-score[top])]
            selected, seen = [], set()
            for index in top:
                doc = documents[int(index)]
                if doc['document_id'] in seen or score[index] <= 0:
                    continue
                seen.add(doc['document_id'])
                candidate = dict(doc, retrieval_score=round(float(score[index]), 5))
                if 'block_index' not in doc:
                    # Choose a literal source substring, without summarization.
                    words = list(re.finditer(r'\S+', doc['text']))
                    slices = [(words[a].start(), words[b-1].end()) for a, b in windows(len(words), 180, 120)]
                    texts = [doc['text'][a:b] for a, b in slices]
                    q = vectorizer.transform(query_parts(row))
                    passage_scores = (vectorizer.transform(texts) @ (.35*q[0]+.45*q[1]+.2*q[2]).T).toarray().ravel()
                    best = int(np.argmax(passage_scores))
                    a, b = slices[best]
                    candidate.update(text=texts[best], source_char_start=a, source_char_end=b,
                                     passage_score=round(float(passage_scores[best]), 5))
                terms = anchors(row)
                hits = terms.intersection(normalized(candidate['text']).split())
                candidate.update(answer_anchor_count=len(terms), answer_anchor_hits=sorted(hits),
                                 answer_anchor_fraction=round(len(hits)/len(terms), 3) if terms else None)
                selected.append(candidate)
                if len(selected) == count:
                    break
            results.append(selected)
        if offset % 512 == 0:
            print(f'retrieved {min(offset+32, len(rows))}/{len(rows)}', flush=True)
    return results


def strong(candidates):
    return any(c.get('passage_score', c['retrieval_score']) >= .18
               and c['answer_anchor_count'] > 0 and c['answer_anchor_fraction'] >= .5 for c in candidates)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((CORPUS / 'manifest.json').read_text(encoding='utf-8'))
    for name in ['train', 'documents']:
        suffix = '.jsonl'
        assert hashlib.sha256((CORPUS / (name+suffix)).read_bytes()).hexdigest() == manifest[name+'_sha256']
    rows = []
    for split in ['train', 'validation', 'test']:
        path = LAB / f'data/knowledge_experiments_v1/{split}.jsonl'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest['sft_source_sha256'][split]
        rows.extend(dict(r, experiment_split=split) for r in load_jsonl(path))
    indexed = {r['source_id']: r for r in rows}
    assert len(rows) == len(indexed) == 6435
    evidence, skipped = reviewed_evidence(indexed)
    docs = {d['id']: d for d in load_jsonl(CORPUS / 'documents.jsonl')}
    tokenizer = AutoTokenizer.from_pretrained(LAB / 'models/Qwen3.5-2B', local_files_only=True)
    passages = actual_passages(tokenizer, docs)
    print(f'Actual training passages: {len(passages)}; reviewed IDs: {len(evidence)}', flush=True)
    actual = retrieve(rows, passages, 3)
    del passages
    raw = []
    for path in manifest['sources']['guidelines']['files']:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=128, columns=['id', 'title', 'url', 'clean_text']):
            for d in batch.to_pylist():
                body = clean_text(d['clean_text'] or '')
                if body:
                    raw.append(dict(document_id=d['id'], title=d['title'] or '', url=d['url'], text=body,
                                    selected_in_v2=d['id'] in docs,
                                    v2_split=docs.get(d['id'], {}).get('split')))
    print(f'Local raw documents: {len(raw)}', flush=True)
    sources = retrieve(rows, raw, 2)
    del raw
    results, reusable, context = [], [], []
    for row, trained, source in zip(rows, actual, sources):
        sid = row['source_id']
        existing = evidence.get(sid, [])
        status = ('reviewed_evidence_available' if existing else
                  'strong_actual_training_candidate' if strong(trained) else
                  'strong_raw_source_candidate' if strong(source) else 'needs_further_local_review')
        results.append(dict(source_id=sid, task=row['task'], split=row['experiment_split'],
                            question=query_parts(row)[0], reference=row['reference'], target=row['target'],
                            reasoning_content=row['reasoning_content'], status=status,
                            reviewed_evidence=existing, actual_training_candidates=trained,
                            raw_source_candidates=source, semantic_coverage='not_determined_by_retrieval'))
        if existing:
            reusable.append(dict(source_id=sid, task=row['task'], evidence=existing))
        if row['task'] == 'context':
            study = row['user_text'].split('Medical research context:\n', 1)[1].split('\n\nQuestion:', 1)[0]
            context.append(dict(source_id=sid, split=row['experiment_split'], text=study,
                                status='existing_study_prose_requires_scope_review'))
    save_jsonl(OUT / 'questions.jsonl', results)
    save_jsonl(OUT / 'reviewed_evidence.jsonl', reusable)
    save_jsonl(OUT / 'existing_study_contexts.jsonl', context)
    save_jsonl(OUT / 'review_queue.jsonl', [r for r in results if r['status'] == 'needs_further_local_review'])
    summary = dict(total_questions=len(rows), model_api_calls=0, gpu_training_started=False,
                   actual_corpus_sha256=manifest['train_sha256'],
                   reviewed_evidence_unique_ids=len(evidence), skipped_review_records=dict(skipped),
                   reviewed_evidence_records=sum(map(len, evidence.values())),
                   standalone_reviewed_ids=sum(any(e['standalone_reviewed'] for e in v) for v in evidence.values()),
                   format_review_flag_ids=sum(any(e['format_review_needed'] for e in v) for v in evidence.values()),
                   reviewed_evidence_characters=sum(len(v[0]['text']) for v in evidence.values()),
                   existing_context_count=len(context), existing_context_characters=sum(len(r['text']) for r in context),
                   status_counts=dict(Counter(r['status'] for r in results)),
                   by_task={task: dict(Counter(r['status'] for r in results if r['task']==task)) for task in sorted({r['task'] for r in results})},
                   by_split={s: dict(Counter(r['status'] for r in results if r['split']==s)) for s in ['train','validation','test']},
                   retrieval_rule='Weighted TF-IDF question .35, rationale .45, target .2; candidate score >=.18 and >=50% literal answer anchors. Heuristic only.',
                   limitations=['No retrieval category proves all necessary facts are present or absent.',
                                'Reviewed rationale/probe evidence does not always have a standalone-prose audit.',
                                'Source evidence is not necessarily inside actual training tokens.',
                                'Context and VQA depend on supplied study/image; CPT text cannot establish visual learning.',
                                'Original questions/answers are retained only in audit output, not added to CPT.',
                                'Confirmed semantic gaps and final paid generation volume remain undetermined.'])
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
