"""Post-hoc retrieval of error evidence from the exact frozen CPT training blocks."""
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer

from data_io import load_jsonl, save_jsonl

LAB = Path(__file__).resolve().parent
DATA = LAB / 'data/cpt_medical_v1'
OUT = LAB / 'outputs/cpt_medical_v1/coverage_audit'
OUT.mkdir(exist_ok=True)
docs = {d['id']: d for d in load_jsonl(DATA / 'documents.jsonl')}
blocks = load_jsonl(DATA / 'train.jsonl')
cache = OUT / 'actual_training_passages.jsonl'
if cache.exists():
    passages = load_jsonl(cache)
else:
    tokenizer = AutoTokenizer.from_pretrained(LAB / 'models/Qwen3.5-2B')
    passages = []
    for index, block in enumerate(blocks):
        text = tokenizer.decode(block['input_ids'], skip_special_tokens=True)
        # Overlap within actual blocks only; never add source text omitted by training.
        spans = list(re.finditer(r'\S+', text))
        for start in range(0, len(spans), 100):
            selected = spans[start:start + 180]
            if not selected:
                continue
            passages.append(dict(block_index=index, document_id=block['document_id'],
                                 text=text[selected[0].start():selected[-1].end()]))
    save_jsonl(cache, passages)
print('actual passages', len(passages), flush=True)
texts = [docs[p['document_id']]['title'] + '\n' + p['text'] for p in passages]
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), min_df=1,
                             max_features=180000, sublinear_tf=True, dtype=np.float32)
matrix = vectorizer.fit_transform(texts)
length_weights = np.array([min(1., len(p['text'].split()) / 60) for p in passages])
evaluation = load_jsonl(DATA / 'evaluation.jsonl')
retrieved = []
for row in evaluation:
    diagnosis = json.loads((OUT.parent / 'diagnostics/questions' / (row['source_id'] + '.json')).read_text(encoding='utf-8'))
    a, b = [diagnosis['models'][arm]['diagnosis']['answer_score'] for arm in ['sft_only', 'cpt_sft']]
    if not diagnosis['reference_valid'] or (a == 2 and b == 2):
        continue
    question = row['user_text'].split('\n\nExplain')[0].split('\n\nAnswer the question')[0]
    query = vectorizer.transform([question, row['reference'], row.get('reasoning_content', '')])
    scores = (matrix @ query.T).toarray()
    rank = 2 * scores[:, 0] + .5 * scores[:, 1] + .5 * scores[:, 2]
    rank *= length_weights
    best = np.argsort(-rank)
    selected = []
    seen = set()
    for i in best:
        p = passages[int(i)]
        if p['document_id'] in seen:
            continue
        seen.add(p['document_id'])
        d = docs[p['document_id']]
        selected.append(dict(p, title=d['title'], url=d['url'], score=float(rank[i]),
                             source_tokens=d.get('source_tokens'), used_tokens=d['used_tokens']))
        if len(selected) == 6:
            break
    retrieved.append(dict(source_id=row['source_id'], task=row['task'], subject=row.get('subject'),
                          question=question, reference=row['reference'], reference_reasoning=row.get('reasoning_content', ''),
                          baseline_score=a, cpt_score=b,
                          cpt_answer=diagnosis['models']['cpt_sft']['final_answer'],
                          cpt_reasoning=diagnosis['models']['cpt_sft']['reasoning'],
                          cpt_diagnosis=diagnosis['models']['cpt_sft']['diagnosis']['explanation'],
                          passages=selected))
save_jsonl(OUT / 'retrieved_evidence.jsonl', retrieved)
print(json.dumps(dict(retrieved=len(retrieved), cpt_errors=Counter(r['task'] for r in retrieved if r['cpt_score'] != 2),
                     regressions=Counter(r['task'] for r in retrieved if r['baseline_score'] == 2 and r['cpt_score'] != 2)), indent=2))
