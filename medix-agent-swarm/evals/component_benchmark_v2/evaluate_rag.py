"""Batch real API embeddings; freeze a dev-selected threshold before evaluating test families."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path

import numpy as np

from api import API, EMBED, read, write

HERE = Path(__file__).resolve().parent
INSTRUCTION = 'Given a medical consultation question, retrieve relevant patient guidance and clinical evidence.'
THRESHOLDS = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def lines(path):
    import json
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]


def verify_freeze(data):
    freeze = read(data / 'freeze.json')
    for name, digest in freeze['sha256'].items():
        if sha256((data / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Dataset changed after freeze: {name}')
    return freeze


def score(row, k, threshold):
    returned = {d['id'] for d in row['ranking'][:k] if d['score'] >= threshold}
    gold = row['case']['gold_groups']
    return {'any_hit': any(returned.intersection(g) for g in gold),
            'all_hit': bool(gold) and all(returned.intersection(g) for g in gold),
            'empty': not returned}


def metrics(rows, k, threshold):
    positive = [r for r in rows if r['case']['kind'] in {'answerable', 'multi_evidence', 'context_supplied'}]
    negative = [r for r in rows if r['case']['kind'] == 'unanswerable']
    hit = sum(score(r, k, threshold)['all_hit'] for r in positive)
    empty = sum(score(r, k, threshold)['empty'] for r in negative)
    return {'k': k, 'threshold': threshold, 'positive_queries': len(positive), 'all_source_hits': hit,
            'any_source_hits': sum(score(r, k, threshold)['any_hit'] for r in positive),
            'negative_queries': len(negative), 'negative_empty': empty,
            'balanced_objective': (hit / len(positive) + empty / len(negative)) / 2}


def main(args):
    data = HERE / 'data'
    freeze = verify_freeze(data)
    corpus, cases = lines(data / 'corpus.jsonl'), lines(data / 'rag_cases.jsonl')
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = {'dataset_freeze': freeze, 'embedding_model': EMBED, 'query_instruction': INSTRUCTION,
                'k_values': [1, 3, 5, 10], 'threshold_grid': THRESHOLDS,
                'selection': 'On dev only: maximize mean(all-source recall, negative empty rate) at K=3; ties choose smaller threshold.',
                'statistics': 'Query counts with shared families; raw context questions reported separately.',
                'scope': 'Frozen MedlinePlus topic-source retrieval; local cosine index; no answer generation.'}
    path = args.output / 'protocol.json'
    if path.exists() and read(path) != protocol:
        raise ValueError('Protocol changed; choose another output directory')
    write(path, protocol)
    api = API(args.cache / 'embeddings', max_requests=500)

    def vectors(prefix, texts):
        batches = [(n, texts[n:n+8]) for n in range(0, len(texts), 8)]
        def fetch(pair):
            n, batch = pair
            return api.embed(f'{prefix}_{n:04}', batch)
        with ThreadPoolExecutor(max_workers=3) as pool:
            result = np.asarray([v for group in pool.map(fetch, batches) for v in group], dtype=np.float64)
        norm = np.linalg.norm(result, axis=1, keepdims=True)
        if not np.isfinite(result).all() or np.any(norm == 0):
            raise ValueError('Invalid embedding vectors')
        return result / norm

    docs = vectors('corpus', [d['content'] for d in corpus])
    selected_path = args.output / 'selection.json'
    groups = {}
    for split in ['dev', 'test']:
        subset = [c for c in cases if c['split'] == split]
        qs = vectors(split, [f'Instruct: {INSTRUCTION}\nQuery: {c["query"]}' for c in subset])
        similarities = qs @ docs.T
        rows = []
        for c, values in zip(subset, similarities):
            order = np.argsort(-values, kind='stable')[:10]
            row = {'case': c, 'ranking': [{'id': corpus[int(i)]['id'], 'score': float(values[i])} for i in order]}
            write(args.output / 'cases' / (c['id'] + '.json'), row)
            rows.append(row)
        groups[split] = rows
        if split == 'dev':
            sweep = [metrics(rows, 3, t) for t in THRESHOLDS]
            chosen = max(sweep, key=lambda r: (r['balanced_objective'], -r['threshold']))
            selection = {'selected_on': 'dev', 'selected_threshold': chosen['threshold'], 'dev_sweep': sweep}
            if selected_path.exists() and read(selected_path) != selection:
                raise ValueError('Frozen dev selection differs')
            write(selected_path, selection)
        print(f'RAG {split}: {len(rows)} queries completed', flush=True)
    summary = {'corpus_documents': len(corpus), 'queries': len(cases),
               'families': len({c['family_id'] for c in cases}),
               'dimensions': int(docs.shape[1]), 'selected_threshold': read(selected_path)['selected_threshold']}
    for split, rows in groups.items():
        summary[split] = {'queries': len(rows), 'families': len({r['case']['family_id'] for r in rows}),
                          'kinds': dict(Counter(r['case']['kind'] for r in rows)),
                          'top_k_at_zero': [metrics(rows, k, 0) for k in [1, 3, 5, 10]],
                          'dev_selected': metrics(rows, 3, summary['selected_threshold'])}
        paired = {}
        for row in rows:
            if row['case']['kind'] in {'needs_history', 'context_supplied'}:
                paired.setdefault(row['case']['family_id'], {})[row['case']['kind']] = score(row, 3, 0)['all_hit']
        summary[split]['context_pairs'] = {'families': len(paired),
                                         'raw_hits': sum(v['needs_history'] for v in paired.values()),
                                         'with_context_hits': sum(v['context_supplied'] for v in paired.values())}
    write(args.output / 'summary.json', summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    main(parser.parse_args())
