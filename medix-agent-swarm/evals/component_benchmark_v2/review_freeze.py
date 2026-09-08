"""Review generated labels before any benchmark retrieval; seal exact dataset bytes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

from api import API, read, write
from evaluate_rag import lines

HERE = Path(__file__).resolve().parent
REVIEW = '''Review synthetic memory retrieval labels against ONLY the supplied fictional dialogue.
Return JSON {"reviews":[{"id":scenario id,"supported":true/false,"reason":"short reason"}]} in input order.
For each scenario verify that EVERY query.expected_facts proposition is entailed by user messages, with
correct speaker, negation, time and uncertainty; assistant suggestions are not patient facts.
Verify that each query actually asks for those facts. Missing details or contradictions mean false.
Do not assess medicine or add new facts. Independently assess the data, not the author.'''
RAG_REVIEW = '''Review source-derived retrieval questions. Return JSON {"reviews":[{"id":same id,"supported":true/false,"reason":"brief"}]} in input order.
For each item verify that the supplied evidence excerpt contains the concrete information asked in both the
English and Chinese question. A general topic relationship is insufficient. Both questions must ask the same
detail. Mark unsupported if a question asks more than the quoted evidence supplies. Use only supplied text.'''


def review_rows(response):
    if isinstance(response,list) and len(response)==1 and 'reviews' in response[0]:
        response=response[0]
    rows=response if isinstance(response,list) else response['reviews']
    for row in rows:
        if row['supported'] in ['true','false']:
            row['supported']=row['supported']=='true'
        if type(row['supported']) is not bool:
            raise ValueError('Review support must be a boolean')
    return rows


def main(args):
    data = HERE/'data'
    if (data/'freeze.json').exists():
        raise RuntimeError('Already frozen')
    corpus = {r['id']:r for r in lines(data/'corpus.jsonl')}
    rag = lines(data/'rag_cases.jsonl')
    mem = [] if args.rag_only else lines(data/'mem0_cases.jsonl')
    assert len(rag) == 400 and (args.rag_only or len(mem) == 120)
    assert len({r['id'] for r in rag}) == 400
    assert args.rag_only or len({r['id'] for r in mem}) == 120
    families = {}
    source_splits = {}
    for c in rag:
        families.setdefault(c['family_id'], set()).add(c['split'])
        for g, evidence in zip(c['gold_groups'], c['evidence']):
            assert evidence in corpus[g[0]]['content'], c['id']
            source_splits.setdefault(g[0], set()).add(c['split'])
    assert all(len(v)==1 for v in families.values())
    assert all(len(v)==1 for v in source_splits.values())
    api = API(args.work/'label_review', max_requests=100)

    source_items = {}
    for r in rag:
        if r['kind'] == 'answerable':
            item = source_items.setdefault(r['family_id'],{'id':r['family_id'],'evidence':r['evidence'][0]})
            item[r['language']] = r['query']
    # Single-source bilingual questions are reviewed; multi/context cases retain the same exact-quote contract.
    source_items = list(source_items.values())
    def review_rag(pair):
        i, cases = pair
        digest=sha256(json.dumps(cases,sort_keys=True).encode()).hexdigest()[:12]
        response = api.json(f'rag_{i:02}_{digest}', RAG_REVIEW, cases, model='minimax/minimax-m2.5',max_tokens=3500)
        results = review_rows(response)
        assert [r['id'] for r in results] == [c['id'] for c in cases]
        return results
    rag_reviewed = []
    if not args.memory_only:
        with ThreadPoolExecutor(max_workers=3) as pool:
            rag_reviewed = [r for group in pool.map(review_rag,[(i,source_items[n:n+10]) for i,n in enumerate(range(0,len(source_items),10))]) for r in group]
    if args.rag_only:
        write(data/'rag_label_review.json',{'review_model':'minimax/minimax-m2.5','reviews':rag_reviewed})
        print('RAG labels reviewed:',len(rag_reviewed),flush=True)
        return

    def review(pair):
        i, cases = pair
        digest=sha256(json.dumps(cases,sort_keys=True).encode()).hexdigest()[:12]
        response = api.json(f'memory_{i:02}_{digest}', REVIEW, cases, model='minimax/minimax-m2.5',max_tokens=3500)
        results = review_rows(response)
        query_ids=[c['id']+f'-Q{q+1}' for c in cases for q in range(len(c['queries']))]
        if [r['id'] for r in results] == query_ids:
            results=[{'id':c['id'],'supported':all(r['supported'] for r in results[2*n:2*n+2]),
                      'reason':' | '.join(r['reason'] for r in results[2*n:2*n+2])} for n,c in enumerate(cases)]
        assert [r['id'] for r in results] == [c['id'] for c in cases]
        return results
    with ThreadPoolExecutor(max_workers=3) as pool:
        reviewed = [r for group in pool.map(review, [(i,mem[n:n+5]) for i,n in enumerate(range(0,len(mem),5))]) for r in group]
    if args.memory_only:
        write(data/'memory_label_review.json',{'review_model':'minimax/minimax-m2.5','reviews':reviewed})
        print('Memory labels reviewed:',len(reviewed),flush=True)
        return
    write(data/'label_review.json', {'review_model':'minimax/minimax-m2.5','memory_reviews':reviewed,
                                   'rag_reviews':rag_reviewed, 'rag_exact_quotes_verified':sum(bool(r['evidence']) for r in rag),
                                   'rag_source_disjoint_dev_test':True,'clinical_review':False})
    issues = [r for r in reviewed+rag_reviewed if r['supported'] is not True]
    if issues:
        print('Review issues:', [r['id'] for r in issues], flush=True)
        raise RuntimeError('Inspect unsupported labels before freezing')
    freeze = {'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
              'sha256':{p.name:sha256(p.read_bytes()).hexdigest() for p in sorted(data.iterdir()) if p.is_file()},
              'selection_protocol':'RAG threshold selected on dev only; memory baseline K3/0.3 and K10/0.3 fixed before execution.',
              'author_model':'qwen/qwen3.5-27b','review_model':'minimax/minimax-m2.5'}
    write(data/'freeze.json', freeze)
    print('Frozen RAG 400 queries / 220 families and Mem0 120 scenarios.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--rag-only', action='store_true')
    parser.add_argument('--memory-only', action='store_true')
    main(parser.parse_args())
