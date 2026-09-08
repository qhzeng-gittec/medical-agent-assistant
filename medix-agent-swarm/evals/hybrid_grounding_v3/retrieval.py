"""Unchanged v2 holdout + new source families; select RRF on dev only."""
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import itertools
import json
import sys
import time

import numpy as np

from common import API, BASE, HERE, RESULTS, ROOT, V2, digest, frozen, lines, read, write
sys.path.insert(0,str(ROOT/'medix-agent-swarm/knowledge'))
from hybrid_retrieval import HybridIndex
from evaluate_rag import INSTRUCTION


def cases():
    old = [dict(c,suite='original',answerability=('needs_history' if c['kind']=='needs_history' else
                'unanswerable' if c['kind']=='unanswerable' else 'answerable')) for c in lines(V2/'data/rag_cases.jsonl')]
    return old + [dict(c,suite='extension') for c in lines(HERE/'data/extension.jsonl')]


def complete(case, ids, k):
    return bool(case['gold_groups']) and all(set(g).intersection(ids[:k]) for g in case['gold_groups'])


def translate_queries(cs,api):
    chinese = [c for c in cs if c.get('language')=='zh' or any('\u3400'<=x<='\u9fff' for x in c['query'])]
    instruction = 'Translate medical search questions into concise English without answering, expanding facts or adding synonyms not present. Preserve negation, numbers and identifiers. Return JSON {"items":[{"id":same id,"query":English translation}]} in input order.'
    def translate(pair):
        i, group = pair
        payload = [{'id':c['id'],'query':c['query']} for c in group]
        result = api.json(f'translate_{i:03}',instruction,payload,max_tokens=4500)['items']
        if [r['id'] for r in result] != [c['id'] for c in group]:
            raise ValueError('Translation IDs differ')
        print(f'Translation batch {i+1} completed',flush=True)
        return result
    translations = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(translate,[(i,chinese[n:n+8]) for i,n in enumerate(range(0,len(chinese),8))]):
            translations.update({r['id']:r['query'] for r in result})
    return translations


def metrics(rows, method):
    positives = [r for r in rows if r['case']['answerability']=='answerable']
    return {'queries':len(rows),'answerable_queries':len(positives),'families':len({r['case']['family_id'] for r in rows}),
            'complete':{str(k):sum(complete(r['case'],r['rankings'][method],k) for r in positives) for k in [1,3,5,10]},
            'empty_top3':sum(not r['rankings'][method][:3] for r in rows),
            'mrr10':float(np.mean([next((1/(i+1) for i,d in enumerate(r['rankings'][method][:10]) if any(d in g for g in r['case']['gold_groups'])),0) for r in positives])) if positives else None}


def paired_interval(rows, baseline, candidate):
    groups = {}
    for r in rows:
        c = r['case']
        if c['answerability']!='answerable':
            continue
        groups.setdefault(c['family_id'],[]).append([int(complete(c,r['rankings'][m],3)) for m in [baseline,candidate]])
    totals = np.asarray([[sum(v[0] for v in values),sum(v[1] for v in values),len(values)] for values in groups.values()])
    rng = np.random.default_rng(20260908)
    samples = totals[rng.integers(0,len(totals),size=(4000,len(totals)))].sum(axis=1)
    delta = (samples[:,1]-samples[:,0])/samples[:,2]
    return {'baseline':baseline,'candidate':candidate,'families':len(totals),'queries':int(totals[:,2].sum()),
            'difference':float((totals[:,1].sum()-totals[:,0].sum())/totals[:,2].sum()),
            'family_bootstrap_95':np.quantile(delta,[.025,.975]).tolist(),'replicates':4000}


def main():
    frozen_data = read(HERE/'data/freeze.json')
    assert frozen_data['sha256']['extension.jsonl']==digest(HERE/'data/extension.jsonl')
    assert frozen_data['sha256']['base_queries']==digest(V2/'data/rag_cases.jsonl')
    assert frozen_data['sha256']['corpus']==digest(V2/'data/corpus.jsonl')
    cs = cases()
    corpus = lines(V2/'data/corpus.jsonl')
    ids = [d['id'] for d in corpus]
    grid = [dict(dense_weight=w,rank_constant=c,window=n) for w,c,n in itertools.product([.25,.5,.75,.9],[10,60],[20,50])]
    protocol = {'dataset':frozen_data,'corpus_documents':1016,'queries':800,'bm25':{'k1':1.2,'b':.75,'text':'same content as dense; no title/label enrichment'},
                'rrf_grid':grid,'selection':'Dev answerable complete@3; tie: larger dense weight, rank constant, smaller window. Select native and translated separately.',
                'query_translation':'Actual Qwen3.5-27B API; only query text and id, no corpus, evidence or paired English questions.',
                'primary':'Top3; same context document budget, no score threshold. RRF scores are not cosine similarities.',
                'embedding_model':'qwen/qwen3-embedding-8b','base_vectors_sha256':digest(BASE/'rag/vectors.npz'),
                'bootstrap':'paired source-family clusters; 4000 resamples; 95% percentile; descriptive secondary strata'}
    frozen(RESULTS/'retrieval/protocol.json',protocol)
    api = API(RESULTS/'retrieval/api_records',max_requests=160)
    translations_path = RESULTS/'retrieval/translations.json'
    translations = read(translations_path) if translations_path.exists() else translate_queries(cs,api)
    frozen(RESULTS/'retrieval/translations.json',translations)
    prior = np.load(BASE/'rag/vectors.npz')
    order = read(BASE/'rag/vector_order.json')
    assert order['corpus']==ids
    vectors = {id_:v for split in ['dev','test'] for id_,v in zip(order[split],prior[split])}
    extension = [c for c in cs if c['suite']=='extension']
    def embed(pair):
        i, group = pair
        return api.embed(f'extension_{i:03}',[f'Instruct: {INSTRUCTION}\nQuery: {c["query"]}' for c in group])
    vector_path = RESULTS/'retrieval/extension_vectors.npz'
    if vector_path.exists():
        assert read(RESULTS/'retrieval/extension_vector_order.json')==[c['id'] for c in extension]
        new_vectors = np.load(vector_path)['queries']
    else:
        with ThreadPoolExecutor(max_workers=3) as pool:
            new_vectors = np.asarray([v for batch in pool.map(embed,[(i,extension[n:n+8]) for i,n in enumerate(range(0,len(extension),8))]) for v in batch])
    norms = np.linalg.norm(new_vectors,axis=1,keepdims=True)
    if not np.isfinite(new_vectors).all() or (norms==0).any():
        raise ValueError('Invalid query embeddings')
    if not vector_path.exists():
        new_vectors /= norms
        np.savez_compressed(vector_path,queries=new_vectors)
    elif not np.allclose(norms,1,atol=1e-6):
        raise ValueError('Stored query vectors must be normalized')
    frozen(RESULTS/'retrieval/extension_vector_order.json',[c['id'] for c in extension])
    vectors.update({c['id']:v for c,v in zip(extension,new_vectors)})
    index = HybridIndex(corpus,prior['corpus'])
    timings = []
    def ranks(c):
        start = time.perf_counter()
        dense = index.rank(index.vectors @ vectors[c['id']])
        native = index.rank(index.lexical_scores(c['query']),True)
        translated = index.rank(index.lexical_scores(translations.get(c['id'],c['query'])),True)
        timings.append((time.perf_counter()-start)*1000)
        return dense,native,translated
    # Do not compute test rankings until both dev selections are saved.
    dev = [c for c in cs if c['split']=='dev']
    dev_ranks = {c['id']:ranks(c) for c in dev}
    selection = {}
    for mode,position in [('native',1),('translated',2)]:
        sweep = []
        for config in grid:
            hit = 0
            for c in dev:
                if c['answerability']!='answerable':
                    continue
                parts = dev_ranks[c['id']]
                fused,_ = index.fuse(parts[0],parts[position],len(ids),**config)
                hit += complete(c,[ids[i] for i in fused[:3]],3)
            sweep.append(dict(config,hits=hit))
        chosen = max(sweep,key=lambda s:(s['hits'],s['dense_weight'],s['rank_constant'],-s['window']))
        selection[mode] = {'selected':{k:chosen[k] for k in grid[0]},'hits':chosen['hits'],'dev_sweep':sweep}
    frozen(RESULTS/'retrieval/selection.json',selection)
    rows = []
    for c in cs:
        parts = dev_ranks[c['id']] if c['split']=='dev' else ranks(c)
        rankings = {'dense':parts[0],'bm25_native':parts[1],'bm25_translated':parts[2]}
        for mode,position in [('native',1),('translated',2)]:
            rankings['hybrid_'+mode],_ = index.fuse(parts[0],parts[position],len(ids),**selection[mode]['selected'])
        rows.append({'case':c,'rankings':{m:[ids[i] for i in r[:10]] for m,r in rankings.items()}})
    write(RESULTS/'retrieval/rankings.json',rows)
    methods = list(rows[0]['rankings'])
    summary = {'queries':len(rows),'methods':methods,'selected':selection,
               'local_three_rankings_ms':{'median':float(np.median(timings)),'p95':float(np.quantile(timings,.95))},
               'splits':{},'strata':{}}
    for split in ['dev','test']:
        rs = [r for r in rows if r['case']['split']==split]
        summary['splits'][split] = {m:metrics(rs,m) for m in methods}
        for key in ['suite','language','kind']:
            for value in sorted({r['case'].get(key,'unspecified') for r in rs}):
                subset = [r for r in rs if r['case'].get(key,'unspecified')==value]
                summary['strata'][f'{split}/{key}/{value}'] = {m:metrics(subset,m) for m in methods}
    test = [r for r in rows if r['case']['split']=='test']
    summary['paired_top3'] = [paired_interval(test,'dense',m) for m in ['hybrid_native','hybrid_translated']]
    write(RESULTS/'retrieval/summary.json',summary)
    print(json.dumps({'test':summary['splits']['test'],'paired':summary['paired_top3']},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
