"""No API required: dense/BM25/RRF on all 400 unchanged original queries."""
import itertools
import argparse
import json
import sys

import numpy as np

from common import BASE, RESULTS, ROOT, V2, digest, frozen, lines, read, write
from retrieval import complete, metrics, paired_interval
sys.path.insert(0,str(ROOT/'medix-agent-swarm/knowledge'))
from hybrid_retrieval import HybridIndex


def main(translated=False):
    output = RESULTS/('baseline_translated' if translated else 'baseline_retrieval')
    corpus = lines(V2/'data/corpus.jsonl')
    cases = [dict(c,suite='original',answerability=('needs_history' if c['kind']=='needs_history' else
                 'unanswerable' if c['kind']=='unanswerable' else 'answerable')) for c in lines(V2/'data/rag_cases.jsonl')]
    vectors = np.load(BASE/'rag/vectors.npz')
    order = read(BASE/'rag/vector_order.json')
    ids = [d['id'] for d in corpus]
    assert ids==order['corpus']
    qs = {id_:v for split in ['dev','test'] for id_,v in zip(order[split],vectors[split])}
    index = HybridIndex(corpus,vectors['corpus'])
    translations = {}
    if translated:
        for path in sorted((RESULTS/'retrieval/api_records').glob('translate_*.json')):
            record = read(path)
            if 'response' in record:
                result = json.loads(record['response']['choices'][0]['message']['content'])
                if isinstance(result,dict) and 'items' in result:
                    translations.update({r['id']:r['query'] for r in result['items']})
        assert len(translations)==240
        frozen(output/'translations.json',translations)
    grid = [dict(dense_weight=w,rank_constant=c,window=n) for w,c,n in itertools.product([.25,.5,.75,.9],[10,60],[20,50])]
    protocol = {'queries':400,'corpus':1016,'base_cases_sha256':digest(V2/'data/rag_cases.jsonl'),
                'base_vectors_sha256':digest(BASE/'rag/vectors.npz'),'grid':grid,'bm25':{'k1':1.2,'b':.75},
                'lexical_text':('Actual Qwen3.5 English query translations, no gold inputs. Same document body as dense.' if translated else
                                'Same content as dense embeddings; no translation, title enrichment or gold labels.'),
                'selection':'Dev answerable complete@3, ties larger dense_weight/rank_constant then smaller window.',
                'scope':'No API calls. Reuse actual published normalized API vectors, unchanged original test set.'}
    frozen(output/'protocol.json',protocol)
    dev = [c for c in cases if c['split']=='dev']
    def ranks(c):
        return index.rank(index.vectors @ qs[c['id']]),index.rank(index.lexical_scores(translations.get(c['id'],c['query'])),True)
    dev_ranks = {c['id']:ranks(c) for c in dev}
    sweep = []
    for config in grid:
        hits = 0
        for c in dev:
            if c['answerability']!='answerable':
                continue
            dense,sparse = dev_ranks[c['id']]
            fused,_ = index.fuse(dense,sparse,len(ids),**config)
            hits += complete(c,[ids[i] for i in fused[:3]],3)
        sweep.append(dict(config,hits=hits))
    chosen = max(sweep,key=lambda s:(s['hits'],s['dense_weight'],s['rank_constant'],-s['window']))
    config = {k:chosen[k] for k in grid[0]}
    frozen(output/'selection.json',{'selected':config,'dev_sweep':sweep})
    rows = []
    for c in cases:
        dense,sparse = dev_ranks[c['id']] if c['split']=='dev' else ranks(c)
        fused,_ = index.fuse(dense,sparse,len(ids),**config)
        equal,_ = index.fuse(dense,sparse,len(ids),.5,60,50)
        rows.append({'case':c,'rankings':{m:[ids[i] for i in values[:10]] for m,values in
                    [('dense',dense),('bm25',sparse),('rrf_equal',equal),('rrf_dev_selected',fused)]}})
    write(output/'rankings.json',rows)
    summary = {'selected':config,'splits':{},'strata':{}}
    methods = list(rows[0]['rankings'])
    for split in ['dev','test']:
        subset = [r for r in rows if r['case']['split']==split]
        summary['splits'][split] = {m:metrics(subset,m) for m in methods}
        for key in ['language','kind']:
            for value in sorted({r['case'].get(key,'unspecified') for r in subset}):
                rs = [r for r in subset if r['case'].get(key,'unspecified')==value]
                summary['strata'][f'{split}/{key}/{value}'] = {m:metrics(rs,m) for m in methods}
    test = [r for r in rows if r['case']['split']=='test']
    summary['paired_top3'] = paired_interval(test,'dense','rrf_dev_selected')
    summary['changed_cases'] = [{'id':r['case']['id'],'query':r['case']['query'],
                               'dense':complete(r['case'],r['rankings']['dense'],3),
                               'hybrid':complete(r['case'],r['rankings']['rrf_dev_selected'],3)} for r in test
                              if r['case']['answerability']=='answerable' and
                              complete(r['case'],r['rankings']['dense'],3)!=complete(r['case'],r['rankings']['rrf_dev_selected'],3)]
    write(output/'summary.json',summary)
    print(json.dumps({'selected':config,'test':summary['splits']['test'],'paired':summary['paired_top3']},ensure_ascii=False))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--translated',action='store_true')
    main(parser.parse_args().translated)
