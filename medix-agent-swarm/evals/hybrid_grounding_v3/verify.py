"""Recompute rankings, dev selection and published answer counts without model calls."""
import argparse
import json
import sys

import numpy as np

from common import BASE, HERE, RESULTS, ROOT, V2, digest, lines, read
from retrieval import cases, complete, metrics
from answers import answer_items, summarize, validate_grades, same_generation, judge_input, has_invalid_citation
sys.path.insert(0,str(ROOT/'medix-agent-swarm/knowledge'))
from hybrid_retrieval import HybridIndex


def verify_retrieval():
    freeze = read(HERE/'data/freeze.json')['sha256']
    for key,path in [('extension.jsonl',HERE/'data/extension.jsonl'),('base_queries',V2/'data/rag_cases.jsonl'),('corpus',V2/'data/corpus.jsonl')]:
        assert freeze[key]==digest(path),key
    cs = cases()
    assert len(cs)==800 and len({c['id'] for c in cs})==800
    for key in ['family_id','source_id']:
        split_sets = [{c[key] for c in cs if c['split']==split and key in c} for split in ['dev','test']]
        assert not split_sets[0].intersection(split_sets[1]),key
    old_gold = {d for c in cs if c['suite']=='original' for g in c['gold_groups'] for d in g}
    assert not old_gold.intersection(c['source_id'] for c in cs if c['suite']=='extension')
    corpus = lines(V2/'data/corpus.jsonl')
    by_id = {d['id']:d for d in corpus}
    ids = list(by_id)
    for c in cs:
        if c['suite']=='extension':
            assert all(q in by_id[c['source_id']]['content'] for q in c['evidence'])
    old_vectors = np.load(BASE/'rag/vectors.npz')
    old_order = read(BASE/'rag/vector_order.json')
    assert ids==old_order['corpus']
    qs = {id_:v for split in ['dev','test'] for id_,v in zip(old_order[split],old_vectors[split])}
    extension_order = read(RESULTS/'retrieval/extension_vector_order.json')
    new_vectors = np.load(RESULTS/'retrieval/extension_vectors.npz')['queries']
    assert len(extension_order)==len(new_vectors)==400
    qs.update(dict(zip(extension_order,new_vectors)))
    index = HybridIndex(corpus,old_vectors['corpus'])
    selection = read(RESULTS/'retrieval/selection.json')
    translations = read(RESULTS/'retrieval/translations.json')
    saved = read(RESULTS/'retrieval/rankings.json')
    assert [r['case'] for r in saved]==cs
    dev_rankings = {}
    for c,row in zip(cs,saved):
        dense = index.rank(index.vectors @ qs[c['id']])
        native = index.rank(index.lexical_scores(c['query']),True)
        translated = index.rank(index.lexical_scores(translations.get(c['id'],c['query'])),True)
        rankings = {'dense':dense,'bm25_native':native,'bm25_translated':translated}
        for mode,sparse in [('native',native),('translated',translated)]:
            rankings['hybrid_'+mode],_ = index.fuse(dense,sparse,len(ids),**selection[mode]['selected'])
        assert row['rankings']=={m:[ids[i] for i in values[:10]] for m,values in rankings.items()},c['id']
        if c['split']=='dev':
            dev_rankings[c['id']] = dense,native,translated
    for mode,pos in [('native',1),('translated',2)]:
        for point in selection[mode]['dev_sweep']:
            hits = 0
            config = {k:point[k] for k in ['dense_weight','rank_constant','window']}
            for c in cs:
                if c['split']!='dev' or c['answerability']!='answerable':
                    continue
                values = dev_rankings[c['id']]
                ranking,_ = index.fuse(values[0],values[pos],len(ids),**config)
                hits += complete(c,[ids[i] for i in ranking[:3]],3)
            assert hits==point['hits']
        best = max(selection[mode]['dev_sweep'],key=lambda s:(s['hits'],s['dense_weight'],s['rank_constant'],-s['window']))
        assert selection[mode]['selected']=={k:best[k] for k in selection[mode]['selected']}
    summary = read(RESULTS/'retrieval/summary.json')
    for split in ['dev','test']:
        subset = [r for r in saved if r['case']['split']==split]
        for method in summary['methods']:
            assert metrics(subset,method)==summary['splits'][split][method]
    return len(saved)


def verify_answers(domain):
    _,items = answer_items(domain)
    output = RESULTS/('memory_answers' if domain=='memory' else 'answers')
    answers = {}
    grades = read(output/'grades.json')
    grading_inputs = read(output/'grading_v2/inputs.json')
    assert len(items)==len(grades)
    assert set(grades)=={i['id'] for i in items}
    for item in items:
        row = read(output/'cases'/(item['id']+'.json'))
        assert same_generation(row['input'],item)
        assert grading_inputs[item['id']]==judge_input(item,row['answer'])
        answers[item['id']] = row['answer']
        verdict = grades[item['id']]
        validate_grades({'items':[dict(verdict)]},[{'id':verdict['id'],'answer':row['answer']}])
    assert summarize(items,answers,grades)==read(output/'summary.json')
    return len(items)


def verify_audit():
    folder = RESULTS/'judge_audit'
    selected = read(folder/'selection.json')['items']
    reviews = {(r['domain'],r['id']):r for r in read(folder/'reviews.json')}
    assert len(reviews)==len(selected)
    for directory,domain in [('answers','rag'),('memory_answers','memory')]:
        _,items = answer_items(domain)
        grades = read(RESULTS/directory/'grades.json')
        inputs = read(RESULTS/directory/'grading_v2/inputs.json')
        failures = set()
        for item in items:
            row = read(RESULTS/directory/'cases'/(item['id']+'.json'))
            if not grades[item['id']]['complete_correct'] or has_invalid_citation(row['answer'],item['context']):
                failures.add(item['id'])
        subset = [r for r in selected if r['domain']==directory]
        assert failures=={r['id'] for r in subset if r['selection']=='all_initial_failures'}
        for row in subset:
            assert row['initial_grade']==grades[row['id']]
            assert row['input']==inputs[row['id']]
            review = reviews[(directory,row['id'])]
            validate_grades({'items':[dict(review['review'])]},[row['input']])
    return len(reviews)


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--part',choices=['all','retrieval','memory','rag'],default='all')
    args = p.parse_args()
    results = {}
    if args.part in {'all','retrieval'}:
        results['retrieval_queries'] = verify_retrieval()
    for domain in ['rag','memory']:
        if args.part in {'all',domain}:
            results[domain+'_answers'] = verify_answers(domain)
    if args.part=='all' and (RESULTS/'judge_audit/reviews.json').exists():
        results['independent_audits'] = verify_audit()
    manifest = RESULTS/'manifest.json'
    if args.part=='all' and manifest.exists():
        for record in read(manifest)['files']:
            path = ROOT/record['path']
            assert path.stat().st_size==record['bytes'],record['path']
            assert digest(path)==record['sha256'],record['path']
        results['manifest_files'] = len(read(manifest)['files'])
    print(json.dumps({'status':'PASS',**results}))
