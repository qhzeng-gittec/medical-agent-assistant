"""Offline standard-library verification of frozen datasets and published benchmark counts."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def lines(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]


def main(root):
    data=HERE/'data'
    for name,digest in read(data/'freeze.json')['sha256'].items():
        assert sha256((data/name).read_bytes()).hexdigest()==digest,name
    summary=read(root/'rag'/'summary.json')
    rows=[read(p) for p in (root/'rag'/'cases').glob('*.json')]
    assert len(rows)==summary['queries']==400
    frozen={r['id']:r for r in lines(data/'rag_cases.jsonl')}
    assert {r['case']['id']:r['case'] for r in rows}==frozen
    selection=read(root/'rag'/'selection.json')
    selected=max(selection['dev_sweep'],key=lambda r:(r['balanced_objective'],-r['threshold']))
    assert selection['selected_on']=='dev'
    assert summary['selected_threshold']==selection['selected_threshold']==selected['threshold']
    for split in ['dev','test']:
        subset=[r for r in rows if r['case']['split']==split]
        assert len(subset)==summary[split]['queries']
        assert len({r['case']['family_id'] for r in subset})==summary[split]['families']
        metrics=summary[split]['top_k_at_zero']+[summary[split]['dev_selected']]
        if split=='dev':
            metrics+=selection['dev_sweep']
        for metric in metrics:
            positive=[r for r in subset if r['case']['kind'] in {'answerable','multi_evidence','context_supplied'}]
            negative=[r for r in subset if r['case']['kind']=='unanswerable']
            def ids(row):
                return {d['id'] for d in row['ranking'][:metric['k']] if d['score']>=metric['threshold']}
            assert metric['positive_queries']==len(positive)
            assert metric['negative_queries']==len(negative)
            assert metric['all_source_hits']==sum(all(ids(r).intersection(g) for g in r['case']['gold_groups']) for r in positive)
            assert metric['any_source_hits']==sum(any(ids(r).intersection(g) for g in r['case']['gold_groups']) for r in positive)
            assert metric['negative_empty']==sum(not ids(r) for r in negative)
            objective=(metric['all_source_hits']/len(positive)+metric['negative_empty']/len(negative))/2
            assert abs(metric['balanced_objective']-objective)<1e-12
    memory=read(root/'memory'/'summary.json')
    protocol=read(root/'memory'/'protocol.json')
    assert sha256((HERE/'runtime/long_term.py').read_bytes()).hexdigest()==protocol['runtime_sha256']
    assert sha256((HERE/'evaluate_memory.py').read_bytes()).hexdigest()==protocol['harness_sha256']
    cases=[read(p) for p in (root/'memory'/'cases').glob('*.json')]
    frozen={r['id']:r for r in lines(data/'mem0_cases.jsonl')}
    assert {r['case']['id']:r['case'] for r in cases}==frozen
    assert len(cases)==memory['planned_scenarios']==memory['completed_scenarios']==120
    assert all(r['status']=='completed' and r['reopened_before_queries'] for r in cases)
    case_lookup={r['case']['id']:r for r in cases}
    grades=read(root/'memory'/'semantic_grades.json')
    for call in grades['api_calls']:
        for attempt in call['attempts']:
            path=root/'memory/judge_records'/attempt['record']
            assert sha256(path.read_bytes()).hexdigest()==attempt['sha256']
            assert read(path)['response']['id']==attempt['response_id']
    inputs={v['id']:v for v in grades['inputs']}
    verdicts={v['id']:v for v in grades['verdicts']}
    assert inputs.keys()==verdicts.keys()
    assert len(grades['mapping'])==480
    assert len({(r['case_id'],r['query_index'],r['k']) for r in grades['mapping']})==480
    for row in grades['mapping']:
        assert row['k'] in {3,10} and row['query_index'] in {0,1}
        case=case_lookup[row['case_id']]
        assert row['split']==case['case']['split'] and row['category']==case['case']['category']
        query=case['queries'][row['query_index']]
        assert query['query']==case['case']['queries'][row['query_index']]
        hits=[h for h in query['hits'][:row['k']] if h['score']>=.3]
        payload={'query':query['query']['query'],'expected_facts':query['query']['expected_facts'],
                 'memories':[{'memory_id':f'm{n}','content':h['content'],
                              'source_session_id':h['metadata']['source_session_id'],
                              'recorded_at':h['metadata']['timestamp']} for n,h in enumerate(hits)]}
        key=sha256(json.dumps(payload,ensure_ascii=False,sort_keys=True).encode()).hexdigest()[:20]
        assert row['item_id']==key and inputs[key]==dict(id=key,**payload)
        text='\n'.join(h['content'] for h in hits).casefold()
        assert row['anchor_all']==all(any(t.casefold() in text for t in g) for g in query['query']['required_groups'])
    for key,verdict in verdicts.items():
        source=inputs[key]
        assert len(source['expected_facts'])==len(verdict['facts'])
        texts={m['memory_id']:m['content'] for m in source['memories']}
        for fact in verdict['facts']:
            assert type(fact['supported']) is bool
            assert not fact['supported'] or fact['evidence']
            for e in fact['evidence']:
                assert e['quote'] and e['quote']==texts[e['memory_id']]
    for split in ['dev','test']:
        subset=[r for r in cases if r['case']['split']==split]
        assert memory[split]['scenarios']==len(subset)
        assert memory[split]['queries']==sum(len(r['queries']) for r in subset)
        assert memory[split]['unknown_user_empty']==sum(not r['unknown_user_hits'] for r in subset)
        assert memory[split]['other_app_empty']==sum(not r['other_app_hits'] for r in subset)
        for metric in memory[split]['k_metrics']:
            group=[r for r in grades['mapping'] if r['split']==split and r['k']==metric['k']]
            assert metric['queries']==len(group)
            assert metric['semantic_all_facts_supported']==sum(all(f['supported'] for f in verdicts[r['item_id']]['facts']) for r in group)
            assert metric['literal_anchor_all']==sum(r['anchor_all'] for r in group)
            assert metric['hard_negative_queries']==len(subset)
            assert metric['hard_negative_empty']==sum(not [h for h in r['unanswerable']['hits'][:metric['k']] if h['score']>=.3] for r in subset)
    print('Verified dataset hashes, 400 RAG rankings, memory semantic evidence quotes and reported denominators.')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--results',type=Path,required=True)
    main(parser.parse_args().results)
