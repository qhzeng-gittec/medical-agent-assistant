"""Resume the same preselected sources with user-requested Codex authorship."""
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import argparse
import json

from common import API, HERE, RESULTS, V2, digest, frozen, lines, read, write
from codex_author import generate, object_schema
from prepare import AUTHOR, REVIEW, source_quote


def main(model):
    plan = read(RESULTS/'authoring/plan.json')
    corpus = {d['id']:d for d in lines(V2/'data/corpus.jsonl')}
    fields = {k:{'type':'string'} for k in ['source_id','positive_en','positive_zh','evidence','expected_answer','missing_en','missing_reason']}
    schema = object_schema({'items':{'type':'array','items':object_schema(fields)}})
    frozen(RESULTS/'authoring/codex_plan.json',{'model':model,'sources':plan['sources'],'user_requested_switch':True,
           'prior_completed_items':'Retained Qwen authored / MiniMax reviewed items; provider provenance is per source.',
           'format':'10 sources per CLI request; input includes only sources, no retrieval results.',
           'evidence_normalization':'Restore source whitespace only when all non-whitespace characters match.'})
    batch_plan = RESULTS/'authoring/codex_batches.json'
    pending = read(batch_plan) if batch_plan.exists() else [s for s in plan['sources'] if not (RESULTS/'authoring/items'/(s['id']+'.json')).exists()]
    frozen(batch_plan,pending)
    def author(pair):
        i, group = pair
        payload = [corpus[s['id']] for s in group]
        value = generate(RESULTS/'authoring/codex',f'group_{i:02}',
                         AUTHOR+'\nReturn one item per input source, in order, with its source_id; wrap items in an items array.',payload,schema,model=model)
        if [v['source_id'] for v in value['items']] != [s['id'] for s in group]:
            raise ValueError('Codex source IDs differ')
        return value['items']
    with ThreadPoolExecutor(max_workers=2) as pool:
        authored = [v for group in pool.map(author,[(i,pending[n:n+10]) for i,n in enumerate(range(0,len(pending),10))]) for v in group]
    api = API(RESULTS/'authoring/codex_reviews',max_requests=200)
    source_splits = {s['id']:s['split'] for s in plan['sources']}
    def review(value):
        value = dict(value)
        id_ = value.pop('source_id')
        if (RESULTS/'authoring/items'/(id_+'.json')).exists():
            return
        doc = corpus[id_]
        value['evidence'] = source_quote(value['evidence'],doc['content'])
        if value['evidence'] not in doc['content']:
            write(RESULTS/'authoring/codex_issues'/(id_+'.json'),{'candidate':value,'validation_error':'Non-verbatim evidence'})
            return False
        result = api.json('review_'+id_,REVIEW,{'source':doc,'candidate':value},model='minimax/minimax-m2.5',max_tokens=2500)
        if not all(result[k] is True for k in ['positive_supported','translation_equivalent','missing_absent']):
            write(RESULTS/'authoring/codex_issues'/(id_+'.json'),{'candidate':value,'review':result})
            return False
        item = {'source_id':id_,'split':source_splits[id_],'value':value,'author_model':model,'review_model':'minimax/minimax-m2.5','review':result}
        write(RESULTS/'authoring/items'/(id_+'.json'),item)
        print(f'Codex source reviewed: {id_}',flush=True)
        return True
    with ThreadPoolExecutor(max_workers=4) as pool:
        reviewed = list(pool.map(review,authored))
    if False in reviewed:
        raise ValueError('Source review found issues; inspect authoring/codex_issues before freezing')
    rows = []
    for i,s in enumerate(plan['sources']):
        item = read(RESULTS/'authoring/items'/(s['id']+'.json'))
        v,doc = item['value'],corpus[s['id']]
        common = {'family_id':f'extension-{i:03}','split':s['split'],'source_id':doc['id'],
                  'expected_answer':v['expected_answer'],'evidence':[v['evidence']],
                  'author_model':item.get('author_model','qwen/qwen3.5-27b'),'review_model':'minimax/minimax-m2.5'}
        scope = f'Use only the indexed MedlinePlus summary titled "{doc["title"]}". '
        for variant,query,language,answerability in [('lexical',v['positive_en'],'en','answerable'),
                ('paraphrase',v['positive_zh'],'zh','answerable'),('near_miss',scope+v['missing_en'],'en','unanswerable'),
                ('partial',scope+v['positive_en']+' Also, '+v['missing_en'],'en','partial')]:
            row = dict(common,id=f'X{i+1:03}-{variant}',query=query,language=language,kind=variant,answerability=answerability,
                       gold_groups=[] if answerability=='unanswerable' else [[doc['id']]])
            if variant in {'near_miss','partial'}:
                row.update(allowed_source_ids=[doc['id']],missing_question=v['missing_en'],missing_reason=v['missing_reason'])
            rows.append(row)
    target = HERE/'data/extension.jsonl'
    target.parent.mkdir(parents=True,exist_ok=True)
    text = ''.join(json.dumps(c,ensure_ascii=False)+'\n' for c in rows)
    if target.exists() and target.read_text(encoding='utf-8')!=text:
        raise ValueError('Frozen extension changed')
    target.write_text(text,encoding='utf-8')
    freeze = {'sha256':{'extension.jsonl':digest(target),'base_queries':digest(V2/'data/rag_cases.jsonl'),'corpus':digest(V2/'data/corpus.jsonl')},
              'counts':dict(Counter(c['split'] for c in rows)),'families':100,'source_overlap_with_old_gold':0,
              'authorship_by_query':dict(Counter(c['author_model'] for c in rows))}
    frozen(HERE/'data/freeze.json',freeze)
    print(json.dumps(freeze,ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',default='gpt-5')
    main(parser.parse_args().model)
