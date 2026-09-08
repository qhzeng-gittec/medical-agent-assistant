"""Evidence-cited semantic coverage, separately from literal anchors and scope isolation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path

from api import API, read, write
from evaluate_memory import anchor_coverage

PROMPT = '''Judge retrieval support, not clinical correctness. Return JSON {"items":[...]} in input order.
For each input item return {"id":same id,"facts":[{"supported":true/false,"evidence":[{"memory_id":id}],"reason":"short"}]}.
One facts entry per expected_fact, in the original order. Judge only whether retrieved memories explicitly
support the requested personal fact, with correct person, event time, current/historical state and uncertainty.
Historical records may coexist with newer ones if chronology resolves the requested state. Mere keyword overlap,
an assistant suggestion or a contradicted current-state assertion does not count. A supported fact requires
at least one supplied memory id; the evaluation program will attach the full original memory text as its quote.
source_session_id and recorded_at preserve the original record order: session-2 follows session-1.
Use an explicit later correction to resolve an earlier assertion. Recording time provides the reference date
for relative dates, not a new claimed date of a past event. Unsupported facts may have empty evidence.
Do not use outside knowledge. Keep each reason to one short sentence.
Each item is independent. A memory id is valid ONLY inside that item's memories, never another item.'''
SINGLE_PROMPT='''Return one JSON object {"supported":true/false,"evidence":[{"memory_id":id}],"reason":"short"}.
Evaluate the entire expected_fact as one claim. All its clauses must be supported to set supported=true.
Return one verdict, not a list of facts. Judge only whether'''+PROMPT.split('Judge only whether',1)[1]


def validate_grades(response, group):
    results=response if isinstance(response,list) else response['items']
    if [r['id'] for r in results]!=[r['id'] for r in group]:
        raise ValueError('Semantic judge ids differ from inputs')
    for result,source in zip(results,group):
        if len(result['facts'])!=len(source['expected_facts']):
            raise ValueError('Semantic judge fact count mismatch')
        memories={m['memory_id']:m['content'] for m in source['memories']}
        for fact in result['facts']:
            if fact['supported'] in ['true','false']:
                fact['supported']=fact['supported']=='true'
            if type(fact['supported']) is not bool:
                raise ValueError('Support verdict must be boolean')
            if fact['supported'] and not fact['evidence']:
                raise ValueError('Supported verdict lacks evidence')
            fact['evidence']=[{'memory_id':e} if isinstance(e,str) else e for e in fact['evidence']]
            for evidence in fact['evidence']:
                if evidence['memory_id'] not in memories:
                    raise ValueError('Judge cites an unknown memory')
                evidence['quote']=memories[evidence['memory_id']]
    return results


def main(args):
    rows = [read(p) for p in sorted((args.results/'cases').glob('*.json'))]
    completed = [r for r in rows if r['status'] == 'completed']
    if len(rows)!=120 or len(completed)!=120:
        raise ValueError('Complete all 120 frozen scenarios before final semantic grading')
    items, mapping = {}, []
    for r in completed:
        for index, q in enumerate(r['queries']):
            for k in [3,10]:
                hits = [h for h in q['hits'][:k] if h['score'] >= .3]
                payload = {'query':q['query']['query'], 'expected_facts':q['query']['expected_facts'],
                           'memories':[{'memory_id':f'm{n}','content':h['content'],
                                        'source_session_id':h['metadata']['source_session_id'],
                                        'recorded_at':h['metadata']['timestamp']} for n,h in enumerate(hits)]}
                key = sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:20]
                items[key] = dict(id=key, **payload)
                mapping.append({'case_id':r['case']['id'], 'split':r['case']['split'], 'category':r['case']['category'],
                                'query_index':index,'k':k,'item_id':key,
                                'anchor_all':all(anchor_coverage(q['query']['required_groups'],hits))})
    api = API(args.cache/'semantic_review_final', max_requests=150)
    # Mix contexts by content hash instead of presenting adjacent K3/K10 variants.
    ordered = sorted(items.values(),key=lambda item:item['id'])

    def grade(pair):
        i, group = pair
        # Content-derived batch key keeps resumed requests immutable as more cases complete.
        batch_id = sha256(json.dumps(group,sort_keys=True).encode()).hexdigest()[:16]
        attempts=[]
        instruction=PROMPT
        results=None
        for attempt in range(3):
            name='batch_'+batch_id+['','_schema_retry','_fact_schema_retry'][attempt]
            path=api.directory/(name+'.json')
            try:
                response=api.json(name,instruction,group,model='minimax/minimax-m2.5',max_tokens=9000)
                results=validate_grades(response,group)
                attempts.append({'record':path.name,'valid':True})
                break
            except (ValueError,KeyError,TypeError) as error:
                attempts.append({'record':path.name,'valid':False,'validation_error':type(error).__name__+': '+str(error)})
                if attempt==2:
                    break
                instruction=PROMPT+'\nThe previous response did not pass output validation. Return all items with valid JSON, exact item IDs and only memory IDs present in the same item.'
                if attempt==1:
                    counts={item['id']:len(item['expected_facts']) for item in group}
                    instruction+='\nRequired facts-array length for each item: '+json.dumps(counts)+'. Never combine expected facts into one verdict. Return one separate verdict per expected fact even when their evidence overlaps.'
        if results is None:
            results=[]
            for source in group:
                facts=[]
                for index,fact in enumerate(source['expected_facts']):
                    name='batch_'+batch_id+'_'+source['id']+f'_fact_{index}'
                    payload={'query':source['query'],'expected_fact':fact,'memories':source['memories']}
                    verdict=api.json(name,SINGLE_PROMPT,payload,model='minimax/minimax-m2.5',max_tokens=9000)
                    single=dict(source,expected_facts=[fact])
                    validated=validate_grades([{'id':source['id'],'facts':[verdict]}],[single])
                    facts.append(validated[0]['facts'][0])
                    path=api.directory/(name+'.json')
                    attempts.append({'record':path.name,'valid':True,'mode':'single_fact',
                                     'item_id':source['id'],'fact_index':index})
                results.append({'id':source['id'],'facts':facts})
        for attempt in attempts:
            attempt_path=api.directory/attempt['record']
            attempt['sha256']=sha256(attempt_path.read_bytes()).hexdigest()
            attempt_response=read(attempt_path)['response']
            attempt.update(response_id=attempt_response['id'],usage=attempt_response.get('usage'))
        record=read(path)
        response=record['response']
        evidence={'batch_id':batch_id,'response_id':response['id'],'model':response['model'],
                  'provider':response.get('provider'),'usage':response.get('usage'),
                  'record_sha256':sha256(path.read_bytes()).hexdigest(),'input_ids':[item['id'] for item in group],
                  'attempts':attempts}
        print(f'Semantic batch {i+1}: {len(group)} inputs validated',flush=True)
        return results,evidence

    with ThreadPoolExecutor(max_workers=4) as pool:
        batches=list(pool.map(grade,[(i,ordered[n:n+4]) for i,n in enumerate(range(0,len(ordered),4))]))
    judged=[r for group,evidence in batches for r in group]
    verdicts = {r['id']:r for r in judged}
    write(args.results/'semantic_grades.json', {'model':'minimax/minimax-m2.5','instruction':PROMPT,'single_fact_instruction':SINGLE_PROMPT,
          'temperature':.2,'max_tokens':9000,'api_calls':[evidence for group,evidence in batches],
          'validation_policy':'First schema-valid batch response; at most two schema retries, then one separate request per expected fact if the batch remains malformed. No retries based on support verdicts.',
          'inputs':ordered,'verdicts':judged,'mapping':mapping})
    summary = {'planned_scenarios':120, 'completed_scenarios':len(completed), 'observed_records':len(rows),
               'judge_model':'minimax/minimax-m2.5','threshold':.3, 'new_memory_backend':'Mem0 OSS / local Qdrant',
               'unique_judge_inputs':len(items), 'query_configuration_observations':len(mapping)}
    for split in ['dev','test']:
        subset = [r for r in completed if r['case']['split']==split]
        summary[split] = {'scenarios':len(subset), 'queries':sum(len(r['queries']) for r in subset),
                          'unknown_user_empty':sum(not r['unknown_user_hits'] for r in subset),
                          'other_app_empty':sum(not r['other_app_hits'] for r in subset),
                          'k_metrics':[]}
        for k in [3,10]:
            rows_k = [r for r in mapping if r['split']==split and r['k']==k]
            summary[split]['k_metrics'].append({'k':k,'queries':len(rows_k),
                'semantic_all_facts_supported':sum(all(f['supported'] for f in verdicts[r['item_id']]['facts']) for r in rows_k),
                'literal_anchor_all':sum(r['anchor_all'] for r in rows_k),
                'hard_negative_queries':len(subset),
                'hard_negative_empty':sum(not [h for h in r['unanswerable']['hits'][:k] if h['score'] >= .3] for r in subset)})
    write(args.results/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--cache',type=Path,required=True)
    main(parser.parse_args())
