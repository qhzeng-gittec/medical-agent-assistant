"""Sequential Gemini-only judge calibration; GPT labels never enter API requests."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import json
import random
import re
import subprocess

import numpy as np
import experiments.compare_gemini_judge as g
import experiments.optimize_gemini_judge as old

ROOT = g.LAB / 'outputs/gemini_iterations_v2'
ARMS = g.diagnostic.ARMS


def prepare():
    if (ROOT/'plan.json').exists():
        return
    previous = g.read(old.ROOT/'plan.json')['partitions']
    development = previous['development'] + previous['holdout']
    seen = set(g.read(g.ROOT/'subset_50.json')['source_ids']) | set(development)
    snapshots = g.read(g.ROOT/'comparison_plan.json')['gpt_snapshots']
    rows = g.diagnostic.load_jsonl(g.SOURCE/'evaluation.jsonl')
    rng = random.Random(202609072)
    holdout = []
    for task in ['knowledge', 'context']:
        eligible = sorted(r['source_id'] for r in rows if r['task']==task and r['source_id'] in snapshots and r['source_id'] not in seen)
        holdout += rng.sample(eligible, 13)
    assert not set(development) & set(holdout)
    g.save(ROOT/'plan.json', dict(development=development, holdout=holdout,
        minimum_sequential_rounds=3, maximum_sequential_rounds=4,
        model=g.MODEL, temperature=.2, thinking_level='medium',
        selection='Development mean of answer and reasoning ordinal exact agreement; ties: lower MAE, then lower absolute full-credit bias. Select only after at least three sequential rounds. Final holdout remains unseen until selection is frozen.',
        acceptance='Replacement pilot target: both answer and reasoning exact agreement >=0.80, both binary full-credit agreement >=0.90, both absolute full-credit bias <=0.05, and ordinal average improvement versus baseline with positive lower bound of question-cluster bootstrap 95% CI. Stability is reported separately. Failure means replacement is not validated.',
        gpt_snapshots={sid:snapshots[sid] for sid in development+holdout},
        previous_baseline_paths={sid:str(old.ROOT/phase/'baseline') for phase,ids in previous.items() for sid in ids},
        limitations=['GPT is an imperfect reference judge, not medical expert ground truth.',
            'Development reuses previously inspected questions. Final validation excludes all previous Gemini questions.',
            'Four model answers and repeated calls on the same question are correlated; uncertainty resamples questions.',
            'Prompts can adapt to generic error categories, never include development medical examples, GPT scores or model identities.']))


def audit_schema(ids, claim_checks=False):
    obj = g.judge.obj
    field = obj({
        'assessment': {'type':'string', 'enum':['fully_correct','limited_defect','fundamental_error']},
        'summary': {'type':'string'},
        'findings': {'type':'array', 'items':obj({
            'type': {'type':'string','enum':g.judge.FLAGS},
            'quote': {'type':'string'}, 'explanation': {'type':'string'},
            'severity': {'type':'string','enum':['limited','fundamental']}})}})
    if claim_checks:
        checks={'type':'array','items':obj({'quote':{'type':'string'},
            'verdict':{'type':'string','enum':['supported','contradicted','not_established']},
            'basis':{'type':'string'}})}
        field=obj({'claim_checks':checks,'findings':field['properties']['findings'],
            'summary':field['properties']['summary'],'assessment':field['properties']['assessment']})
    return obj({'reference_valid':{'type':'boolean'}, 'reference_comment':{'type':'string'},
        'candidates':{'type':'array','items':obj({'id':{'type':'string','enum':ids},
            'final_audit':copy.deepcopy(field), 'reasoning_audit':copy.deepcopy(field)})}})


def parse_visible(raw):
    assert raw['requested_model']==g.MODEL and raw['returned_model']==g.MODEL
    assert raw['finish_reason']=='stop', raw['finish_reason']
    visible=raw['message']['content'].lstrip()
    while visible.startswith('<thought>'):
        _, separator, visible=visible.partition('</thought>')
        assert separator, 'Incomplete thought envelope'
        visible=visible.lstrip()
    fenced=re.fullmatch(r'```(?:json)?\s*([\s\S]*?)\s*```', visible.strip(), re.I)
    return json.loads(fenced[1] if fenced else visible)


def invoke(request, out, sid, tag):
    g.save(out/'requests'/(sid+'.json'),request)
    key=g.judge.digest(dict(request=request,model=g.MODEL,run_tag=tag))
    call=out/'calls'/(key+'.json')
    if call.exists():
        raw=g.read(call)
    else:
        process=subprocess.run(['node',str(g.LAB/'gemini_judge_bridge.mjs')],
            input=json.dumps(request,ensure_ascii=False),text=True,encoding='utf-8',capture_output=True,timeout=90)
        if process.returncode:
            raise RuntimeError(f'{sid}: Gemini bridge failed: {process.stderr[-1500:]}')
        raw=json.loads(process.stdout)
        g.save(call,raw)
    return parse_visible(raw),key


def score(sid, phase, variant, repeat, *, output_root=None, responses=None):
    out=ROOT/phase/variant/str(repeat) if output_root is None else output_root
    target=out/'diagnostics/questions'/(sid+'.json')
    if target.exists():
        return
    row=next(r for r in g.diagnostic.load_jsonl(g.SOURCE/'evaluation.jsonl') if r['source_id']==sid)
    if responses is None:
        responses={arm+'__native':g.read(g.SOURCE/'generation'/(arm+'__native')/(sid+'.json')) for arm in ARMS}
    unique={g.judge.response_id(r):dict(id=g.judge.response_id(r), reasoning=r['reasoning'], final_answer=r['final_answer']) for r in responses.values()}
    candidates=list(unique.values())
    random.Random(g.judge.SEED).shuffle(candidates)
    payload=dict(task=row['task'],question=row['user_text'],reference=row['reference'],
        reference_explanation=row.get('reference_explanation',''),candidates=candidates)
    rubric=g.judge.RUBRIC+('\n'+g.judge.CONTEXT_CALIBRATION if row['task']=='context' else '')
    if variant=='baseline':
        schema=copy.deepcopy(g.judge.SCORE_SCHEMA)
        schema['properties']['candidates']['items']['properties']['id']['enum']=list(unique)
    else:
        config=g.read(ROOT/'rounds'/(variant+'.json'))
        rubric=config['prompt'] if config.get('replace_rubric') else rubric+'\n\n'+config['prompt']
        schema=audit_schema(list(unique),config.get('claim_checks',False))
    tag=f'v2:{phase}:{variant}:{repeat}'
    if variant!='baseline' and config.get('split_fields'):
        parts=[]
        keys=[]
        for text_field,audit_field in [('final_answer','final_audit'),('reasoning','reasoning_audit')]:
            partial_payload=copy.deepcopy(payload)
            partial_payload['candidates']=[dict(id=c['id'],**{text_field:c[text_field]}) for c in candidates]
            partial_schema=copy.deepcopy(schema)
            candidate_schema=partial_schema['properties']['candidates']['items']
            candidate_schema['properties']={k:v for k,v in candidate_schema['properties'].items() if k in ['id',audit_field]}
            candidate_schema['required']=['id',audit_field]
            partial_rubric=rubric+f'\nThis call evaluates ONLY {text_field}; the other field is deliberately withheld. Return ONLY {audit_field} for each candidate. Do not penalize the absence of the withheld field. Establish correctness from the question and evidence independently.'
            request=dict(rubric=partial_rubric,payload=partial_payload,schema=partial_schema)
            if config.get('thinking_level'):
                request['thinking_level']=config['thinking_level']
            part,key=invoke(request,out,sid+'_'+text_field,tag+':'+text_field)
            assert len(part['candidates'])==len(unique) and {c['id'] for c in part['candidates']}==set(unique)
            parts.append(part)
            keys.append(key)
        judged=dict(reference_valid=all(p['reference_valid'] for p in parts),
            reference_comment=' | '.join(p['reference_comment'] for p in parts),candidates=[])
        for cid in unique:
            combined=dict(id=cid)
            for part in parts:
                combined.update(next(c for c in part['candidates'] if c['id']==cid))
            judged['candidates'].append(combined)
        key=keys
    else:
        judged,key=invoke(dict(rubric=rubric,payload=payload,schema=schema),out,sid,tag)
    assert type(judged['reference_valid']) is bool
    assert len(judged['candidates'])==len(unique) and {c['id'] for c in judged['candidates']}==set(unique)
    score_map={'fully_correct':2,'limited_defect':1,'fundamental_error':0}
    by_id={}
    for candidate in judged['candidates']:
        cid=candidate['id']
        if variant=='baseline':
            result=candidate
            for field in ['answer_score','reasoning_score']:
                assert type(result[field]) is int and result[field] in [0,1,2]
        else:
            findings=[]
            result=dict(id=cid,audits=candidate)
            for name,field,text_field in [('final_audit','answer_score','final_answer'),('reasoning_audit','reasoning_score','reasoning')]:
                audit=candidate[name]
                result[field]=score_map[audit['assessment']]
                for finding in audit['findings']:
                    assert finding['type'] in g.judge.FLAGS
                    assert finding['severity'] in ['limited','fundamental']
                    # Preserve unsupported quotations for audit; never silently change scores.
                    findings.append(dict(finding,field=text_field,quote_verified=bool(finding['quote']) and finding['quote'] in unique[cid][text_field]))
            result.update(findings=findings,flags=sorted({f['type'] for f in findings}),
                explanation='Final: '+candidate['final_audit']['summary']+' Reasoning: '+candidate['reasoning_audit']['summary'])
        result['joint_score']=min(result['answer_score'],result['reasoning_score'])
        by_id[cid]=result
    g.save(target,dict(source_id=sid,task=row['task'],reference_valid=judged['reference_valid'],
        reference_comment=judged['reference_comment'],judge_model=g.MODEL,call_sha256=key,
        models={arm:dict(response,diagnosis=by_id[g.judge.response_id(response)]) for arm,response in responses.items()}))


def results(phase, variant, repeats):
    plan=g.read(ROOT/'plan.json')
    items=[]
    for sid in plan[phase]:
        truth=g.read(g.SOURCE/'diagnostics/questions'/(sid+'.json'))
        runs=[]
        for repeat in repeats:
            path=(g.LAB/plan['previous_baseline_paths'][sid] if phase=='development' and variant=='baseline' else ROOT/phase/variant)
            runs.append(g.read(path/str(repeat)/'diagnostics/questions'/(sid+'.json')))
        item=dict(source_id=sid,task=truth['task'],judge_model=plan['gpt_snapshots'][sid]['judge_model'],fields={})
        for field in ['answer_score','reasoning_score','joint_score']:
            pairs=[(truth['models'][arm+'__native']['diagnosis'][field],run['models'][arm+'__native']['diagnosis'][field]) for run in runs for arm in ARMS]
            a,b=np.array(pairs).T
            item['fields'][field]=dict(exact=float(np.mean(a==b)),binary=float(np.mean((a==2)==(b==2))),
                mae=float(np.mean(abs(a-b))),bias=float(np.mean(b==2)-np.mean(a==2)),pairs=pairs)
        items.append(item)
    fields={field:{key:float(np.mean([i['fields'][field][key] for i in items])) for key in ['exact','binary','mae','bias']} for field in ['answer_score','reasoning_score','joint_score']}
    for field in fields:
        pairs=np.array([p for i in items for p in i['fields'][field]['pairs']])
        matrix=np.zeros((3,3),int)
        for a,b in pairs: matrix[a,b]+=1
        fields[field]['confusion']=matrix.tolist()
    result=dict(phase=phase,variant=variant,questions=len(items),repeats=repeats,fields=fields,
        objective=(fields['answer_score']['exact']+fields['reasoning_score']['exact'])/2,items=items)
    g.save(ROOT/(phase+'_'+variant+'_results.json'),result)
    print(json.dumps({k:v for k,v in result.items() if k!='items'},ensure_ascii=False),flush=True)
    return result


def run(phase, variant, repeats):
    plan=g.read(ROOT/'plan.json')
    status_path=ROOT/f'{phase}_{variant}_status.json'
    if phase=='holdout':
        selected=g.read(ROOT/'selection.json')['selected']
        assert variant in ['baseline',selected]
    jobs=[(sid,phase,variant,repeat) for repeat in repeats for sid in plan[phase]]
    errors=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures={pool.submit(score,*job):job for job in jobs}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as error:
                errors.append(dict(job=futures[future],error=repr(error)))
                g.save(ROOT/f'{phase}_{variant}_errors.json',errors)
                print('ERROR',errors[-1],flush=True)
            complete=len(list((ROOT/phase/variant).glob('*/diagnostics/questions/*.json')))
            g.save(status_path,dict(status='running',phase=phase,variant=variant,complete=complete,total=len(jobs),errors=len(errors)))
            print(phase,variant,complete,'/',len(jobs),flush=True)
    if errors:
        raise RuntimeError(f'{len(errors)} API/parsing failures; inspect errors.json before retrying')
    results(phase,variant,repeats)
    g.save(status_path,dict(status='round_complete',phase=phase,variant=variant,complete=len(jobs)))


def finish():
    plan=g.read(ROOT/'plan.json')
    selection=g.read(ROOT/'selection.json')
    chosen=selection['selected']
    baseline=results('holdout','baseline',[0,1])
    candidate=results('holdout',chosen,[0,1])
    differences={}
    rng=np.random.default_rng(20260907)
    indices=rng.integers(0,len(plan['holdout']),(20000,len(plan['holdout'])))
    for field in ['answer_score','reasoning_score','joint_score']:
        differences[field]={}
        for key in ['exact','binary','mae','bias']:
            delta=np.array([b['fields'][field][key]-a['fields'][field][key] for a,b in zip(baseline['items'],candidate['items'],strict=True)])
            differences[field][key]=dict(delta=float(delta.mean()),ci95=np.quantile(delta[indices].mean(axis=1),[.025,.975]).tolist())
    ordinal_delta=np.array([np.mean([b['fields'][f]['exact']-a['fields'][f]['exact'] for f in ['answer_score','reasoning_score']]) for a,b in zip(baseline['items'],candidate['items'],strict=True)])
    improvement_ci=np.quantile(ordinal_delta[indices].mean(axis=1),[.025,.975]).tolist()
    accepted=all(candidate['fields'][f]['exact']>=.8 and candidate['fields'][f]['binary']>=.9 and abs(candidate['fields'][f]['bias'])<=.05 for f in ['answer_score','reasoning_score']) and improvement_ci[0]>0
    strata={}
    for category in ['task','judge_model']:
        strata[category]={}
        for label in sorted({item[category] for item in baseline['items']}):
            strata[category][label]={}
            for result in [baseline,candidate]:
                selected=[item for item in result['items'] if item[category]==label]
                strata[category][label][result['variant']]=dict(questions=len(selected),fields={f:{k:float(np.mean([i['fields'][f][k] for i in selected])) for k in ['exact','binary','mae','bias']} for f in ['answer_score','reasoning_score','joint_score']})
    stability={}
    by_arm={}
    audit_quality={}
    for variant in ['baseline',chosen]:
        same={f:[] for f in ['answer_score','reasoning_score','joint_score']}
        by_arm[variant]={}
        for arm in ARMS:
            accuracy={f:dict(gpt=[],gemini=[]) for f in same}
            for sid in plan['holdout']:
                truth=g.read(g.SOURCE/'diagnostics/questions'/(sid+'.json'))['models'][arm+'__native']['diagnosis']
                pair=[g.read(ROOT/'holdout'/variant/str(repeat)/'diagnostics/questions'/(sid+'.json'))['models'][arm+'__native']['diagnosis'] for repeat in [0,1]]
                for f in same:
                    same[f].append(pair[0][f]==pair[1][f])
                    accuracy[f]['gpt'].extend([truth[f]==2]*2)
                    accuracy[f]['gemini'].extend([run[f]==2 for run in pair])
            by_arm[variant][arm]={f:{name:float(np.mean(values)) for name,values in scores.items()} for f,scores in accuracy.items()}
        stability[variant]={f:float(np.mean(values)) for f,values in same.items()}
        quality=dict(reference_validity_disagreements=0,findings=0,
            nonliteral_findings=0 if variant!='baseline' else None,
            assessment_finding_conflicts=0 if variant!='baseline' else None,identical_final_score_conflicts=0)
        for repeat in [0,1]:
            for sid in plan['holdout']:
                run=g.read(ROOT/'holdout'/variant/str(repeat)/'diagnostics/questions'/(sid+'.json'))
                truth=g.read(g.SOURCE/'diagnostics/questions'/(sid+'.json'))
                quality['reference_validity_disagreements']+=run['reference_valid']!=truth['reference_valid']
                finals={}
                for response in run['models'].values():
                    diagnosis=response['diagnosis']
                    quality['findings']+=len(diagnosis['findings'])
                    finals.setdefault(response['final_answer'],set()).add(diagnosis['answer_score'])
                    if 'audits' in diagnosis:
                        for name in ['final_audit','reasoning_audit']:
                            audit=diagnosis['audits'][name]
                            expected='fundamental_error' if any(f['severity']=='fundamental' for f in audit['findings']) else ('limited_defect' if audit['findings'] else 'fully_correct')
                            quality['assessment_finding_conflicts']+=audit['assessment']!=expected
                        for finding in diagnosis['findings']:
                            quality['nonliteral_findings']+=not finding['quote_verified']
                quality['identical_final_score_conflicts']+=sum(len(scores)>1 for scores in finals.values())
        audit_quality[variant]=quality
    original_plan=g.read(g.ROOT/'comparison_plan.json')
    for sid,meta in original_plan['gpt_snapshots'].items():
        assert g.sha(g.SOURCE/'diagnostics/questions'/(sid+'.json'))==meta['sha256']
    for name,expected in original_plan['generation_hashes'].items():
        assert g.sha(g.SOURCE/name)==expected
    selection_time=(ROOT/'selection.json').stat().st_mtime_ns
    assert all(p.stat().st_mtime_ns>=selection_time for p in (ROOT/'holdout').glob('*/*/requests/*.json'))
    call_files=list(ROOT.glob('*/*/*/calls/*.json'))
    usage={}
    for p in call_files:
        raw=g.read(p)
        assert raw['requested_model']==g.MODEL and raw['returned_model']==g.MODEL
        for k,v in (raw.get('usage') or {}).items():
            if isinstance(v,(int,float)):
                usage[k]=usage.get(k,0)+v
    for path in (ROOT/'rounds').glob('*.json'):
        config=g.read(path)
        assert g.judge.digest(config['prompt'])==config['prompt_sha256']
    for number in range(2,selection['rounds_completed']+1):
        assert (ROOT/'rounds'/f'round{number}.json').stat().st_mtime_ns>(ROOT/f'development_round{number-1}_results.json').stat().st_mtime_ns
    report=dict(selected=chosen,replacement_validated=bool(accepted),holdout_baseline=baseline,
        holdout_candidate=candidate,differences=differences,ordinal_average_improvement=float(ordinal_delta.mean()),
        ordinal_average_improvement_ci95=improvement_ci,strata=strata,repeat_exact_stability=stability,
        full_credit_rates_by_arm=by_arm,audit_quality=audit_quality,new_gemini_calls=len(call_files),usage=usage,limitations=plan['limitations'])
    g.save(ROOT/'results.json',report)
    compact=dict(selected=chosen,rounds_completed=selection['rounds_completed'],
        development={name:g.read(ROOT/f'development_{name}_results.json')['fields'] for name in ['baseline']+list(selection['development_metrics'])},
        holdout={r['variant']:r['fields'] for r in [baseline,candidate]},
        holdout_questions=len(plan['holdout']),native_model_answers=len(plan['holdout'])*len(ARMS),repeats=2,
        differences=differences,repeat_exact_stability=stability,full_credit_rates_by_arm=by_arm,
        replacement_validated=bool(accepted),ordinal_average_improvement_ci95=improvement_ci,
        new_gemini_calls=len(call_files),new_gpt_calls=0,limitations=plan['limitations'])
    g.save(ROOT/'summary.json',compact)
    g.save(ROOT/'final_verification.json',dict(complete=True,original_gpt_labels_unchanged=len(original_plan['gpt_snapshots']),
        original_generations_unchanged=len(original_plan['generation_hashes']),holdout_started_after_selection=True,
        prompt_hashes_unchanged=True,sequential_adaptation_after_previous_results=True,new_gpt_calls=0,new_gemini_calls=len(call_files)))
    g.save(ROOT/'status.json',dict(status='complete',selected=chosen,replacement_validated=bool(accepted),new_gemini_calls=len(call_files)))
    print(json.dumps(dict(selected=chosen,replacement_validated=bool(accepted),ordinal_average_improvement=float(ordinal_delta.mean()),ci95=improvement_ci,stability=stability,calls=len(call_files)),ensure_ascii=False),flush=True)


def select():
    assert not (ROOT/'selection.json').exists(), 'Selection is already frozen'
    rounds=sorted((ROOT/'rounds').glob('round*.json'))
    assert 3<=len(rounds)<=4
    candidates=[g.read(ROOT/f'development_{p.stem}_results.json') for p in rounds]
    best=max(candidates,key=lambda r:(r['objective'],
        -sum(r['fields'][f]['mae'] for f in ['answer_score','reasoning_score']),
        -sum(abs(r['fields'][f]['bias']) for f in ['answer_score','reasoning_score'])))
    decision=dict(selected=best['variant'],rounds_completed=len(rounds),
        development_metrics={r['variant']:{k:v for k,v in r.items() if k!='items'} for r in candidates},
        selection_rule=g.read(ROOT/'plan.json')['selection'],selected_before_holdout=True)
    g.save(ROOT/'selection.json',decision)
    print(json.dumps(dict(selected=best['variant'],objective=best['objective']),ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','run','metrics','select','finish'])
    parser.add_argument('--phase',default='development',choices=['development','holdout'])
    parser.add_argument('--variant',default='baseline')
    parser.add_argument('--repeats',type=int,default=1)
    args=parser.parse_args()
    prepare()
    if args.stage=='run': run(args.phase,args.variant,list(range(args.repeats)))
    elif args.stage=='metrics': results(args.phase,args.variant,list(range(args.repeats)))
    elif args.stage=='finish': finish()
    elif args.stage=='select': select()
