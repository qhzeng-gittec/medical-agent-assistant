"""Audit answer content using the user's unchanged Gemini round2 protocol."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import random
from threading import Lock

import experiments.iterate_gemini_judge as judge
from data_processing.prepare_medical_holdout600 import ROOT,DATA
from experiments.run_causal_fact_experiment import read,save,sha
from experiments.run_medical_holdout600 import prompt

ARMS=['base','old_cpt','old_sft','old_cpt_sft']
LOCK=Lock()


def prepare():
    path=ROOT/'semantic_scoring_plan.json'
    if path.exists():return read(path)
    excluded=set(read(ROOT/'input_quality_review.json')['missing_input_exclusions'])
    rows=[json.loads(l) for l in (DATA/'evaluation.jsonl').read_text(encoding='utf-8').splitlines()]
    hashes={}
    for arm in ARMS:
        assert read(ROOT/'runs'/arm/'complete.json')['n']==600
        hashes.update({str(p):sha(p) for p in (ROOT/'runs'/arm/'rows').glob('*.json')})
    config=read(judge.ROOT/'rounds/round2.json')
    assert read(judge.ROOT/'selection.json')['selected']=='round2' and config['split_fields']
    plan=dict(model=judge.g.MODEL,protocol=config,protocol_sha256=sha(judge.ROOT/'rounds/round2.json'),
        data_sha256=sha(DATA/'evaluation.jsonl'),generation_hashes=hashes,
        ids=[r['source_id'] for r in rows if r['source_id'] not in excluded],maximum_calls=600-len(excluded),concurrency=2,
        purpose='Score semantic content of existing32-token nonthinking outputs with original MCQ options present. Letter formatting is separately scored locally; do not equate missing parsed letter with wrong medicine.',
        reference_scope='Original MedMCQA labels; judge may flag reference invalidity. No claim of clinician validation.',
        limitations='Length-limited outputs remain as generated. This audit does not establish performance with a longer answer budget or native reasoning.',
        unknown_request_policy='Persist attempt before invoke; if attempt exists without cached raw response, stop instead of repeating a possibly paid request.')
    save(path,plan);return plan


def score(row):
    plan=read(ROOT/'semantic_scoring_plan.json');out=ROOT/'semantic_scoring'
    target=out/'judgments'/f"{row['source_id']}.json"
    if target.exists():return
    config=plan['protocol']
    responses={}
    for arm in ARMS:
        path=ROOT/'runs'/arm/'rows'/f"{row['source_id']}.json"
        assert sha(path)==plan['generation_hashes'][str(path)]
        responses[arm]=dict(reasoning='',final_answer=read(path)['prediction'])
    unique={judge.g.judge.response_id(r):dict(id=judge.g.judge.response_id(r),final_answer=r['final_answer']) for r in responses.values()}
    candidates=list(unique.values());random.Random(judge.g.judge.SEED).shuffle(candidates)
    rubric=config['prompt'] if config.get('replace_rubric') else judge.g.judge.RUBRIC+'\n\n'+config['prompt']
    schema=judge.audit_schema(list(unique),config.get('claim_checks',False))
    item=schema['properties']['candidates']['items']
    item['properties']={k:v for k,v in item['properties'].items() if k in ['id','final_audit']}
    item['required']=['id','final_audit']
    request=dict(rubric=rubric+'\nThis call evaluates ONLY final_answer; the other field is deliberately withheld. Return ONLY final_audit for each candidate. Do not penalize the absence of the withheld field. Establish correctness from the question and evidence independently.',
        payload=dict(task='knowledge',question=prompt(row),reference=row['reference'],
            reference_explanation=row['reference_explanation'],candidates=candidates),schema=schema)
    if config.get('thinking_level'):request['thinking_level']=config['thinking_level']
    tag='medical_holdout600_v1:round2:0:final_answer'
    key=judge.g.judge.digest(dict(request=request,model=judge.g.MODEL,run_tag=tag))
    call=out/'calls'/f'{key}.json';attempt=out/'attempts'/f'{key}.json'
    with LOCK:
        if not call.exists():
            attempt.parent.mkdir(parents=True,exist_ok=True)
            if len(list(attempt.parent.glob('*.json')))>=plan['maximum_calls']:raise RuntimeError('Frozen Gemini budget exhausted')
            with attempt.open('x',encoding='utf-8') as handle:json.dump(dict(source_id=row['source_id'],request_sha256=key),handle)
    part,returned_key=judge.invoke(request,out,row['source_id'],tag)
    assert key==returned_key
    assert len(part['candidates'])==len(unique) and {c['id'] for c in part['candidates']}==set(unique)
    mapping={'fully_correct':2,'limited_defect':1,'fundamental_error':0}
    models={}
    for arm,response in responses.items():
        cid=judge.g.judge.response_id(response)
        audit=next(c['final_audit'] for c in part['candidates'] if c['id']==cid)
        models[arm]=dict(answer_score=mapping[audit['assessment']],audit=audit,
            quote_checks=[bool(f['quote']) and f['quote'] in response['final_answer'] for f in audit['findings']])
    save(target,dict(source_id=row['source_id'],reference_valid=part['reference_valid'],reference_comment=part['reference_comment'],
        models=models,judge_model=judge.g.MODEL,call_sha256=key))
    with LOCK:
        done=len(list((out/'judgments').glob('*.json')))
        save(ROOT/'semantic_scoring_status.json',dict(state='running',completed=done,expected=len(plan['ids'])))
    print('GEMINI_CONTENT',done,'/',len(plan['ids']),flush=True)


def summarize():
    import numpy as np
    plan=read(ROOT/'semantic_scoring_plan.json')
    judgments=[read(ROOT/'semantic_scoring/judgments'/f'{sid}.json') for sid in plan['ids']]
    valid=[j for j in judgments if j['reference_valid']]
    assert valid
    metrics={arm:dict(n=len(valid),fully_correct=sum(j['models'][arm]['answer_score']==2 for j in valid),
        partial=sum(j['models'][arm]['answer_score']==1 for j in valid)) for arm in ARMS}
    pairs=[]
    for left,right in [('base','old_cpt'),('base','old_sft'),('old_sft','old_cpt_sft'),('old_cpt','old_cpt_sft')]:
        delta=np.array([int(j['models'][right]['answer_score']==2)-int(j['models'][left]['answer_score']==2) for j in valid])
        rng=np.random.default_rng(20260908)
        boot=delta[rng.integers(len(valid),size=(10000,len(valid)))].mean(axis=1)*100
        pairs.append(dict(before=left,after=right,n=len(valid),repaired=int((delta==1).sum()),regressed=int((delta==-1).sum()),
            delta_pp=float(delta.mean()*100),paired_bootstrap95_pp=np.quantile(boot,[.025,.975]).tolist()))
    save(ROOT/'semantic_results.json',dict(judged_questions=len(judgments),reference_valid_questions=len(valid),metrics=metrics,comparisons=pairs,
        invalid_reference_ids=[j['source_id'] for j in judgments if not j['reference_valid']],
        meets_500_independent_items=len(valid)>=500,new_gpt_calls=0,gemini_model=plan['model'],
        limitations=['Existing32-token outputs only; truncation can affect content scores.','Gemini semantic assessments, not clinical expert ground truth.','Nonthinking MCQ inputs retain options; not an option-free question benchmark.']))
    save(ROOT/'semantic_scoring_status.json',dict(state='complete',completed=len(judgments),reference_valid=len(valid)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','run','summarize']);args=parser.parse_args()
    plan=prepare()
    if args.stage=='run':
        assert sha(DATA/'evaluation.jsonl')==plan['data_sha256']
        assert sha(judge.ROOT/'rounds/round2.json')==plan['protocol_sha256'] and judge.g.MODEL==plan['model']
        try:
            allowed=set(plan['ids'])
            rows=[json.loads(l) for l in (DATA/'evaluation.jsonl').read_text(encoding='utf-8').splitlines()]
            pending=[r for r in rows if r['source_id'] in allowed]
            with ThreadPoolExecutor(max_workers=2) as pool:
                for offset in range(0,len(pending),2):list(pool.map(score,pending[offset:offset+2]))
            summarize()
        except Exception as error:
            save(ROOT/'semantic_scoring_status.json',dict(state='failed',error=repr(error)));raise
    elif args.stage=='summarize':summarize()
