"""Gemini scoring and paired comparison against preserved GPT judgments; no GPT calls."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import json
from pathlib import Path
import random
import re
import shutil
import subprocess

import numpy as np
import experiments.diagnose_reasoning_effects as judge
import experiments.run_cpt_diagnosis as diagnostic

LAB = Path(__file__).resolve().parents[1]
SOURCE = LAB/'outputs/cpt_diagnosis_v3'
ROOT = LAB/'outputs/gemini_judge_v1'
MODEL = 'gemini-3.5-flash'
read,save,sha = diagnostic.read,diagnostic.save,diagnostic.sha


def prepare():
    if (ROOT/'comparison_plan.json').exists():
        return
    ROOT.mkdir(exist_ok=True)
    for name in ['plan.json','evaluation.jsonl','evidence.json']:
        shutil.copyfile(SOURCE/name,ROOT/name)
    originals=list((SOURCE/'diagnostics/questions').glob('*.json'))
    save(ROOT/'comparison_plan.json',dict(model=MODEL,thinking_level='medium',temperature=.2,
        adapter='Existing WhatsApp project LlmClient, OpenAI-compatible chat completions; no WhatsApp operations.',
        total_questions=247,gpt_overlap=len(originals),
        gpt_snapshots={p.stem:dict(sha256=sha(p),judge_model=read(p).get('judge_model','gpt-5.6-sol')) for p in originals},
        generation_hashes={str(p.relative_to(SOURCE)):sha(p) for p in (SOURCE/'generation').glob('*/*.json')},
        rubric_sha256=judge.digest(dict(rubric=judge.RUBRIC,context=judge.CONTEXT_CALIBRATION,schema=judge.SCORE_SCHEMA)),
        source_files={name:sha(SOURCE/name) for name in ['plan.json','evaluation.jsonl','evidence.json']},
        primary='Within-question agreement on strict correctness across four native model outputs; per-arm model accuracy differences.',
        limitations=['GPT labels are a comparator, not an expert gold standard; agreement is not true medical accuracy.',
            'GPT overlap is the completed subset, not a new random sample; report task and original-judge strata.',
            'Multiple candidates per question are correlated; bootstrap by question.',
            'All variants are secondary and error-conditioned; do not extrapolate their agreement to general accuracy.',
            'Models use their existing interface settings; judge identity and generation configuration are not isolated factors.']))
    save(SOURCE/'status.json',dict(status='stopped',stage='scoring',complete=len(originals),total=247,
        reason='User stopped GPT scoring; Gemini comparison now runs in outputs/gemini_judge_v1.'))


def score_one(row, *, output_root=None, rubric_suffix='', run_tag='', native_only=False):
    root=ROOT if output_root is None else output_root
    sid=row['source_id']
    target=root/'diagnostics/questions'/(sid+'.json')
    if target.exists():
        return read(target)
    plan,evidence=read(root/'plan.json'),read(root/'evidence.json')
    responses={variant:read(SOURCE/'generation'/variant/(sid+'.json'))
        for arm in diagnostic.ARMS for variant,_,_ in diagnostic.variants(row,arm,plan,evidence)
        if not native_only or variant.endswith('__native')}
    unique={judge.response_id(r):dict(id=judge.response_id(r),reasoning=r['reasoning'],final_answer=r['final_answer']) for r in responses.values()}
    candidates=list(unique.values())
    random.Random(judge.SEED).shuffle(candidates)
    payload=dict(task=row['task'],question=row['user_text'],reference=row['reference'],
        reference_explanation=row.get('reference_explanation',''),candidates=candidates)
    rubric=judge.RUBRIC+('\n'+judge.CONTEXT_CALIBRATION if row['task']=='context' else '')
    if rubric_suffix:
        rubric+='\n\n'+rubric_suffix
    schema=copy.deepcopy(judge.SCORE_SCHEMA)
    schema['properties']['candidates']['items']['properties']['id']['enum']=list(unique)
    request=dict(rubric=rubric,payload=payload,schema=schema)
    identity=dict(request=request,model=MODEL)
    if run_tag:
        identity['run_tag']=run_tag
    key=judge.digest(identity)
    call=root/'calls'/(key+'.json')
    save(root/'requests'/(sid+'.json'),request)
    if call.exists():
        raw=read(call)
    else:
        command=['node',str(LAB/'gemini_judge_bridge.mjs')]
        process=subprocess.run(command,input=json.dumps(request,ensure_ascii=False),text=True,
            encoding='utf-8',capture_output=True,timeout=90)
        if process.returncode:
            raise RuntimeError(f'{sid}: Gemini bridge failed: {process.stderr[-2000:]}')
        raw=json.loads(process.stdout)
        save(call,raw)
    assert raw['requested_model']==MODEL
    if raw['finish_reason']!='stop':
        raise ValueError(f'{sid}: incomplete judge output: {raw["finish_reason"]}')
    # Match the existing project's thought-summary.js envelope handling; retain raw calls.
    visible=raw['message']['content'].lstrip()
    while visible.startswith('<thought>'):
        _,separator,visible=visible.partition('</thought>')
        assert separator, 'Incomplete thought envelope'
        visible=visible.lstrip()
    fenced=re.fullmatch(r'```(?:json)?\s*([\s\S]*?)\s*```',visible.strip(),re.I)
    judged=json.loads(fenced[1] if fenced else visible)
    assert type(judged['reference_valid']) is bool
    assert len(judged['candidates'])==len(unique) and {c['id'] for c in judged['candidates']}==set(unique)
    repairs=[]
    for candidate in judged['candidates']:
        assert type(candidate['answer_score']) is int and candidate['answer_score'] in [0,1,2]
        assert type(candidate['reasoning_score']) is int and candidate['reasoning_score'] in [0,1,2]
        assert set(candidate['flags'])<=set(judge.FLAGS)
        flags={f['type'] for f in candidate['findings']}
        assert flags<=set(judge.FLAGS)
        if set(candidate['flags'])!=flags:
            repairs.append(dict(id=candidate['id'],original_flags=candidate['flags'],derived_flags=sorted(flags)))
            candidate['flags']=sorted(flags)
        candidate['joint_score']=min(candidate['answer_score'],candidate['reasoning_score'])
    by_id={c['id']:c for c in judged['candidates']}
    result=dict(source_id=sid,task=row['task'],cohort=row['cohort'],question=row['user_text'],reference=row['reference'],
        reference_valid=judged['reference_valid'],reference_comment=judged['reference_comment'],
        call_sha256=key,judge_model=MODEL,returned_model=raw.get('returned_model'),metadata_repairs=repairs,
        models={arm:dict(response,diagnosis=by_id[judge.response_id(response)]) for arm,response in responses.items()})
    save(target,result)
    return result


def score(smoke=False):
    rows=diagnostic.load_jsonl(ROOT/'evaluation.jsonl')
    selected=set(read(ROOT/'subset_50.json')['source_ids'])
    rows=[r for r in rows if r['source_id'] in selected]
    if smoke:
        rows=[next(r for r in rows if r['task']=='knowledge')]
    pending=[r for r in rows if not (ROOT/'diagnostics/questions'/(r['source_id']+'.json')).exists()]
    errors=[]
    save(ROOT/'status.json',dict(stage='scoring',complete=len(rows)-len(pending),total=len(rows),status='running',judge=MODEL))
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs={pool.submit(score_one,r):r['source_id'] for r in pending}
        for future in as_completed(jobs):
            try:
                result=future.result()
                print('judged',result['source_id'],flush=True)
            except Exception as error:
                errors.append(dict(source_id=jobs[future],error=repr(error)))
                save(ROOT/'scoring_errors.json',errors)
                print('scoring error',errors[-1],flush=True)
            save(ROOT/'status.json',dict(stage='scoring',complete=len(list((ROOT/'diagnostics/questions').glob('*.json'))),
                total=len(rows),failures=len(errors),status='running',judge=MODEL))
    if errors:
        raise RuntimeError(f'{len(errors)} Gemini scoring failures; see scoring_errors.json')


def agreement(pairs):
    # Each item is one question's list of paired native candidate scores.
    matrix=np.zeros((3,3),dtype=int)
    for row in pairs:
        for gpt,gemini in row:
            matrix[gpt,gemini]+=1
    n=int(matrix.sum())
    exact=float(matrix.trace()/n)
    expected=float((matrix.sum(axis=0)*matrix.sum(axis=1)).sum()/n**2)
    per_question=np.array([np.mean([(a==2)==(b==2) for a,b in row]) for row in pairs])
    boot=per_question[np.random.default_rng(42).integers(0,len(pairs),(20000,len(pairs)))].mean(axis=1)
    return dict(questions=len(pairs),responses=n,score_confusion_gpt_rows_gemini_columns=matrix.tolist(),
        exact_score_agreement=exact,cohen_kappa=(exact-expected)/(1-expected) if expected<1 else None,
        strict_correctness_agreement=float(per_question.mean()),strict_agreement_ci95=np.quantile(boot,[.025,.975]).tolist())


def summarize():
    plan=read(ROOT/'comparison_plan.json')
    subset=read(ROOT/'subset_50.json')
    rows=[read(ROOT/'diagnostics/questions'/(sid+'.json')) for sid in subset['source_ids']]
    assert len(rows)==50 and len({r['source_id'] for r in rows})==50
    overlap=[]
    disagreement=[]
    validity=[]
    for row in rows:
        sid=row['source_id']
        if sid not in plan['gpt_snapshots']:
            continue
        prior=read(SOURCE/'diagnostics/questions'/(sid+'.json'))
        if prior['reference_valid']!=row['reference_valid']:
            validity.append(dict(source_id=sid,gpt=prior['reference_valid'],gemini=row['reference_valid']))
        if not (prior['reference_valid'] and row['reference_valid']):
            continue
        overlap.append((prior,row))
        for variant,response in row['models'].items():
            left=prior['models'][variant]['diagnosis']; right=response['diagnosis']
            if left['answer_score']!=right['answer_score']:
                disagreement.append(dict(source_id=sid,task=row['task'],variant=variant,
                    gpt_score=left['answer_score'],gemini_score=right['answer_score'],
                    gpt_explanation=left['explanation'],gemini_explanation=right['explanation'],final_answer=response['final_answer']))
    comparisons={}
    strata={'all':overlap}
    strata.update({task:[(a,b) for a,b in overlap if a['task']==task] for task in ['knowledge','context']})
    strata.update({model:[(a,b) for a,b in overlap if plan['gpt_snapshots'][a['source_id']]['judge_model']==model]
        for model in sorted({m['judge_model'] for m in plan['gpt_snapshots'].values()})})
    for task,subset in strata.items():
        pairs=[[(a['models'][arm+'__native']['diagnosis']['answer_score'],b['models'][arm+'__native']['diagnosis']['answer_score']) for arm in diagnostic.ARMS] for a,b in subset]
        reasoning_pairs=[[(a['models'][arm+'__native']['diagnosis']['reasoning_score'],b['models'][arm+'__native']['diagnosis']['reasoning_score']) for arm in diagnostic.ARMS] for a,b in subset]
        comparisons[task]=dict(agreement=agreement(pairs),reasoning_agreement=agreement(reasoning_pairs),by_arm={})
        for arm in diagnostic.ARMS:
            paired_rows=[dict(models=dict(gpt=a['models'][arm+'__native'],gemini=b['models'][arm+'__native'])) for a,b in subset]
            comparisons[task]['by_arm'][arm]=diagnostic.paired(paired_rows,'gpt','gemini')
    save(ROOT/'comparison_results.json',dict(comparisons=comparisons,validity_disagreements=validity,
        common_valid_questions=len(overlap),gpt_overlap=50,disagreements=disagreement,
        gpt_strata=dict(Counter(plan['gpt_snapshots'][r['source_id']]['judge_model'] for r in rows)),
        selection=subset,limitations=plan['limitations']+['User limited this pilot to 50 questions; reused completed Gemini judgments and selected remaining knowledge questions before scoring. Unequal task mix; not a full medical-model evaluation.']))
    for sid,metadata in plan['gpt_snapshots'].items():
        assert sha(SOURCE/'diagnostics/questions'/(sid+'.json'))==metadata['sha256']
    for name,expected in plan['generation_hashes'].items():
        assert sha(SOURCE/name)==expected
    for name,expected in plan['source_files'].items():
        assert sha(SOURCE/name)==expected
    assert len(list((ROOT/'diagnostics/questions').glob('*.json')))==50
    save(ROOT/'final_verification.json',dict(gemini_questions=50,gpt_overlap=50,original_gpt_and_generations_unchanged=True,complete=True))
    save(ROOT/'status.json',dict(status='complete',stage='complete',judge=MODEL,complete=50,total=50))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','smoke','score','summarize','all'])
    args=parser.parse_args()
    prepare()
    if args.stage=='all': score(); summarize()
    elif args.stage=='smoke': score(smoke=True)
    elif args.stage=='score': score()
    elif args.stage=='summarize': summarize()
