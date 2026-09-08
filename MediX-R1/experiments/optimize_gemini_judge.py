"""Frozen generic prompt experiment: 12 development and 12 fresh holdout questions."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import random
import shutil

import numpy as np
import experiments.compare_gemini_judge as gemini

ROOT=gemini.LAB/'outputs/gemini_prompt_stability_v1'
ARMS=gemini.diagnostic.ARMS
PROMPTS={
    'baseline':'',
    'field_audit':'''FIELD-ISOLATED FACT AUDIT
Treat reasoning and final_answer as two separate fields. Evaluate the ENTIRE final_answer field, including explanations, caveats, lists and text under headings such as "Reasoning" inside that field. A heading does not move text into the separate reasoning field.
For each field, identify its central conclusion and its supporting factual claims. Check each material claim against the supplied evidence and relevant domain knowledge. Do not assume that a correct keyword makes its surrounding assertions correct. Conversely, do not invent an error merely because the response is brief, uses a valid synonym, or omits unrequested detail.
Assign answer_score using only final_answer: 2 requires a correct requested conclusion and no material false or unsupported claim in that field; 1 fits a substantially correct conclusion with limited incorrect supporting content or a genuinely required missing qualifier; 0 fits an incorrect central conclusion, no usable answer, or errors that fundamentally undermine the requested conclusion. Distinguish harmless imprecision from a substantive factual error. An error confined to the separate reasoning field must not lower answer_score.
Assign reasoning_score independently: a concise accurate justification can earn 2; minor inaccuracies or a necessary omitted step merit 1; a materially false premise, unsupported decisive inference or rationale that fails to support the conclusion merits 0. A correct final choice does not repair faulty reasoning.
Before returning JSON, check every full-credit field against the errors you identified in THAT field. If your own findings establish a material error there, full credit is inconsistent: reassess either the finding or the score based on the evidence. Never lower a score just to make the response look critical. Keep identical final answers equally scored, copy candidate IDs exactly, and make flags exactly the distinct types in findings.''',
    'evidence_gates':'''EVIDENCE-FIRST SCORING AND CONSISTENCY GATES
For each anonymous candidate, first determine what the question actually requires and what evidence supports an acceptable answer. Reference wording is a guide, not a mandatory string match. Preserve the question's implicit population, context and scope.
Then audit the two supplied fields independently. Identify concrete supported or contradicted assertions before assigning scores. Every alleged material error needs a quote from the field where it occurs and a specific evidence-based explanation. A label without such evidence is not a reason to deduct points. Consider the strongest reasonable interpretation of concise answers and valid paraphrases; do not add requirements that were not asked.
Apply these gates to the whole final_answer, including all attached explanations: the requested conclusion must be correct; required distinctions must be present; accompanying factual claims must not be materially false or unsupported. Award 2 only when all applicable gates pass. Use 1 for a substantially correct answer with limited but substantive defects, and 0 when the central answer is incorrect, unusable or fundamentally undermined. Do not confuse a correct conclusion with a fully correct final-answer field.
Apply independent gates to reasoning: are the relevant premises accurate, are the inferences supported, and is the rationale sufficient for this question? Score correctness and sufficiency, not length. Do not transfer errors across fields unless the erroneous claim occurs in both. Do not infer an internal contradiction from an omission or from two claims that are consistently wrong.
Finally, challenge the proposed grades in both directions: could full credit be overlooking an identified material defect, or could a deduction be punishing harmless wording, brevity or appropriate uncertainty? Resolve conflicts using the supplied evidence, not by aiming for any pass rate or model ranking. Return only the required judgment JSON with exact candidate IDs and mutually consistent findings, flags, scores and explanations.''',
}


def prepare():
    if (ROOT/'plan.json').exists():
        return
    ROOT.mkdir(exist_ok=True)
    old=gemini.read(gemini.ROOT/'comparison_plan.json')['gpt_snapshots']
    seen=set(gemini.read(gemini.ROOT/'subset_50.json')['source_ids'])
    rows=gemini.diagnostic.load_jsonl(gemini.SOURCE/'evaluation.jsonl')
    rng=random.Random(202609071)
    partitions={name:[] for name in ['development','holdout']}
    for task in ['knowledge','context']:
        for name,predicate in [('development',lambda sid:sid in seen),('holdout',lambda sid:sid not in seen)]:
            candidates=sorted(r['source_id'] for r in rows if r['task']==task and r['source_id'] in old and predicate(r['source_id']))
            partitions[name]+=rng.sample(candidates,6)
    assert not set(partitions['holdout']) & seen
    for name,prompt in PROMPTS.items():
        (ROOT/(name+'.txt')).write_text(prompt,encoding='utf-8')
    snapshots={sid:old[sid] for ids in partitions.values() for sid in ids}
    gemini.save(ROOT/'plan.json',dict(partitions=partitions,prompts=PROMPTS,prompt_hashes={k:gemini.judge.digest(v) for k,v in PROMPTS.items()},
        model=gemini.MODEL,temperature=.2,thinking_level='medium',repeats=2,max_calls=120,
        responses='Four native arms only, identical candidate order and schema across variants and repeats.',
        selection='Choose B/C once using development score: mean(balanced joint-full-credit agreement with GPT, repeated joint-full-credit stability) minus 0.25*field_score_conflict_rate. No further prompt edits. Test selected challenger and baseline twice on fresh holdout, even if development gain is weak.',
        acceptance='Promising only if holdout balanced joint agreement and repeat stability each decrease by no more than 0.03, conflict rate does not increase, and their penalized composite improves by at least 0.02. Otherwise keep baseline; no retuning on holdout.',
        gpt_snapshots=snapshots,limitations=['Only 12 questions per split and two repeats; exploratory, not proof of robust medical judging.',
            'GPT labels are imperfect proxy labels, never passed into the Gemini prompt.',
            'Development is sampled from already inspected pilot; holdout is outside all 50 prior pilot questions.',
            'Scores from multiple arms on a question are correlated; uncertainty resamples questions.',
            'No disease-specific examples, expected scores, desired pass rate or model names are provided to Gemini.']))


def run_phase(phase,variants):
    plan=gemini.read(ROOT/'plan.json')
    rows={r['source_id']:r for r in gemini.diagnostic.load_jsonl(gemini.SOURCE/'evaluation.jsonl')}
    jobs=[]
    for variant in variants:
        assert gemini.judge.digest(PROMPTS[variant])==plan['prompt_hashes'][variant]
        for repeat in [0,1]:
            out=ROOT/phase/variant/str(repeat)
            out.mkdir(parents=True,exist_ok=True)
            for name in ['plan.json','evidence.json']:
                if not (out/name).exists():
                    shutil.copyfile(gemini.SOURCE/name,out/name)
            for sid in plan['partitions'][phase]:
                if not (out/'diagnostics/questions'/(sid+'.json')).exists():
                    jobs.append((variant,repeat,rows[sid],out))
    errors=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures={pool.submit(gemini.score_one,row,output_root=out,rubric_suffix=PROMPTS[variant],
            run_tag=f'{phase}:{variant}:{repeat}',native_only=True):(variant,repeat,row['source_id']) for variant,repeat,row,out in jobs}
        for future in as_completed(futures):
            identity=futures[future]
            try:
                future.result()
            except Exception as error:
                errors.append(dict(job=identity,error=repr(error)))
                gemini.save(ROOT/(phase+'_errors.json'),errors)
                print('error',identity,repr(error),flush=True)
            done=len(list((ROOT/phase).glob('*/*/diagnostics/questions/*.json')))
            gemini.save(ROOT/'status.json',dict(status='running',phase=phase,phase_complete=done,
                phase_total=12*len(variants)*2,failures=len(errors),max_calls=120))
            print(phase,done,'completed',identity,flush=True)
    if errors:
        raise RuntimeError(f'{phase}: {len(errors)} failed judgments; see error file')


def metrics(phase,variant):
    plan=gemini.read(ROOT/'plan.json')
    items=[]
    for sid in plan['partitions'][phase]:
        prior=gemini.read(gemini.SOURCE/'diagnostics/questions'/(sid+'.json'))
        runs=[gemini.read(ROOT/phase/variant/str(i)/'diagnostics/questions'/(sid+'.json')) for i in [0,1]]
        labels=[]; predictions=[]; answer_agree=[]; reasoning_agree=[]; conflicts=[]; stability=[]
        for arm in ARMS:
            key=arm+'__native'
            original=prior['models'][key]['diagnosis']
            pair=[r['models'][key]['diagnosis'] for r in runs]
            stability.append((pair[0]['joint_score']==2)==(pair[1]['joint_score']==2))
            for run,score in zip(runs,pair):
                labels.append(original['joint_score']==2)
                predictions.append(score['joint_score']==2)
                answer_agree.append(original['answer_score']==score['answer_score'])
                reasoning_agree.append(original['reasoning_score']==score['reasoning_score'])
                response=run['models'][key]
                field_conflict=False
                for finding in score['findings']:
                    quote=finding['quote']
                    if finding['type'] not in ['medical_fact_error','source_misread','unsupported_inference'] or not quote:
                        continue
                    in_answer=quote in response['final_answer']; in_reasoning=quote in response['reasoning']
                    field_conflict |= in_answer and not in_reasoning and score['answer_score']==2
                    field_conflict |= in_reasoning and not in_answer and score['reasoning_score']==2
                conflicts.append(bool(field_conflict))
        items.append(dict(source_id=sid,labels=labels,predictions=predictions,answer_exact=np.mean(answer_agree),
            reasoning_exact=np.mean(reasoning_agree),repeat_joint_stability=np.mean(stability),conflict_rate=np.mean(conflicts),
            validity_disagreement=any(r['reference_valid']!=prior['reference_valid'] for r in runs)))
    flat_labels=np.array([v for q in items for v in q['labels']]); flat_predictions=np.array([v for q in items for v in q['predictions']])
    positive=float((flat_predictions[flat_labels]==True).mean())
    negative=float((flat_predictions[~flat_labels]==False).mean())
    balanced=(positive+negative)/2
    stability=float(np.mean([q['repeat_joint_stability'] for q in items]))
    conflict=float(np.mean([q['conflict_rate'] for q in items]))
    return dict(variant=variant,questions=len(items),scored_responses=96,
        answer_exact=float(np.mean([q['answer_exact'] for q in items])),reasoning_exact=float(np.mean([q['reasoning_exact'] for q in items])),
        joint_balanced_agreement=balanced,joint_positive_agreement=positive,joint_negative_agreement=negative,
        repeat_joint_stability=stability,field_score_conflict_rate=conflict,
        validity_disagreement_questions=sum(q['validity_disagreement'] for q in items),
        composite=(balanced+stability)/2-.25*conflict,items=items)


def select():
    if (ROOT/'selection.json').exists():
        return gemini.read(ROOT/'selection.json')['selected']
    results={variant:metrics('development',variant) for variant in PROMPTS}
    selected=max(['field_audit','evidence_gates'],key=lambda name:results[name]['composite'])
    gemini.save(ROOT/'development_results.json',results)
    gemini.save(ROOT/'selection.json',dict(selected=selected,selected_before_holdout=True,rule=gemini.read(ROOT/'plan.json')['selection']))
    return selected


def summarize():
    plan=gemini.read(ROOT/'plan.json')
    selected=gemini.read(ROOT/'selection.json')['selected']
    original=metrics('holdout','baseline'); candidate=metrics('holdout',selected)
    accepted=(candidate['joint_balanced_agreement']>=original['joint_balanced_agreement']-.03
        and candidate['repeat_joint_stability']>=original['repeat_joint_stability']-.03
        and candidate['field_score_conflict_rate']<=original['field_score_conflict_rate']
        and candidate['composite']>=original['composite']+.02
        and candidate['validity_disagreement_questions']<=original['validity_disagreement_questions'])
    differences={}
    for key in ['answer_exact','reasoning_exact','repeat_joint_stability','conflict_rate']:
        delta=np.array([b[key]-a[key] for a,b in zip(original['items'],candidate['items'],strict=True)])
        boot=delta[np.random.default_rng(42).integers(0,12,(20000,12))].mean(axis=1)
        differences[key]=dict(delta=float(delta.mean()),ci95=np.quantile(boot,[.025,.975]).tolist())
    for sid,meta in plan['gpt_snapshots'].items():
        assert gemini.sha(gemini.SOURCE/'diagnostics/questions'/(sid+'.json'))==meta['sha256']
    outputs=list(ROOT.glob('*/*/*/diagnostics/questions/*.json'))
    assert len(outputs)==120
    result=dict(selected=selected,promising_on_holdout=accepted,holdout_baseline=original,holdout_candidate=candidate,
        paired_differences=differences,limitations=plan['limitations'],calls=120)
    gemini.save(ROOT/'results.json',result)
    chosen=selected if accepted else 'baseline'
    text=gemini.judge.RUBRIC+'\n\nFor research-context tasks additionally apply:\n'+gemini.judge.CONTEXT_CALIBRATION
    if PROMPTS[chosen]: text+='\n\n'+PROMPTS[chosen]
    (ROOT/'recommended_prompt.txt').write_text(text,encoding='utf-8')
    gemini.save(ROOT/'status.json',dict(status='complete',calls=120,selected=selected,promising_on_holdout=accepted))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','all','summarize'])
    args=parser.parse_args()
    prepare()
    if args.stage=='all':
        run_phase('development',list(PROMPTS)); chosen=select(); run_phase('holdout',['baseline',chosen]); summarize()
    elif args.stage=='summarize': summarize()
