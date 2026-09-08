"""Paired question-level analysis of complete frozen holdout arms."""
from collections import defaultdict
import json

from data_processing.prepare_medical_holdout600 import ROOT,DATA
from experiments.run_causal_fact_experiment import read,save,sha


def analyze():
    import numpy as np
    from scipy.stats import binomtest
    plan=read(ROOT/'evaluation_plan.json')
    assert sha(DATA/'evaluation.jsonl')==plan['data_sha256']
    rows=[json.loads(line) for line in (DATA/'evaluation.jsonl').read_text(encoding='utf-8').splitlines()]
    overlap=set(read(ROOT/'pre_generation_amendment.json')['exclude_from_sensitivity'])
    sensitivity=np.array([r['source_id'] not in overlap for r in rows])
    missing_input=set(read(ROOT/'input_quality_review.json')['missing_input_exclusions'])
    valid_input=np.array([r['source_id'] not in missing_input for r in rows])
    subjects=defaultdict(list)
    for i,row in enumerate(rows):subjects[row['subject']].append(i)
    def interval(delta,keep):
        rng=np.random.default_rng(20260908);sums=np.zeros(10000)
        for ids in subjects.values():
            indices=np.array(ids)[keep[ids]]
            if len(indices):sums+=delta[rng.choice(indices,size=(10000,len(indices)),replace=True)].sum(axis=1)
        return [float(x) for x in np.quantile(sums/int(keep.sum())*100,[.025,.975])]
    records={};summaries={}
    for path in sorted((ROOT/'runs').glob('*/complete.json')):
        arm=path.parent.name
        values=[read(path.parent/'rows'/f"{r['source_id']}.json") for r in rows]
        assert len(values)==600
        assert all(v['source_id']==r['source_id'] and v['reference_label']==r['answer_label'] for v,r in zip(values,rows,strict=True))
        records[arm]=values
        summaries[arm]=dict(n=600,**{metric:sum(v[metric] for v in values) for metric in ['ranked_correct','free_correct','strict_correct']},
            input_complete=dict(n=int(valid_input.sum()),ranked_correct=sum(values[i]['ranked_correct'] for i in np.flatnonzero(valid_input)),
                free_correct=sum(values[i]['free_correct'] for i in np.flatnonzero(valid_input))),
            missing_explicit_letter=sum(v['explicit_letter'] is None for v in values),
            subjects={s:dict(n=len(ids),ranked_correct=sum(values[i]['ranked_correct'] for i in ids),
                free_correct=sum(values[i]['free_correct'] for i in ids)) for s,ids in subjects.items()})
    pairs=plan['comparisons']+[
        ['retention_source_disabled','retention_attention'],['retention_source_disabled','retention_attention_ffn'],
        ['retention_attention','retention_attention_ffn']]
    for scope in ['attention','attention_ffn']:
        previous='retention_source_disabled'
        for step in [250,500,750,1000,1250]:
            current=f'retention_{scope}_step{step}'
            pairs.append([previous,current]);previous=current
        pairs.append([previous,f'retention_{scope}'])
    comparisons=[]
    for before,after in pairs:
        if before not in records or after not in records:continue
        for metric in ['ranked_correct','free_correct']:
            a=np.array([v[metric] for v in records[before]],dtype=int)
            b=np.array([v[metric] for v in records[after]],dtype=int)
            delta=b-a
            fixed=int(((a==0)&(b==1)).sum());regressed=int(((a==1)&(b==0)).sum())
            low,high=interval(delta,np.ones(600,dtype=bool))
            comparisons.append(dict(before=before,after=after,metric=metric,n=600,
                repaired=fixed,regressed=regressed,net=fixed-regressed,delta_pp=float(delta.mean()*100),
                phrase_overlap_sensitivity=dict(n=int(sensitivity.sum()),
                    repaired=int((delta[sensitivity]==1).sum()),regressed=int((delta[sensitivity]==-1).sum()),
                    delta_pp=float(delta[sensitivity].mean()*100)),
                input_complete_posthoc=dict(n=int(valid_input.sum()),
                    repaired=int((delta[valid_input]==1).sum()),regressed=int((delta[valid_input]==-1).sum()),
                    delta_pp=float(delta[valid_input].mean()*100),bootstrap95_pp=interval(delta,valid_input)),
                input_complete_without_phrase_overlap=dict(n=int((valid_input&sensitivity).sum()),
                    delta_pp=float(delta[valid_input&sensitivity].mean()*100)),
                paired_stratified_bootstrap95_pp=[float(low),float(high)],
                mcnemar_exact_p=float(binomtest(fixed,fixed+regressed,.5).pvalue) if fixed+regressed else 1.,
                repaired_ids=[rows[i]['source_id'] for i in np.flatnonzero(delta==1)],
                regressed_ids=[rows[i]['source_id'] for i in np.flatnonzero(delta==-1)]))
    disabled_checks={}
    source=records.get('retention_source_disabled')
    if source:
        for arm in ['retention_attention_disabled_final','retention_attention_ffn_disabled_final']:
            if arm not in records:continue
            values=records[arm]
            disabled_checks[arm]=dict(prediction_differences=sum(a['prediction']!=b['prediction'] for a,b in zip(source,values,strict=True)),
                rank_differences=sum(a['ranked_letter']!=b['ranked_letter'] for a,b in zip(source,values,strict=True)),
                maximum_letter_logit_difference=max(abs(a['letter_logits'][letter]-b['letter_logits'][letter]) for a,b in zip(source,values,strict=True) for letter in 'ABCD'))
    save(ROOT/'results.json',dict(independent_questions=600,completed_arms=list(records),summaries=summaries,
        comparisons=comparisons,disabled_controls=disabled_checks,
        interpretation='Original-label MCQ scores, not open-ended answer accuracy. P values descriptive across multiple comparisons. Training-seed comparisons shown separately. Pilot adapters were trained on12 facts; this holdout measures their broader effects, not general acquisition of600 trained facts.'))
    print('Analyzed',len(records),'complete600-question arms and',len(comparisons),'paired metrics.',flush=True)


if __name__=='__main__':analyze()
