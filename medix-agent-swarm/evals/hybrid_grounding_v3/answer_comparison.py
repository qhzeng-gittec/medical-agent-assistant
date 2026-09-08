"""Paired answer comparisons and a control for identical retrieved contexts."""
import numpy as np
from common import RESULTS, read, write
from answers import has_invalid_citation


def interval(pairs):
    groups = {}
    for family,a,b in pairs:
        groups.setdefault(family,[]).append((a,b))
    totals = np.asarray([[sum(a for a,b in values),sum(b for a,b in values),len(values)] for values in groups.values()])
    rng = np.random.default_rng(20260908)
    samples = totals[rng.integers(0,len(totals),(4000,len(totals)))].sum(axis=1)
    return {'families':len(totals),'queries':int(totals[:,2].sum()),'baseline_pass':int(totals[:,0].sum()),
            'candidate_pass':int(totals[:,1].sum()),'delta':float((totals[:,1].sum()-totals[:,0].sum())/totals[:,2].sum()),
            'family_bootstrap_95':np.quantile((samples[:,1]-samples[:,0])/samples[:,2],[.025,.975]).tolist()}


def main():
    output = RESULTS/'answers'
    grades = read(output/'grades.json')
    method = read(output/'protocol.json')['selected_method']
    rows = {p.stem:read(p) for p in (output/'cases').glob('*.json')}
    def passed(id_):
        row = rows[id_]
        return int(grades[id_]['complete_correct'] and not has_invalid_citation(row['answer'],row['input']['context']))
    raw,controlled = [],[]
    same = []
    for id_,row in rows.items():
        if row['input']['method']!='dense':
            continue
        c = row['input']['case']
        other_id = c['id']+'__'+method
        other = rows[other_id]
        a,b = passed(id_),passed(other_id)
        identical = row['input']['context']==other['input']['context']
        raw.append((c['family_id'],a,b))
        controlled.append((c['family_id'],a,a if identical else b))
        if identical:
            same.append({'case_id':c['id'],'dense_pass':a,'hybrid_pass':b})
    prompt_pairs = []
    prompt_summary = {}
    for label,suffix in [('basic',method+'_basic_prompt'),('guarded',method)]:
        subset = [r for r in rows.values() if r['input']['method']==suffix and
                  r['input']['case']['suite']=='extension' and r['input']['case']['kind'] in {'near_miss','partial'}]
        prompt_summary[label] = {'n':len(subset),'complete_correct':sum(passed(r['input']['id']) for r in subset),
                                 'unsupported':sum(grades[r['input']['id']]['has_unsupported_claim'] for r in subset),
                                 'false_search':sum(grades[r['input']['id']]['false_search_claim'] for r in subset)}
        if label=='basic':
            for row in subset:
                c = row['input']['case']
                prompt_pairs.append((c['family_id'],passed(row['input']['id']),passed(c['id']+'__'+method)))
    assert len(raw)==640 and len(prompt_pairs)==160
    result = {'independent_generations':interval(raw),'identical_context_control':interval(controlled),
              'identical_context_cases':len(same),'identical_context_different_pass_flags':sum(r['dense_pass']!=r['hybrid_pass'] for r in same),
              'identical_context_details':same,'prompt_ablation':prompt_summary,'prompt_paired':interval(prompt_pairs),
              'interpretation':'One temperature-0.2 generation per arm. The control reuses the dense answer/verdict for BOTH arms only when the entire ordered context is identical; it does not discard or regenerate either original response. Differences under identical inputs expose generation/judge variation.'}
    write(output/'paired_comparison.json',result)
    print({k:v for k,v in result.items() if k!='identical_context_details'})


if __name__=='__main__':
    main()
