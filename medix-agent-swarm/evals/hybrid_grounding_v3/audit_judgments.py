"""Independent Codex review of all failures and a fixed sample of passes.

This is an enriched audit sample, never an alternative estimate for all answers.
Original MiniMax scores are retained regardless of reviewer agreement.
"""
from concurrent.futures import ThreadPoolExecutor
import random

from answers import JUDGE, has_invalid_citation, validate_grades
from codex_author import generate, object_schema
from common import RESULTS, frozen, read, write


def main():
    output = RESULTS/'judge_audit'
    selected = []
    rng = random.Random(20260908)
    for directory in ['answers','memory_answers']:
        folder = RESULTS/directory
        assert read(folder/'execution_status.json')['completed']
        grades = read(folder/'grades.json')
        inputs = read(folder/'grading_v2/inputs.json')
        passes = {}
        for id_,grade in sorted(grades.items()):
            row = read(folder/'cases'/(id_+'.json'))
            invalid = has_invalid_citation(row['answer'],row['input']['context'])
            entry = {'domain':directory,'id':id_,'input':inputs[id_],'initial_grade':grade,
                     'invalid_citation':invalid,'selection':'all_initial_failures'}
            if not grade['complete_correct'] or invalid:
                selected.append(entry)
            else:
                passes.setdefault(row['input']['method'],[]).append(entry)
        for group in passes.values():
            for entry in rng.sample(group,min(8,len(group))):
                entry['selection'] = 'seeded_pass_sample'
                selected.append(entry)
    selected.sort(key=lambda r:r['input']['id'])
    frozen(output/'selection.json',{'seed':20260908,'rule':'Every initial failure, plus 8 passes per method. Enriched audit, not prevalence estimation.',
                                  'items':selected})
    fields = {'id':{'type':'string'},'context_sufficiency':{'type':'string','enum':['full','partial','none']},
              **{k:{'type':'boolean'} for k in ['has_unsupported_claim','false_search_claim','acknowledges_limit','answers_supported_part','complete_correct']},
              **{k:{'type':'string'} for k in ['unsupported_quote','search_quote','reason']}}
    schema = object_schema({'items':{'type':'array','items':object_schema(fields)}})
    instruction = JUDGE+'''
Independently review each item. Only the context field was shown to the answerer.
Gold evidence and expected answers are evaluator references, NOT additional retrieved context.
If gold evidence is absent from context, an admission of missing information is not hallucination or a false search claim.
Read all context documents before judging. Do not mark a statement unsupported merely because one gold excerpt is narrower.
An explicit claim about what the provided documents contain is not a claim of external database search.
An answer may fail completeness without fabricating facts. Preserve this distinction rigorously.'''
    def batch(pair):
        number,group = pair
        values = [r['input'] for r in group]
        result = generate(output/'codex_records',f'batch_{number:03d}',instruction,values,schema,model='gpt-5.5')
        by_id = {r['id']:r for r in result['items']}
        repairs = read(output/'quote_repairs.json') if (output/'quote_repairs.json').exists() else {}
        for id_,repair in repairs.items():
            if id_ in by_id:
                assert by_id[id_][repair['field']]==repair['before']
                by_id[id_][repair['field']] = repair['after']
        if len(by_id)!=len(result['items']) or set(by_id)-{v['id'] for v in values}:
            raise ValueError('Duplicate or unexpected audit IDs')
        # Preserve every returned verdict; request only an omitted item.
        reviewed = []
        for value in values:
            if value['id'] not in by_id:
                write(output/'format_issues'/f'{value["id"]}.json',{'batch':number,'error':'Missing item; existing verdicts preserved'})
                missing = generate(output/'codex_records','missing_'+value['id'],instruction,[value],schema,model='gpt-5.5')
                reviewed.extend(validate_grades(missing,[value]))
            else:
                reviewed.extend(validate_grades({'items':[by_id[value['id']]]},[value]))
        return reviewed
    groups = [(n,selected[i:i+6]) for n,i in enumerate(range(0,len(selected),6))]
    judgments = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        for n,values in enumerate(pool.map(batch,groups),1):
            judgments.update({v['id']:v for v in values})
            print(f'Independent audit {n}/{len(groups)}',flush=True)
    records = []
    for row in selected:
        review = judgments[row['input']['id']]
        records.append({k:v for k,v in row.items() if k!='input'} | {'review':review})
    summary = {}
    for row in records:
        for stratum in [row['domain']+'/all',row['domain']+'/'+row['selection']]:
            values = summary.setdefault(stratum,{'n':0,'initial_unsupported':0,'review_unsupported':0,
                          'initial_false_search':0,'review_false_search':0,'pass_disagreements':0})
            values['n'] += 1
            for label,key in [('unsupported','has_unsupported_claim'),('false_search','false_search_claim')]:
                values['initial_'+label] += row['initial_grade'][key]
                values['review_'+label] += row['review'][key]
            values['pass_disagreements'] += row['initial_grade']['complete_correct']!=row['review']['complete_correct']
    write(output/'reviews.json',records)
    write(output/'summary.json',summary)
    print(summary)


if __name__=='__main__':
    main()
