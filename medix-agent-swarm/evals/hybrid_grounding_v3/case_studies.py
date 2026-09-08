"""Expose every changed Top-3 outcome, plus all answer failures, without cherry-picking."""
from common import RESULTS, V2, lines, read, write
from retrieval import complete
from answers import has_invalid_citation


def main():
    corpus = {d['id']:d for d in lines(V2/'data/corpus.jsonl')}
    rows = read(RESULTS/'retrieval/rankings.json')
    changed = []
    for row in rows:
        c = row['case']
        if c['split']!='test' or c['answerability']!='answerable':
            continue
        a,b = [complete(c,row['rankings'][m],3) for m in ['dense','hybrid_translated']]
        if a==b:
            continue
        changed.append({'id':c['id'],'query':c['query'],'suite':c['suite'],'dense_complete':a,'hybrid_complete':b,
                        'targets':[[{'id':d,'title':corpus[d]['title']} for d in group] for group in c['gold_groups']],
                        'rankings':{m:row['rankings'][m] for m in ['dense','bm25_translated','hybrid_translated']},
                        'gold_evidence':c.get('evidence',[])})
    write(RESULTS/'retrieval/changed_cases.json',changed)
    for directory in ['answers','memory_answers']:
        grades = read(RESULTS/directory/'grades.json')
        failures = []
        for id_,grade in grades.items():
            row = read(RESULTS/directory/'cases'/(id_+'.json'))
            invalid = has_invalid_citation(row['answer'],row['input']['context'])
            if not grade['complete_correct'] or invalid:
                failures.append({'id':id_,'question':row['input']['case']['query'],'answer':row['answer'],
                                 'grade':grade,'invalid_citation':invalid,'record':f'cases/{id_}.json'})
        write(RESULTS/directory/'failures.json',failures)
    print(f'Published {len(changed)} changed retrieval outcomes and answer failure indexes')


if __name__=='__main__':
    main()
