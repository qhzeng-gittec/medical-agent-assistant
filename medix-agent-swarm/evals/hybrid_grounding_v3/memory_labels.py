"""Explicit source-based annotation corrections; preserve v2 data and results."""
import re
from common import HERE, V2, digest, frozen, lines


def main():
    cases = {c['id']:c for c in lines(V2/'data/mem0_cases.jsonl')}
    changes = []
    for number in range(1,11):
        c = cases[f'M06-{number:02}']
        q = c['queries'][0]
        before = q['expected_facts']
        assert len(before)==1 and re.search(r'202[23]年',before[0])
        after = [re.sub(r'202[23]年','',before[0])]
        changes.append({'case_id':c['id'],'query_id':q['id'],'before':before,'after':after,
                        'source_user_turns':[t['user'] for t in c['turns']],
                        'reason':'The authored label invents 2022/2023; the dialogue supplies relative year or only month/day. Retain the stated month/day. Check any generated absolute year against the recorded session date separately.'})
    c = cases['M11-07']
    q = c['queries'][0]
    assert q['expected_facts'][-1]=='患者症状好转后未再复发。'
    changes.append({'case_id':c['id'],'query_id':q['id'],'before':q['expected_facts'],
                    'after':q['expected_facts'][:-1]+['患者当时理疗后症状好转；第二次会话补充两年前曾复发。'],
                    'source_user_turns':[t['user'] for t in c['turns']],
                    'reason':'Second session explicitly reports recurrence. The all-time no-recurrence label contradicts the complete dialogue.'})
    frozen(HERE/'data/memory_label_corrections.json',{'base_dataset_sha256':digest(V2/'data/mem0_cases.jsonl'),
            'review':'Assistant source audit; not clinical expert review. All ten timeline cases corrected, including prior passes.',
            'interpretation':'Annotation repair, not a memory system improvement; old labels/results remain available.',
            'changes':changes})


def corrected_facts(case_id, query):
    from common import read
    corrections = read(HERE/'data/memory_label_corrections.json')['changes']
    match = next((c for c in corrections if c['case_id']==case_id and c['query_id']==query['id']),None)
    return match['after'] if match else query.get('expected_facts',[])


if __name__=='__main__':
    main()
