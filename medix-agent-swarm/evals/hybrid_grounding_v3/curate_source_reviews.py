"""Explicit adjudication of inspected translation-review contradictions, before retrieval."""
from common import RESULTS, V2, digest, frozen, lines, read, write


def main():
    inspected = ['MLP-1250','MLP-211','MLP-3873','MLP-5709','MLP-6033','MLP-608','MLP-6206','MLP-4370','MLP-4055']
    corpus = {d['id']:d for d in lines(V2/'data/corpus.jsonl')}
    splits = {s['id']:s['split'] for s in read(RESULTS/'authoring/plan.json')['sources']}
    records = []
    for id_ in inspected:
        path = RESULTS/'authoring/codex_issues'/(id_+'.json')
        issue = read(path)
        value,review = issue['candidate'],issue['review']
        assert review['positive_supported'] is True and review['missing_absent'] is True
        assert review['translation_equivalent'] is False
        assert value['evidence'] in corpus[id_]['content']
        note = {'source_id':id_,'review_record_sha256':digest(path),'decision':'accept unchanged authored questions',
                'reviewer':'assistant source/translation audit, not clinical expert review',
                'reason':'The displayed Chinese and English questions ask the same information need and neither embeds the answer. The reviewer either supplied no reason for its false translation flag or confused the answer being present in the source/evidence with it being present in the question. Original review and unchanged questions are retained.',
                'phase':'before freezing extension and before its retrieval outcomes'}
        records.append(note)
        write(RESULTS/'authoring/items'/(id_+'.json'),{'source_id':id_,'split':splits[id_],'value':value,
              'author_model':'gpt-5.5','review_model':'minimax/minimax-m2.5','review':review,'adjudication':note})
    id_ = 'MLP-1608'
    path = RESULTS/'authoring/codex_issues'/(id_+'.json')
    before = read(path)['candidate']
    value = dict(before,positive_en='How does mucus buildup in chronic bronchitis affect breathing?',
                 positive_zh='慢性支气管炎的黏液积聚会怎样影响呼吸？',
                 evidence='The irritation of the tubes causes mucus to build up. This mucus and the swelling of the tubes make it harder for your lungs to move oxygen in and carbon dioxide out of your body.',
                 expected_answer='Mucus buildup and swelling of the bronchial tubes make it harder for the lungs to move oxygen in and carbon dioxide out of the body.')
    assert value['evidence'] in corpus[id_]['content']
    note = {'source_id':id_,'review_record_sha256':digest(path),'decision':'narrow the question and restore a contiguous source excerpt',
            'reviewer':'assistant source/translation audit, not clinical expert review','before':before,'after':value,
            'reason':'The author concatenated nonadjacent passages. Keep the mucus/breathing question and its exact contiguous source evidence; remove the additional COPD expansion subquestion.',
            'phase':'before freezing extension and before its retrieval outcomes'}
    records.append(note)
    write(RESULTS/'authoring/items'/(id_+'.json'),{'source_id':id_,'split':splits[id_],'value':value,
          'author_model':'gpt-5.5','review_model':'assistant source audit after verbatim validation failure','adjudication':note})
    frozen(RESULTS/'authoring/manual_adjudications_v2.json',records)


if __name__=='__main__':
    main()
