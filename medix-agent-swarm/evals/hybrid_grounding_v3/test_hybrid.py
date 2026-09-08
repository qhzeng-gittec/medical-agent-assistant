import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'knowledge'))
from hybrid_retrieval import HybridIndex
from answers import has_invalid_citation, validate_grades
from prepare import source_quote
from retrieval import metrics


def test_zero_lexical_overlap_does_not_invent_candidates_or_change_dense_order():
    index = HybridIndex([{'id':'a','content':'aspirin dose'},{'id':'b','content':'kidney disease'}],np.eye(2))
    sparse = index.lexical_scores('完全无重叠')
    assert not index.rank(sparse,True).size
    order,_ = index.fuse([1,0],[],2)
    assert order.tolist()==[1,0]


def test_reciprocal_rank_fusion_uses_rank_not_raw_score():
    order,score = HybridIndex.fuse([0,1],[1,2],3,.5,10,2)
    assert order[0]==1
    assert score[1]==pytest.approx(.5/12+.5/11)


def test_bm25_rare_term_and_document_length():
    index = HybridIndex([{'content':'rare common'},{'content':'common common common common'}],np.eye(2))
    assert index.lexical_scores('rare')[0]>0
    assert index.lexical_scores('rare')[1]==0


def test_restore_xml_whitespace_never_changes_medical_words():
    assert source_quote('called atherosclerosis.','called atherosclerosis .')=='called atherosclerosis .'
    assert source_quote('called asthma.','called atherosclerosis .')=='called asthma.'


def verdict():
    return {'id':'a','context_sufficiency':'none','has_unsupported_claim':False,'unsupported_quote':'',
            'false_search_claim':True,'search_quote':'searched PubMed','acknowledges_limit':True,
            'answers_supported_part':False,'complete_correct':True,'reason':'test'}


def test_false_search_claim_cannot_pass_despite_abstention():
    value = validate_grades({'items':[verdict()]},[{'id':'a','answer':'I searched PubMed but found nothing.'}])[0]
    assert value['complete_correct'] is False


def test_judge_must_quote_actual_answer():
    with pytest.raises(ValueError,match='actual answer'):
        validate_grades({'items':[verdict()]},[{'id':'a','answer':'I cannot tell from the supplied material.'}])


def test_complete_retrieval_requires_every_group_and_excludes_partial_cases():
    rows = [{'case':{'family_id':'one','answerability':'answerable','gold_groups':[['a','a2'],['b']]},
             'rankings':{'dense':['a2','x','y','b']}},
            {'case':{'family_id':'two','answerability':'partial','gold_groups':[['a']]},'rankings':{'dense':['a']}}]
    result = metrics(rows,'dense')
    assert result['answerable_queries']==1
    assert result['complete']['3']==0
    assert result['complete']['5']==1


def test_invalid_vectors_fail_before_retrieval():
    with pytest.raises(ValueError,match='unit vectors'):
        HybridIndex([{'content':'test'}],[[0.,0.]])


def test_multiple_citations_validate_each_source_id():
    context = [{'id':'MLP-123'},{'id':'MLP-456'}]
    assert not has_invalid_citation('Supported [MLP-123, MLP-456].',context)
    assert has_invalid_citation('Invented [MLP-123, MLP-999].',context)
    assert has_invalid_citation('Invented memory [MEM-7].',context)
