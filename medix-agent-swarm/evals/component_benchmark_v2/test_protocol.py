from evaluate_rag import metrics, score
from evaluate_memory import anchor_coverage
from grade_memory import validate_grades
import pytest


def row(kind, gold, ranking):
    return {'case':{'kind':kind,'gold_groups':gold},'ranking':[{'id':i,'score':s} for i,s in ranking]}


def test_partial_multisource_is_not_complete_coverage():
    r=row('multi_evidence',[['a','a2'],['b']],[('a2',.8),('c',.7),('b',.4)])
    assert score(r,3,.5)=={'any_hit':True,'all_hit':False,'empty':False}
    assert score(r,3,.3)['all_hit']


def test_context_controls_do_not_inflate_answerable_denominator():
    rows=[row('answerable',[['a']],[('a',.8)]),row('needs_history',[['a']],[('c',.9)]),
          row('unanswerable',[],[('x',.2)])]
    m=metrics(rows,3,.3)
    assert m['positive_queries']==1 and m['negative_queries']==1
    assert m['balanced_objective']==1


def test_anchor_alternatives_are_literal_not_regex():
    assert anchor_coverage([['a.b']], [{'content':'axb'}])==[False]
    assert anchor_coverage([['哮喘','asthma']], [{'content':'Asthma'}])==[True]


def test_judge_cannot_borrow_evidence_from_another_item():
    inputs=[{'id':'a','expected_facts':['no leave'],'memories':[{'memory_id':'m0','content':'no cold'}]}]
    response={'items':[{'id':'a','facts':[{'supported':True,'evidence':[{'memory_id':'m3'}]}]}]}
    with pytest.raises(ValueError,match='unknown memory'):
        validate_grades(response,inputs)


def test_judge_must_preserve_expected_fact_count():
    inputs=[{'id':'a','expected_facts':['started running','stopped running'],'memories':[{'memory_id':'m0','content':'history'}]}]
    response=[{'id':'a','facts':[{'supported':True,'evidence':[{'memory_id':'m0'}]}]}]
    with pytest.raises(ValueError,match='fact count mismatch'):
        validate_grades(response,inputs)
