import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import evaluation.score_medical_holdout600 as scorer
from experiments.run_causal_fact_experiment import read,save,sha


class SemanticScoringTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.root_patch=patch.object(scorer,'ROOT',self.root);self.root_patch.start()
        self.row=dict(source_id='fixture',question='Which term?',options=dict(A='Alpha',B='Beta',C='Gamma',D='Delta'),reference='Alpha',reference_explanation='Source explanation.')
        hashes={}
        for arm in scorer.ARMS:
            path=self.root/'runs'/arm/'rows/fixture.json';save(path,dict(prediction='A'))
            hashes[str(path)]=sha(path)
        protocol=dict(prompt='Evaluate only the supplied answer.', replace_rubric=True, split_fields=True)
        save(self.root/'semantic_scoring_plan.json',dict(protocol=protocol,generation_hashes=hashes,maximum_calls=1,ids=['fixture']))

    def tearDown(self):
        self.root_patch.stop();self.temp.cleanup()

    def test_one_audit_per_unique_answer_and_completed_cache(self):
        def invoke(request,out,sid,tag):
            self.assertEqual(len(request['payload']['candidates']),1)
            self.assertIn('A. Alpha',request['payload']['question'])
            self.assertEqual(set(request['payload']['candidates'][0]),{'id','final_answer'})
            self.assertEqual(request['schema']['properties']['candidates']['items']['required'],['id','final_audit'])
            key=scorer.judge.g.judge.digest(dict(request=request,model=scorer.judge.g.MODEL,run_tag=tag))
            cid=request['payload']['candidates'][0]['id']
            return dict(reference_valid=True,reference_comment='',candidates=[dict(id=cid,final_audit=dict(assessment='fully_correct',summary='Correct.',findings=[]))]),key
        with patch.object(scorer.judge,'invoke',side_effect=invoke) as mock:
            scorer.score(self.row);scorer.score(self.row)
            self.assertEqual(mock.call_count,1)
        result=read(self.root/'semantic_scoring/judgments/fixture.json')
        self.assertTrue(all(v['answer_score']==2 for v in result['models'].values()))

    def test_uncertain_request_does_not_repeat_a_paid_call(self):
        with patch.object(scorer.judge,'invoke',side_effect=TimeoutError('unknown remote state')) as mock:
            with self.assertRaises(TimeoutError):scorer.score(self.row)
            with self.assertRaises(RuntimeError):scorer.score(self.row)
            self.assertEqual(mock.call_count,1)


if __name__=='__main__':unittest.main()
