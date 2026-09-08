import json
import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch

from data_processing.prepare_medical_holdout600 import ROOT,DATA,normalize
from experiments.run_causal_fact_experiment import read,sha
from experiments.run_medical_holdout600 import parse_letter,prompt
import experiments.analyze_medical_holdout600 as analysis


class HoldoutTests(unittest.TestCase):
    def test_frozen_population_and_label_integrity(self):
        if not (DATA/'evaluation.jsonl').exists():
            self.skipTest('Requires the locally prepared 600-question evaluation data.')
        plan=read(ROOT/'data_plan.json')
        rows=[json.loads(line) for line in (DATA/'evaluation.jsonl').read_text(encoding='utf-8').splitlines()]
        self.assertEqual(sha(DATA/'evaluation.jsonl'),plan['evaluation_sha256'])
        self.assertEqual(len(rows),600)
        self.assertEqual(len({r['source_id'] for r in rows}),600)
        self.assertEqual(len({normalize(r['question']) for r in rows}),600)
        for row in rows:self.assertEqual(row['reference'],row['options'][row['answer_label']])

    def test_explanations_are_not_supplied_to_model(self):
        row=dict(question='Fixture question?',options=dict(A='Alpha',B='Beta',C='Gamma',D='Delta'),reference_explanation='SECRET EXPLANATION',answer_label='B')
        rendered=prompt(row)
        self.assertNotIn('SECRET',rendered)
        self.assertEqual(rendered,'Fixture question?\n\nA. Alpha\nB. Beta\nC. Gamma\nD. Delta')

    def test_letter_extraction_rejects_incidental_letters(self):
        self.assertEqual(parse_letter('B'),('B','B'))
        self.assertEqual(parse_letter('Answer: C. Explanation'),(None,'C'))
        self.assertEqual(parse_letter('**D**'),(None,'D'))
        self.assertEqual(parse_letter('Because vitamin A is involved'),(None,None))
        self.assertEqual(parse_letter('CD21'),(None,None))
        self.assertEqual(parse_letter('The answer may be A or B'),(None,None))
        self.assertEqual(parse_letter('A patient has vitamin deficiency'),(None,None))
        self.assertEqual(parse_letter('A or B'),(None,None))

    def test_paired_analysis_preserves_repairs_and_regressions(self):
        from experiments.run_causal_fact_experiment import save
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);data=root/'data';data.mkdir()
            rows=[dict(source_id=f'fixture-{i}',subject='Fixture',answer_label='A') for i in range(600)]
            dataset=data/'evaluation.jsonl'
            dataset.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
            save(root/'evaluation_plan.json',dict(data_sha256=sha(dataset),comparisons=[['before','after']]))
            save(root/'pre_generation_amendment.json',dict(exclude_from_sensitivity=['fixture-0','fixture-300']))
            save(root/'input_quality_review.json',dict(missing_input_exclusions=['fixture-1','fixture-301','fixture-590','fixture-599']))
            for arm in ['before','after']:
                for i,row in enumerate(rows):
                    correct=i<300
                    if arm=='after':correct=5<=i<320
                    save(root/'runs'/arm/'rows'/f"{row['source_id']}.json",dict(**row,reference_label='A',
                        ranked_correct=correct,free_correct=correct,strict_correct=correct,explicit_letter='A' if correct else 'B'))
                save(root/'runs'/arm/'complete.json',dict(n=600))
            with patch.object(analysis,'ROOT',root),patch.object(analysis,'DATA',data):analysis.analyze()
            result=read(root/'results.json')['comparisons'][0]
            self.assertEqual((result['repaired'],result['regressed'],result['net']),(20,5,15))
            self.assertAlmostEqual(result['delta_pp'],2.5)
            self.assertEqual(len(result['repaired_ids']),20)
            self.assertEqual(result['phrase_overlap_sensitivity']['n'],598)
            self.assertEqual(result['phrase_overlap_sensitivity']['repaired'],19)
            self.assertEqual(result['phrase_overlap_sensitivity']['regressed'],4)
            self.assertEqual(result['input_complete_posthoc']['n'],596)
            self.assertEqual(result['input_complete_without_phrase_overlap']['n'],594)


if __name__=='__main__':unittest.main()
