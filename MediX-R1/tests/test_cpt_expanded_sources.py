import json
import sys
import tempfile
import unittest
from collections import Counter
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import experiments.mine_cpt_sources as mining
import experiments.run_cpt_coverage_experiment as experiment
import data_processing.download_cpt_sources as downloading
from data_processing.prepare_cpt_expanded_candidate import prose_spans, subdivide_spans


class ExpandedSourcesTests(unittest.TestCase):
    def test_subdivision_preserves_selected_tokens_and_gaps(self):
        selected = [(0, 4095), (9000, 11047)]
        spans = subdivide_spans(selected)
        self.assertTrue(all(0 < b-a <= 2047 for a,b in spans))
        self.assertEqual([i for a,b in spans for i in range(a,b)],
                         [i for a,b in selected for i in range(a,b)])

    def test_sentence_chunks_retain_all_content_without_overlap(self):
        text = 'First sentence. Second longer sentence. Final sentence.'
        encoded = dict(input_ids=list(range(len(text))), offset_mapping=[(i,i+1) for i in range(len(text))])
        spans = prose_spans(text, encoded, width=24)
        self.assertTrue(all(0 < b-a <= 24 for a,b in spans))
        self.assertEqual([i for a,b in spans for i in range(a,b)], list(range(len(text))))
        self.assertEqual(spans[0][1], len('First sentence. '))

    def test_exploratory_authorization_does_not_change_semantic_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            coverage = dict(ready_for_training=False, required=6435, accepted=0, corpus_manifest_sha256='fixed')
            plan = dict(exploratory=False, sft_train_sha256='fixed', sft_validation_sha256='fixed',
                        evaluation_sha256='fixed', baseline_adapter_sha256='fixed', corpus_manifest_sha256='fixed',
                        sft=dict(lora_scope='attention'), baseline_adapter_dir=str(experiment.BASELINE))
            manifest = dict(experiment_type='expanded_candidate_not_full_coverage', semantic_full_coverage_verified=False,
                            **{name+'_sha256':'fixed' for name in ['train','validation','documents','evaluation','candidate_map']})
            for name, data in [('coverage_status', coverage), ('plan', plan), ('manifest', manifest), ('audit', dict(status='passed'))]:
                (root/f'{name}.json').write_text(json.dumps(data), encoding='utf-8')
            with patch.object(experiment, 'ROOT', root), patch.object(experiment, 'DATA', root), patch.object(experiment, 'sha', return_value='fixed'):
                with patch.object(experiment, 'EXPLORATORY', False), self.assertRaisesRegex(ValueError, 'Full coverage'):
                    experiment.verify()
                plan['exploratory'] = True
                (root/'plan.json').write_text(json.dumps(plan), encoding='utf-8')
                with patch.object(experiment, 'EXPLORATORY', True):
                    experiment.verify()
                    manifest['semantic_full_coverage_verified'] = True
                    (root/'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
                    with self.assertRaisesRegex(ValueError, 'explicitly unverified'):
                        experiment.verify()
                self.assertEqual(json.loads((root/'coverage_status.json').read_text()), coverage)

    def test_download_resumes_after_broken_chunk_and_verifies_completed_partial(self):
        from unittest.mock import MagicMock
        import hashlib
        with tempfile.TemporaryDirectory() as directory, patch.object(downloading, 'ROOT', Path(directory)), patch.object(downloading.time, 'sleep'):
            root = Path(directory)
            item = dict(path='sample.jsonl', url='https://example.test/source', bytes=6, sha256=hashlib.sha256(b'abcdef').hexdigest())
            first, second = MagicMock(), MagicMock()
            for response in [first, second]:
                response.__enter__.return_value = response
            def broken_chunks(*args):
                yield b'abc'
                raise downloading.requests.exceptions.ChunkedEncodingError('connection interrupted')
            first.status_code = 200
            first.iter_content.side_effect = broken_chunks
            second.status_code = 206
            second.headers = {'Content-Range':'bytes 3-5/6'}
            second.iter_content.return_value = [b'def']
            with patch.object(downloading.requests, 'get', side_effect=[first, second]) as get:
                downloading.download(item)
                self.assertEqual(get.call_args.kwargs['headers']['Range'], 'bytes=3-')
            self.assertEqual((root/'sample.jsonl').read_bytes(), b'abcdef')
            completed = dict(item, path='complete.jsonl')
            (root/'complete.jsonl.partial').write_bytes(b'abcdef')
            with patch.object(downloading.requests, 'get') as get:
                downloading.download(completed)
                get.assert_not_called()

    def test_xml_keeps_prose_without_question_sections_or_duplicate_nested_text(self):
        xml = '<book-part><book-part-meta><title-group><title>Topic</title></title-group></book-part-meta><body><sec id="a"><title>Overview</title><p>'+('Supported medical explanation. '*5)+'</p><sec id="b"><title>Details</title><p>'+('Further explanatory detail. '*5)+'</p></sec></sec><sec id="q"><title>Review Questions</title><p>'+('Question to exclude. '*10)+'</p></sec></body></book-part>'
        rows = list(mining.xml_passages(xml, 'chapter.nxml'))
        self.assertEqual(len(rows), 2)
        self.assertEqual(len({r[0] for r in rows}), 2)
        self.assertTrue(all('Question to exclude' not in r[3] for r in rows))

    def test_question_queries_do_not_prioritize_case_age_and_vitals(self):
        row = dict(user_text='Medical question: A 50 year old with splenic injury and pressure 120.',
                   task='case', target='Observation.', reference='Observation.', reasoning_content='Stable splenic injury permits observation.')
        queries = mining.queries(row, Counter())
        self.assertNotIn('"120"', queries[0])
        self.assertNotIn('"50"', queries[0])
        self.assertIn('"splenic"', queries[0])

    def test_completed_input_is_not_indexed_twice(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(mining, 'ROOT', Path(directory)):
            root = Path(directory)
            folder = root/'textbooks/chunk'
            folder.mkdir(parents=True)
            path = folder/'sample.jsonl'
            path.write_text(json.dumps(dict(id='sample',title='Topic',content='Evidence prose.'))+'\n', encoding='utf-8')
            path.with_suffix('.jsonl.receipt.json').write_text(json.dumps(dict(actual_sha256='fixed')), encoding='utf-8')
            (root/'textbooks_status.json').write_text(json.dumps(dict(state='complete')), encoding='utf-8')
            mining.index_dataset('textbooks', False)
            mining.index_dataset('textbooks', False)
            with closing(mining.connect('textbooks')) as db:
                self.assertEqual(db.execute('SELECT count(*) FROM passages').fetchone()[0], 1)

    def test_training_gate_blocks_before_starting_subprocess(self):
        with patch.object(experiment, 'verify', side_effect=ValueError('coverage pending')), patch.object(experiment.subprocess, 'run') as run:
            with self.assertRaisesRegex(ValueError, 'coverage pending'):
                experiment.train()
            run.assert_not_called()

    def test_both_sft_arms_use_identical_attention_only_recipe(self):
        jobs = experiment.training_jobs()
        self.assertEqual([job[0] for job in jobs], ['cpt', 'sft', 'sft_only'])
        adapted, baseline = jobs[1][1], jobs[2][1]
        self.assertEqual(adapted[adapted.index('--lora-scope')+1], 'attention')
        self.assertEqual(baseline[baseline.index('--lora-scope')+1], 'attention')
        self.assertEqual(adapted[:adapted.index('--output-root')], baseline[:baseline.index('--output-root')])
        self.assertEqual(adapted[adapted.index('--base-adapter-dir')+1], str(experiment.CPT))
        self.assertNotIn('--base-adapter-dir', baseline)
        self.assertNotIn('attention_ffn', str(jobs[1:]))

    def test_adoption_waits_for_existing_cpt_without_starting_another_trainer(self):
        from unittest.mock import MagicMock
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cpt = root/'cpt_seed42/final_adapter'
            cpt.mkdir(parents=True)
            (cpt/'adapter_model.safetensors').touch()
            (cpt/'adapter_config.json').write_text('{}')
            (cpt.parent/'run_config.json').write_text(json.dumps(dict(expected_steps=1252)))
            process = MagicMock()
            process.cmdline.return_value = ['python', '-u', '-m', 'training.cpt', '--output-dir', str(cpt.parent)]
            def finish():
                (cpt.parent/'training_complete.json').write_text(json.dumps(dict(probe_only=False, global_step=1252)))
            process.wait.side_effect = finish
            with patch.object(experiment, 'ROOT', root), patch.object(experiment, 'CPT', cpt), patch.object(experiment.psutil, 'Process', return_value=process), patch.object(experiment.subprocess, 'run') as run:
                experiment.adopt_training(123, 'cpt')
                process.wait.assert_called_once()
                run.assert_not_called()
                process.cmdline.return_value = ['python', 'unrelated.py']
                with self.assertRaisesRegex(ValueError, 'does not belong'):
                    experiment.adopt_training(123, 'cpt')


if __name__ == '__main__':
    unittest.main()
