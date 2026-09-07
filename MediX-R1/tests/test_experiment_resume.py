import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from filelock import Timeout

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_knowledge_experiments as queue


class ExperimentResumeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.root_patch = patch.object(queue, 'ROOT', self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        self.state = queue.initial_state()

    def checkpoint(self, job, step):
        path = queue.job_output(job) / f'checkpoint-{step}'
        path.mkdir(parents=True)
        for name in ('adapter_model.safetensors', 'adapter_config.json', 'optimizer.pt',
                     'scheduler.pt', 'rng_state.pth', 'training_args.bin'):
            (path / name).write_bytes(b'test')
        (path / 'trainer_state.json').write_text(json.dumps({'global_step': step}))
        return path

    def test_resume_uses_latest_complete_checkpoint_and_keeps_hyperparameters(self):
        job = self.state['jobs'][2]
        self.checkpoint(job, 250)
        expected = self.checkpoint(job, 500)
        incomplete = expected.parent / 'checkpoint-700'
        incomplete.mkdir()
        (incomplete / 'trainer_state.json').write_text('{}')
        command = queue.prepare_command(job)
        self.assertEqual(command[command.index('--resume-from-checkpoint') + 1], str(expected))
        self.assertEqual(command[command.index('--batch-size') + 1], '2')
        self.assertEqual(command[command.index('--gradient-accumulation-steps') + 1], '2')
        self.assertEqual(command[command.index('--gpu-memory-fraction') + 1], '0.8')

    def test_stale_resume_argument_is_replaced(self):
        job = self.state['jobs'][2]
        expected = self.checkpoint(job, 500)
        job['command'] += ['--resume-from-checkpoint', 'old', '--gpu-memory-fraction', '1']
        command = queue.prepare_command(job)
        self.assertEqual(command.count('--resume-from-checkpoint'), 1)
        self.assertEqual(command[-1], str(expected))
        self.assertEqual(command.count('--gpu-memory-fraction'), 1)

    def test_dry_run_does_not_change_saved_state_or_outputs(self):
        queue.save_state(self.state)
        before = (self.root / 'queue_status.json').read_bytes()
        with patch.object(queue, 'assert_no_active_job'), patch('builtins.print'):
            queue.run_queue(copy.deepcopy(self.state), dry_run=True)
        self.assertEqual((self.root / 'queue_status.json').read_bytes(), before)

    def test_completed_jobs_are_skipped_and_logs_preserved(self):
        self.state['jobs'] = self.state['jobs'][:2]
        self.state['jobs'][0]['status'] = 'complete'
        old_log = self.root / 'old.log'
        old_log.write_text('previous attempt')
        self.state['jobs'][1]['log'] = str(old_log)
        process = Mock(pid=1234)
        process.wait.return_value = 0
        with patch.object(queue, 'assert_no_active_job'), patch.object(queue.subprocess, 'Popen', return_value=process) as launch:
            queue.run_queue(self.state)
        launch.assert_called_once()
        self.assertEqual(self.state['status'], 'complete')
        self.assertEqual(old_log.read_text(), 'previous attempt')
        self.assertNotEqual(self.state['jobs'][1]['log'], str(old_log))

    def test_failure_stops_queue_and_records_exit_code(self):
        self.state['jobs'] = self.state['jobs'][:2]
        process = Mock(pid=1234)
        process.wait.return_value = 7
        with patch.object(queue, 'assert_no_active_job'), patch.object(queue.subprocess, 'Popen', return_value=process) as launch:
            with self.assertRaises(queue.subprocess.CalledProcessError):
                queue.run_queue(self.state)
        launch.assert_called_once()
        saved = json.loads((self.root / 'queue_status.json').read_text())
        self.assertEqual(saved['status'], 'failed')
        self.assertEqual(saved['jobs'][0]['returncode'], 7)
        self.assertEqual(saved['jobs'][1]['status'], 'pending')

    def test_completed_artifact_recovers_stale_job_status(self):
        job = self.state['jobs'][2]
        self.state['jobs'] = [job]
        output = queue.job_output(job)
        output.mkdir(parents=True)
        (output / 'training_complete.json').write_text('{"global_step":730}')
        with patch.object(queue, 'assert_no_active_job'), patch.object(queue.subprocess, 'Popen') as launch:
            queue.run_queue(self.state)
        launch.assert_not_called()
        self.assertEqual(job['status'], 'complete')

    def test_partial_evaluation_is_preserved_before_retry(self):
        job = self.state['jobs'][6]
        output = queue.job_output(job)
        output.mkdir(parents=True)
        (output / 'vqa_generations.jsonl').write_text('partial results')
        queue.archive_partial_output(job)
        self.assertFalse(output.exists())
        archive = Path(job['preserved_partial_outputs'][0])
        self.assertEqual((archive / 'vqa_generations.jsonl').read_text(), 'partial results')
        self.assertIn('--gpu-memory-fraction', queue.prepare_command(job))

    def test_refuses_to_move_outputs_outside_experiment(self):
        job = self.state['jobs'][6]
        command = job['command']
        command[command.index('--output-root') + 1] = str(self.root.parent)
        with patch.object(Path, 'exists', return_value=True):
            with self.assertRaises(ValueError):
                queue.archive_partial_output(job)

    def test_live_orphan_blocks_duplicate_job(self):
        process = Mock(pid=4321)
        process.info = {'cmdline': self.state['jobs'][2]['command']}
        with patch.object(queue.psutil, 'process_iter', return_value=[process]):
            with self.assertRaisesRegex(RuntimeError, '4321'):
                queue.assert_no_active_job()

    def test_second_queue_cannot_acquire_running_queue_lock(self):
        path = self.root / 'queue.lock'
        with queue.FileLock(path, timeout=0):
            with self.assertRaises(Timeout):
                with queue.FileLock(path, timeout=0):
                    self.fail('Duplicate queue acquired the lock')


if __name__ == '__main__':
    unittest.main()
