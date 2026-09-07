import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.evaluate_native_reasoning import GeneratedPresencePenalty, qwen_sampling_parameters
from common.calibration_stopping import RepetitionStop, repeated_line


class CalibrationTests(unittest.TestCase):
    def test_repetition_requires_six_long_lines(self):
        line = 'This is a sufficiently long sentence about the same image finding.'
        self.assertFalse(repeated_line('\n'.join([line] * 5)))
        self.assertTrue(repeated_line('\n'.join([line, '  ' + line, line, line, line, line])))
        self.assertFalse(repeated_line('Yes\n' * 100))

    def test_repetition_ignores_prompt_and_uses_recent_window(self):
        class Tokenizer:
            def decode(self, ids, **kwargs):
                self.ids = ids
                return ('A sufficiently long repeated sentence describing the same finding.\n' * 6
                        if 9 in ids else 'No repetition')

        tokenizer = Tokenizer()
        stop = RepetitionStop(tokenizer, 512)
        self.assertFalse(stop(torch.tensor([[9] * 512 + [1] * 512]), None))
        self.assertFalse(stop(torch.tensor([[1] * 512 + [9] * 511]), None))
        self.assertTrue(stop(torch.tensor([[1] * 512 + [9] * 512]), None))
        stop = RepetitionStop(tokenizer, 512)
        self.assertFalse(stop(torch.tensor([[9] * 512 + [9] * 512 + [1] * 2048]), None))
        self.assertEqual(len(tokenizer.ids), 2048)

    def test_presence_penalty_counts_once_and_excludes_prompt(self):
        ids = torch.tensor([[1, 2, 3, 3, 4], [4, 1, 2, 2, 2]])
        scores = torch.zeros((2, 6))
        actual = GeneratedPresencePenalty(2, 1.5)(ids, scores)
        self.assertTrue(torch.equal(actual, torch.tensor([[0., 0., 0., -1.5, -1.5, 0.], [0., 0., -1.5, 0., 0., 0.]])))
        self.assertTrue(torch.equal(scores, torch.zeros_like(scores)))

    def test_no_generated_tokens_have_no_penalty(self):
        scores = torch.ones((1, 6))
        self.assertTrue(torch.equal(GeneratedPresencePenalty(2, 2.)(torch.tensor([[1, 2]]), scores), scores))

    def test_official_mode_specific_profiles(self):
        self.assertEqual(qwen_sampling_parameters(True, True), dict(temperature=.6, top_p=.95, top_k=20, presence_penalty=0.))
        self.assertEqual(qwen_sampling_parameters(True, False)['presence_penalty'], 1.5)
        self.assertEqual(qwen_sampling_parameters(False, True)['top_p'], .8)
        self.assertEqual(qwen_sampling_parameters(False, False)['presence_penalty'], 2.)


if __name__ == '__main__':
    unittest.main()
