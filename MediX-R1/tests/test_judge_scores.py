import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_judge import summarize_judgments


class JudgeScoreTests(unittest.TestCase):
    def test_percentage_and_unjudgeable_denominator(self):
        metrics = summarize_judgments([
            {"score": 2, "verdict": "correct"}, {"score": 1, "verdict": "partial"},
            {"score": 0, "verdict": "incorrect"}, {"score": None, "verdict": "unjudgeable"},
        ])
        self.assertEqual(metrics["score_0_to_100"], 50)
        self.assertEqual(metrics["judged_samples"], 3)
        self.assertEqual(metrics["unjudgeable_samples"], 1)
        self.assertEqual(metrics["samples"], 4)

    def test_all_unjudgeable_has_no_score(self):
        self.assertIsNone(summarize_judgments([{"score": None, "verdict": "unjudgeable"}])["score_0_to_100"])

    def test_example_twenty_questions_scores_72_point_5(self):
        rows = [{"score": 2, "verdict": "correct"}]*12 + [{"score": 1, "verdict": "partial"}]*5 + [{"score": 0, "verdict": "incorrect"}]*3
        self.assertEqual(summarize_judgments(rows)["score_0_to_100"], 72.5)


if __name__ == "__main__":
    unittest.main()
