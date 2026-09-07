import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

LAB_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB_DIR))
from complete_reasoning import annotate_batch


class TeacherBatchTests(unittest.TestCase):
    def test_missing_record_is_repaired_without_regenerating_accepted_records(self):
        items = [{"id": key, "image_path": None, "question": "Example?", "reference_answer": "Example."} for key in ("a", "b")]
        replies = [{"annotations": [{"id": key, "status": "keep", "reasoning_content": f"Supported explanation {key}.", "final_answer": "Example.", "review_note": ""}]} for key in ("a", "b")]

        def teacher(prompt, images, schema, output_path, model):
            reply = replies.pop(0)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(reply), encoding="utf-8")
            return reply

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory).resolve()
            with patch("complete_reasoning.call_teacher", side_effect=teacher) as call:
                annotations = annotate_batch(items, output, "fixture")
                self.assertEqual({r["id"] for r in annotations}, {"a", "b"})
                self.assertEqual(call.call_count, 2)
                retry_prompt = call.call_args.args[0].split("\n\n", 1)[1]
                self.assertEqual([r["id"] for r in json.loads(retry_prompt)], ["b"])
                annotate_batch(items, output, "fixture")
                self.assertEqual(call.call_count, 2)
            audits = list((output / "incomplete_batches").glob("*.json"))
            self.assertEqual(len(audits), 1)
            self.assertEqual([r["id"] for r in json.loads(audits[0].read_text(encoding="utf-8"))["annotations"]], ["a"])


if __name__ == "__main__":
    unittest.main()
