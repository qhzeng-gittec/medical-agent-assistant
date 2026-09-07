import hashlib
import json
import sys
import unittest
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor

LAB_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB_DIR))

from native_reasoning import SYSTEM_PROMPT, missing_case_image_reason, parse_completion, source_explanation
from train_multitask_lora import MultitaskCollator
from data_io import load_jsonl
from evaluate_native_reasoning import ThinkingBudget
from original_images import load_model_image
from prepare_open_cases import CHOICE_REFERENCE, accepted_cases


class NativeReasoningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.processor = AutoProcessor.from_pretrained(LAB_DIR / "models" / "Qwen3.5-2B", do_resize=False)
        cls.collator = MultitaskCollator(cls.processor, SYSTEM_PROMPT)
        cls.data_dir = LAB_DIR / "data" / "native_reasoning"
        cls.rows = {
            s: load_jsonl(cls.data_dir / f"{s}.jsonl")
            for s in ("train", "validation", "test")
        }

    def test_native_reasoning_and_final_are_supervised(self):
        for task in ("vqa", "context", "case"):
            candidates = [r for r in self.rows["train"] if r["task"] == task]
            for row in (candidates[0], max(candidates, key=lambda r: r["supervised_tokens"])):
                with self.subTest(task=task, source=row["source_id"]):
                    batch = self.collator([row])
                    mask = batch["labels"][0] != -100
                    target = self.processor.tokenizer.decode(batch["labels"][0][mask], skip_special_tokens=False)
                    self.assertIn(row["reasoning_content"], target)
                    self.assertIn("</think>", target)
                    self.assertIn(row["target"], target)
                    self.assertNotIn("<evidence>", target)
                    self.assertNotIn("<answer>", target)
                    self.assertEqual(int(mask.sum()), row["supervised_tokens"])
                    self.assertTrue(torch.equal(batch["labels"][0][mask], batch["input_ids"][0][mask]))
                    self.assertTrue(parse_completion(target.replace("<|im_end|>", "").strip())["response_complete"])

    def test_image_dependency_boundary(self):
        cases = [
            ("This technetium-99m sulfur colloid scan was performed after abdominal pain.", True),
            ("Chest X-ray is shown below. What is the diagnosis?", True),
            ("MRI revealed a 5-cm mass in the left kidney. What is the diagnosis?", False),
            ("The lab reports given below pH = 7.2, HCO3 = 10, PCO2 = 30.", False),
            ("Proctitis may be shown by the presence of which of the following?", False),
        ]
        for question, rejected in cases:
            self.assertEqual(bool(missing_case_image_reason({"case_question": question, "options": {"A": "a"}})), rejected)
        self.assertEqual(missing_case_image_reason({"case_question": "Diagnosis?", "options": {"A": "image_question"}}), "missing_option_image")

    def test_mixed_batch_matches_individual_examples(self):
        collator = MultitaskCollator(self.processor, SYSTEM_PROMPT, image_max_edge=768)
        visual = [r for r in self.rows['train'] if r['task'] == 'vqa']
        text = [r for r in self.rows['train'] if r['task'] == 'context']
        examples = [text[0], visual[0], text[1], visual[-1]]
        mixed = collator(examples)
        singles = [collator([r]) for r in examples]
        for i, single in enumerate(singles):
            length = int(single['attention_mask'].sum())
            self.assertTrue(torch.equal(mixed['input_ids'][i, :length], single['input_ids'][0]))
            self.assertTrue(torch.equal(mixed['labels'][i, :length], single['labels'][0]))
            self.assertTrue((mixed['labels'][i, length:] == -100).all())
        for field in ('pixel_values', 'image_grid_thw'):
            self.assertTrue(torch.equal(mixed[field], torch.cat([s[field] for s in singles if field in s])))

    def test_optional_image_downscale_preserves_small_originals(self):
        path = LAB_DIR / 'data/vqa_rad_original/images/46bf5432d3296d99.png'
        scaled = load_model_image(path, 768)
        self.assertLessEqual(max(scaled.size), 768)
        with Image.open(path) as source:
            expected = source.convert('RGB')
            expected.thumbnail((768, 768), Image.Resampling.LANCZOS)
            self.assertEqual(scaled.crop((0, 0, *expected.size)).tobytes(), expected.tobytes())
        with __import__('tempfile').TemporaryDirectory() as directory:
            small = Path(directory) / 'small.png'
            Image.new('RGB', (70, 50), 'red').save(small)
            self.assertEqual(load_model_image(small, 768).crop((0, 0, 70, 50)).size, (70, 50))
            self.assertEqual(load_model_image(small, 768).size, (96, 64))

    def test_final_answer_is_natural_language_and_requires_thinking_closure(self):
        unfinished = parse_completion("A is unlikely; B is likely")
        self.assertEqual(unfinished["final_answer"], "")
        self.assertFalse(unfinished["response_complete"])
        for final in ("Answer: B", "The findings suggest pneumonia.", "This study supports an association."):
            parsed = parse_completion("Reasoning</think>" + final)
            self.assertEqual(parsed["final_answer"], final)
            self.assertTrue(parsed["response_complete"])

    def test_no_cross_split_source_or_image_leakage(self):
        for task in ("vqa", "context", "case"):
            for field in ("source_id", "source_group"):
                sets = [{r[field] for r in rows if r["task"] == task} for rows in self.rows.values()]
                for i in range(3):
                    for j in range(i + 1, 3):
                        self.assertFalse(sets[i] & sets[j])

    def test_complete_teacher_coverage_and_reasoning_only_recipe(self):
        annotation_dir = LAB_DIR / "data/reasoning_full"
        manifest = json.loads((annotation_dir / "manifest.json").read_text(encoding="utf-8"))
        self.assertTrue(manifest["complete"])
        annotations = {r["id"]: r for r in load_jsonl(annotation_dir / "annotations.jsonl")}
        open_annotations = {r["id"]: r for r in load_jsonl(LAB_DIR / "data/open_cases/annotations.jsonl")}
        self.assertEqual(set(open_annotations), {r["id"] for r in accepted_cases()})
        from complete_reasoning import source_items
        self.assertEqual(set(annotations), {r["id"] for r in source_items()})
        for rows in self.rows.values():
            for row in rows:
                annotation = (open_annotations if row["task"] == "case" else annotations)[row["source_id"]]
                self.assertEqual(annotation["status"], "keep")
                self.assertEqual(row["reasoning_content"], annotation["reasoning_content"])
                self.assertEqual(row["target"], annotation["final_answer"])
        recipe = load_jsonl(self.data_dir / "recipes/all_reasoning/train.jsonl")
        self.assertEqual({r["source_id"] for r in recipe}, {r["source_id"] for r in self.rows["train"]})
        self.assertEqual(len(recipe), len(self.rows["train"]))

    def test_cases_have_standalone_questions_and_text_references(self):
        import re
        seen_questions = set()
        for rows in self.rows.values():
            for row in rows:
                if row["task"] != "case":
                    continue
                text = row["user_text"] + "\n" + row["reasoning_content"] + "\n" + row["target"]
                with self.subTest(source=row["source_id"]):
                    self.assertNotIn(row["normalized_question"], seen_questions)
                    seen_questions.add(row["normalized_question"])
                    self.assertFalse(CHOICE_REFERENCE.search(text))
                    self.assertFalse(re.search(r"(?m)^\s*[A-D][.)]\s", row["user_text"]))
                    self.assertLess(len(re.findall(r"\b[a-d]\)", row["user_text"])), 2)
                    self.assertNotIn(row["reference"].strip(), ("A", "B", "C", "D", "ab", "ac", "ad"))
                    self.assertEqual(row["reference"], row["target"])

    def test_budget_covers_supervision_without_truncation(self):
        manifest = json.loads((self.data_dir / "manifest.json").read_text(encoding="utf-8"))
        for rows in self.rows.values():
            for row in rows:
                if row["has_reasoning_reference"]:
                    self.assertLess(row["supervised_tokens"], manifest["generation_max_new_tokens"][row["task"]])
                self.assertLess(row["final_tokens"], manifest["generation_final_reserve_tokens"][row["task"]])
        self.assertEqual(source_explanation("Ref: Book page 2. Explanation: Necessary clinical content."), "Ref: Book page 2. Explanation: Necessary clinical content.")

    def test_reasoning_budget_reserves_final_answer_without_changing_natural_closure(self):
        close_id = 8
        limiter = ThinkingBudget(prompt_length=2, reasoning_budget=3, close_token_id=close_id)
        scores = torch.zeros(1, 10)
        self.assertTrue(torch.equal(limiter(torch.tensor([[1, 2, 3, 4]]), scores.clone()), scores))
        forced = limiter(torch.tensor([[1, 2, 3, 4, 5]]), scores.clone())
        self.assertEqual(forced.argmax().item(), close_id)
        self.assertEqual(torch.isfinite(forced).sum().item(), 1)
        self.assertTrue(limiter.forced)
        self.assertTrue(torch.equal(limiter(torch.tensor([[1, 2, 3, 4, 5, close_id]]), scores.clone()), scores))
        natural = ThinkingBudget(2, 3, close_id)
        self.assertTrue(torch.equal(natural(torch.tensor([[1, 2, close_id, 4, 5]]), scores.clone()), scores))
        self.assertFalse(natural.forced)

    def test_original_image_pixels_and_processor_grid_are_preserved(self):
        path = LAB_DIR / "data/vqa_rad_original/images/46bf5432d3296d99.png"
        image = load_model_image(path)
        with Image.open(path) as source:
            source = source.convert("RGB")
            self.assertEqual(image.crop((0, 0, source.width, source.height)).tobytes(), source.tobytes())
            self.assertLess(image.width - source.width, 32)
            self.assertLess(image.height - source.height, 32)
        self.assertFalse(self.processor.image_processor.do_resize)
        batch = self.processor.image_processor(images=[image], return_tensors="pt")
        self.assertEqual(batch["image_grid_thw"][0].tolist(), [1, image.height // 16, image.width // 16])

    def test_all_saved_originals_match_source_pixels(self):
        directory = LAB_DIR / "data/vqa_rad_original"
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        for image_id, expected in manifest["images"].items():
            with self.subTest(image=image_id), Image.open(directory / "images" / f"{image_id}.png") as source:
                image = source.convert("RGB")
                self.assertEqual(image.size, (expected["width"], expected["height"]))
                self.assertEqual(hashlib.sha256(image.tobytes()).hexdigest(), expected["pixel_sha256"])


if __name__ == "__main__":
    unittest.main()
