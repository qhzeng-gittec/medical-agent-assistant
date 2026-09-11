from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load(name: str) -> dict:
    with (ROOT / name).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


manifest = load("manifest.json")
for relative, expected in manifest["files"].items():
    path = ROOT / relative
    assert path.stat().st_size == expected["bytes"], relative
    assert sha256(path) == expected["sha256"], relative

scale = load("sft-scale/results.json")
assert scale["expected_independent_questions"] == 1200
assert scale["model_responses"] == 8400
assert scale["automatic_judgments"] == 1199

medical = scale["summary"]["medical"]
general = scale["summary"]["general"]
expected_medical = {
    "base": 246,
    "cpt": 260,
    "old_cpt_sft": 285,
    "cpt_5k": 292,
    "cpt_20k": 300,
    "cpt_5k_repeated": 288,
    "base_20k": 302,
}
expected_general = {
    "base": 249,
    "cpt": 269,
    "old_cpt_sft": 220,
    "cpt_5k": 321,
    "cpt_20k": 326,
    "cpt_5k_repeated": 324,
    "base_20k": 324,
}
for arm, count in expected_medical.items():
    assert medical[arm]["answer_score"]["fully_correct"] == count, arm
    assert medical[arm]["answer_score"]["denominator"] == 599, arm
for arm, count in expected_general.items():
    assert general[arm]["answer_score"]["fully_correct"] == count, arm
    assert general[arm]["answer_score"]["denominator"] == 600, arm

keyword = load("cpt/keyword_experiment.json")
assert keyword["corpus"]["train_tokens"] == 8_081_489
expected_ranked = {"base": 305, "cpt": 325, "sft_only": 293, "cpt_sft": 323}
for arm, count in expected_ranked.items():
    assert keyword["holdout600"]["summaries"][arm]["ranked_correct"] == count, arm

stage = load("cpt/stage_results.json")
assert stage["reference_valid_questions"] == 590
expected_content = {"base": 220, "cpt": 212, "sft_only": 228, "high_lr": 199, "low_lr": 178}
for arm, count in expected_content.items():
    assert stage["metrics"][arm]["fully_correct"] == count, arm

print(f"verified {len(manifest['files'])} files and all published headline counts")
