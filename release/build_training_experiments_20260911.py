from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sanitize(value: Any, source_root: Path) -> Any:
    if isinstance(value, dict):
        return {sanitize(key, source_root): sanitize(item, source_root) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize(item, source_root) for item in value]
    if isinstance(value, str):
        normalized = value.replace("\\", "/")
        root = source_root.as_posix()
        return normalized.replace(root, "<EXPERIMENT_ROOT>")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(source_root: Path, repo_root: Path) -> None:
    output = repo_root / "results" / "training-scale-cpt-2026-09-11"
    output.mkdir(parents=True, exist_ok=True)

    scale_root = source_root / "outputs" / "sft_scale_reasoning_v1"
    scale_analysis = load_json(scale_root / "evaluation" / "reviewed_analysis.json")
    scale_public = {
        key: scale_analysis[key]
        for key in (
            "created_at_utc",
            "status",
            "automatic_scoring_complete",
            "automatic_judgments",
            "expected_independent_questions",
            "model_responses",
            "missing_automatic_judgments",
            "summary",
            "paired",
            "cpt20k_vs_cpt_medical_stratified_by_cpt_length_stop",
            "stratification_limit",
            "spot_review",
            "limitations",
        )
    }
    write_json(output / "sft-scale" / "results.json", sanitize(scale_public, source_root))
    write_json(
        output / "sft-scale" / "protocol.json",
        sanitize(load_json(scale_root / "experiment_plan.json"), source_root),
    )
    write_json(
        output / "sft-scale" / "data_plan.json",
        sanitize(load_json(scale_root / "final_data_plan.json"), source_root),
    )
    write_json(
        output / "sft-scale" / "spot_audit.json",
        sanitize(load_json(scale_root / "evaluation" / "spot_audit_review.json"), source_root),
    )
    write_json(
        output / "sft-scale" / "code_checks.json",
        sanitize(load_json(scale_root / "evaluation" / "code_audit_checks.json"), source_root),
    )

    corpus_root = source_root / "data" / "cpt_keyword_training_v1"
    write_json(
        output / "cpt" / "corpus_manifest.json",
        sanitize(load_json(corpus_root / "manifest.json"), source_root),
    )
    write_json(
        output / "cpt" / "corpus_audit.json",
        sanitize(load_json(corpus_root / "audit.json"), source_root),
    )
    write_json(
        output / "cpt" / "coverage_status.json",
        sanitize(load_json(corpus_root / "coverage_status.json"), source_root),
    )

    keyword_report = load_json(source_root / "outputs" / "cpt_keyword_experiment_v1" / "final_report.json")
    keyword_public = {
        key: keyword_report[key]
        for key in (
            "experiment_complete",
            "cpt_sft_training_complete",
            "corpus",
            "collected_candidates",
            "training",
            "holdout600",
            "rl_decision",
            "limitations",
        )
    }
    write_json(output / "cpt" / "keyword_experiment.json", sanitize(keyword_public, source_root))

    stage_root = source_root / "outputs" / "stage_explanation_control_v1" / "semantic_codex_gpt55"
    stage_analysis = load_json(stage_root / "stage_analysis.json")
    stage_public = {
        key: stage_analysis[key]
        for key in (
            "independent_questions",
            "reference_valid_questions",
            "common_eos_n",
            "reference_flags",
            "comparisons",
            "output_mechanisms",
            "limitations",
        )
        if key in stage_analysis
    }
    write_json(output / "cpt" / "stage_results.json", sanitize(load_json(stage_root / "results.json"), source_root))
    write_json(output / "cpt" / "stage_analysis.json", sanitize(stage_public, source_root))

    lr_root = source_root / "outputs" / "sft_lr_control_v1" / "semantic_codex_gpt55"
    write_json(output / "cpt" / "lr_results.json", sanitize(load_json(lr_root / "results.json"), source_root))
    lr_analysis = load_json(lr_root / "mechanism_analysis.json")
    lr_public = {
        key: lr_analysis[key]
        for key in ("metrics", "repairs_summary", "regressions_summary", "limitations")
        if key in lr_analysis
    }
    write_json(output / "cpt" / "lr_analysis.json", sanitize(lr_public, source_root))

    supplemental = {
        "scope": "Mechanism probes; not headline effect estimates.",
        "hparam_pilot": sanitize(
            load_json(source_root / "outputs" / "cpt_hparam_pilot_v1" / "results.json"), source_root
        ),
        "recall_diagnostic": sanitize(
            load_json(source_root / "outputs" / "cpt_recall_diagnostic_v1" / "reviewed_analysis.json"),
            source_root,
        ),
    }
    write_json(output / "cpt" / "supplemental_probes.json", supplemental)

    files = sorted(
        path
        for path in output.rglob("*")
        if path.is_file() and path.name not in {"README.md", "manifest.json", "reproduce.py"}
    )
    manifest = {
        "release": "training-scale-cpt-2026-09-11",
        "source_policy": "Curated completed results and audits; raw CPT prose excluded pending source-level redistribution review.",
        "files": {
            path.relative_to(output).as_posix(): {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in files
        },
    }
    write_json(output / "manifest.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()
    build(args.source_root.resolve(), args.repo_root.resolve())


if __name__ == "__main__":
    main()
