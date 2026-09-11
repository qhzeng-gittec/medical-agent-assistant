"""Build a compact, auditable publication from the two 2026-09-10 Agent studies."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil
import zipfile


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path(__file__).resolve().parents[3]
OUTPUT = REPO_ROOT / "results" / "agent-architecture-comparisons-2026-09-10"
ARCH_SOURCE = SOURCE_ROOT / "medix-agent-swarm" / "evals" / "results" / "architecture_value_20260910"
COVERAGE_SOURCE = SOURCE_ROOT / "medix-agent-swarm" / "evals" / "results" / "coverage_study_20260910"
SECRET_PATTERN = re.compile(rb"(?:sk-(?:or-v1-)?[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9_]{20,})")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def scrub_string(value: str) -> str:
    variants = {
        str(SOURCE_ROOT),
        SOURCE_ROOT.as_posix(),
        str(SOURCE_ROOT).replace("\\", "/"),
    }
    for prefix in sorted(variants, key=len, reverse=True):
        value = value.replace(prefix, "[SOURCE_ROOT]")
    return value


def sanitize(value):
    if isinstance(value, dict):
        return {scrub_string(str(key)): sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, str):
        return scrub_string(value)
    return value


def json_bytes(path: Path) -> bytes:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    return (json.dumps(sanitize(value), ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def jsonl_bytes(path: Path) -> bytes:
    rows = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            rows.append(json.dumps(sanitize(json.loads(line)), ensure_ascii=False))
    return (("\n".join(rows) + "\n") if rows else "").encode("utf-8")


def write_public_json(source: Path, destination: Path, source_hashes: dict[str, str]) -> None:
    data = json_bytes(source)
    if SECRET_PATTERN.search(data):
        raise ValueError(f"potential secret in {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    source_hashes[destination.relative_to(OUTPUT).as_posix()] = sha256(source.read_bytes())


def archive_bytes(path: Path) -> bytes:
    if path.suffix == ".json":
        return json_bytes(path)
    if path.suffix == ".jsonl":
        return jsonl_bytes(path)
    return scrub_string(path.read_text(encoding="utf-8-sig")).encode("utf-8")


def add_archive_file(
    archive: zipfile.ZipFile,
    source: Path,
    archive_name: str,
    archive_manifest: dict[str, dict[str, str]],
) -> None:
    data = archive_bytes(source)
    if SECRET_PATTERN.search(data):
        raise ValueError(f"potential secret in {source}")
    info = zipfile.ZipInfo(archive_name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    archive_manifest[archive_name] = {
        "source_sha256": sha256(source.read_bytes()),
        "published_sha256": sha256(data),
    }


def selected_run_files(root: Path, phases: tuple[str, ...], kinds: set[str]):
    for phase in phases:
        phase_root = root / phase
        for run in sorted(path for path in phase_root.iterdir() if path.is_dir()):
            for path in sorted(run.rglob("*")):
                if not path.is_file():
                    continue
                relative = path.relative_to(run)
                if path.name in kinds or relative.parts[0] == "ledgers":
                    yield phase, run.name, relative, path


def main() -> None:
    if REPO_ROOT not in OUTPUT.parents:
        raise ValueError(f"output escaped repository: {OUTPUT}")
    if not ARCH_SOURCE.is_dir() or not COVERAGE_SOURCE.is_dir():
        raise FileNotFoundError("source Agent evaluation directories are missing")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    cache = OUTPUT / "__pycache__"
    if cache.exists():
        shutil.rmtree(cache)
    for name in ("architecture", "coverage"):
        target = OUTPUT / name
        if target.exists():
            shutil.rmtree(target)
    for name in ("evidence.zip", "manifest.json"):
        target = OUTPUT / name
        if target.exists():
            target.unlink()

    source_hashes: dict[str, str] = {}
    architecture_files = (
        "analysis.json",
        "test.json",
        "design.json",
        "test_freeze.json",
        "qualitative_review.json",
        "final_verification.json",
    )
    coverage_files = (
        "combined_analysis.json",
        "fixed_test_analysis.json",
        "case_evidence.json",
        "test.json",
        "confirm.json",
        "fixed_test.json",
        "design.json",
        "fixed_protocol.json",
        "final_audit.json",
        "verification.json",
    )
    for name in architecture_files:
        write_public_json(ARCH_SOURCE / name, OUTPUT / "architecture" / name, source_hashes)
    for name in coverage_files:
        write_public_json(COVERAGE_SOURCE / name, OUTPUT / "coverage" / name, source_hashes)

    archive_manifest: dict[str, dict[str, str]] = {}
    archive_path = OUTPUT / "evidence.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        architecture_kinds = {"prompt.json", "result.json", "trace.json", "grade_0.json", "grade_1.json"}
        for phase, run, relative, source in selected_run_files(
            ARCH_SOURCE, ("test",), architecture_kinds
        ):
            name = Path("architecture", phase, run, relative).as_posix()
            add_archive_file(archive, source, name, archive_manifest)

        coverage_kinds = {"result.json", "trace.json"}
        for phase, run, relative, source in selected_run_files(
            COVERAGE_SOURCE, ("test", "confirm", "fixed_test"), coverage_kinds
        ):
            name = Path("coverage", phase, run, relative).as_posix()
            add_archive_file(archive, source, name, archive_manifest)

    public_files = {
        path.relative_to(OUTPUT).as_posix(): sha256(path.read_bytes())
        for path in sorted(OUTPUT.rglob("*"))
        if path.is_file()
        and path.name != "manifest.json"
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    }
    manifest = {
        "scope": {
            "architecture": "96 primary executions plus 12 serial ablations",
            "coverage": "112 primary formal executions plus 16 frozen follow-up executions",
        },
        "normalization": "JSON formatting normalized and local source-root paths replaced with [SOURCE_ROOT].",
        "files": public_files,
        "source_sha256": source_hashes,
        "archive_entries": archive_manifest,
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "public_files": len(public_files),
                "archive_entries": len(archive_manifest),
                "archive_mib": round(archive_path.stat().st_size / 1024 / 1024, 2),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
