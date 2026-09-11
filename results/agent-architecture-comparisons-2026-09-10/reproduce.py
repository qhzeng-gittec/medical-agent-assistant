"""Verify hashes and headline counts for the public Agent architecture studies."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parent


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


manifest = load("manifest.json")
for relative, expected in manifest["files"].items():
    path = ROOT / relative
    assert path.is_file(), relative
    assert sha256(path.read_bytes()) == expected, relative

with zipfile.ZipFile(ROOT / "evidence.zip") as archive:
    names = set(archive.namelist())
    assert names == set(manifest["archive_entries"])
    for name, hashes in manifest["archive_entries"].items():
        assert sha256(archive.read(name)) == hashes["published_sha256"], name

architecture = load("architecture/analysis.json")
expected_architecture = {
    "single": (24, 24, 21, 18, 49.15283333339418),
    "single_review": (24, 24, 20, 15, 81.2114583333799),
    "swarm": (24, 24, 22, 17, 77.18808333335134),
    "swarm_tasks": (24, 24, 21, 19, 81.22837499994785),
    "swarm_tasks_serial": (12, 12, 12, 8, 69.18383333350842),
}
for arm, expected in expected_architecture.items():
    group = architecture["groups"][arm]
    actual = (
        group["planned"],
        group["completed"],
        group["valid_grade"],
        group["passed"],
        group["mean_seconds"],
    )
    assert actual == expected, (arm, actual)
assert architecture["groups"]["swarm"]["duplicate_rejections"] == 9
assert architecture["groups"]["swarm_tasks"]["duplicate_rejections"] == 0

coverage = load("coverage/combined_analysis.json")
single = coverage["groups"]["single_audit"]
swarm = coverage["groups"]["coverage_swarm"]
assert (single["planned"], single["exact"], single["input_tokens"]) == (32, 13, 2_486_128)
assert (swarm["planned"], swarm["exact"], swarm["input_tokens"]) == (32, 13, 627_044)
assert round((1 - swarm["seconds"] / single["seconds"]) * 100, 1) == 41.3
assert round((1 - swarm["input_tokens"] / single["input_tokens"]) * 100, 1) == 74.8
assert swarm["parallel_runs"] == 24

followup = load("coverage/fixed_test_analysis.json")
assert followup["groups"]["single_audit"]["exact"] == 2
assert followup["groups"]["coverage_swarm"]["exact"] == 3

print(
    "verified "
    f"{len(manifest['files'])} files, {len(manifest['archive_entries'])} evidence entries, "
    "108 architecture executions and 128 coverage executions"
)
