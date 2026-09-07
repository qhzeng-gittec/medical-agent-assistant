"""Explicit-file, exact/embedded-text source audit; not semantic contamination detection."""

import argparse
import hashlib
import json
import unicodedata
from pathlib import Path

from dataset_tools import DEFAULT_DATASET, read_jsonl


def normalize(value: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKC", value).casefold() if c.isalnum())


def audit(root: Path, paths: list[Path]) -> dict:
    sources = read_jsonl(root / "private/source_records.jsonl")
    needles = {s["source_id"]: normalize(s["record"].get("question", s["record"].get("description", "")))
               for s in sources}
    fields = ("user_text", "question", "description", "normalized_question", "original_normalized_question")
    matches, coverage = [], []
    for path in paths:
        count = 0
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for count, line in enumerate(handle, 1):
                digest.update(line)
                row = json.loads(line)
                values = [normalize(row[field]) for field in fields if isinstance(row.get(field), str)]
                for source_id, needle in needles.items():
                    if any(needle in value for value in values):
                        matches.append({"source_id": source_id, "file": str(path.resolve()),
                                        "line": count, "training_source_id": row.get("source_id")})
        coverage.append({"path": str(path.resolve()), "rows": count, "sha256": digest.hexdigest()})
    return {"method": "NFKC+casefold+alphanumeric; source question/description contained in selected fields",
            "checked_fields": list(fields), "sources": len(sources), "files": coverage,
            "matches": matches, "match_count": len(matches),
            "limitations": ["Only these explicitly supplied files; no other historical training files checked.",
                            "No paraphrase/translation/semantic matching, no pretraining contamination claims.",
                            "Public questions may already be known to base models; all cases remain dev."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args()
    print(json.dumps(audit(args.dataset, args.files), ensure_ascii=False, indent=2))
