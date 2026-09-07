import asyncio
import copy
import json
import shutil

import pytest

from dataset_tools import DEFAULT_DATASET, current_user_message, read_jsonl, validate
from run_components import run


def test_manifest_and_references_are_valid():
    result = validate(DEFAULT_DATASET)
    assert result["cases"] == 30
    assert result["suites"] == {"medical": 8, "system": 14, "component": 8}


def test_current_user_message_does_not_expose_future_or_private_fields():
    case = next(c for c in read_jsonl(DEFAULT_DATASET / "cases.jsonl") if c["case_id"] == "SYS-10")
    case = copy.deepcopy(case)
    case["gold"] = "SECRET_EVALUATOR_SENTINEL"
    message = current_user_message(case, 1)
    assert set(message) == {"role", "content"}
    assert "37.2" in message["content"]
    assert "38.2" not in message["content"]
    assert "SECRET_EVALUATOR_SENTINEL" not in json.dumps(message)


@pytest.mark.parametrize("damage", ["duplicate_id", "future_check", "unknown_document", "gold_field", "bad_hash"])
def test_invalid_datasets_are_rejected(tmp_path, damage):
    root = tmp_path / "dataset"
    shutil.copytree(DEFAULT_DATASET, root)
    filename = {"future_check": "private/rubrics.jsonl",
                "unknown_document": "private/fixtures.jsonl"}.get(damage, "cases.jsonl")
    path = root / filename
    rows = read_jsonl(path)
    if damage == "duplicate_id":
        rows[1]["case_id"] = rows[0]["case_id"]
    elif damage == "future_check":
        rows[0]["checks"][0]["turn"] = 999
    elif damage == "unknown_document":
        fixture = next(r for r in rows if r["case_id"] == "RAG-01")
        fixture["component_actions"][0]["document_keys"] = ["NONEXISTENT"]
    elif damage == "gold_field":
        rows[0]["gold"] = "B"
    else:
        rows[0]["title"] = "tampered"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    with pytest.raises(ValueError):
        validate(root, verify_hashes=(damage == "bad_hash"))


def test_all_eight_component_contracts_against_real_functions():
    result = asyncio.run(run(DEFAULT_DATASET))
    assert result["model_calls"] == 0
    assert result["total"] == result["passed"] == 8
    paraphrase = next(r for r in result["results"] if r["case_id"] == "RAG-02")
    assert paraphrase["observed"]["backend_calls"] == 2
    assert paraphrase["observed"]["new_bodies_per_call"] == [1, 0]
