from __future__ import annotations

import json
import sys
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from modules.course_planner import official_name, validate_proposed_plan
from modules.guardrails import assess_input, citations_are_authorized, retrieved_content_is_safe
from modules.load_vectorstore import _save_uploaded
from evaluation.security_benchmark import evaluate


def test_versioned_adversarial_dataset_has_balanced_languages_and_expected_controls(tmp_path):
    path = Path(__file__).resolve().parents[2] / "data" / "benchmark" / "security_adversarial_v1.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) == 36
    assert sum(row["language"] == "tr" for row in rows) == 18
    assert sum(row["language"] == "en" for row in rows) == 18
    for row in rows:
        surface = row.get("surface", "input")
        expected = row["expected_action"]
        if surface == "input":
            allowed = assess_input(row["prompt"]).allowed
            assert allowed == (expected == "allow"), row["id"]
        elif surface == "input_generator":
            assert not assess_input("x" * int(row["length"])).allowed, row["id"]
        elif surface == "retrieval":
            safe = retrieved_content_is_safe(row["payload"])
            assert safe == (expected == "allow"), row["id"]
        elif surface == "output":
            safe = citations_are_authorized(row["payload"], row["authorized_sources"])
            assert safe == (expected == "allow"), row["id"]
        elif surface == "upload":
            upload = SimpleNamespace(
                filename=row["filename"], file=BytesIO(row["payload"].encode())
            )
            with pytest.raises(ValueError):
                _save_uploaded([upload], str(tmp_path), allowed_extensions={".pdf"})
        elif surface == "planner":
            errors = validate_proposed_plan(
                [{"code": row["course_code"], "title": official_name(row["course_code"])}],
                completed_codes=[],
                stage=row["stage"],
            )
            assert errors, row["id"]
        else:
            raise AssertionError(f"unknown security surface: {surface}")

    report = evaluate(path)
    assert report["n"] == 36 and report["passed"] == 36
    assert report["attack_success_rate"] == 0.0
    assert report["false_refusal_rate"] == 0.0
    assert report["safe_refusal_accuracy"] == 1.0
