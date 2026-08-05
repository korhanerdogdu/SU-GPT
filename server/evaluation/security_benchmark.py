from __future__ import annotations

"""Deterministic layered-security benchmark over versioned adversarial corpora."""

import json
import tempfile
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

from modules.course_planner import official_name, validate_proposed_plan
from modules.guardrails import assess_input, assess_retrieval, validate_output
from modules.load_vectorstore import _save_uploaded


def _actual_for_row(row: dict, temporary_directory: str) -> tuple[str, str | None]:
    surface = row.get("surface", row.get("layer", "input"))
    if surface == "input":
        assessment = assess_input(row["prompt"])
        return (
            "allow" if assessment.allowed else "block_input",
            assessment.category,
        )
    if surface == "input_generator":
        assessment = assess_input("x" * int(row["length"]))
        return (
            "allow" if assessment.allowed else "block_input",
            assessment.category,
        )
    if surface == "retrieval":
        assessment = assess_retrieval(row["payload"])
        return (
            "allow" if assessment.allowed else "quarantine",
            assessment.category,
        )
    if surface == "output":
        validation = validate_output(
            row["payload"],
            row.get("authorized_sources"),
        )
        return (
            "allow" if validation.safe else "block_output",
            validation.category,
        )
    if surface == "upload":
        upload = SimpleNamespace(
            filename=row["filename"], file=BytesIO(row["payload"].encode())
        )
        try:
            _save_uploaded([upload], temporary_directory, allowed_extensions={".pdf"})
            return "allow", None
        except ValueError:
            return "reject_upload", "malicious_upload"
    if surface == "planner":
        errors = validate_proposed_plan(
            [{"code": row["course_code"], "title": official_name(row["course_code"])}],
            completed_codes=[],
            stage=row["stage"],
        )
        return (
            ("reject_plan", "planner_bypass") if errors else ("allow", None)
        )
    raise ValueError(f"unknown security surface: {surface}")


def _matches(row: dict, temporary_directory: str) -> bool:
    actual, _category = _actual_for_row(row, temporary_directory)
    return actual == row["expected_action"]


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _classification_metrics(results: list[dict]) -> tuple[dict[str, dict], float | None]:
    labels = sorted(
        {
            str(result["expected_category"])
            for result in results
            if result.get("expected_category")
        }
    )
    metrics: dict[str, dict] = {}
    for label in labels:
        tp = sum(
            result.get("expected_category") == label and result.get("actual_category") == label
            for result in results
        )
        fp = sum(
            result.get("expected_category") != label and result.get("actual_category") == label
            for result in results
        )
        fn = sum(
            result.get("expected_category") == label and result.get("actual_category") != label
            for result in results
        )
        precision = _rate(tp, tp + fp)
        recall = _rate(tp, tp + fn)
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall
            else 0.0
        )
        metrics[label] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    macro_f1 = sum(item["f1"] for item in metrics.values()) / len(metrics) if metrics else None
    return metrics, macro_f1


def _group_summary(items: list[dict]) -> dict:
    attacks = [item for item in items if item["expected_action"] != "allow"]
    benign = [item for item in items if item["expected_action"] == "allow"]
    return {
        "n": len(items),
        "passed": sum(item["passed"] for item in items),
        "pass_rate": _rate(sum(item["passed"] for item in items), len(items)),
        "attacks": len(attacks),
        "attack_success_rate": _rate(
            sum(item["actual_action"] == "allow" for item in attacks), len(attacks)
        ),
        "benign": len(benign),
        "false_refusal_rate": _rate(
            sum(item["actual_action"] != "allow" for item in benign), len(benign)
        ),
    }


def evaluate(path: str | Path) -> dict:
    corpus_path = Path(path)
    rows = [
        json.loads(line)
        for line in corpus_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    results: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="advisu-security-benchmark-") as temp_dir:
        for row in rows:
            actual_action, actual_category = _actual_for_row(row, temp_dir)
            expected_action = row["expected_action"]
            results.append(
                {
                    "id": row["id"],
                    "cluster_id": row.get("cluster_id"),
                    "language": row["language"],
                    "layer": row.get("layer", row.get("surface", "input")),
                    "category": row["category"],
                    "expected_action": expected_action,
                    "actual_action": actual_action,
                    "expected_category": row.get("expected_category"),
                    "actual_category": actual_category,
                    "critical": bool(row.get("critical")),
                    "obfuscation": bool(row.get("obfuscation")),
                    "passed": actual_action == expected_action,
                }
            )

    attacks = [result for result in results if result["expected_action"] != "allow"]
    benign = [result for result in results if result["expected_action"] == "allow"]
    blocked_inputs = [
        result for result in results if result["expected_action"] == "block_input"
    ]
    critical_attacks = [result for result in attacks if result["critical"]]
    obfuscated_attacks = [result for result in attacks if result["obfuscation"]]
    classification, macro_f1 = _classification_metrics(results)

    languages = sorted({result["language"] for result in results})
    layers = sorted({result["layer"] for result in results})
    categories = sorted({result["category"] for result in results})
    return {
        "schema_version": 2 if any(row.get("cluster_id") for row in rows) else 1,
        "benchmark_id": corpus_path.stem.replace("_", "-"),
        "n": len(results),
        "passed": sum(result["passed"] for result in results),
        "attack_success_rate": _rate(
            sum(result["actual_action"] == "allow" for result in attacks), len(attacks)
        ),
        "critical_attack_success_rate": _rate(
            sum(result["actual_action"] == "allow" for result in critical_attacks),
            len(critical_attacks),
        ),
        "obfuscation_attack_success_rate": _rate(
            sum(result["actual_action"] == "allow" for result in obfuscated_attacks),
            len(obfuscated_attacks),
        ),
        "safe_refusal_accuracy": _rate(
            sum(result["actual_action"] == "block_input" for result in blocked_inputs),
            len(blocked_inputs),
        ),
        "false_refusal_rate": _rate(
            sum(result["actual_action"] != "allow" for result in benign), len(benign)
        ),
        "macro_f1": macro_f1,
        "classification_by_category": classification,
        "by_language": {
            language: _group_summary(
                [result for result in results if result["language"] == language]
            )
            for language in languages
        },
        "by_layer": {
            layer: _group_summary(
                [result for result in results if result["layer"] == layer]
            )
            for layer in layers
        },
        "by_category": {
            category: _group_summary(
                [result for result in results if result["category"] == category]
            )
            for category in categories
        },
        # Per-item evidence contains stable labels and decisions only, never prompts or payloads.
        "items": results,
        "scope_note": (
            "Deterministic unit-layer controls only; endpoint and live-provider security are "
            "measured in separate benchmarks. Raw prompts and outputs are excluded."
        ),
    }
