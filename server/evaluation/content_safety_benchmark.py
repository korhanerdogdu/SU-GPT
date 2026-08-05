from __future__ import annotations

"""Frozen comparison of the original guardrail, the contributed classifier, and their union."""

import argparse
import json
import sys
from pathlib import Path

SERVER_ROOT = Path(__file__).resolve().parents[1]
if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))

from modules import content_safety
from modules.guardrails import assess_input


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data/benchmark/security_adversarial_v2.jsonl"
SUPPLEMENT = ROOT / "data/benchmark/content_safety_supplement_v1.jsonl"


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def corpus() -> list[dict]:
    base = [row for row in _rows(BASE) if row.get("surface", row.get("layer")) == "input"]
    return [*base, *_rows(SUPPLEMENT)]


def _blocked(row: dict, method: str) -> bool:
    prompt = str(row["prompt"])
    content_blocked = content_safety.classify(prompt).blocked
    guardrail_blocked = not assess_input(prompt).allowed
    if method == "content_classifier":
        return content_blocked
    if method == "guardrails":
        return guardrail_blocked
    if method == "layered":
        return content_blocked or guardrail_blocked
    raise ValueError(f"unknown method: {method}")


def evaluate() -> dict:
    rows = corpus()
    methods: dict[str, dict] = {}
    for method in ("guardrails", "content_classifier", "layered"):
        tp = tn = fp = fn = 0
        missed: list[str] = []
        false_refusals: list[str] = []
        for row in rows:
            expected_block = row["expected_action"] == "block_input"
            actual_block = _blocked(row, method)
            if expected_block and actual_block:
                tp += 1
            elif expected_block:
                fn += 1
                missed.append(row["id"])
            elif actual_block:
                fp += 1
                false_refusals.append(row["id"])
            else:
                tn += 1
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        methods[method] = {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_refusal_rate": fp / (fp + tn) if fp + tn else 0.0,
            "missed_ids": missed,
            "false_refusal_ids": false_refusals,
        }
    return {
        "benchmark_id": "content-safety-layered-v1",
        "n": len(rows),
        "source_corpora": [str(BASE.relative_to(ROOT)), str(SUPPLEMENT.relative_to(ROOT))],
        "methods": methods,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate()
    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
