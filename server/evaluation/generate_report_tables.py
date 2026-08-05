"""Generate Section 6 report-ready CSV tables.

The script reads measured evaluation outputs from
`outputs/evaluation_runs/<timestamp>/results.jsonl` and `summary_metrics.json`
plus the hand-reviewed `outputs/failure_analysis/failure_cases.csv`. It does
not invent missing metrics: unanswered/manual grading columns stay blank until
a real grading pass fills them.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

EVALUATION_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVALUATION_DIR.parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SU-GPT Section 6 report tables.")
    parser.add_argument("--run-dir", default=None, help="Specific outputs/evaluation_runs/<timestamp> directory.")
    parser.add_argument(
        "--failure-cases",
        default=str(REPO_ROOT / "outputs" / "failure_analysis" / "failure_cases.csv"),
        help="Failure-analysis CSV path.",
    )
    parser.add_argument(
        "--tables-dir",
        default=str(REPO_ROOT / "outputs" / "tables"),
        help="Directory where report-ready CSV tables are written.",
    )
    parser.add_argument(
        "--ablation-dir",
        default=str(REPO_ROOT / "outputs" / "evaluation_runs" / "ablations"),
        help="Directory containing ablation_runner.py JSON outputs.",
    )
    return parser.parse_args()


def latest_run_dir() -> Path:
    root = REPO_ROOT / "outputs" / "evaluation_runs"
    candidates = [
        path
        for path in root.iterdir()
        if path.is_dir() and (path / "results.jsonl").exists() and (path / "summary_metrics.json").exists()
    ]
    if not candidates:
        raise SystemExit(f"No evaluation run found under {root}; run Section 5 first or pass --run-dir.")
    return sorted(candidates)[-1]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def numeric_average(rows: list[dict[str, Any]], field: str) -> str:
    values = [float(row[field]) for row in rows if isinstance(row.get(field), (int, float))]
    return f"{sum(values) / len(values):.6f}" if values else ""


def generate_retrieval_metrics(summary: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    fieldnames = [
        "mode",
        "num_rows",
        "avg_recall_at_k",
        "avg_mrr_at_k",
        "avg_ndcg_at_k",
        "chunk_hit_rate",
        "source_hit_rate",
    ]
    rows = []
    for mode, metrics in sorted((summary.get("modes") or {}).items()):
        rows.append({"mode": mode, **{field: metrics.get(field, "") for field in fieldnames if field != "mode"}})
    return fieldnames, rows


def generate_answer_metrics(results: list[dict[str, Any]]) -> tuple[list[str], list[dict[str, Any]]]:
    fieldnames = [
        "mode",
        "num_rows",
        "answer_correctness_avg",
        "citation_correctness_avg",
        "faithfulness_avg",
        "answer_relevancy_avg",
        "hallucination_flag_rate",
        "note",
    ]
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        by_mode[str(row.get("mode") or "unknown")].append(row)

    rows = []
    for mode, mode_rows in sorted(by_mode.items()):
        hallucination_values = [
            1.0 if row.get("hallucination_flag") is True else 0.0
            for row in mode_rows
            if row.get("hallucination_flag") is not None
        ]
        rows.append(
            {
                "mode": mode,
                "num_rows": len(mode_rows),
                "answer_correctness_avg": numeric_average(mode_rows, "answer_correctness"),
                "citation_correctness_avg": numeric_average(mode_rows, "citation_correctness"),
                "faithfulness_avg": numeric_average(mode_rows, "faithfulness"),
                "answer_relevancy_avg": numeric_average(mode_rows, "answer_relevancy"),
                "hallucination_flag_rate": (
                    f"{sum(hallucination_values) / len(hallucination_values):.6f}"
                    if hallucination_values
                    else ""
                ),
                "note": "Manual/automated answer grading not run yet; blank fields are not fabricated.",
            }
        )
    return fieldnames, rows


def generate_efficiency_metrics(summary: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    fieldnames = [
        "mode",
        "num_rows",
        "avg_latency_ms",
        "avg_retrieval_latency_ms",
        "avg_rerank_latency_ms",
        "avg_generation_latency_ms",
        "total_estimated_cost_usd",
    ]
    rows = []
    for mode, metrics in sorted((summary.get("modes") or {}).items()):
        rows.append({"mode": mode, **{field: metrics.get(field, "") for field in fieldnames if field != "mode"}})
    return fieldnames, rows


def generate_failure_summary(failure_cases_path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    fieldnames = ["error_type", "mode", "count"]
    if not failure_cases_path.exists():
        return fieldnames, []
    with failure_cases_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        counts = Counter((row.get("error_type") or "unknown", row.get("mode") or "unknown") for row in reader)
    return fieldnames, [
        {"error_type": error_type, "mode": mode, "count": count}
        for (error_type, mode), count in sorted(counts.items())
    ]


def generate_ablation_summary(ablation_dir: Path) -> tuple[list[str], list[dict[str, Any]]]:
    fieldnames = [
        "file",
        "status",
        "mode",
        "top_k",
        "limit",
        "num_questions",
        "avg_recall_at_k",
        "avg_mrr_at_k",
        "avg_ndcg_at_k",
        "hit_rate",
        "avg_latency_ms",
        "note",
    ]
    rows = []
    if ablation_dir.exists():
        for path in sorted(ablation_dir.glob("mode-*.json")):
            payload = load_json(path)
            summary = payload.get("summary") or {}
            rows.append(
                {
                    "file": path.name,
                    "status": payload.get("status", ""),
                    "mode": payload.get("mode", ""),
                    "top_k": payload.get("top_k", ""),
                    "limit": payload.get("limit", ""),
                    "num_questions": payload.get("num_questions", ""),
                    "avg_recall_at_k": summary.get("avg_recall_at_k", ""),
                    "avg_mrr_at_k": summary.get("avg_mrr_at_k", ""),
                    "avg_ndcg_at_k": summary.get("avg_ndcg_at_k", ""),
                    "hit_rate": summary.get("hit_rate", ""),
                    "avg_latency_ms": summary.get("avg_latency_ms", ""),
                    "note": "Measured by ablation_runner.py.",
                }
            )
    if not rows:
        rows.append(
            {
                "file": "",
                "status": "not_run",
                "note": "Run server/evaluation/ablation_runner.py to produce real ablation measurements.",
            }
        )
    return fieldnames, rows


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else latest_run_dir()
    results = load_jsonl(run_dir / "results.jsonl")
    summary = load_json(run_dir / "summary_metrics.json")
    tables_dir = Path(args.tables_dir)

    for filename, generator_args in {
        "retrieval_metrics.csv": generate_retrieval_metrics(summary),
        "answer_metrics.csv": generate_answer_metrics(results),
        "efficiency_metrics.csv": generate_efficiency_metrics(summary),
        "failure_summary.csv": generate_failure_summary(Path(args.failure_cases)),
        "ablation_summary.csv": generate_ablation_summary(Path(args.ablation_dir)),
    }.items():
        fieldnames, rows = generator_args
        write_csv(tables_dir / filename, fieldnames, rows)

    print(f"Wrote report tables to {tables_dir}")


if __name__ == "__main__":
    main()
