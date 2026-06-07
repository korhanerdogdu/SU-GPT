"""Run lightweight Section 6 retrieval ablations.

This runner is intentionally conservative: it executes only real retrieval
runs through `modules.retrieval_modes.run_retrieval_mode`, writes measured
metrics, and skips anything that cannot run today (for example, few-shot and
expert-routed prompts from Section 4). It never fabricates ablation results.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EVALUATION_DIR = Path(__file__).resolve().parent
SERVER_ROOT = EVALUATION_DIR.parents[0]
REPO_ROOT = EVALUATION_DIR.parents[1]
sys.path.insert(0, str(SERVER_ROOT))

try:
    import yaml
except ImportError as exc:  # pragma: no cover - depends on local environment
    raise SystemExit("PyYAML is required for ablation_runner.py. Install pyyaml first.") from exc

from modules.load_vectorstore import get_vectorstore  # noqa: E402
from modules.retrieval_modes import run_retrieval_mode  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cheap SU-GPT Section 6 retrieval ablations.")
    parser.add_argument(
        "--config",
        default=str(EVALUATION_DIR / "ablation_configs.yaml"),
        help="Ablation YAML config path.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Number of benchmark questions per combo.")
    parser.add_argument("--modes", nargs="+", default=None, help="Subset of modes to run.")
    parser.add_argument("--top-k-values", nargs="+", type=int, default=None, help="Subset of top_k values.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run one tiny real combination only (default-friendly smoke test).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to config runner.output_dir under repo root.",
    )
    return parser.parse_args()


def load_questions(path: Path, limit: int) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            questions.append(json.loads(line))
            if len(questions) >= limit:
                break
    return questions


def metadata_filter_for(question: dict[str, Any]) -> dict[str, str] | None:
    document_type = str(question.get("documentType") or "").strip()
    if document_type in {"course", "review", "exam"}:
        return {"documentType": document_type}
    return None


def result_chunk_id(result: dict[str, Any]) -> str:
    metadata = result.get("metadata") or {}
    return str(result.get("chunk_id") or metadata.get("chunk_id") or metadata.get("chunkId") or "")


def result_source(result: dict[str, Any]) -> str:
    metadata = result.get("metadata") or {}
    return str(metadata.get("source") or metadata.get("file_name") or metadata.get("sourceId") or "unknown")


def retrieval_metrics(
    retrieved_chunk_ids: list[str],
    retrieved_sources: list[str],
    expected_chunk_ids: list[str],
    expected_sources: list[str],
    *,
    k: int,
) -> dict[str, float | bool]:
    """Small local metric helper so Section 6 does not depend on Section 5 files."""
    use_chunk_level = bool(expected_chunk_ids)
    retrieved = retrieved_chunk_ids[:k] if use_chunk_level else retrieved_sources[:k]
    expected = set(expected_chunk_ids if use_chunk_level else expected_sources)
    if not expected:
        return {"hit": False, "recall_at_k": 0.0, "mrr_at_k": 0.0, "ndcg_at_k": 0.0}

    hits = [1 if item in expected else 0 for item in retrieved]
    hit_count = sum(hits)
    recall = hit_count / len(expected)
    first_hit_rank = next((idx + 1 for idx, hit in enumerate(hits) if hit), None)
    mrr = 1.0 / first_hit_rank if first_hit_rank else 0.0
    dcg = sum(hit / (1 if idx == 0 else __import__("math").log2(idx + 2)) for idx, hit in enumerate(hits))
    ideal_hits = [1] * min(len(expected), k)
    idcg = sum(hit / (1 if idx == 0 else __import__("math").log2(idx + 2)) for idx, hit in enumerate(ideal_hits))
    ndcg = dcg / idcg if idcg else 0.0
    return {
        "hit": bool(hit_count),
        "recall_at_k": round(recall, 6),
        "mrr_at_k": round(mrr, 6),
        "ndcg_at_k": round(ndcg, 6),
    }


def average(values: list[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def combo_path(output_dir: Path, mode: str, top_k: int, limit: int) -> Path:
    return output_dir / f"mode-{mode}__topk-{top_k}__limit-{limit}.json"


def run_combo(
    *,
    mode: str,
    top_k: int,
    limit: int,
    questions: list[dict[str, Any]],
    vectorstore: Any,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        return {"status": "skipped", "reason": "output already exists", "path": str(output_path)}

    rows: list[dict[str, Any]] = []
    for question in questions:
        started = time.perf_counter()
        report = run_retrieval_mode(
            mode,
            str(question.get("question") or ""),
            vectorstore,
            top_k=top_k,
            metadata_filter=metadata_filter_for(question),
        )
        latency_ms = round((time.perf_counter() - started) * 1000.0, 2)
        results = report.get("results") or []
        retrieved_chunk_ids = [result_chunk_id(result) for result in results]
        retrieved_sources = [result_source(result) for result in results]
        expected_chunk_ids = list(question.get("expected_chunk_ids") or [])
        expected_sources = list(question.get("expected_sources") or [])
        metrics = retrieval_metrics(
            retrieved_chunk_ids,
            retrieved_sources,
            expected_chunk_ids,
            expected_sources,
            k=top_k,
        )
        timings = report.get("timings_ms") or {}
        rows.append(
            {
                "query_id": question.get("id"),
                "mode": mode,
                "top_k": top_k,
                "retrieved_chunk_ids": retrieved_chunk_ids,
                "retrieved_sources": retrieved_sources,
                "expected_chunk_ids": expected_chunk_ids,
                "expected_sources": expected_sources,
                "latency_ms": latency_ms,
                "retrieval_latency_ms": timings.get("retrieval_ms", 0.0),
                "rerank_latency_ms": timings.get("rerank_ms", 0.0),
                **metrics,
            }
        )

    payload = {
        "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "top_k": top_k,
        "limit": limit,
        "num_questions": len(rows),
        "summary": {
            "avg_recall_at_k": average([float(row["recall_at_k"]) for row in rows]),
            "avg_mrr_at_k": average([float(row["mrr_at_k"]) for row in rows]),
            "avg_ndcg_at_k": average([float(row["ndcg_at_k"]) for row in rows]),
            "hit_rate": average([1.0 if row["hit"] else 0.0 for row in rows]),
            "avg_latency_ms": average([float(row["latency_ms"]) for row in rows]),
            "avg_retrieval_latency_ms": average([float(row["retrieval_latency_ms"]) for row in rows]),
            "avg_rerank_latency_ms": average([float(row["rerank_latency_ms"]) for row in rows]),
        },
        "rows": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return {"status": "completed", "path": str(output_path), "summary": payload["summary"]}


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    runner_config = config.get("runner") or {}
    limit = args.limit or int(runner_config.get("max_questions_per_combo") or 5)
    if args.quick:
        limit = min(limit, 2)

    modes = args.modes or list(config.get("modes") or [])
    top_k_values = args.top_k_values or list(config.get("top_k_values") or [])
    if args.quick:
        modes = modes[:1]
        top_k_values = top_k_values[:1]

    output_dir = Path(args.output_dir or (REPO_ROOT / str(runner_config.get("output_dir"))))
    benchmark_path = REPO_ROOT / str(runner_config.get("benchmark_path"))
    questions = load_questions(benchmark_path, limit)

    prompt_strategy_status = []
    for strategy in config.get("prompt_strategies") or []:
        name = strategy.get("name") if isinstance(strategy, dict) else str(strategy)
        implemented = bool(strategy.get("implemented")) if isinstance(strategy, dict) else name == "basic"
        if not implemented:
            prompt_strategy_status.append(
                {"prompt_strategy": name, "status": "skipped", "reason": "not implemented yet"}
            )

    combo_specs = [
        (mode, int(top_k), combo_path(output_dir, mode, int(top_k), limit))
        for mode in modes
        for top_k in top_k_values
    ]

    results = []
    pending_specs = []
    for mode, top_k, output_path in combo_specs:
        if output_path.exists():
            results.append(
                {
                    "mode": mode,
                    "top_k": top_k,
                    "status": "skipped",
                    "reason": "output already exists",
                    "path": str(output_path),
                }
            )
        else:
            pending_specs.append((mode, top_k, output_path))

    vectorstore = get_vectorstore() if pending_specs else None
    for mode, top_k, output_path in pending_specs:
        results.append(
            {
                "mode": mode,
                "top_k": top_k,
                **run_combo(
                    mode=mode,
                    top_k=top_k,
                    limit=limit,
                    questions=questions,
                    vectorstore=vectorstore,
                    output_path=output_path,
                ),
            }
        )

    manifest_path = output_dir / "latest_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "status": "completed",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "config": str(config_path),
                "quick": bool(args.quick),
                "limit": limit,
                "results": results,
                "prompt_strategy_status": prompt_strategy_status,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
    print(json.dumps({"manifest": str(manifest_path), "results": results}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
