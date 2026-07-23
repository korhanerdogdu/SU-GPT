from __future__ import annotations

"""
Ablation sweep over the retrieval grid (CLAUDE.md Section 6.3).

Runs every (mode x top_k x candidate_k x prompt_strategy) cell from ablation_configs.yaml against
the benchmark and writes one summary row per cell. Cells already present in the output file are
skipped, so an interrupted sweep resumes instead of re-running expensive work (--force overrides).

Defaults to retrieval-only: the ablation question is about ranking, and generating answers for
every cell would multiply LLM cost by the grid size.

Usage:
  python server/evaluation/ablation_runner.py
  python server/evaluation/ablation_runner.py --with-answers --force
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SERVER_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SERVER_ROOT.parent
sys.path.insert(0, str(SERVER_ROOT))
os.environ.setdefault("CHROMA_PERSIST_DIR", str(SERVER_ROOT / "chroma_store"))
os.environ.setdefault("MONGO_URI", "mongodb://localhost:27017")

import yaml  # noqa: E402

from evaluation.run_evaluation import load_benchmark, run_question, summarize  # noqa: E402
from modules.load_vectorstore import get_vectorstore  # noqa: E402


def cell_key(mode: str, top_k: int, candidate_k: int, strategy: str) -> str:
    return f"{mode}|top_k={top_k}|cand_k={candidate_k}|prompt={strategy}"


def load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            done.add(json.loads(line)["cell"])
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the adviSU ablation grid.")
    parser.add_argument("--config", default=str(Path(__file__).parent / "ablation_configs.yaml"))
    parser.add_argument("--benchmark", default=str(PROJECT_ROOT / "data" / "benchmark" / "questions.jsonl"))
    parser.add_argument("--out", default=str(PROJECT_ROOT / "outputs" / "evaluation_runs" / "ablations.jsonl"))
    parser.add_argument("--with-answers", action="store_true", help="override the config and generate answers")
    parser.add_argument("--force", action="store_true", help="re-run cells that already have results")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    modes = config.get("modes", [])
    top_ks = config.get("top_k_values", [5])
    candidate_ks = config.get("candidate_k_values", [20])
    strategies = config.get("prompt_strategies", ["basic"])
    with_answers = args.with_answers or bool(config.get("with_answers", False))

    items = load_benchmark(Path(args.benchmark), args.limit)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set() if args.force else load_done(out_path)

    cells = [(m, t, c, s) for m in modes for t in top_ks for c in candidate_ks for s in strategies]
    todo = [cell for cell in cells if args.force or cell_key(*cell) not in done]
    print(f"grid: {len(cells)} cells | to run: {len(todo)} | skipped (already done): {len(cells) - len(todo)}")
    if with_answers:
        print(f"WARNING: generating answers for {len(todo)} cells x {len(items)} questions "
              f"= up to {len(todo) * len(items)} LLM calls")

    if not todo:
        print("nothing to do (use --force to re-run)")
        return

    vs = get_vectorstore()
    mode_open = "a" if (out_path.exists() and not args.force) else "w"
    with out_path.open(mode_open, encoding="utf-8") as fh:
        for mode, top_k, candidate_k, strategy in todo:
            key = cell_key(mode, top_k, candidate_k, strategy)
            started = time.perf_counter()
            ks = sorted({1, 3, 5, 10, top_k})
            rows = [run_question(vs, item, mode, top_k, candidate_k, ks, with_answers) for item in items]
            summary = summarize(rows, ks)[mode]
            chunk = summary["retrieval_chunk_level"]
            record = {
                "cell": key,
                "mode": mode,
                "top_k": top_k,
                "candidate_k": candidate_k,
                "prompt_strategy": strategy,
                "with_answers": with_answers,
                "questions": len(items),
                f"recall@{top_k}": chunk.get(f"recall@{top_k}"),
                f"mrr@{top_k}": chunk.get(f"mrr@{top_k}"),
                f"ndcg@{top_k}": chunk.get(f"ndcg@{top_k}"),
                "retrieval_chunk_level": chunk,
                "efficiency": summary["efficiency"],
                "answer_signals": summary["answer_signals"],
                "wall_clock_s": round(time.perf_counter() - started, 1),
                "timestamp_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()  # a killed sweep keeps every finished cell
            recall = record[f"recall@{top_k}"]
            shown = f"{recall:.3f}" if isinstance(recall, float) else "n/a"
            print(f"  {key:<48} recall@{top_k}={shown} ({record['wall_clock_s']}s)")

    print(f"\nablation results -> {out_path}")


if __name__ == "__main__":
    main()
