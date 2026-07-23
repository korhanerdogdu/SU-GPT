from __future__ import annotations

"""
Turn an evaluation run into report-ready tables (CLAUDE.md Sections 6.4 and 6.5).

Reads outputs/evaluation_runs/<run>/results.jsonl (defaults to the newest run) and writes:

  outputs/tables/retrieval_metrics.csv    Recall/Precision/MRR/nDCG per mode at every k
  outputs/tables/answer_metrics.csv       objective answer signals + empty manual-label columns
  outputs/tables/efficiency_metrics.csv   latency split, context size, prompt-token estimate, cost
  outputs/tables/failure_summary.csv      failure counts per mode x error_type
  outputs/tables/ablation_summary.csv     one row per ablation cell (if ablations.jsonl exists)
  outputs/failure_analysis/failure_cases.csv   one row per failing question x mode

ERROR-TYPE CLASSIFICATION (rule-based, and the rules are the whole story - no model judges this):
  retrieval_miss        answerable, and no gold chunk appeared even among the pre-cut candidates
  retrieval_noise       a gold chunk WAS retrieved but ranked out of the final top-k context
  generation_hallucination  gold chunk was in context, yet the answer omits the ground-truth number
  insufficient_context  the system refused although a gold chunk was in context
  unsupported_answer    unanswerable question that did NOT get a refusal (the risky failure)
root_cause / proposed_fix / before_after_evidence are left blank on purpose: those are human
analysis, not measurements, and CLAUDE.md forbids inventing them.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

SERVER_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SERVER_ROOT.parent
RUNS_DIR = PROJECT_ROOT / "outputs" / "evaluation_runs"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
FAILURES_DIR = PROJECT_ROOT / "outputs" / "failure_analysis"

# Groq pricing is not hard-coded: an unset rate yields an empty cost column rather than a wrong one.
COST_PER_1K_INPUT = None
COST_PER_1K_OUTPUT = None


def latest_run(runs_dir: Path) -> Path:
    candidates = sorted(p for p in runs_dir.glob("*") if (p / "results.jsonl").exists())
    if not candidates:
        sys.exit(f"no evaluation runs with results.jsonl under {runs_dir}\n"
                 "Run: python server/evaluation/run_evaluation.py")
    return candidates[-1]


def read_rows(run_dir: Path) -> list[dict]:
    path = run_dir / "results.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def fmt(value, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if value != value:  # NaN
            return ""
        return f"{value:.{digits}f}"
    return str(value)


def classify(row: dict) -> str | None:
    """Return an error_type, or None when the row is not a failure."""
    gold = set(row.get("expected_chunk_ids") or [])
    final = set(row.get("retrieved_chunk_ids") or [])
    candidates = set(row.get("candidate_chunk_ids") or []) or final
    answer = row.get("answer")
    refused = bool(row.get("auto_refusal_detected"))
    # A provider failure (rate limit, timeout) is an infrastructure event, not a system defect.
    # Retrieval for that row is still valid, so only the generation-side verdicts are suppressed.
    generation_failed = bool(row.get("generation_error")) or (answer is not None and not answer)

    if not row.get("answerable"):
        if generation_failed:
            return None
        # For unanswerable/misleading questions the only failure mode is answering anyway.
        if answer and not refused:
            return "unsupported_answer"
        return None

    # llm_only retrieves nothing by definition, so "it missed the gold chunk" is not a finding
    # about it - counting that as a retrieval failure would just restate the mode's definition.
    if row["mode"] != "llm_only" and gold and not (gold & final):
        return "retrieval_miss" if not (gold & candidates) else "retrieval_noise"

    if answer is None or generation_failed:
        return None  # retrieval-only run, or the call never completed
    if refused:
        return "insufficient_context"
    if row.get("auto_reference_number_match") is False:
        return "generation_hallucination"
    return None


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"  {path.relative_to(PROJECT_ROOT)}  ({len(rows)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build report tables from an evaluation run.")
    parser.add_argument("--run", default=None, help="run directory for retrieval/efficiency/failures (default: newest)")
    parser.add_argument("--answers-run", default=None,
                        help="separate run to source answer metrics from, when the retrieval run "
                             "was retrieval-only (e.g. answers were generated in an earlier run)")
    parser.add_argument("--ablations", default=str(RUNS_DIR / "ablations.jsonl"))
    args = parser.parse_args()

    run_dir = Path(args.run) if args.run else latest_run(RUNS_DIR)
    rows = read_rows(run_dir)
    summary = json.loads((run_dir / "summary_metrics.json").read_text(encoding="utf-8"))
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    print(f"run: {run_dir.name}  ({len(rows)} result rows)")

    answers_dir = Path(args.answers_run) if args.answers_run else run_dir
    answers_summary = summary
    if answers_dir != run_dir:
        answers_summary = json.loads((answers_dir / "summary_metrics.json").read_text(encoding="utf-8"))
        print(f"answer metrics sourced from: {answers_dir.name}")
        # Graft the answer columns onto the authoritative retrieval rows, keyed by
        # (query_id, mode). Concatenating instead would double-count every retrieval failure.
        answer_fields = ("answer", "sources", "generation_error", "generation_ms",
                         "auto_refusal_detected", "auto_reference_number_match",
                         "prompt_tokens_estimate", "prompt_chars_estimate")
        by_key = {(r["query_id"], r["mode"]): r for r in read_rows(answers_dir)}
        for row in rows:
            source = by_key.get((row["query_id"], row["mode"]))
            if source:
                row.update({f: source.get(f) for f in answer_fields if f in source})

    # ---- retrieval_metrics.csv ---------------------------------------------------------
    metric_keys: list[str] = []
    for mode_data in summary.values():
        for key in mode_data["retrieval_chunk_level"]:
            if key not in metric_keys:
                metric_keys.append(key)
    write_csv(
        TABLES_DIR / "retrieval_metrics.csv",
        ["mode", "granularity", "answerable_questions", *metric_keys],
        [
            [mode, granularity, data["answerable_questions"],
             *[fmt(data[f"retrieval_{granularity}_level"].get(k)) for k in metric_keys]]
            for mode, data in summary.items()
            for granularity in ("chunk", "source")
        ],
    )

    # ---- answer_metrics.csv ------------------------------------------------------------
    write_csv(
        TABLES_DIR / "answer_metrics.csv",
        ["mode", "generation_coverage", "answers_scored", "unanswerable_answered",
         "reference_number_match_rate", "reference_number_checked",
         "refusal_rate_on_unanswerable", "generation_errors",
         "answer_correctness", "citation_correctness", "faithfulness",
         "answer_relevancy", "hallucination_rate"],
        [
            [mode, fmt(data["answer_signals"].get("generation_coverage"), 2),
             data["answer_signals"].get("answers_scored", ""),
             data["answer_signals"]["unanswerable_questions"],
             fmt(data["answer_signals"]["reference_number_match_rate"]),
             data["answer_signals"]["reference_number_checked"],
             fmt(data["answer_signals"]["refusal_rate_on_unanswerable"]),
             data["answer_signals"]["generation_errors"],
             "", "", "", "", ""]  # manual labels - Section 5.5
            for mode, data in answers_summary.items()
        ],
    )

    # ---- efficiency_metrics.csv --------------------------------------------------------
    efficiency_rows = []
    for mode, data in summary.items():
        eff = data["efficiency"]
        tokens = eff.get("mean_prompt_tokens_estimate")
        cost = ""
        if tokens is not None and COST_PER_1K_INPUT is not None:
            cost = fmt(tokens / 1000 * COST_PER_1K_INPUT, 6)
        efficiency_rows.append([
            mode,
            fmt(eff["mean_retrieval_ms"], 1), fmt(eff["mean_rerank_ms"], 1),
            fmt(eff["mean_generation_ms"], 1), fmt(eff["mean_total_ms"], 1),
            fmt(eff["mean_context_chunks"], 2), fmt(tokens, 1), cost,
        ])
    write_csv(
        TABLES_DIR / "efficiency_metrics.csv",
        ["mode", "mean_retrieval_ms", "mean_rerank_ms", "mean_generation_ms", "mean_total_ms",
         "mean_context_chunks", "mean_prompt_tokens_estimate", "estimated_cost_usd_per_query"],
        efficiency_rows,
    )

    # ---- failure_cases.csv + failure_summary.csv ---------------------------------------
    failure_rows = []
    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        error_type = classify(row)
        if not error_type:
            continue
        counts[(row["mode"], error_type)] = counts.get((row["mode"], error_type), 0) + 1
        failure_rows.append([
            row["query_id"], row["question"], row["mode"],
            row.get("prompt_strategy", ""), row.get("expert_mode", ""),
            (row.get("answer") or "")[:800],
            row.get("reference_answer", ""),
            " | ".join(row.get("sources") or []),
            error_type,
            "", "", "",  # root_cause / proposed_fix / before_after_evidence - human analysis
        ])
    write_csv(
        FAILURES_DIR / "failure_cases.csv",
        ["query_id", "question", "mode", "prompt_strategy", "expert_mode", "model_output",
         "reference_answer", "retrieved_sources", "error_type", "root_cause", "proposed_fix",
         "before_after_evidence"],
        failure_rows,
    )
    write_csv(
        TABLES_DIR / "failure_summary.csv",
        ["mode", "error_type", "count"],
        [[mode, error_type, count] for (mode, error_type), count in sorted(counts.items())],
    )

    # ---- ablation_summary.csv ----------------------------------------------------------
    ablation_path = Path(args.ablations)
    if ablation_path.exists():
        records = [json.loads(l) for l in ablation_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        write_csv(
            TABLES_DIR / "ablation_summary.csv",
            ["cell", "mode", "top_k", "candidate_k", "prompt_strategy", "questions",
             "recall@top_k", "mrr@top_k", "ndcg@top_k", "mean_total_ms", "wall_clock_s"],
            [[r["cell"], r["mode"], r["top_k"], r["candidate_k"], r["prompt_strategy"], r["questions"],
              fmt(r.get(f"recall@{r['top_k']}")), fmt(r.get(f"mrr@{r['top_k']}")),
              fmt(r.get(f"ndcg@{r['top_k']}")), fmt(r["efficiency"].get("mean_total_ms"), 1),
              r.get("wall_clock_s")] for r in records],
        )
    else:
        print(f"  (no ablations.jsonl at {ablation_path} - skipping ablation_summary.csv)")

    (TABLES_DIR / "run_provenance.json").write_text(
        json.dumps({"run_dir": run_dir.name, **config}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\ntables written under {TABLES_DIR.relative_to(PROJECT_ROOT)}")
    print("answer_correctness / citation_correctness / faithfulness / answer_relevancy / "
          "hallucination_rate columns are EMPTY by design - fill them by manual labelling.")


if __name__ == "__main__":
    main()
