from __future__ import annotations

"""
Turn a run directory from run_retrieval_lab.py into the report's tables and statistics.

Reads only the machine-readable artifacts (rankings/*.jsonl, summary.json, subgroups.json)
and writes CSV/JSON. Nothing here recomputes retrieval, and no metric is entered by hand, so
the tables cannot drift from the run that produced them.

Outputs (under <run_dir>/):
    retrieval_metrics.csv / .json   headline table, one row per method
    per_query_results.csv           every query x method, for auditing
    subgroup_metrics.csv            metric x subgroup x method
    latency_metrics.csv             single-query latency + index build cost
    significance_results.json       paired bootstrap + permutation test vs the baseline

Usage:
    python server/evaluation/analyze_retrieval_lab.py <run_dir> [--baseline bm25_full_corpus]
"""

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retrieval_lab.metrics import K_VALUES, paired_bootstrap

HEADLINE = (
    [f"recall@{k}" for k in K_VALUES]
    + ["mrr@10", "ndcg@10", "hit@1"]
    + [f"evidence_set_recall@{k}" for k in (5, 10, 20)]
    + [f"all_gold@{k}" for k in (5, 10, 20)]
)


def load_run(run_dir: Path) -> tuple[dict, dict, dict[str, list[dict]]]:
    summary = json.loads((run_dir / "summary.json").read_text())
    subgroups = json.loads((run_dir / "subgroups.json").read_text())
    rankings: dict[str, list[dict]] = {}
    for path in sorted((run_dir / "rankings").glob("*.jsonl")):
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        rankings[path.stem] = [r for r in rows if r.get("scorable")]
    return summary, subgroups, rankings


def write_metrics(run_dir: Path, summary: dict) -> None:
    cols = ["method", "n_queries"] + HEADLINE + [
        "latency_mean_ms", "latency_p50_ms", "latency_p95_ms",
        "index_build_seconds", "index_bytes", "build_notes",
    ]
    with (run_dir / "retrieval_metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for name, s in summary.items():
            row = {"method": name}
            row.update({c: s.get(c, "") for c in cols[1:]})
            w.writerow(row)
    (run_dir / "retrieval_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def write_per_query(run_dir: Path, rankings: dict[str, list[dict]]) -> None:
    cols = ["method", "query_id", "num_gold", "multi_evidence", "rank_of_first_gold"] + \
           [f"recall@{k}" for k in K_VALUES] + ["mrr@10", "ndcg@10", "hit@1", "subgroups"]
    with (run_dir / "per_query_results.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for method, rows in rankings.items():
            for r in rows:
                out = dict(r)
                out["method"] = method
                out["subgroups"] = "|".join(r.get("subgroups", []))
                w.writerow(out)


def write_subgroups(run_dir: Path, subgroups: dict) -> None:
    with (run_dir / "subgroup_metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "subgroup", "n_queries", "recall@1", "recall@5", "recall@10",
                    "recall@20", "mrr@10", "ndcg@10", "hit@1"])
        for method, groups in subgroups.items():
            for g, m in sorted(groups.items()):
                w.writerow([method, g, int(m.get("n_queries", 0)),
                            f"{m.get('recall@1', 0):.4f}", f"{m.get('recall@5', 0):.4f}",
                            f"{m.get('recall@10', 0):.4f}", f"{m.get('recall@20', 0):.4f}",
                            f"{m.get('mrr@10', 0):.4f}", f"{m.get('ndcg@10', 0):.4f}",
                            f"{m.get('hit@1', 0):.4f}"])


def write_latency(run_dir: Path, summary: dict) -> None:
    with (run_dir / "latency_metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "latency_mean_ms", "latency_p50_ms", "latency_p95_ms",
                    "index_build_seconds", "index_bytes", "latency_sample_n"])
        for name, s in summary.items():
            w.writerow([name, f"{s.get('latency_mean_ms', 0):.2f}", f"{s.get('latency_p50_ms', 0):.2f}",
                        f"{s.get('latency_p95_ms', 0):.2f}", f"{s.get('index_build_seconds', 0):.2f}",
                        int(s.get("index_bytes", 0)), int(s.get("latency_sample_n", 0))])


def significance(rankings: dict[str, list[dict]], baseline: str, metrics: tuple[str, ...]) -> dict:
    if baseline not in rankings:
        raise SystemExit(f"baseline '{baseline}' not in run (have: {sorted(rankings)})")
    base_rows = {r["query_id"]: r for r in rankings[baseline]}
    out: dict[str, dict] = {}
    for method, rows in rankings.items():
        if method == baseline:
            continue
        cand_rows = {r["query_id"]: r for r in rows}
        per_metric = {}
        for metric in metrics:
            b = {q: float(r[metric]) for q, r in base_rows.items() if metric in r}
            c = {q: float(r[metric]) for q, r in cand_rows.items() if metric in r}
            if not (set(b) & set(c)):
                continue
            cmp = paired_bootstrap(
                b, c, metric=metric, baseline_name=baseline, candidate_name=method
            )
            per_metric[metric] = asdict(cmp)
        out[method] = per_metric
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir")
    ap.add_argument("--baseline", default="bm25_full_corpus")
    ap.add_argument("--metrics", default="recall@10,recall@5,recall@1,mrr@10,ndcg@10")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    summary, subgroups, rankings = load_run(run_dir)

    write_metrics(run_dir, summary)
    write_per_query(run_dir, rankings)
    write_subgroups(run_dir, subgroups)
    write_latency(run_dir, summary)

    metrics = tuple(m.strip() for m in args.metrics.split(",") if m.strip())
    sig = significance(rankings, args.baseline, metrics)
    (run_dir / "significance_results.json").write_text(json.dumps(sig, indent=2), encoding="utf-8")

    print(f"baseline: {args.baseline}\n")
    print(f"{'method':30s} {'R@10':>8s} {'delta':>9s} {'95% CI':>20s} {'p':>8s} {'win':>5s} {'loss':>5s} {'tie':>5s}")
    print("-" * 100)
    base_r10 = summary[args.baseline].get("recall@10", 0.0)
    print(f"{args.baseline:30s} {base_r10:8.4f} {'-':>9s} {'-':>20s} {'-':>8s} {'-':>5s} {'-':>5s} {'-':>5s}")
    rows = []
    for method, per in sig.items():
        d = per.get("recall@10")
        if not d:
            continue
        rows.append((d["delta"], method, d))
    for delta, method, d in sorted(rows, reverse=True):
        ci = f"[{d['ci_low']:+.4f},{d['ci_high']:+.4f}]"
        star = "*" if d["significant"] else " "
        print(f"{method:30s} {d['candidate_mean']:8.4f} {delta:+9.4f} {ci:>20s} {d['p_value']:8.4f}{star} "
              f"{d['improved']:5d} {d['harmed']:5d} {d['tied']:5d}")
    print("\n* = 95% bootstrap CI excludes zero")
    print(f"\nwrote retrieval_metrics.csv/json, per_query_results.csv, subgroup_metrics.csv, "
          f"latency_metrics.csv, significance_results.json -> {run_dir}")


if __name__ == "__main__":
    main()
