from __future__ import annotations

"""
Candidate-pool ceiling and oracle-reranking analysis.

A reranker can only reorder what the first stage handed it. Before spending compute on
cross-encoders, this establishes, per candidate depth N, the best any reranker could
possibly do: the oracle places every relevant candidate found in the top N ahead of every
non-relevant one, and its scores are therefore hard upper bounds.

Two ceilings matter and they are different:

  * Candidate Hit@N  - can the pool reach the query at all? A query whose gold is absent from
    the top N is unreachable at any effort; reranking cannot invent it.
  * Oracle Hit@1 / Recall@3 / MRR@10 - the best achievable ORDERING given that pool. This is
    the number to compare an actual reranker against; the gap between achieved and oracle is
    the real remaining headroom.

If Oracle Hit@1 at depth 10 already equals Oracle Hit@1 at depth 50, deeper reranking buys
nothing but latency, and the depth ablation should say so.

Usage:
    python server/evaluation/candidate_ceiling.py <run_dir> --method hybrid_bm25f_e5_meta
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retrieval_lab import metrics as M
from retrieval_lab.corpus import PROJECT_ROOT

DEPTHS = (10, 20, 25, 50, 100)


def load_rankings(run_dir: Path, method: str) -> dict[str, list[str]]:
    path = run_dir / "rankings" / f"{method}.jsonl"
    if not path.exists():
        sys.exit(f"no rankings for '{method}' in {run_dir}")
    out: dict[str, list[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("scorable"):
            out[row["query_id"]] = row.get("retrieved") or []
    return out


def load_gold(split: str) -> dict[str, set[str]]:
    path = PROJECT_ROOT / "data" / "benchmark" / f"retrieval_{split}.jsonl"
    gold: dict[str, set[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("expected_chunk_ids"):
            gold[row["id"]] = set(row["expected_chunk_ids"])
    return gold


def oracle_order(candidates: list[str], gold: set[str]) -> list[str]:
    """Perfect reranking of a fixed pool: every relevant candidate first, order otherwise kept."""
    rel = [c for c in candidates if c in gold]
    non = [c for c in candidates if c not in gold]
    return rel + non


def analyse(rankings: dict[str, list[str]], gold: dict[str, set[str]]) -> list[dict]:
    qids = sorted(set(rankings) & set(gold))
    rows: list[dict] = []
    for depth in DEPTHS:
        cand_hit = cand_allgold = 0
        no_gold = 0
        o_hit1 = o_r1 = o_r3 = o_r5 = o_r10 = o_mrr = o_ndcg = o_allgold10 = 0.0
        for q in qids:
            g = gold[q]
            pool = rankings[q][:depth]
            found = g & set(pool)
            if found:
                cand_hit += 1
            else:
                no_gold += 1
            if g <= set(pool):
                cand_allgold += 1
            o = oracle_order(pool, g)
            o_hit1 += M.hit_at_k(o, g, 1)
            o_r1 += M.true_recall_at_k(o, g, 1)
            o_r3 += M.true_recall_at_k(o, g, 3)
            o_r5 += M.true_recall_at_k(o, g, 5)
            o_r10 += M.hit_at_k(o, g, 10)
            o_mrr += M.reciprocal_rank(o, g, 10)
            o_ndcg += M.ndcg_at_k(o, g, 10)
            o_allgold10 += M.all_gold_at_k(o, g, 10)
        n = len(qids) or 1
        rows.append({
            "candidate_depth": depth,
            "n_queries": len(qids),
            "candidate_hit": cand_hit / n,
            "candidate_allgold": cand_allgold / n,
            "queries_without_gold": no_gold,
            "oracle_hit@1": o_hit1 / n,
            "oracle_recall@1": o_r1 / n,
            "oracle_recall@3": o_r3 / n,
            "oracle_recall@5": o_r5 / n,
            "oracle_hit@10": o_r10 / n,
            "oracle_mrr@10": o_mrr / n,
            "oracle_ndcg@10": o_ndcg / n,
            "oracle_allgold@10": o_allgold10 / n,
        })
    return rows


def achieved(rankings: dict[str, list[str]], gold: dict[str, set[str]]) -> dict[str, float]:
    qids = sorted(set(rankings) & set(gold))
    n = len(qids) or 1
    agg = {"hit@1": 0.0, "recall@1": 0.0, "recall@3": 0.0, "hit@10": 0.0, "mrr@10": 0.0, "ndcg@10": 0.0}
    for q in qids:
        g, r = gold[q], rankings[q]
        agg["hit@1"] += M.hit_at_k(r, g, 1)
        agg["recall@1"] += M.true_recall_at_k(r, g, 1)
        agg["recall@3"] += M.true_recall_at_k(r, g, 3)
        agg["hit@10"] += M.hit_at_k(r, g, 10)
        agg["mrr@10"] += M.reciprocal_rank(r, g, 10)
        agg["ndcg@10"] += M.ndcg_at_k(r, g, 10)
    return {k: v / n for k, v in agg.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir")
    ap.add_argument("--method", default="hybrid_bm25f_e5_meta")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    rankings = load_rankings(run_dir, args.method)
    gold = load_gold(args.split)
    depth_saved = max((len(v) for v in rankings.values()), default=0)

    rows = analyse(rankings, gold)
    got = achieved(rankings, gold)

    out_dir = run_dir / "reranking"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "candidate_ceiling.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out_dir / "oracle_metrics.json").write_text(
        json.dumps({"method": args.method, "split": args.split, "achieved": got,
                    "max_saved_depth": depth_saved, "by_depth": rows}, indent=2),
        encoding="utf-8",
    )

    print(f"first stage: {args.method}   split={args.split}   n={rows[0]['n_queries']}")
    print(f"ranked ids persisted per query: {depth_saved}"
          f"{'  (WARNING: shallower than the deepest depth analysed)' if depth_saved < DEPTHS[-1] else ''}\n")
    print(f"{'depth':>6} {'candHit':>8} {'candAllG':>9} {'noGold':>7} "
          f"{'oHit@1':>8} {'oRec@1':>8} {'oRec@3':>8} {'oMRR@10':>8} {'oNDCG@10':>9}")
    print("-" * 82)
    for r in rows:
        if r["candidate_depth"] > depth_saved:
            continue
        print(f"{r['candidate_depth']:>6} {r['candidate_hit']:>8.4f} {r['candidate_allgold']:>9.4f} "
              f"{r['queries_without_gold']:>7} {r['oracle_hit@1']:>8.4f} {r['oracle_recall@1']:>8.4f} "
              f"{r['oracle_recall@3']:>8.4f} {r['oracle_mrr@10']:>8.4f} {r['oracle_ndcg@10']:>9.4f}")

    print(f"\nACHIEVED by the first stage (no reranking):")
    print(f"       Hit@1={got['hit@1']:.4f}  Recall@1={got['recall@1']:.4f}  Recall@3={got['recall@3']:.4f}  "
          f"MRR@10={got['mrr@10']:.4f}  nDCG@10={got['ndcg@10']:.4f}  Hit@10={got['hit@10']:.4f}")
    top = next(r for r in rows if r["candidate_depth"] == min(DEPTHS))
    print(f"\nHEADROOM at depth 10: Hit@1 {got['hit@1']:.4f} -> oracle {top['oracle_hit@1']:.4f} "
          f"(+{top['oracle_hit@1'] - got['hit@1']:.4f} available)")
    print(f"wrote {out_dir}/candidate_ceiling.csv and oracle_metrics.json")


if __name__ == "__main__":
    main()
