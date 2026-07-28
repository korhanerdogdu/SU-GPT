from __future__ import annotations

"""
Fuse two already-scored rerankings offline, without re-running any model.

At candidate depth 10 the saved `top10` field IS the complete permutation the reranker
produced, so Reciprocal Rank Fusion over saved rankings is exactly equal to running the
FusionReranker live — and costs nothing. This matters because the Qwen3 reranker takes ~17 s
per query; re-running it once per fusion variant would cost hours for a result that is
arithmetically determined by rankings already on disk.

Only valid when every saved ranking is the full pool (depth <= len(top10)). The script
refuses to run otherwise rather than silently fusing truncated lists.

Usage:
    python server/evaluation/posthoc_fusion.py \
        --a outputs/reranking/dev__qwenfull/rankings/qwen3_06b_domain_at_10.jsonl \
        --b outputs/reranking/dev__cheap/rankings/original_ranking_at_10.jsonl \
        --name fuse_qwen_first --split dev --out outputs/reranking/dev__posthoc
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retrieval_lab import metrics as M
from retrieval_lab.corpus import PROJECT_ROOT
from retrieval_lab.fusion import reciprocal_rank_fusion

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_reranking_lab import per_query_metrics  # noqa: E402


def load(path: Path) -> dict[str, dict]:
    return {r["query_id"]: r for r in (json.loads(l) for l in path.read_text().splitlines() if l.strip())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--weights", default="1,1")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    A, B = load(Path(args.a)), load(Path(args.b))
    gold = {}
    for line in (PROJECT_ROOT / "data" / "benchmark" / f"retrieval_{args.split}.jsonl").read_text(
        encoding="utf-8"
    ).splitlines():
        if line.strip():
            r = json.loads(line)
            if r["expected_chunk_ids"]:
                gold[r["id"]] = set(r["expected_chunk_ids"])

    qids = sorted(set(A) & set(B) & set(gold))
    if not qids:
        sys.exit("no overlapping queries")

    w = [float(x) for x in args.weights.split(",")]
    depth = A[qids[0]].get("depth", 10)
    rows, agg = [], {}
    for q in qids:
        ra, rb = A[q]["top10"], B[q]["top10"]
        if len(ra) < min(depth, 10) or len(rb) < min(depth, 10):
            sys.exit(f"saved ranking for {q} is shorter than the pool; post-hoc fusion would be wrong")
        fused = [c for c, _ in reciprocal_rank_fusion(
            [[(c, 1.0 / (i + 1)) for i, c in enumerate(ra)],
             [(c, 1.0 / (i + 1)) for i, c in enumerate(rb)]],
            k=args.rrf_k, weights=w, top=max(len(ra), len(rb)))]
        m = per_query_metrics(fused, gold[q])
        for k, v in m.items():
            agg[k] = agg.get(k, 0.0) + v
        rows.append({"query_id": q, "reranker": args.name, "depth": depth,
                     "rank_of_first_gold": next((r for r, c in enumerate(fused, 1) if c in gold[q]), None),
                     "orig_rank_of_first_gold": B[q].get("orig_rank_of_first_gold"),
                     "num_gold": len(gold[q]), "subgroups": B[q].get("subgroups", []),
                     **m, "top10": fused[:10]})

    n = len(rows)
    summary = {f"{args.name}@{depth}": {**{k: v / n for k, v in agg.items()}, "n_queries": n,
                                        "depth": depth, "latency_p95_ms": None,
                                        "note": "post-hoc RRF over saved rankings; latency = sum of components"}}
    out = Path(args.out)
    (out / "rankings").mkdir(parents=True, exist_ok=True)
    with (out / "rankings" / f"{args.name}_at_{depth}.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    path = out / "reranker_metrics.json"
    existing = json.loads(path.read_text()) if path.exists() else {}
    existing.update(summary)
    path.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    s = summary[f"{args.name}@{depth}"]
    print(f"[{args.name}@{depth}] Hit@1={s['hit@1']:.4f} R@1={s['recall@1']:.4f} R@3={s['recall@3']:.4f} "
          f"MRR@10={s['mrr@10']:.4f} nDCG@10={s['ndcg@10']:.4f} Hit@10={s['hit@10']:.4f}  (n={n})")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
