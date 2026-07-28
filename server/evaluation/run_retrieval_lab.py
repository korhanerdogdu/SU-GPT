from __future__ import annotations

"""
Run retrieval experiments over the adviSU benchmark splits and write machine-readable results.

Everything the report and the figures show is produced here; no metric is ever typed into a
plotting script by hand.

LATENCY IS MEASURED TWICE, ON PURPOSE
Ranking quality is computed from a batched pass (one forward pass per split for dense
models), because that is the cheap way to score 500 queries. But batched latency is not
serving latency, so a separate single-query pass over a sample re-measures each retriever
one query at a time - that is the number reported as mean/p50/p95, and it is the one that
reflects what a user waits for.

Usage:
    python server/evaluation/run_retrieval_lab.py --split dev --methods bm25_original,bm25f
    python server/evaluation/run_retrieval_lab.py --split test --methods all --out-tag final
"""

import argparse
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retrieval_lab import metrics as M
from retrieval_lab.corpus import PROJECT_ROOT, load_corpus
from retrieval_lab.retrievers import BuildInfo

OUT_ROOT = PROJECT_ROOT / "outputs" / "retrieval_lab"
TOP_K_MAX = 100
LATENCY_SAMPLE = 60


# --- subgroups ---------------------------------------------------------------------------

def subgroups_for(item: dict) -> list[str]:
    """Every subgroup an item belongs to. A query can be in several."""
    groups: list[str] = [f"type:{item.get('question_type', '?')}"]
    groups.append(f"lang:{item.get('language', '?')}")
    q = item.get("question", "")
    from retrieval_lab.text import extract_course_codes, extract_term_codes

    if extract_course_codes(q):
        groups.append("has_course_code")
    if extract_term_codes(q):
        groups.append("has_catalog_year")
    if item.get("program"):
        groups.append("program_specific")
    if item.get("multi_evidence"):
        groups.append("multi_evidence")
    else:
        groups.append("single_evidence")
    if item.get("intent") == "minor":
        groups.append("minor")
    if item.get("language") == "tr":
        # Turkish query over an English corpus: every adviSU document body is English.
        groups.append("cross_language")
    if item.get("question_type") in {"catalog_year", "program_specific"}:
        groups.append("hard_negatives")
    if item.get("origin") == "original_benchmark":
        groups.append("original_benchmark")
    return groups


# --- provenance --------------------------------------------------------------------------

def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def hardware() -> dict:
    info = {"platform": platform.platform(), "python": sys.version.split()[0], "machine": platform.machine()}
    try:
        import torch

        info["torch"] = torch.__version__
        info["device"] = "mps" if torch.backends.mps.is_available() else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    except Exception:
        info["device"] = "cpu"
    return info


# --- loading -------------------------------------------------------------------------------

def load_split(split: str) -> list[dict]:
    path = PROJECT_ROOT / "data" / "benchmark" / f"retrieval_{split}.jsonl"
    if not path.exists():
        sys.exit(f"missing split: {path}\nRun: python server/evaluation/build_retrieval_benchmark.py")
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def build_registry(corpus, include_dense: bool, include_rerank: bool) -> dict:
    from retrieval_lab.retrievers import (
        DenseRetriever,
        FusionRetriever,
        MetadataRetriever,
        RerankRetriever,
        sparse_factories,
    )

    reg = dict(sparse_factories())
    if include_dense:
        for key in ("minilm", "e5_small", "e5_base", "e5_large_instruct", "bge_m3", "e5_small_contextual"):
            reg[f"dense_{key}"] = (lambda k: (lambda c: DenseRetriever(f"dense_{k}", c, k)))(key)

    # metadata-aware sparse (no model needed)
    reg["metadata_bm25f"] = lambda c: MetadataRetriever(
        "metadata_bm25f", c, sparse_factories()["bm25f"](c), policy="boost"
    )
    reg["metadata_filter_bm25f"] = lambda c: MetadataRetriever(
        "metadata_filter_bm25f", c, sparse_factories()["bm25f"](c), policy="filter"
    )

    if include_dense:
        def hybrid(name, dense_key, metadata_policy="none", weights=None):
            def make(c):
                sf = sparse_factories()
                return FusionRetriever(
                    name, c,
                    [sf["bm25f"](c), DenseRetriever(f"dense_{dense_key}", c, dense_key)],
                    metadata_policy=metadata_policy, weights=weights,
                )
            return make

        reg["hybrid_bm25f_e5"] = hybrid("hybrid_bm25f_e5", "e5_small")
        reg["hybrid_bm25f_e5base"] = hybrid("hybrid_bm25f_e5base", "e5_base")
        reg["hybrid_bm25f_bge"] = hybrid("hybrid_bm25f_bge", "bge_m3")
        reg["hybrid_bm25f_e5_meta"] = hybrid("hybrid_bm25f_e5_meta", "e5_small", metadata_policy="boost")
        reg["hybrid_bm25f_bge_meta"] = hybrid("hybrid_bm25f_bge_meta", "bge_m3", metadata_policy="boost")

        if include_rerank:
            for depth in (20, 50, 100):
                def mk(d):
                    def make(c):
                        sf = sparse_factories()
                        base = FusionRetriever(
                            "first", c,
                            [sf["bm25f"](c), DenseRetriever("dense_e5_small", c, "e5_small")],
                            metadata_policy="boost",
                        )
                        return RerankRetriever(f"hybrid_meta_rerank{d}", c, base, candidate_depth=d)
                    return make
                reg[f"hybrid_meta_rerank{depth}"] = mk(depth)
    return reg


# --- running ---------------------------------------------------------------------------------

def run_method(name, factory, corpus, items, *, latency_sample: int, save_depth: int = 100) -> dict:
    print(f"  [{name}] building index...", flush=True)
    t0 = time.perf_counter()
    retriever = factory(corpus)
    build_s = time.perf_counter() - t0
    binfo: BuildInfo = getattr(retriever, "build", BuildInfo())

    queries = [i["question"] for i in items]
    print(f"  [{name}] scoring {len(queries)} queries...", flush=True)
    t0 = time.perf_counter()
    rankings = retriever.search_batch(queries, TOP_K_MAX)
    batch_s = time.perf_counter() - t0

    # separate single-query latency pass (serving-realistic)
    sample = queries[:latency_sample]
    lat: list[float] = []
    for q in sample:
        t = time.perf_counter()
        retriever.search_batch([q], TOP_K_MAX)
        lat.append((time.perf_counter() - t) * 1000.0)
    lat.sort()

    scores: list[M.QueryScore] = []
    per_query: list[dict] = []
    for item, ranking in zip(items, rankings):
        gold = set(item.get("expected_chunk_ids") or [])
        ranked_ids = [cid for cid, _ in ranking]
        if not gold:
            per_query.append({
                "query_id": item["id"], "method": name, "scorable": False,
                "retrieved": ranked_ids[:save_depth],
            })
            continue
        s = M.score_query(item["id"], ranked_ids, gold)
        scores.append(s)
        per_query.append({
            "query_id": item["id"], "method": name, "scorable": True,
            "num_gold": len(gold),
            "multi_evidence": bool(item.get("multi_evidence")),
            "subgroups": subgroups_for(item),
            **{f"recall@{k}": s.recall[k] for k in M.K_VALUES},
            "mrr@10": s.mrr10, "ndcg@10": s.ndcg10, "hit@1": s.hit1,
            **{f"evidence_set_recall@{k}": s.evidence_recall[k] for k in (5, 10, 20)},
            **{f"all_gold@{k}": s.all_gold[k] for k in (5, 10, 20)},
            "rank_of_first_gold": next((r for r, c in enumerate(ranked_ids, 1) if c in gold), None),
            # Persist the full candidate list, not just the top 10: reranking experiments
            # need the deep pool, and R@20/R@50 must be independently recomputable from the
            # saved artifacts rather than taken on trust from summary.json.
            "retrieved": ranked_ids[:save_depth],
        })

    agg = M.aggregate(scores)
    agg.update({
        "latency_mean_ms": sum(lat) / len(lat) if lat else 0.0,
        "latency_p50_ms": lat[len(lat) // 2] if lat else 0.0,
        "latency_p95_ms": lat[min(len(lat) - 1, int(0.95 * len(lat)))] if lat else 0.0,
        "batch_seconds_total": batch_s,
        "index_build_seconds": build_s,
        "index_bytes": binfo.index_bytes,
        "build_notes": binfo.notes,
        "latency_sample_n": len(lat),
    })

    # subgroup metrics
    sub: dict[str, dict] = {}
    by_group: dict[str, list[M.QueryScore]] = {}
    score_by_id = {s.query_id: s for s in scores}
    for row in per_query:
        if not row.get("scorable"):
            continue
        for g in row["subgroups"]:
            by_group.setdefault(g, []).append(score_by_id[row["query_id"]])
    for g, ss in sorted(by_group.items()):
        sub[g] = M.aggregate(ss)

    return {"name": name, "summary": agg, "subgroups": sub, "per_query": per_query}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    ap.add_argument("--methods", default="sparse",
                    help="comma-separated names, or 'sparse' / 'all'")
    ap.add_argument("--limit", type=int, default=0, help="smoke-test on the first N queries")
    ap.add_argument("--out-tag", default="")
    ap.add_argument("--latency-sample", type=int, default=LATENCY_SAMPLE)
    ap.add_argument("--save-depth", type=int, default=100,
                    help="how many ranked chunk ids to persist per query (reranking needs the deep pool)")
    ap.add_argument("--no-dense", action="store_true")
    ap.add_argument("--no-rerank", action="store_true")
    args = ap.parse_args()

    corpus = load_corpus()
    items = load_split(args.split)
    if args.limit:
        items = items[: args.limit]

    registry = build_registry(corpus, include_dense=not args.no_dense, include_rerank=not args.no_rerank)
    if args.methods == "all":
        names = list(registry)
    elif args.methods == "sparse":
        names = [n for n in registry if not n.startswith(("dense_", "hybrid_"))]
    else:
        names = [n.strip() for n in args.methods.split(",") if n.strip()]
    unknown = [n for n in names if n not in registry]
    if unknown:
        sys.exit(f"unknown methods: {unknown}\navailable: {sorted(registry)}")

    tag = args.out_tag or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_dir = OUT_ROOT / f"{args.split}__{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "rankings").mkdir(exist_ok=True)

    print(f"corpus: {len(corpus):,} chunks (fingerprint {corpus.fingerprint})")
    print(f"split : {args.split}  n={len(items)}  scorable={sum(1 for i in items if i['expected_chunk_ids'])}")
    print(f"out   : {out_dir}\n")

    results = []
    for name in names:
        try:
            res = run_method(name, registry[name], corpus, items, latency_sample=args.latency_sample,
                             save_depth=args.save_depth)
        except Exception as exc:  # a broken candidate must not abort the whole sweep
            print(f"  [{name}] FAILED: {type(exc).__name__}: {exc}")
            (out_dir / "failures.jsonl").open("a").write(
                json.dumps({"method": name, "error": f"{type(exc).__name__}: {exc}"}) + "\n"
            )
            continue
        results.append(res)
        s = res["summary"]
        print(f"  [{name}] R@10={s.get('recall@10', 0):.4f}  R@1={s.get('recall@1', 0):.4f}  "
              f"MRR@10={s.get('mrr@10', 0):.4f}  nDCG@10={s.get('ndcg@10', 0):.4f}  "
              f"p95={s.get('latency_p95_ms', 0):.1f}ms\n", flush=True)
        with (out_dir / "rankings" / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
            for row in res["per_query"]:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    (out_dir / "summary.json").write_text(
        json.dumps({r["name"]: r["summary"] for r in results}, indent=2), encoding="utf-8"
    )
    (out_dir / "subgroups.json").write_text(
        json.dumps({r["name"]: r["subgroups"] for r in results}, indent=2), encoding="utf-8"
    )
    (out_dir / "run_config.json").write_text(json.dumps({
        "split": args.split,
        "methods": names,
        "n_items": len(items),
        "n_scorable": sum(1 for i in items if i["expected_chunk_ids"]),
        "corpus_fingerprint": corpus.fingerprint,
        "corpus_chunks": len(corpus),
        "git_commit": git_commit(),
        "hardware": hardware(),
        "top_k_max": TOP_K_MAX,
        "latency_sample": args.latency_sample,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, indent=2), encoding="utf-8")

    print(f"\nwrote {out_dir}/summary.json, subgroups.json, rankings/*.jsonl, run_config.json")


if __name__ == "__main__":
    main()
