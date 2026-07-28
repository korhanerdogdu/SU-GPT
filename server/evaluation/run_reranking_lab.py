from __future__ import annotations

"""
Run reranking experiments over FROZEN first-stage candidate pools.

Candidates are read from a completed run of run_retrieval_lab.py, never regenerated. Every
reranker therefore sees byte-identical input, so a difference between two rows of the output
is a difference in ordering ability and nothing else. Candidate depth is the only exception:
it is an explicit experimental variable (`--depths`).

Metric semantics (see retrieval_lab/metrics.py for formulas):
  Hit@K        - at least one gold chunk in the top K
  Recall@K     - FRACTION of the gold set in the top K (capped at K/|gold|)
  MRR@10, nDCG@10, MAP@10
  EvidenceSetRecall@K / AllGold@K - multi-evidence coverage
Hit@1 and Recall@1 are reported separately and are NOT interchangeable for multi-gold queries.

Usage:
    python server/evaluation/run_reranking_lab.py --run outputs/retrieval_lab/dev__X \
        --split dev --rerankers original_ranking,metadata_rule_rerank --depths 10,25,50
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retrieval_lab import metrics as M
from retrieval_lab.corpus import PROJECT_ROOT, load_corpus
from retrieval_lab.fusion import build_program_lexicon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from candidate_ceiling import load_gold, load_rankings  # noqa: E402

OUT_ROOT = PROJECT_ROOT / "outputs" / "reranking"


def per_query_metrics(ranked: list[str], gold: set[str]) -> dict[str, float]:
    return {
        "hit@1": M.hit_at_k(ranked, gold, 1),
        "hit@3": M.hit_at_k(ranked, gold, 3),
        "hit@10": M.hit_at_k(ranked, gold, 10),
        "recall@1": M.true_recall_at_k(ranked, gold, 1),
        "recall@3": M.true_recall_at_k(ranked, gold, 3),
        "recall@5": M.true_recall_at_k(ranked, gold, 5),
        "recall@10": M.true_recall_at_k(ranked, gold, 10),
        "mrr@10": M.reciprocal_rank(ranked, gold, 10),
        "ndcg@10": M.ndcg_at_k(ranked, gold, 10),
        "map@10": _ap_at_k(ranked, gold, 10),
        "evidence_set_recall@5": M.evidence_set_recall_at_k(ranked, gold, 5),
        "evidence_set_recall@10": M.evidence_set_recall_at_k(ranked, gold, 10),
        "all_gold@5": M.all_gold_at_k(ranked, gold, 5),
        "all_gold@10": M.all_gold_at_k(ranked, gold, 10),
    }


def _ap_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    """Average precision: mean of precision@i over the positions where a gold chunk lands."""
    if not gold:
        return 0.0
    hits = 0
    total = 0.0
    for i, cid in enumerate(ranked[:k], 1):
        if cid in gold:
            hits += 1
            total += hits / i
    return total / min(len(gold), k)


def build_rerankers(names: list[str], corpus, lexicon) -> dict:
    from retrieval_lab.rerank import (
        QWEN_DEFAULT_INSTRUCTION,
        QWEN_DOMAIN_INSTRUCTION,
        CrossEncoderReranker,
        FusionReranker,
        IdentityReranker,
        MetadataRuleReranker,
        Qwen3Reranker,
    )

    def ce(name, model, doc="contextual", q="raw", ml=512, bs=32):
        return lambda: CrossEncoderReranker(corpus, lexicon, name=name, model_name=model,
                                            doc_format=doc, query_format=q, max_length=ml,
                                            batch_size=bs)

    def qwen(name, instr, doc="contextual", q="raw"):
        return lambda: Qwen3Reranker(corpus, lexicon, name=name, instruction=instr,
                                     doc_format=doc, query_format=q)

    reg: dict[str, callable] = {
        "original_ranking": lambda: IdentityReranker(),
        "metadata_rule_rerank": lambda: MetadataRuleReranker(corpus, lexicon),
        # existing repo reranker - the incumbent that must be reproduced, not silently replaced
        "existing_msmarco": ce("existing_msmarco", "cross-encoder/ms-marco-MiniLM-L-6-v2",
                               doc="raw", ml=512, bs=64),
        "msmarco_contextual": ce("msmarco_contextual", "cross-encoder/ms-marco-MiniLM-L-6-v2",
                                 doc="contextual", ml=512, bs=64),
        "bge_v2m3": ce("bge_v2m3", "BAAI/bge-reranker-v2-m3", doc="contextual", ml=512, bs=16),
        "bge_v2m3_raw": ce("bge_v2m3_raw", "BAAI/bge-reranker-v2-m3", doc="raw", ml=512, bs=16),
        "bge_v2m3_structq": ce("bge_v2m3_structq", "BAAI/bge-reranker-v2-m3", doc="contextual",
                               q="structured", ml=512, bs=16),
        "qwen3_06b_domain": qwen("qwen3_06b_domain", QWEN_DOMAIN_INSTRUCTION),
        "qwen3_06b_default": qwen("qwen3_06b_default", QWEN_DEFAULT_INSTRUCTION),
        "qwen3_06b_raw": qwen("qwen3_06b_raw", QWEN_DOMAIN_INSTRUCTION, doc="raw"),
    }
    # fusions of a neural reranker with the first-stage order and the rule reranker
    # Fuse the BEST-performing BGE input format (contextual doc + structured query), not the
    # default one: fusing a weaker variant would understate what fusion can do.
    reg["fuse_bge_first"] = lambda: FusionReranker(
        "fuse_bge_first", [reg["bge_v2m3_structq"](), IdentityReranker()])
    reg["fuse_bge_first_w2"] = lambda: FusionReranker(
        "fuse_bge_first_w2", [reg["bge_v2m3_structq"](), IdentityReranker()], weights=[1.0, 2.0])
    reg["fuse_bge_first_rules"] = lambda: FusionReranker(
        "fuse_bge_first_rules", [reg["bge_v2m3_structq"](), IdentityReranker(),
                                 MetadataRuleReranker(corpus, lexicon)])
    reg["fuse_rules_first"] = lambda: FusionReranker(
        "fuse_rules_first", [MetadataRuleReranker(corpus, lexicon), IdentityReranker()])
    reg["fuse_qwen_first"] = lambda: FusionReranker(
        "fuse_qwen_first", [reg["qwen3_06b_domain"](), IdentityReranker()])
    reg["fuse_qwen_first_rules"] = lambda: FusionReranker(
        "fuse_qwen_first_rules", [reg["qwen3_06b_domain"](), IdentityReranker(),
                                  MetadataRuleReranker(corpus, lexicon)])

    unknown = [n for n in names if n not in reg]
    if unknown:
        sys.exit(f"unknown rerankers: {unknown}\navailable: {sorted(reg)}")
    return {n: reg[n] for n in names}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="first-stage run dir holding rankings/")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--method", default="hybrid_bm25f_e5_meta", help="first-stage method")
    ap.add_argument("--rerankers", default="original_ranking,metadata_rule_rerank")
    ap.add_argument("--depths", default="25")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-tag", default="")
    args = ap.parse_args()

    corpus = load_corpus()
    lexicon = build_program_lexicon(corpus)
    rankings = load_rankings(Path(args.run), args.method)
    gold = load_gold(args.split)
    qids = sorted(set(rankings) & set(gold))
    if args.limit:
        qids = qids[: args.limit]

    questions = {}
    for line in (PROJECT_ROOT / "data" / "benchmark" / f"retrieval_{args.split}.jsonl").read_text(
        encoding="utf-8"
    ).splitlines():
        if line.strip():
            r = json.loads(line)
            questions[r["id"]] = r

    names = [n.strip() for n in args.rerankers.split(",") if n.strip()]
    depths = [int(d) for d in args.depths.split(",") if d.strip()]
    factories = build_rerankers(names, corpus, lexicon)

    tag = args.out_tag or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_dir = OUT_ROOT / f"{args.split}__{tag}"
    (out_dir / "rankings").mkdir(parents=True, exist_ok=True)

    print(f"first stage : {args.method}  ({args.run})")
    print(f"split       : {args.split}  n={len(qids)}")
    print(f"rerankers   : {names}")
    print(f"depths      : {depths}")
    print(f"out         : {out_dir}\n")

    summary: dict[str, dict] = {}
    for depth in depths:
        for name in names:
            key = f"{name}@{depth}"
            try:
                rr = factories[name]()
            except Exception as exc:
                print(f"  [{key}] BUILD FAILED: {type(exc).__name__}: {exc}")
                continue
            lat: list[float] = []
            rows = []
            agg: dict[str, float] = {}
            try:
                # Periodic progress. Without it a stalled run is indistinguishable from a slow
                # one: a Qwen3 run once wedged on an MPS out-of-memory error and spent 2h
                # emitting nothing but Metal errors, which looked identical to "still working".
                progress_every = max(1, len(qids) // 10)
                for done, qid in enumerate(qids, 1):
                    if done % progress_every == 0 or done == len(qids):
                        rate = sum(lat) / len(lat) if lat else 0.0
                        print(f"    [{key}] {done}/{len(qids)} queries  "
                              f"({rate:.0f} ms/query avg)", flush=True)
                    pool = rankings[qid][:depth]
                    base = [(c, 1.0 / (i + 1)) for i, c in enumerate(pool)]
                    t0 = time.perf_counter()
                    out = rr.rerank(questions[qid]["question"], pool, base)
                    lat.append((time.perf_counter() - t0) * 1000.0)
                    ranked = [c for c, _ in out]
                    # A reranker must return a permutation - never drop a candidate.
                    if len(set(ranked)) != len(set(pool)):
                        missing = set(pool) - set(ranked)
                        raise RuntimeError(f"{key} lost {len(missing)} candidates on {qid}")
                    m = per_query_metrics(ranked, gold[qid])
                    for k, v in m.items():
                        agg[k] = agg.get(k, 0.0) + v
                    rows.append({
                        "query_id": qid, "reranker": name, "depth": depth,
                        "rank_of_first_gold": next(
                            (r for r, c in enumerate(ranked, 1) if c in gold[qid]), None),
                        "orig_rank_of_first_gold": next(
                            (r for r, c in enumerate(rankings[qid], 1) if c in gold[qid]), None),
                        "num_gold": len(gold[qid]),
                        "subgroups": _subgroups(questions[qid]),
                        **m,
                        "top10": ranked[:10],
                    })
            except Exception as exc:
                print(f"  [{key}] FAILED: {type(exc).__name__}: {exc}")
                (out_dir / "failures.jsonl").open("a").write(
                    json.dumps({"reranker": key, "error": f"{type(exc).__name__}: {exc}"}) + "\n")
                continue

            n = len(rows) or 1
            s = {k: v / n for k, v in agg.items()}
            lat.sort()
            s.update({
                "n_queries": len(rows), "depth": depth,
                "latency_mean_ms": sum(lat) / len(lat) if lat else 0.0,
                "latency_p50_ms": lat[len(lat) // 2] if lat else 0.0,
                "latency_p95_ms": lat[min(len(lat) - 1, int(0.95 * len(lat)))] if lat else 0.0,
                "model_load_seconds": getattr(rr, "load_seconds", 0.0),
            })
            if hasattr(rr, "trunc"):
                s.update({f"trunc_{k}": v for k, v in rr.trunc.as_dict().items()})
            summary[key] = s
            with (out_dir / "rankings" / f"{key.replace('@', '_at_')}.jsonl").open(
                "w", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"  [{key}] Hit@1={s['hit@1']:.4f} R@1={s['recall@1']:.4f} R@3={s['recall@3']:.4f} "
                  f"MRR@10={s['mrr@10']:.4f} nDCG@10={s['ndcg@10']:.4f} Hit@10={s['hit@10']:.4f} "
                  f"p95={s['latency_p95_ms']:.0f}ms", flush=True)

    (out_dir / "reranker_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "run_config.json").write_text(json.dumps({
        "first_stage_run": str(args.run), "first_stage_method": args.method,
        "split": args.split, "rerankers": names, "depths": depths,
        "n_queries": len(qids), "corpus_fingerprint": corpus.fingerprint,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out_dir}/reranker_metrics.json")


def _subgroups(item: dict) -> list[str]:
    from retrieval_lab.text import extract_course_codes, extract_term_codes

    g = [f"type:{item.get('question_type','?')}", f"lang:{item.get('language','?')}"]
    if extract_course_codes(item["question"]):
        g.append("has_course_code")
    if extract_term_codes(item["question"]):
        g.append("has_catalog_year")
    if item.get("multi_evidence"):
        g.append("multi_evidence")
    else:
        g.append("single_evidence")
    if item.get("language") == "tr":
        g.append("cross_language")
    if item.get("intent") == "minor":
        g.append("minor")
    if item.get("question_type") in {"catalog_year", "program_specific"}:
        g.append("hard_negatives")
    return g


if __name__ == "__main__":
    main()
