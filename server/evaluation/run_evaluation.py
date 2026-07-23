from __future__ import annotations

"""
Benchmark runner: every question x every retrieval mode (CLAUDE.md Section 5.2/5.3).

WHAT IS MEASURED
  retrieval  chunk-level Recall@k / Precision@k / MRR@k / nDCG@k against ground truth read out of
             the corpus by build_benchmark.py. Source-level is also emitted but is near-ceiling
             for this corpus (see metrics.py) - quote the chunk-level numbers.
  answers    generated with the SHIPPED prompt and the SHIPPED context formatting (imported from
             main.py, not reimplemented, so the evaluation cannot drift from production).
  efficiency retrieval / rerank / generation latency split, context sizes, prompt-char estimate.

WHAT IS NOT MEASURED (stated so the report does not overclaim)
  - The per-student layer: MongoDB course history and the deterministic degree audit are NOT
    injected. The benchmark asks curriculum questions that need no student record, and the audit
    is deterministic code already covered by server/tests/test_advising.py.
  - answer_correctness / faithfulness / citation_correctness / hallucination_flag are left EMPTY
    for human labelling (Section 5.5). Only two objective automatic signals are filled in:
      auto_refusal_detected      did the answer decline for lack of evidence (phrase match)
      auto_reference_number_match did every ground-truth number appear in the answer
    Neither is a correctness judgement; both are defined precisely and reproducibly.

Usage:
  python server/evaluation/run_evaluation.py                     # all 5 modes, with answers
  python server/evaluation/run_evaluation.py --retrieval-only    # no LLM calls, no API cost
  python server/evaluation/run_evaluation.py --modes dense bm25 --limit 5
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SERVER_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SERVER_ROOT.parent
sys.path.insert(0, str(SERVER_ROOT))

# CHROMA_PERSIST_DIR defaults to the cwd-relative "./chroma_store"; anchor it before importing
# config so the runner works from any working directory.
os.environ.setdefault("CHROMA_PERSIST_DIR", str(SERVER_ROOT / "chroma_store"))
os.environ.setdefault("MONGO_URI", "mongodb://localhost:27017")  # lazy; never dialled here

from evaluation import metrics as M  # noqa: E402
from modules import retrieval_modes  # noqa: E402
from modules.catalog_retriever import retrieve_documents  # noqa: E402
from modules.config import (  # noqa: E402
    CROSS_ENCODER_MODEL_NAME,
    EMBEDDING_MODEL_NAME,
    GROQ_MODEL_NAME,
    LLM_PROVIDER,
    RERANK_TOP_K,
    RETRIEVAL_CANDIDATE_K,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)
from modules.load_vectorstore import get_vectorstore  # noqa: E402
from modules.retrieval_policy import build_metadata_filter  # noqa: E402

COURSE_CODE_RE = re.compile(r"\b[A-Z]{2,5}\s*\d{3}[A-Z]?\b")
NUMBER_RE = re.compile(r"\d+")


def load_benchmark(path: Path, limit: int | None = None) -> list[dict]:
    if not path.exists():
        sys.exit(f"benchmark not found: {path}\nRun: python server/evaluation/build_benchmark.py")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:limit] if limit else rows


def reference_numbers(reference_answer: str) -> list[str]:
    """Ground-truth numbers, with course codes stripped so 'ACC 301' does not become a value."""
    stripped = COURSE_CODE_RE.sub(" ", reference_answer or "")
    return NUMBER_RE.findall(stripped)


def answer_contains_reference_numbers(answer: str, reference_answer: str) -> bool | None:
    """True iff every ground-truth number occurs in the answer. None when there is none to check."""
    wanted = reference_numbers(reference_answer)
    if not wanted:
        return None
    text = answer or ""
    return all(re.search(rf"\b{re.escape(n)}\b", text) for n in wanted)


def _generate(question: str, context_docs, intent: str, retries: int = 3):
    """Answer with the production chain. Retries on transient provider errors (rate limits)."""
    from main import StaticRetriever  # shipped helpers - keeps eval faithful to production
    from modules.llm import get_llm_chain
    from modules.query_handlers import query_chain

    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            chain = get_llm_chain(StaticRetriever(documents=context_docs), intent=intent)
            return query_chain(chain, question), None
        except Exception as exc:  # noqa: BLE001 - provider errors must not kill the whole run
            last_error = exc
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
    return None, str(last_error)


def run_question(vs, item: dict, mode: str, top_k: int, candidate_k: int, ks, with_answers: bool) -> dict:
    intent = item.get("intent") or "ders_ayrintisi"
    profile = {"major": item.get("program"), "curriculum_term": item.get("curriculum_term")}
    metadata_filter = build_metadata_filter(intent, profile)
    question = item["question"]

    started = time.perf_counter()
    # Keep the pre-cut candidate ids: distinguishing "never retrieved" (retrieval_miss) from
    # "retrieved but ranked out of the context window" (retrieval_noise) is the whole point of
    # the Section 6 failure analysis, and it is unrecoverable after truncation.
    candidate_ids: list[str] = []

    def _hybrid_with_capture() -> list:
        docs = retrieve_documents(vs, question, k=candidate_k, metadata_filter=metadata_filter)
        candidate_ids.extend((d.metadata or {}).get("chunk_id", "") for d in docs)
        return docs

    outcome = retrieval_modes.retrieve(
        mode,
        vectorstore=vs,
        query=question,
        top_k=top_k,
        candidate_k=candidate_k,
        metadata_filter=metadata_filter,
        hybrid_search=_hybrid_with_capture,
    )

    retrieved_chunk_ids = [(d.metadata or {}).get("chunk_id", "") for d in outcome.documents]
    if not candidate_ids:  # bm25 / dense / llm_only never call the hybrid path
        candidate_ids = list(retrieved_chunk_ids)
    retrieved_sources = [(d.metadata or {}).get("source_document", "") for d in outcome.documents]

    gold_chunks = item.get("expected_chunk_ids") or []
    gold_sources = item.get("expected_sources") or []
    chunk_metrics = M.evaluate_retrieval(retrieved_chunk_ids, gold_chunks, ks)
    source_metrics = M.evaluate_retrieval(retrieved_sources, gold_sources, ks)

    row = {
        "query_id": item["id"],
        "question": question,
        "language": item.get("language"),
        "question_type": item.get("question_type"),
        "answerable": item.get("answerable"),
        "mode": mode,
        "intent": intent,
        "prompt_strategy": "basic",
        "expert_mode": "auto",
        "top_k": top_k,
        "metadata_filter": metadata_filter,
        "reference_answer": item.get("reference_answer"),
        "expected_chunk_ids": gold_chunks,
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "candidate_chunk_ids": candidate_ids,
        "num_retrieved_chunks": outcome.candidate_count,
        "num_final_context_chunks": len(outcome.documents),
        "reranked": outcome.reranked,
        "retrieval_ms": outcome.retrieval_ms,
        "rerank_ms": outcome.rerank_ms,
        "chunk_metrics": chunk_metrics,
        "source_metrics": source_metrics,
        # Section 5.5 - left empty for manual/RAGAS labelling, never auto-filled.
        "answer_correctness": None,
        "citation_correctness": None,
        "faithfulness": None,
        "answer_relevancy": None,
        "hallucination_flag": None,
        "manual_notes": "",
    }

    if not with_answers:
        row.update(
            {"answer": None, "sources": [], "generation_ms": None, "total_ms": round((time.perf_counter() - started) * 1000, 2)}
        )
        return row

    gen_started = time.perf_counter()
    if mode == "llm_only":
        from modules.llm import answer_without_context

        try:
            answer, sources, error = answer_without_context(question), [], None
        except Exception as exc:  # noqa: BLE001
            answer, sources, error = "", [], str(exc)
        result = {"response": answer, "sources": sources, "source_chunk_ids": []}
    else:
        from main import _format_for_context, _intent_context_document

        context_docs = _format_for_context(outcome.documents)
        context_docs.insert(0, _intent_context_document(intent, []))
        result, error = _generate(question, context_docs, intent)
        if result is None:
            result = {"response": "", "sources": [], "source_chunk_ids": []}
    generation_ms = round((time.perf_counter() - gen_started) * 1000, 2)

    answer = result.get("response", "") or ""
    prompt_chars = sum(len(d.page_content) for d in outcome.documents) + len(question)
    row.update({
        "answer": answer,
        "sources": result.get("sources", []),
        "answer_source_chunk_ids": result.get("source_chunk_ids", []),
        "generation_error": error,
        "generation_ms": generation_ms,
        "total_ms": round((time.perf_counter() - started) * 1000, 2),
        "prompt_chars_estimate": prompt_chars,
        "prompt_tokens_estimate": round(prompt_chars / 4),  # ~4 chars/token, ESTIMATE only
        "auto_refusal_detected": M.looks_like_refusal(answer),
        "auto_reference_number_match": answer_contains_reference_numbers(answer, item.get("reference_answer", "")),
    })
    return row


def summarize(rows: list[dict], ks) -> dict:
    """Aggregate per mode. Retrieval metrics use answerable questions only (gold exists there)."""
    summary: dict[str, dict] = {}
    for mode in sorted({r["mode"] for r in rows}):
        mode_rows = [r for r in rows if r["mode"] == mode]
        answerable = [r for r in mode_rows if r["answerable"]]
        answered = [r for r in mode_rows if r.get("answer")]

        # A generation that never happened (provider 429/timeout) is NOT a wrong answer. Scoring
        # it as one silently turns an outage into a quality result, so failed rows are excluded
        # from every answer-side statistic and reported separately as generation_coverage.
        def produced(row: dict) -> bool:
            return not row.get("generation_error") and bool(row.get("answer"))

        scorable = [r for r in answerable if produced(r)]
        unanswerable = [r for r in mode_rows if not r["answerable"] and produced(r)]
        attempted = [r for r in mode_rows if r.get("generation_ms") is not None]

        number_checks = [r["auto_reference_number_match"] for r in scorable
                         if r.get("auto_reference_number_match") is not None]
        refusals_on_unanswerable = [r for r in unanswerable if r.get("auto_refusal_detected")]

        summary[mode] = {
            "questions": len(mode_rows),
            "answerable_questions": len(answerable),
            "retrieval_chunk_level": M.aggregate([r["chunk_metrics"] for r in answerable]),
            "retrieval_source_level": M.aggregate([r["source_metrics"] for r in answerable]),
            "efficiency": {
                "mean_retrieval_ms": M.mean_ignoring_nan([r["retrieval_ms"] for r in mode_rows]),
                "mean_rerank_ms": M.mean_ignoring_nan([r["rerank_ms"] for r in mode_rows]),
                "mean_generation_ms": M.mean_ignoring_nan(
                    [r["generation_ms"] for r in answered if r.get("generation_ms") is not None]
                ),
                "mean_total_ms": M.mean_ignoring_nan([r["total_ms"] for r in mode_rows]),
                "mean_context_chunks": M.mean_ignoring_nan([r["num_final_context_chunks"] for r in mode_rows]),
                "mean_prompt_tokens_estimate": M.mean_ignoring_nan(
                    [r.get("prompt_tokens_estimate") for r in answered if r.get("prompt_tokens_estimate") is not None]
                ),
            },
            "answer_signals": {
                "reference_number_match_rate": (
                    sum(1 for v in number_checks if v) / len(number_checks) if number_checks else None
                ),
                "reference_number_checked": len(number_checks),
                "refusal_rate_on_unanswerable": (
                    len(refusals_on_unanswerable) / len(unanswerable) if unanswerable else None
                ),
                "unanswerable_questions": len(unanswerable),
                "generation_errors": sum(1 for r in mode_rows if r.get("generation_error")),
                # How much of this mode's answer column is actually usable. Anything below 1.0
                # means the provider failed on some questions; treat the answer metrics for that
                # mode as partial and say so in the report.
                "generation_coverage": (
                    len([r for r in attempted if produced(r)]) / len(attempted) if attempted else None
                ),
                "answers_scored": len(scorable),
            },
            "manual_labels_pending": True,
        }
    return summary


def print_summary(summary: dict, k: int) -> None:
    def fmt(value, spec=".3f"):
        return format(value, spec) if isinstance(value, (int, float)) else "n/a"

    print(f"\n{'mode':<15}{'recall@'+str(k):>11}{'mrr@'+str(k):>10}{'ndcg@'+str(k):>10}"
          f"{'numMatch':>10}{'refuse':>9}{'answers':>9}{'total ms':>10}")
    for mode, data in summary.items():
        chunk = data["retrieval_chunk_level"]
        sig, eff = data["answer_signals"], data["efficiency"]
        coverage = sig.get("generation_coverage")
        print(f"{mode:<15}{fmt(chunk.get(f'recall@{k}')):>11}{fmt(chunk.get(f'mrr@{k}')):>10}"
              f"{fmt(chunk.get(f'ndcg@{k}')):>10}{fmt(sig['reference_number_match_rate']):>10}"
              f"{fmt(sig['refusal_rate_on_unanswerable']):>9}{fmt(coverage, '.0%'):>9}"
              f"{fmt(eff['mean_total_ms'], '.0f'):>10}")
    incomplete = [m for m, d in summary.items()
                  if (d["answer_signals"].get("generation_coverage") or 1.0) < 1.0]
    if incomplete:
        print(f"\n!! partial answer coverage for: {', '.join(incomplete)} "
              f"- the provider failed on some questions. Answer-side numbers for those modes are "
              f"computed over the answers that DID come back; do not quote them as complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the adviSU retrieval/answer benchmark.")
    parser.add_argument("--benchmark", default=str(PROJECT_ROOT / "data" / "benchmark" / "questions.jsonl"))
    parser.add_argument("--modes", nargs="+", default=list(retrieval_modes.RETRIEVAL_MODES))
    parser.add_argument("--top-k", type=int, default=RERANK_TOP_K)
    parser.add_argument("--candidate-k", type=int, default=RETRIEVAL_CANDIDATE_K)
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 3, 5, 10])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--retrieval-only", action="store_true", help="skip all LLM calls (no API cost)")
    parser.add_argument("--out-root", default=str(PROJECT_ROOT / "outputs" / "evaluation_runs"))
    parser.add_argument("--tag", default="", help="suffix for the run directory name")
    parser.add_argument("--summarize-only", default=None,
                        help="re-aggregate an existing run directory in place (no retrieval, no LLM calls)")
    args = parser.parse_args()

    if args.summarize_only:
        run_dir = Path(args.summarize_only)
        rows = [json.loads(l) for l in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        summary = summarize(rows, args.ks)
        (run_dir / "summary_metrics.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print_summary(summary, args.top_k)
        print(f"\nre-summarized {len(rows)} rows in {run_dir}")
        return

    modes = [retrieval_modes.normalize_mode(m) for m in args.modes]
    # The headline number is reported at top_k, so it must always be one of the measured cutoffs.
    args.ks = sorted({*args.ks, args.top_k})
    with_answers = not args.retrieval_only
    items = load_benchmark(Path(args.benchmark), args.limit)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(args.out_root) / (f"{stamp}_{args.tag}" if args.tag else stamp)
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"benchmark: {len(items)} questions | modes: {', '.join(modes)} | answers: {with_answers}")
    print(f"output:    {run_dir}")

    vs = get_vectorstore()
    run_started = time.perf_counter()
    rows: list[dict] = []
    for mode in modes:
        for index, item in enumerate(items, start=1):
            row = run_question(vs, item, mode, args.top_k, args.candidate_k, args.ks, with_answers)
            rows.append(row)
            recall = row["chunk_metrics"].get(f"recall@{args.top_k}")
            shown = f"{recall:.2f}" if isinstance(recall, float) and recall == recall else "n/a"
            flag = "" if row["answerable"] else " (unanswerable)"
            print(f"  [{mode:<13}] {item['id']} {index:>2}/{len(items)} "
                  f"recall@{args.top_k}={shown:<5} {row['total_ms']:>8.0f}ms{flag}")

    summary = summarize(rows, args.ks)

    # ---- artifacts (Section 5.3) ------------------------------------------------------
    with (run_dir / "results.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    with (run_dir / "retrieved_chunks.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps({
                "query_id": row["query_id"],
                "mode": row["mode"],
                "expected_chunk_ids": row["expected_chunk_ids"],
                "retrieved_chunk_ids": row["retrieved_chunk_ids"],
                "hit": bool(set(row["retrieved_chunk_ids"]) & set(row["expected_chunk_ids"])),
            }, ensure_ascii=False) + "\n")

    (run_dir / "summary_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        import ragas  # noqa: F401
        ragas_available = True
    except Exception:
        ragas_available = False  # Section 5.6: RAGAS is optional and skipped when absent

    (run_dir / "run_config.json").write_text(json.dumps({
        "timestamp_utc": stamp,
        "benchmark": str(Path(args.benchmark)),
        "benchmark_questions": len(items),
        "modes": modes,
        "top_k": args.top_k,
        "candidate_k": args.candidate_k,
        "ks": args.ks,
        "answers_generated": with_answers,
        "llm_provider": LLM_PROVIDER,
        "llm_model": GROQ_MODEL_NAME,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "cross_encoder_model": CROSS_ENCODER_MODEL_NAME,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "ragas_available": ragas_available,
        "wall_clock_s": round(time.perf_counter() - run_started, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print_summary(summary, args.top_k)
    print(f"\nartifacts written to {run_dir}")
    print("answer_correctness / faithfulness / citation_correctness / hallucination_flag are "
          "intentionally empty - label them manually before quoting answer quality.")


if __name__ == "__main__":
    main()
