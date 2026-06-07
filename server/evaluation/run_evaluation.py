"""Run SU-GPT benchmark questions across retrieval modes.

This script implements CLAUDE.md Section 5 only. It intentionally does not
depend on logging_utils or Section 6 ablation/failure-analysis modules so the
evaluation runner can be used while those tasks evolve independently.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

EVALUATION_DIR = Path(__file__).resolve().parent
SERVER_ROOT = EVALUATION_DIR.parents[0]
REPO_ROOT = EVALUATION_DIR.parents[1]
sys.path.insert(0, str(EVALUATION_DIR))
sys.path.insert(0, str(SERVER_ROOT))

from metrics import retrieval_metrics  # noqa: E402
from modules.load_vectorstore import get_vectorstore  # noqa: E402
from modules.llm import get_llm_chain  # noqa: E402
from modules.query_handlers import query_chain  # noqa: E402
from modules.retrieval_modes import SUPPORTED_MODES, run_retrieval_mode  # noqa: E402


PROMPT_STRATEGY = "basic"
EXPERT_MODE = "auto"
TOKEN_CHARS_PER_TOKEN = 4
PROMPT_COST_PER_1K = 0.00005
COMPLETION_COST_PER_1K = 0.00008


class StaticRetriever(BaseRetriever):
    documents: list[Document] = Field(default_factory=list)

    def _get_relevant_documents(self, query: str) -> list[Document]:
        return self.documents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SU-GPT Section 5 benchmark evaluation.")
    parser.add_argument(
        "--benchmark-path",
        default=str(REPO_ROOT / "data" / "benchmark" / "questions.jsonl"),
        help="Benchmark JSONL path.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=SUPPORTED_MODES,
        default=list(SUPPORTED_MODES),
        help="Retrieval modes to evaluate. Default: all six modes.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N questions.")
    parser.add_argument("--top-k", type=int, default=None, help="Final chunk count per retrieval mode.")
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Skip LLM generation and compute retrieval metrics only.",
    )
    parser.add_argument(
        "--output-root",
        default=str(REPO_ROOT / "outputs" / "evaluation_runs"),
        help="Directory where timestamped evaluation runs are written.",
    )
    return parser.parse_args()


def load_questions(path: Path, limit: int | None) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            questions.append(json.loads(line))
            if limit is not None and len(questions) >= limit:
                break
    return questions


def metadata_filter_for(question: dict[str, Any], mode: str) -> dict[str, Any] | None:
    if mode == "llm_only":
        return None
    document_type = str(question.get("documentType") or "").strip()
    if document_type in {"course", "review", "exam"}:
        return {"documentType": document_type}
    return None


def format_source(metadata: dict[str, Any]) -> str:
    source = metadata.get("source") or metadata.get("file_name") or metadata.get("sourceId") or "unknown"
    location_parts = []
    if metadata.get("page") is not None:
        location_parts.append(f"page {metadata['page']}")
    if metadata.get("slide") is not None:
        location_parts.append(f"slide {metadata['slide']}")
    if metadata.get("section"):
        location_parts.append(f"section '{metadata['section']}'")
    location = ", ".join(location_parts)
    return f"{source}, {location}" if location else str(source)


def format_for_context(docs: list[Document]) -> list[Document]:
    formatted: list[Document] = []
    for doc in docs:
        metadata = dict(doc.metadata or {})
        header = f"[Source: {format_source(metadata)}]"
        formatted.append(Document(page_content=f"{header}\n{doc.page_content}", metadata=metadata))
    return formatted


def unique_in_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            output.append(value)
    return output


def chunk_id_for(result: dict[str, Any]) -> str:
    metadata = result.get("metadata") or {}
    return str(result.get("chunk_id") or metadata.get("chunk_id") or metadata.get("chunkId") or "")


def source_for(result: dict[str, Any]) -> str:
    metadata = result.get("metadata") or {}
    return str(metadata.get("source") or metadata.get("file_name") or metadata.get("sourceId") or "unknown")


def estimate_tokens(text: str) -> int:
    return max(0, len(text or "") // TOKEN_CHARS_PER_TOKEN)


def estimate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    cost = (prompt_tokens / 1000.0) * PROMPT_COST_PER_1K
    cost += (completion_tokens / 1000.0) * COMPLETION_COST_PER_1K
    return round(cost, 8)


def prompt_text_estimate(question: str, context_documents: list[Document], llm_question: str) -> str:
    context = "\n\n".join(doc.page_content for doc in context_documents)
    return f"{context}\n\n{llm_question or question}"


def generate_answer(
    *,
    question: dict[str, Any],
    mode: str,
    context_documents: list[Document],
    retrieval_only: bool,
) -> tuple[str, float, int, int, float]:
    if retrieval_only:
        return "", 0.0, 0, 0, 0.0

    original_question = str(question.get("question") or "")
    if mode == "llm_only":
        llm_question = (
            "Evaluation mode: LLM-only baseline. No retrieved context is available; "
            "answer from internal knowledge and state uncertainty for course-specific claims.\n\n"
            f"Original user question: {original_question}"
        )
        final_context_documents: list[Document] = []
    else:
        llm_question = original_question
        final_context_documents = context_documents

    started = time.perf_counter()
    retriever = StaticRetriever(documents=final_context_documents)
    chain = get_llm_chain(retriever, intent=str(question.get("intent") or "diger"))
    result = query_chain(chain, llm_question)
    generation_ms = (time.perf_counter() - started) * 1000

    answer = str(result.get("response") or "")
    prompt_tokens = estimate_tokens(prompt_text_estimate(original_question, final_context_documents, llm_question))
    completion_tokens = estimate_tokens(answer)
    return answer, generation_ms, prompt_tokens, completion_tokens, estimate_cost(prompt_tokens, completion_tokens)


def row_for_result(
    *,
    question: dict[str, Any],
    mode: str,
    report: dict[str, Any],
    context_documents: list[Document],
    answer: str,
    row_started: float,
    generation_latency_ms: float,
    prompt_tokens: int,
    completion_tokens: int,
    estimated_cost_usd: float,
    top_k: int,
) -> dict[str, Any]:
    results = report.get("results") or []
    retrieved_chunk_ids = unique_in_order([chunk_id_for(result) for result in results])
    retrieved_sources = unique_in_order([source_for(result) for result in results])
    expected_chunk_ids = list(question.get("expected_chunk_ids") or [])
    expected_sources = list(question.get("expected_sources") or [])
    metric_values = retrieval_metrics(
        retrieved_chunk_ids,
        retrieved_sources,
        expected_chunk_ids,
        expected_sources,
        k=top_k,
    )
    timings = report.get("timings_ms") or {}
    return {
        "query_id": str(question.get("id") or ""),
        "question": str(question.get("question") or ""),
        "language": str(question.get("language") or ""),
        "question_type": str(question.get("question_type") or ""),
        "mode": mode,
        "prompt_strategy": PROMPT_STRATEGY,
        "expert_mode": EXPERT_MODE,
        "answer": answer,
        "answerable_expected": bool(question.get("answerable")),
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "retrieved_sources": retrieved_sources,
        "expected_chunk_ids": expected_chunk_ids,
        "expected_sources": expected_sources,
        "retrieval_hit_chunk": bool(metric_values["retrieval_hit_chunk"]),
        "retrieval_hit_source": bool(metric_values["retrieval_hit_source"]),
        "recall_at_k": float(metric_values["recall_at_k"]),
        "mrr_at_k": float(metric_values["mrr_at_k"]),
        "ndcg_at_k": float(metric_values["ndcg_at_k"]),
        "latency_ms": round((time.perf_counter() - row_started) * 1000, 2),
        "retrieval_latency_ms": float(timings.get("retrieval_ms") or 0.0),
        "rerank_latency_ms": float(timings.get("rerank_ms") or 0.0),
        "generation_latency_ms": round(generation_latency_ms, 2),
        "num_retrieved_chunks": len(results),
        "num_final_context_chunks": 0 if mode == "llm_only" else len(context_documents),
        "prompt_tokens_estimate": prompt_tokens,
        "completion_tokens_estimate": completion_tokens,
        "estimated_cost_usd": estimated_cost_usd,
        "answer_correctness": None,
        "citation_correctness": None,
        "faithfulness": None,
        "answer_relevancy": None,
        "hallucination_flag": None,
        "manual_notes": "",
    }


def retrieved_chunk_rows(question: dict[str, Any], mode: str, report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank, result in enumerate(report.get("results") or [], start=1):
        metadata = result.get("metadata") or {}
        rows.append(
            {
                "query_id": str(question.get("id") or ""),
                "mode": mode,
                "rank": rank,
                "chunk_id": chunk_id_for(result),
                "source": source_for(result),
                "retriever": result.get("retriever"),
                "score": result.get("score"),
                "rrf_score": result.get("rrf_score"),
                "rerank_score": result.get("rerank_score"),
                "text": result.get("text") or "",
                "metadata": metadata,
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_mode: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_mode.setdefault(row["mode"], []).append(row)

    summary: dict[str, Any] = {"overall": {"num_rows": len(rows)}, "modes": {}}
    for mode, mode_rows in sorted(by_mode.items()):
        summary["modes"][mode] = {
            "num_rows": len(mode_rows),
            "avg_recall_at_k": _mean(row["recall_at_k"] for row in mode_rows),
            "avg_mrr_at_k": _mean(row["mrr_at_k"] for row in mode_rows),
            "avg_ndcg_at_k": _mean(row["ndcg_at_k"] for row in mode_rows),
            "chunk_hit_rate": _mean(1.0 if row["retrieval_hit_chunk"] else 0.0 for row in mode_rows),
            "source_hit_rate": _mean(1.0 if row["retrieval_hit_source"] else 0.0 for row in mode_rows),
            "avg_latency_ms": _mean(row["latency_ms"] for row in mode_rows),
            "avg_retrieval_latency_ms": _mean(row["retrieval_latency_ms"] for row in mode_rows),
            "avg_rerank_latency_ms": _mean(row["rerank_latency_ms"] for row in mode_rows),
            "avg_generation_latency_ms": _mean(row["generation_latency_ms"] for row in mode_rows),
            "total_estimated_cost_usd": round(sum(row["estimated_cost_usd"] for row in mode_rows), 8),
        }
    return summary


def _mean(values: Any) -> float:
    materialized = [float(value) for value in values]
    return round(statistics.fmean(materialized), 6) if materialized else 0.0


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def main() -> None:
    args = parse_args()
    benchmark_path = Path(args.benchmark_path).expanduser().resolve()
    questions = load_questions(benchmark_path, args.limit)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root).expanduser().resolve() / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    modes = list(args.modes)
    top_k = int(args.top_k or 6)
    needs_vectorstore = any(mode != "llm_only" for mode in modes)
    vectorstore = get_vectorstore() if needs_vectorstore else None

    rows: list[dict[str, Any]] = []
    chunk_rows: list[dict[str, Any]] = []

    run_config = {
        "timestamp": timestamp,
        "benchmark_path": str(benchmark_path),
        "num_questions": len(questions),
        "modes": modes,
        "limit": args.limit,
        "top_k": top_k,
        "retrieval_only": bool(args.retrieval_only),
        "prompt_strategy": PROMPT_STRATEGY,
        "expert_mode": EXPERT_MODE,
        "token_estimate_heuristic": f"len(text)//{TOKEN_CHARS_PER_TOKEN}",
        "estimated_cost_rates_per_1k_tokens": {
            "prompt": PROMPT_COST_PER_1K,
            "completion": COMPLETION_COST_PER_1K,
        },
    }

    for question_index, question in enumerate(questions, start=1):
        query = str(question.get("question") or "")
        for mode in modes:
            row_started = time.perf_counter()
            metadata_filter = metadata_filter_for(question, mode)
            report = run_retrieval_mode(
                mode,
                query,
                vectorstore,
                top_k=top_k,
                metadata_filter=metadata_filter,
            )
            context_documents = format_for_context(report.get("context_documents") or [])
            answer, generation_ms, prompt_tokens, completion_tokens, estimated_cost_usd = generate_answer(
                question=question,
                mode=mode,
                context_documents=context_documents,
                retrieval_only=bool(args.retrieval_only),
            )
            row = row_for_result(
                question=question,
                mode=report.get("mode") or mode,
                report=report,
                context_documents=context_documents,
                answer=answer,
                row_started=row_started,
                generation_latency_ms=generation_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                estimated_cost_usd=estimated_cost_usd,
                top_k=top_k,
            )
            rows.append(row)
            chunk_rows.extend(retrieved_chunk_rows(question, report.get("mode") or mode, report))
            print(
                f"[{question_index}/{len(questions)}] {question.get('id')} {mode}: "
                f"recall={row['recall_at_k']:.3f} mrr={row['mrr_at_k']:.3f} "
                f"ndcg={row['ndcg_at_k']:.3f} latency_ms={row['latency_ms']:.1f}",
                flush=True,
            )

    write_jsonl(output_dir / "results.jsonl", rows)
    write_jsonl(output_dir / "retrieved_chunks.jsonl", chunk_rows)
    (output_dir / "summary_metrics.json").write_text(
        json.dumps(summarize(rows), ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"Evaluation complete. Outputs: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
