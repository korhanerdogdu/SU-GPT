"""CLI: run the same question through multiple retrieval modes side by side.

Demonstrates the Section 3 / Step 2 contract -- `run_retrieval_mode` lets any
question be compared across llm_only/bm25/dense/dense_rerank/hybrid/hybrid_rerank
on equal footing. Useful for manual spot checks and as a template for the
Section 5 evaluation runner.

Usage:
    python scripts/compare_retrieval_modes.py "CS412 dersini kim veriyor?"
    python scripts/compare_retrieval_modes.py "What is BERT?" --modes dense hybrid_rerank --top-k 5
    python scripts/compare_retrieval_modes.py "Bu donem hangi dersler aciliyor?" --filter-document-type course
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.load_vectorstore import get_vectorstore
from modules.retrieval_modes import SUPPORTED_MODES, run_retrieval_mode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare SU-GPT retrieval modes for one question.")
    parser.add_argument("question", help="The question to run through each retrieval mode.")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=SUPPORTED_MODES,
        default=list(SUPPORTED_MODES),
        help="Which retrieval modes to run (default: all six).",
    )
    parser.add_argument(
        "--top-k", type=int, default=None, help="Final chunk count per mode (default: RERANK_TOP_K)."
    )
    parser.add_argument(
        "--filter-document-type",
        choices=["course", "review", "exam"],
        default=None,
        help="Optional Chroma documentType filter applied identically to every mode.",
    )
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=160,
        help="How many characters of each chunk's text to print (default: 160).",
    )
    return parser.parse_args()


def _format_source(metadata: dict) -> str:
    label = metadata.get("source") or metadata.get("file_name") or metadata.get("sourceId") or "unknown source"
    location = metadata.get("page") or metadata.get("slide") or metadata.get("section")
    return f"{label}, {location}" if location else str(label)


def _print_result(rank: int, result: dict, preview_chars: int) -> None:
    metadata = result.get("metadata") or {}
    pieces = [f"retriever={result.get('retriever')}"]
    if result.get("score") is not None:
        pieces.append(f"score={result['score']:.4f}")
    if result.get("rrf_score") is not None:
        pieces.append(f"rrf_score={result['rrf_score']:.4f}")
    if result.get("rerank_score") is not None:
        pieces.append(f"rerank_score={result['rerank_score']:.4f}")
    text_preview = (result.get("text") or "").strip().replace("\n", " ")[:preview_chars]
    print(f"  {rank}. [{_format_source(metadata)}]  ({', '.join(pieces)})")
    print(f"     chunk_id={result.get('chunk_id')}")
    print(f"     {text_preview}...")


def main() -> None:
    args = parse_args()
    vectorstore = get_vectorstore()
    metadata_filter = {"documentType": args.filter_document_type} if args.filter_document_type else None

    print(f"Question: {args.question}")
    if metadata_filter:
        print(f"Metadata filter: {metadata_filter}")
    print(f"Modes: {', '.join(args.modes)}")
    print("=" * 78)

    for mode in args.modes:
        report = run_retrieval_mode(
            mode,
            args.question,
            vectorstore,
            top_k=args.top_k,
            metadata_filter=metadata_filter,
        )
        timings = report["timings_ms"]
        print(f"\nMode: {report['mode']}  --  {report['description']}")
        print(
            f"  timings_ms: retrieval={timings['retrieval_ms']} "
            f"rerank={timings['rerank_ms']} total={timings['total_ms']}"
        )
        for warning in report["warnings"]:
            print(f"  WARNING: {warning}")
        if not report["results"]:
            print("  (no chunks retrieved)")
            continue
        for rank, result in enumerate(report["results"], start=1):
            _print_result(rank, result, args.preview_chars)

    print("\n" + "=" * 78)
    print("Done -- the same question ran through every selected mode via run_retrieval_mode().")


if __name__ == "__main__":
    main()
