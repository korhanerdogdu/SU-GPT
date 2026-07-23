from __future__ import annotations

import argparse
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_chroma import Chroma
from modules.catalog_data_loader import iter_catalog_documents
from modules.config import (
    CATALOG_TERM_CODES,
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    EMBEDDING_MODEL_NAME,
)
from modules.load_vectorstore import (
    get_chroma_collection_configuration,
    get_embedding_model,
    upsert_documents,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Sabanci JSONL catalog data into RAG-ready Chroma vectors."
    )
    parser.add_argument(
        "--data-dir",
        default=str(Path.home() / "data"),
        help="Directory containing the JSONL data tree. Default: ~/data",
    )
    parser.add_argument(
        "--persist-dir",
        default=CHROMA_PERSIST_DIR,
        help=f"Chroma persist directory. Default: {CHROMA_PERSIST_DIR}",
    )
    parser.add_argument(
        "--collection",
        default=CHROMA_COLLECTION_NAME,
        help=f"Chroma collection name. Default: {CHROMA_COLLECTION_NAME}",
    )
    parser.add_argument(
        "--embedding-model",
        default=EMBEDDING_MODEL_NAME,
        help=f"SentenceTransformers embedding model. Default: {EMBEDDING_MODEL_NAME}",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--terms",
        default=",".join(CATALOG_TERM_CODES),
        help=(
            "Comma-separated term codes to ingest. Default comes from "
            "CATALOG_TERM_CODES. Pass an empty value to ingest every historical term."
        ),
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the persist directory before ingesting.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build documents and print stats without writing vectors.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional document limit for smoke tests. 0 means no limit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()
    persist_dir = Path(args.persist_dir).expanduser().resolve()
    term_codes = [value.strip() for value in args.terms.split(",") if value.strip()]

    if args.reset and persist_dir.exists() and not args.dry_run:
        shutil.rmtree(persist_dir)

    vectorstore = None
    if not args.dry_run:
        persist_dir.mkdir(parents=True, exist_ok=True)
        if args.embedding_model != EMBEDDING_MODEL_NAME:
            raise ValueError(
                "--embedding-model must match EMBEDDING_MODEL_NAME so ingestion "
                "and query use the same cached embedding model."
            )
        embeddings = get_embedding_model()
        vectorstore = Chroma(
            collection_name=args.collection,
            persist_directory=str(persist_dir),
            embedding_function=embeddings,
            collection_configuration=get_chroma_collection_configuration(),
        )

    stats: Counter[str] = Counter()
    batch = []
    ids = []
    total = 0
    written = 0
    started_at = time.monotonic()

    def flush() -> None:
        nonlocal written
        if not batch:
            return
        if vectorstore is not None:
            upsert_documents(vectorstore, batch, ids)
        written += len(batch)
        elapsed = max(time.monotonic() - started_at, 0.001)
        print(
            f"Processed: {written:,} documents "
            f"({written / elapsed:.1f} docs/sec)",
            flush=True,
        )
        batch.clear()
        ids.clear()

    for document in iter_catalog_documents(data_dir, term_codes=term_codes):
        total += 1
        stats[document.metadata.get("document_type", "unknown")] += 1
        if not args.dry_run:
            batch.append(document)
            ids.append(str(document.metadata["chunk_id"]))
            if len(batch) >= args.batch_size:
                flush()
        if args.limit and total >= args.limit:
            break

    flush()

    print(f"Data directory: {data_dir}")
    print(f"Persist directory: {persist_dir}")
    print(f"Collection: {args.collection}")
    print(f"Embedding model: {args.embedding_model}")
    print(f"Selected terms: {', '.join(term_codes) if term_codes else 'all historical terms'}")
    print(f"Documents prepared: {total}")
    for kind, count in stats.most_common():
        print(f"  {kind}: {count}")
    if args.dry_run:
        print("Dry run only: no vectors were written.")
    else:
        actual_count = vectorstore._collection.count()
        if actual_count != total:
            raise RuntimeError(
                f"Chroma count mismatch: expected {total}, found {actual_count}."
            )
        print("Verifying persisted HNSW index with a similarity query...", flush=True)
        verification = vectorstore.similarity_search(
            "Sabanci University course catalog verification",
            k=1,
        )
        if not verification:
            raise RuntimeError("Chroma verification query returned no documents.")
        print(f"Verification query succeeded. Collection count: {actual_count:,}")
        print("Chroma vector database updated.")


if __name__ == "__main__":
    main()
