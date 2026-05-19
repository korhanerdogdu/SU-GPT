from __future__ import annotations

import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from modules.catalog_data_loader import iter_catalog_documents
from modules.config import CHROMA_COLLECTION_NAME, CHROMA_PERSIST_DIR, EMBEDDING_MODEL_NAME
from modules.load_vectorstore import upsert_documents


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

    if args.reset and persist_dir.exists() and not args.dry_run:
        shutil.rmtree(persist_dir)

    vectorstore = None
    if not args.dry_run:
        persist_dir.mkdir(parents=True, exist_ok=True)
        embeddings = HuggingFaceEmbeddings(model_name=args.embedding_model)
        vectorstore = Chroma(
            collection_name=args.collection,
            persist_directory=str(persist_dir),
            embedding_function=embeddings,
        )

    stats: Counter[str] = Counter()
    batch = []
    ids = []
    total = 0

    def flush() -> None:
        if not batch:
            return
        if vectorstore is not None:
            upsert_documents(vectorstore, batch, ids)
        batch.clear()
        ids.clear()

    for document in iter_catalog_documents(data_dir):
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
    print(f"Documents prepared: {total}")
    for kind, count in stats.most_common():
        print(f"  {kind}: {count}")
    if args.dry_run:
        print("Dry run only: no vectors were written.")
    else:
        print("Chroma vector database updated.")


if __name__ == "__main__":
    main()
