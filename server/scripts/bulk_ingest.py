from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.documents import Document

from modules.catalog_data_loader import iter_catalog_documents
from modules.config import CATALOG_DATA_DIR, EXAMS_DIR, REVIEWS_DIR, SOURCES_DIR
from modules.document_loaders import SUPPORTED_EXTENSIONS
from modules.load_vectorstore import get_vectorstore, ingest_file_paths, upsert_documents
from modules.source_of_truth import (
    content_hash_text,
    ensure_source_of_truth_indexes,
    source_id_for,
    upsert_source_document,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Idempotent SU-GPT bulk ingestion.")
    parser.add_argument("--target", choices=["all", "courses", "reviews", "exams"], default="all")
    parser.add_argument("--catalog-dir", default=CATALOG_DATA_DIR)
    parser.add_argument("--sources-dir", default=SOURCES_DIR)
    parser.add_argument("--reviews-dir", default=REVIEWS_DIR)
    parser.add_argument("--exams-dir", default=EXAMS_DIR)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_source_of_truth_indexes()

    totals: Counter[str] = Counter()
    if args.target in {"all", "courses"}:
        totals["course_catalog_chunks"] += ingest_catalog_courses(args.catalog_dir, batch_size=args.batch_size)
        totals["course_source_chunks"] += ingest_files(args.sources_dir, "course", exclude_dirs={"reviews", "exams"})
    if args.target in {"all", "reviews"}:
        totals["review_chunks"] += ingest_files(args.reviews_dir, "review")
    if args.target in {"all", "exams"}:
        totals["exam_chunks"] += ingest_files(args.exams_dir, "exam")

    print(f"Target: {args.target}")
    for key, value in totals.items():
        print(f"{key}: {value}")
    print("Bulk ingest completed idempotently.")


def ingest_catalog_courses(catalog_dir: str, *, batch_size: int = 256) -> int:
    root = Path(catalog_dir).expanduser().resolve()
    if not root.exists():
        print(f"Catalog directory not found, skipping: {root}")
        return 0

    vectorstore = get_vectorstore()
    batch: list[Document] = []
    ids: list[str] = []
    per_source_chunks: Counter[str] = Counter()
    per_source_name: dict[str, str] = {}
    source_counters: defaultdict[str, int] = defaultdict(int)
    total = 0

    def flush() -> None:
        if not batch:
            return
        upsert_documents(vectorstore, batch, ids)
        batch.clear()
        ids.clear()

    for document in iter_catalog_documents(root):
        normalized, chunk_id, source_id, source_name = normalize_catalog_document(document, source_counters)
        batch.append(normalized)
        ids.append(chunk_id)
        per_source_chunks[source_id] += 1
        per_source_name[source_id] = source_name
        total += 1
        if len(batch) >= batch_size:
            flush()
    flush()

    for source_id, chunks_created in per_source_chunks.items():
        source_name = per_source_name.get(source_id, "catalog")
        upsert_source_document(
            source_id=source_id,
            document_type="course",
            file_name=source_name,
            storage_key=str(root),
            content_hash=content_hash_text(source_name),
            status="indexed",
            created_by="bulk-script",
            metadata={"documentType": "course", "sourceId": source_id, "source": source_name},
            chunks_created=chunks_created,
        )
    return total


def normalize_catalog_document(
    document: Document,
    source_counters: defaultdict[str, int],
) -> tuple[Document, str, str, str]:
    metadata = dict(document.metadata or {})
    source_name = str(metadata.get("source_path") or metadata.get("source") or "catalog")
    source_id = str(metadata.get("sourceId") or source_id_for("course", content_hash_text(source_name)))
    chunk_index = source_counters[source_id]
    source_counters[source_id] += 1
    content_hash = content_hash_text(document.page_content)
    chunk_id = f"{source_id}:{chunk_index}:{content_hash[:16]}"

    metadata.update(
        {
            "documentType": "course",
            "sourceId": source_id,
            "contentHash": content_hash,
            "chunkIndex": chunk_index,
            "chunk_id": chunk_id,
        }
    )
    return Document(page_content=document.page_content, metadata=metadata), chunk_id, source_id, source_name


def ingest_files(root_dir: str, document_type: str, *, exclude_dirs: set[str] | None = None) -> int:
    paths = list(iter_supported_files(root_dir, exclude_dirs=exclude_dirs or set()))
    if not paths:
        print(f"No {document_type} files found in {Path(root_dir).expanduser()}")
        return 0
    return ingest_file_paths(paths, document_type=document_type, created_by="bulk-script")


def iter_supported_files(root_dir: str, *, exclude_dirs: set[str]) -> list[str]:
    root = Path(root_dir).expanduser().resolve()
    if not root.exists():
        return []
    paths: list[str] = []
    excluded = {name.lower() for name in exclude_dirs}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if any(part.lower() in excluded for part in path.relative_to(root).parts[:-1]):
            continue
        paths.append(str(path))
    return paths


if __name__ == "__main__":
    main()
