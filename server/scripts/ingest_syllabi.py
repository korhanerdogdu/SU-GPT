from __future__ import annotations

"""Upsert the pre-chunked ``data/syllabi/*.chunks.jsonl`` corpus into Chroma."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from modules.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_HNSW_BATCH_SIZE,
    CHROMA_HNSW_SYNC_THRESHOLD,
    CHROMA_PERSIST_DIR,
    DEGREE_DATA_DIR,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DEVICE,
    EMBEDDING_MODEL_NAME,
    SERVER_ROOT,
)


_PERSIST = (
    CHROMA_PERSIST_DIR
    if Path(CHROMA_PERSIST_DIR).is_absolute()
    else str((SERVER_ROOT / CHROMA_PERSIST_DIR).resolve())
)


def _device() -> str:
    if EMBEDDING_DEVICE != "auto":
        return EMBEDDING_DEVICE
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def get_vectorstore() -> Chroma:
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": _device()},
        encode_kwargs={"batch_size": EMBEDDING_BATCH_SIZE},
    )
    return Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=_PERSIST,
        embedding_function=embeddings,
        collection_configuration={
            "hnsw": {"batch_size": CHROMA_HNSW_BATCH_SIZE, "sync_threshold": CHROMA_HNSW_SYNC_THRESHOLD}
        },
    )


def _scalarize(row: dict) -> dict:
    metadata: dict = {}
    for key, value in row.items():
        if key == "text" or value is None:
            continue
        metadata[key] = value if isinstance(value, (str, int, float, bool)) else json.dumps(value, ensure_ascii=False)
    metadata.setdefault("documentType", "course")
    metadata.setdefault("data_role", "course_syllabus")
    metadata.setdefault("source", row.get("source_document") or row.get("source_url") or "syllabus")
    return metadata


def iter_rows(data_dir: Path, terms: set[str]):
    for path in sorted(data_dir.glob("*.chunks.jsonl")):
        term = path.name.split(".", 1)[0]
        if terms and term not in terms:
            continue
        with path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                text = row.get("text")
                chunk_id = row.get("chunk_id")
                if not text or not chunk_id:
                    raise ValueError(f"{path}:{line_no}: syllabus chunk requires text and chunk_id")
                yield str(chunk_id), Document(page_content=str(text), metadata=_scalarize(row))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(Path(DEGREE_DATA_DIR) / "syllabi"))
    parser.add_argument("--terms", nargs="*", default=[])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--reset", action="store_true", help="Remove only existing course_syllabus vectors first")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    rows = iter_rows(data_dir, set(args.terms))
    vectorstore = None if args.dry_run else get_vectorstore()
    if args.reset and vectorstore is not None:
        vectorstore._collection.delete(where={"data_role": "course_syllabus"})
        print("Reset: removed existing course_syllabus vectors.")

    documents: list[Document] = []
    ids: list[str] = []
    total = 0

    def flush() -> None:
        if documents and vectorstore is not None:
            texts = [document.page_content for document in documents]
            vectorstore._collection.upsert(
                ids=list(ids),
                documents=texts,
                metadatas=[document.metadata for document in documents],
                embeddings=vectorstore._embedding_function.embed_documents(texts),
            )
        documents.clear()
        ids.clear()

    for chunk_id, document in rows:
        ids.append(chunk_id)
        documents.append(document)
        total += 1
        if len(documents) >= args.batch_size:
            flush()
    flush()
    print(f"Syllabus chunks {'validated' if args.dry_run else 'ingested'}: {total}")
    if vectorstore is not None:
        print(f"Collection now holds {vectorstore._collection.count():,} vectors total.")


if __name__ == "__main__":
    main()
