from __future__ import annotations

"""
Ingest the pre-chunked degree-requirement corpus into ChromaDB.

Unlike catalog_data_loader (which synthesizes prose from raw catalog rows), the files
under data/degree_requirements/<PROGRAM>/<TERM>.jsonl, data/minors/<CODE>/<TERM>.jsonl,
and data/suggested_programs/<PROGRAM>/*.jsonl
are ALREADY one-chunk-per-line: every row carries a ready-made `text`, a stable
`chunk_id`, and flat metadata (data_role, program, curriculum_term, course_id, ...).

So this ingester just:
  - reads each JSONL row,
  - uses row["text"] as the document body and row["chunk_id"] as the vector id,
  - scalarizes metadata (Chroma only stores str/int/float/bool; lists/dicts are
    JSON-stringified, None is dropped),
  - adds two compatibility aliases so the existing router/retriever keep working:
        documentType  = "course" for majors, "minor" for minors
        term_code     = curriculum_term
  - upserts into the shared `su_knowledge` collection.

Usage:
    python server/scripts/ingest_degree_requirements.py [--data-dir DIR] [--reset]
                                                        [--dry-run] [--limit N]
"""

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

# The app runs from server/ so CHROMA_PERSIST_DIR ("./chroma_store") means server/chroma_store.
# Anchor to SERVER_ROOT so ingestion writes exactly where the running app reads, regardless
# of the directory this script is launched from.
_PERSIST = CHROMA_PERSIST_DIR if Path(CHROMA_PERSIST_DIR).is_absolute() else str((SERVER_ROOT / CHROMA_PERSIST_DIR).resolve())

# Ingestion writes only to Chroma; it must NOT import the Mongo-backed modules
# (load_vectorstore -> source_of_truth eagerly resolves the Mongo SRV URI and can
# hang when the DB is unreachable). So we build the vectorstore locally.


def _resolve_device(configured: str) -> str:
    if configured != "auto":
        return configured
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def get_vectorstore() -> Chroma:
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": _resolve_device(EMBEDDING_DEVICE)},
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


def upsert_documents(vectorstore: Chroma, documents: list[Document], ids: list[str]) -> None:
    if not documents:
        return
    collection = vectorstore._collection
    embed = vectorstore._embedding_function
    collection.upsert(
        ids=ids,
        documents=[d.page_content for d in documents],
        metadatas=[d.metadata for d in documents],
        embeddings=embed.embed_documents([d.page_content for d in documents]),
    )

# data_role -> coarse documentType understood by rag_router / catalog_retriever.
_DOCUMENT_TYPE_ALIAS = {
    "curriculum_requirement": "course",
    "minor_requirement": "minor",
    "suggested_program": "course",
}
_SCALAR = (str, int, float, bool)


def _scalarize(row: dict) -> dict:
    """Chroma metadata must be primitive. Keep scalars, JSON-encode lists/dicts, drop None."""
    meta: dict = {}
    for key, value in row.items():
        if key == "text" or value is None:
            continue
        if isinstance(value, bool) or isinstance(value, _SCALAR):
            meta[key] = value
        else:  # list / dict
            meta[key] = json.dumps(value, ensure_ascii=False)
    return meta


def _iter_rows(data_dir: Path):
    """Yield every pre-chunked curriculum, minor, and suggested-program JSONL row."""
    roots = [data_dir / "degree_requirements", data_dir / "minors", data_dir / "suggested_programs"]
    files = sorted(p for root in roots if root.is_dir() for p in root.rglob("*.jsonl"))
    for path in files:
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                text = row.get("text")
                chunk_id = row.get("chunk_id")
                if not text or not chunk_id:
                    continue
                data_role = row.get("data_role", "")
                meta = _scalarize(row)
                # compatibility aliases for the existing retrieval stack
                meta.setdefault("documentType", _DOCUMENT_TYPE_ALIAS.get(data_role, "course"))
                if row.get("course_code") and not meta.get("course_id"):
                    meta["course_id"] = row["course_code"]
                if row.get("curriculum_term"):
                    meta.setdefault("term_code", row["curriculum_term"])
                meta["source"] = row.get("source_document") or path.name
                yield str(chunk_id), Document(page_content=text, metadata=meta)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DEGREE_DATA_DIR))
    parser.add_argument("--persist-dir", default=_PERSIST)
    parser.add_argument("--collection", default=None, help="(informational) uses CHROMA_COLLECTION_NAME")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--reset", action="store_true",
                        help="Delete existing curriculum_requirement/minor_requirement vectors first.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    vectorstore = None if args.dry_run else get_vectorstore()

    if args.reset and vectorstore is not None:
        collection = getattr(vectorstore, "_collection", None)
        if collection is not None:
            collection.delete(where={"data_role": {"$in": [
                "curriculum_requirement", "minor_requirement", "suggested_program",
            ]}})
            print("Reset: removed existing degree/minor/suggested-program vectors.")

    batch_docs: list[Document] = []
    batch_ids: list[str] = []
    total = 0
    by_role: dict[str, int] = {}
    by_program: dict[str, int] = {}

    def flush() -> None:
        if batch_docs and vectorstore is not None:
            upsert_documents(vectorstore, batch_docs, batch_ids)
        batch_docs.clear()
        batch_ids.clear()

    for chunk_id, doc in _iter_rows(data_dir):
        total += 1
        by_role[doc.metadata.get("data_role", "?")] = by_role.get(doc.metadata.get("data_role", "?"), 0) + 1
        prog = doc.metadata.get("program", "?")
        by_program[prog] = by_program.get(prog, 0) + 1
        batch_docs.append(doc)
        batch_ids.append(chunk_id)
        if len(batch_docs) >= args.batch_size:
            flush()
        if args.limit and total >= args.limit:
            break
    flush()

    print(f"Data dir: {data_dir}")
    print(f"Rows ingested: {total}")
    print("By data_role: " + ", ".join(f"{k}={v}" for k, v in sorted(by_role.items())))
    print(f"Programs/minors: {len(by_program)}")
    if args.dry_run:
        print("Dry run: nothing written.")
    else:
        count = vectorstore._collection.count()
        print(f"Collection now holds {count:,} vectors total.")


if __name__ == "__main__":
    main()
