from __future__ import annotations

from pathlib import Path

from modules.document_loaders import SUPPORTED_EXTENSIONS, load_document
from modules.load_vectorstore import _records_to_chunks, get_vectorstore


SOURCE_BATCH_SIZE = 512


def ensure_sources_indexed(sources_dir: str) -> int:
    vectorstore = get_vectorstore()
    collection = getattr(vectorstore, "_collection", None)
    if collection is not None:
        existing = collection.get(
            where={"source_collection": "sources"},
            limit=1,
            include=["metadatas"],
        )
        if existing.get("ids"):
            return 0
    return ingest_source_directory(sources_dir)


def ingest_source_directory(sources_dir: str) -> int:
    root = Path(sources_dir).expanduser().resolve()
    if not root.exists():
        return 0

    all_chunks = []
    all_ids = []
    for path in _iter_source_files(root):
        records = load_document(str(path))
        rel_path = path.relative_to(root).as_posix()
        module = path.relative_to(root).parts[0] if len(path.relative_to(root).parts) > 1 else "sources"
        for record in records:
            metadata = record.setdefault("metadata", {})
            metadata["source"] = rel_path
            metadata["source_path"] = rel_path
            metadata["source_collection"] = "sources"
            metadata["module"] = module
        chunks, ids = _records_to_chunks(records)
        all_chunks.extend(chunks)
        all_ids.extend(ids)

    if not all_chunks:
        return 0

    vectorstore = get_vectorstore()
    for start in range(0, len(all_chunks), SOURCE_BATCH_SIZE):
        end = start + SOURCE_BATCH_SIZE
        vectorstore.add_documents(documents=all_chunks[start:end], ids=all_ids[start:end])
    return len(all_chunks)


def _iter_source_files(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path
