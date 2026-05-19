from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from modules.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOCUMENT_STORAGE_DIR,
    EMBEDDING_MODEL_NAME,
)
from modules.document_cleaner import clean_text
from modules.document_loaders import SUPPORTED_EXTENSIONS, load_document
from modules.source_of_truth import (
    content_hash_text,
    file_content_hash,
    finish_ingestion_job,
    source_id_for,
    start_ingestion_job,
    upsert_source_document,
)


UPLOAD_DIR = "./uploaded_pdfs"
DOCUMENTS_UPLOAD_DIR = DOCUMENT_STORAGE_DIR


def get_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def get_vectorstore():
    return Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=get_embedding_model(),
    )


def canonical_document_type(value: str | None) -> str:
    raw = (value or "").strip().lower()
    if raw in {"review", "whatsapp", "instructor_review"}:
        return "review"
    if raw in {"exam", "exam_pdf", "pdf_exam", "midterm", "final", "quiz"}:
        return "exam"
    return "course"


def _chunk_id(source_id: str, chunk_index: int, text: str) -> str:
    digest = sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"{source_id}:{chunk_index}:{digest}"


def _location_key(metadata: dict) -> str:
    if metadata.get("page") is not None:
        return f"p{metadata['page']}"
    if metadata.get("slide") is not None:
        return f"s{metadata['slide']}"
    if metadata.get("section"):
        return f"sec-{str(metadata['section'])[:20]}"
    return ""


def _save_uploaded(uploaded_files, target_dir: str) -> list[str]:
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    saved_paths: list[str] = []
    for file in uploaded_files:
        save_path = Path(target_dir) / file.filename
        with open(save_path, "wb") as f:
            f.write(file.file.read())
        saved_paths.append(str(save_path))
    return saved_paths


def _chroma_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Chroma accepts only primitive metadata values."""
    allowed = (str, int, float, bool)
    return {k: v for k, v in metadata.items() if v is not None and isinstance(v, allowed)}


def _records_to_chunks(records: list[dict]) -> tuple[list[Document], list[str]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks: list[Document] = []
    ids: list[str] = []
    source_counters: dict[str, int] = {}

    for record in records:
        cleaned = clean_text(record.get("text", ""))
        if not cleaned:
            continue

        base_metadata = record.get("metadata", {}) or {}
        document_type = canonical_document_type(
            base_metadata.get("documentType") or base_metadata.get("document_type")
        )
        content_hash = str(base_metadata.get("contentHash") or content_hash_text(cleaned))
        source_id = str(base_metadata.get("sourceId") or source_id_for(document_type, content_hash))
        loc_key = _location_key(base_metadata)

        for piece in splitter.split_text(cleaned):
            chunk_index = source_counters.get(source_id, 0)
            source_counters[source_id] = chunk_index + 1
            cid = _chunk_id(source_id, chunk_index, piece)
            metadata = {
                **base_metadata,
                "source": base_metadata.get("source"),
                "file_name": base_metadata.get("source"),
                "document_type": base_metadata.get("document_type") or document_type,
                "documentType": document_type,
                "sourceId": source_id,
                "contentHash": content_hash,
                "page": base_metadata.get("page"),
                "slide": base_metadata.get("slide"),
                "section": base_metadata.get("section"),
                "language": base_metadata.get("language"),
                "chunk_id": cid,
                "chunkIndex": chunk_index,
            }
            if loc_key:
                metadata["location_key"] = loc_key
            metadata = _chroma_metadata(metadata)

            chunks.append(Document(page_content=piece, metadata=metadata))
            ids.append(cid)

    return chunks, ids


def upsert_documents(vectorstore: Chroma, documents: list[Document], ids: list[str]) -> None:
    if not documents:
        return
    collection = getattr(vectorstore, "_collection", None)
    embedding_function = getattr(vectorstore, "_embedding_function", None)
    if collection is None or embedding_function is None:
        vectorstore.add_documents(documents=documents, ids=ids)
        return
    collection.upsert(
        ids=ids,
        documents=[doc.page_content for doc in documents],
        metadatas=[doc.metadata for doc in documents],
        embeddings=embedding_function.embed_documents([doc.page_content for doc in documents]),
    )


def ingest_file_paths(
    file_paths: list[str],
    *,
    document_type: str = "course",
    created_by: str = "system",
    extra_metadata: dict[str, Any] | None = None,
) -> int:
    total_chunks = 0
    for raw_path in file_paths:
        path = Path(raw_path)
        canonical_type = canonical_document_type(document_type)
        file_hash = file_content_hash(path)
        source_id = source_id_for(canonical_type, file_hash)
        metadata_seed = {
            "sourceId": source_id,
            "documentType": canonical_type,
            "contentHash": file_hash,
            "storageKey": str(path),
            "createdBy": created_by,
            **(extra_metadata or {}),
        }

        upsert_source_document(
            source_id=source_id,
            document_type=canonical_type,
            file_name=path.name,
            storage_key=str(path),
            content_hash=file_hash,
            status="uploaded",
            created_by=created_by,
            metadata=metadata_seed,
        )
        job_id = start_ingestion_job(source_id)
        try:
            records = load_document(str(path))
            for record in records:
                metadata = record.setdefault("metadata", {})
                metadata.update(metadata_seed)
                metadata.setdefault("source", path.name)
                metadata.setdefault("source_path", str(path))

            chunks, ids = _records_to_chunks(records)
            if chunks:
                Path(CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
                upsert_documents(get_vectorstore(), chunks, ids)
            finish_ingestion_job(
                job_id=job_id,
                source_id=source_id,
                status="indexed",
                chunks_created=len(chunks),
            )
            total_chunks += len(chunks)
        except Exception as exc:
            finish_ingestion_job(
                job_id=job_id,
                source_id=source_id,
                status="failed",
                chunks_created=0,
                error=str(exc),
            )
            raise
    return total_chunks


def ingest_text_source(
    *,
    text: str,
    source_name: str,
    document_type: str,
    source_id: str | None = None,
    storage_key: str = "",
    created_by: str = "system",
    extra_metadata: dict[str, Any] | None = None,
) -> int:
    canonical_type = canonical_document_type(document_type)
    content_hash = content_hash_text(text)
    source_id = source_id or source_id_for(canonical_type, content_hash)
    metadata = {
        "source": source_name,
        "sourceId": source_id,
        "documentType": canonical_type,
        "contentHash": content_hash,
        "storageKey": storage_key,
        "createdBy": created_by,
        **(extra_metadata or {}),
    }

    upsert_source_document(
        source_id=source_id,
        document_type=canonical_type,
        file_name=source_name,
        storage_key=storage_key,
        content_hash=content_hash,
        status="uploaded",
        created_by=created_by,
        metadata=metadata,
    )
    job_id = start_ingestion_job(source_id)
    try:
        chunks, ids = _records_to_chunks([{"text": text, "metadata": metadata}])
        if chunks:
            Path(CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
            upsert_documents(get_vectorstore(), chunks, ids)
        finish_ingestion_job(
            job_id=job_id,
            source_id=source_id,
            status="indexed",
            chunks_created=len(chunks),
        )
        return len(chunks)
    except Exception as exc:
        finish_ingestion_job(
            job_id=job_id,
            source_id=source_id,
            status="failed",
            chunks_created=0,
            error=str(exc),
        )
        raise


def _ingest_paths(file_paths: list[str]) -> int:
    return ingest_file_paths(file_paths, document_type="course")


def load_vectorstore(uploaded_files) -> int:
    file_paths = _save_uploaded(uploaded_files, UPLOAD_DIR)
    pdf_paths = [p for p in file_paths if Path(p).suffix.lower() == ".pdf"]
    return ingest_file_paths(pdf_paths, document_type="exam")


def load_vectorstore_multi(uploaded_files) -> dict:
    Path(DOCUMENTS_UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

    accepted: list[str] = []
    skipped: list[str] = []
    for file in uploaded_files:
        ext = Path(file.filename).suffix.lower()
        if ext in SUPPORTED_EXTENSIONS:
            save_path = Path(DOCUMENTS_UPLOAD_DIR) / file.filename
            with open(save_path, "wb") as f:
                f.write(file.file.read())
            accepted.append(str(save_path))
        else:
            skipped.append(file.filename)

    chunk_count = ingest_file_paths(accepted, document_type="course") if accepted else 0
    return {
        "chunks": chunk_count,
        "accepted_files": [Path(p).name for p in accepted],
        "skipped_files": skipped,
        "supported_extensions": sorted(SUPPORTED_EXTENSIONS),
    }
