from __future__ import annotations

from functools import lru_cache
from hashlib import sha256
from pathlib import Path
import re
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from modules.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_HNSW_BATCH_SIZE,
    CHROMA_HNSW_SYNC_THRESHOLD,
    CHROMA_PERSIST_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOCUMENT_STORAGE_DIR,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DEVICE,
    EMBEDDING_MODEL_NAME,
    REVIEWS_DIR,
)
from modules.document_cleaner import clean_text
from modules.document_loaders import SUPPORTED_EXTENSIONS, load_document
from modules.resource_controls import MAX_UPLOAD_BYTES, MAX_UPLOAD_FILES, MAX_UPLOAD_TOTAL_BYTES
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

_ALLOWED_MIME_TYPES: dict[str, frozenset[str]] = {
    ".pdf": frozenset({"application/pdf"}),
    ".pptx": frozenset(
        {"application/vnd.openxmlformats-officedocument.presentationml.presentation"}
    ),
    ".docx": frozenset(
        {"application/vnd.openxmlformats-officedocument.wordprocessingml.document"}
    ),
    ".md": frozenset({"text/markdown", "text/plain"}),
    ".txt": frozenset({"text/plain"}),
    ".html": frozenset({"text/html"}),
    ".htm": frozenset({"text/html"}),
    ".ipynb": frozenset({"application/json", "text/json"}),
    ".json": frozenset({"application/json", "text/json"}),
}


def _validate_upload_batch(uploaded_files) -> list[Any]:
    files = list(uploaded_files)
    if not files:
        raise ValueError("At least one upload is required")
    if len(files) > MAX_UPLOAD_FILES:
        raise ValueError("Upload exceeds the configured file-count limit")
    return files


def _read_validated_upload(file: Any, extension: str) -> bytes:
    content_type = str(getattr(file, "content_type", "") or "").split(";", 1)[0].lower()
    allowed_mime_types = _ALLOWED_MIME_TYPES.get(extension, frozenset())
    if hasattr(file, "content_type") and not content_type:
        raise ValueError("Uploaded file is missing a MIME type")
    if content_type and content_type not in allowed_mime_types:
        raise ValueError("Uploaded file has an invalid MIME type")
    content = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("Upload exceeds the configured file-size limit")
    if extension == ".pdf" and not content.startswith(b"%PDF"):
        raise ValueError("Uploaded PDF has an invalid file signature")
    if extension in {".docx", ".pptx"} and not content.startswith(b"PK"):
        raise ValueError("Uploaded Office document has an invalid file signature")
    return content


def _resolve_torch_device(configured_device: str) -> str:
    if configured_device != "auto":
        return configured_device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


@lru_cache(maxsize=1)
def get_embedding_model():
    device = _resolve_torch_device(EMBEDDING_DEVICE)
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": device},
        encode_kwargs={"batch_size": EMBEDDING_BATCH_SIZE},
    )


def get_chroma_collection_configuration() -> dict:
    return {
        "hnsw": {
            "batch_size": CHROMA_HNSW_BATCH_SIZE,
            "sync_threshold": CHROMA_HNSW_SYNC_THRESHOLD,
        }
    }


def get_vectorstore():
    return Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=get_embedding_model(),
        collection_configuration=get_chroma_collection_configuration(),
    )


_PRIVATE_INGEST_PATH_PREFIXES = (
    "review",
    "whatsapp",
    "telegram",
    "discord",
    "privatechat",
    "instructorreview",
)


def _private_ingest_label(value: object) -> bool:
    compacts: list[str] = []
    for token in re.split(r"[/\\]", str(value or "")):
        # Preserve separator boundaries before compacting. Uploaded files are stored as
        # ``<digest>-<original name>``; checking only the compacted prefix would therefore turn
        # ``abc123-whatsapp.txt`` into ``abc123whatsapptxt`` and miss the private marker.
        segments = [
            "".join(character for character in part.casefold() if character.isalnum())
            for part in re.split(r"[\W_]+", token, flags=re.UNICODE)
            if part
        ]
        compact = "".join(character for character in token.casefold() if character.isalnum())
        stem = "".join(
            character for character in Path(token).stem.casefold() if character.isalnum()
        )
        compacts.append(stem or compact)
        if any(
            compact.startswith(prefix) or stem.startswith(prefix)
            for prefix in _PRIVATE_INGEST_PATH_PREFIXES
        ) or any(
            segment.startswith(prefix)
            for segment in segments
            for prefix in _PRIVATE_INGEST_PATH_PREFIXES
        ):
            return True
    joined_pairs = {"".join(compacts[index:index + 2]) for index in range(len(compacts) - 1)}
    if any(
        pair.startswith("privatechat") or pair.startswith("instructorreview")
        for pair in joined_pairs
    ):
        return True
    return False


def validate_public_ingest_source(
    *,
    document_type: object = "",
    source_name: object = "",
    storage_key: object = "",
) -> None:
    """One normalized deny policy shared by file, text, bulk, and automatic ingestion."""

    if any(_private_ingest_label(value) for value in (document_type, source_name, storage_key)):
        raise ValueError(
            "Private-chat and instructor-review ingestion is permanently disabled"
        )


def canonical_document_type(value: str | None) -> str:
    raw = (value or "").strip().lower()
    validate_public_ingest_source(document_type=raw)
    if raw in {"exam", "exam_pdf", "pdf_exam", "midterm", "final", "quiz"}:
        return "exam"
    return "course"


def validate_public_ingest_path(
    value: str | Path,
    *,
    allowed_root: str | Path | None = None,
) -> Path:
    """Reject private-corpus path conventions and paths escaping a declared source root.

    This is a shared fail-closed policy for bulk and direct file ingestion.  It is deliberately
    independent of the caller's requested ``document_type``: relabeling a private export as a
    course document must not make it ingestible.
    """

    path = Path(value).expanduser()
    resolved = path.resolve()
    if allowed_root is not None:
        root = Path(allowed_root).expanduser().resolve()
        if not resolved.is_relative_to(root):
            raise ValueError("Ingestion path escapes the configured public source root")
        inspected_parts = path.absolute().relative_to(Path(allowed_root).expanduser().absolute()).parts
    else:
        inspected_parts = path.parts
    validate_public_ingest_source(storage_key="/".join(inspected_parts))
    configured_reviews = Path(REVIEWS_DIR).expanduser().resolve()
    if resolved == configured_reviews or resolved.is_relative_to(configured_reviews):
        raise ValueError("Private-chat and instructor-review ingestion is permanently disabled")
    return resolved


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


def _save_uploaded(
    uploaded_files,
    target_dir: str,
    *,
    allowed_extensions: set[str] | None = None,
) -> list[str]:
    uploaded_files = _validate_upload_batch(uploaded_files)
    prepared: list[tuple[str, bytes]] = []
    total_bytes = 0
    for file in uploaded_files:
        filename = Path(str(file.filename or "upload")).name
        if filename in {"", ".", ".."}:
            raise ValueError("Invalid upload filename")
        validate_public_ingest_source(source_name=filename)
        extension = Path(filename).suffix.lower()
        effective_extensions = allowed_extensions if allowed_extensions is not None else SUPPORTED_EXTENSIONS
        if extension not in effective_extensions:
            raise ValueError("Unsupported upload file type")
        content = _read_validated_upload(file, extension)
        total_bytes += len(content)
        if total_bytes > MAX_UPLOAD_TOTAL_BYTES:
            raise ValueError("Upload exceeds the configured total-size limit")
        prepared.append((filename, content))

    # Nothing is persisted until the entire batch has passed validation, including the aggregate
    # size limit. This avoids orphaning a valid prefix when a later file makes the batch invalid.
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    saved_paths: list[str] = []
    for filename, content in prepared:
        digest = sha256(content).hexdigest()[:12]
        save_path = Path(target_dir) / f"{digest}-{filename}"
        with open(save_path, "wb") as f:
            f.write(content)
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
    max_batch_size = getattr(getattr(collection, "_client", None), "max_batch_size", None) or 5000
    for start in range(0, len(documents), max_batch_size):
        batch_docs = documents[start : start + max_batch_size]
        batch_ids = ids[start : start + max_batch_size]
        page_contents = [doc.page_content for doc in batch_docs]
        collection.upsert(
            ids=batch_ids,
            documents=page_contents,
            metadatas=[doc.metadata for doc in batch_docs],
            embeddings=embedding_function.embed_documents(page_contents),
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
        path = validate_public_ingest_path(raw_path)
        canonical_type = canonical_document_type(document_type)
        validate_public_ingest_source(
            document_type=document_type,
            source_name=path.name,
            storage_key=path,
        )
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
    validate_public_ingest_source(
        document_type=document_type,
        source_name=source_name,
        storage_key=storage_key,
    )
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
    file_paths = _save_uploaded(uploaded_files, UPLOAD_DIR, allowed_extensions={".pdf"})
    return ingest_file_paths(file_paths, document_type="exam")


def load_vectorstore_multi(uploaded_files) -> dict:
    uploaded_files = _validate_upload_batch(uploaded_files)
    prepared: list[tuple[str, bytes]] = []
    skipped: list[str] = []
    total_bytes = 0
    for file in uploaded_files:
        filename = Path(str(file.filename or "upload")).name
        validate_public_ingest_source(source_name=filename)
        ext = Path(filename).suffix.lower()
        if ext in SUPPORTED_EXTENSIONS:
            content = _read_validated_upload(file, ext)
            total_bytes += len(content)
            if total_bytes > MAX_UPLOAD_TOTAL_BYTES:
                raise ValueError("Upload exceeds the configured total-size limit")
            prepared.append((filename, content))
        else:
            skipped.append(filename)

    Path(DOCUMENTS_UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
    accepted: list[str] = []
    for filename, content in prepared:
        digest = sha256(content).hexdigest()[:12]
        save_path = Path(DOCUMENTS_UPLOAD_DIR) / f"{digest}-{filename}"
        with open(save_path, "wb") as f:
            f.write(content)
        accepted.append(str(save_path))

    chunk_count = ingest_file_paths(accepted, document_type="course") if accepted else 0
    return {
        "chunks": chunk_count,
        "accepted_files": [Path(p).name for p in accepted],
        "skipped_files": skipped,
        "supported_extensions": sorted(SUPPORTED_EXTENSIONS),
    }
