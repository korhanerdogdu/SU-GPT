from hashlib import sha1
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from modules.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL_NAME,
)
from modules.document_cleaner import clean_text
from modules.document_loaders import SUPPORTED_EXTENSIONS, load_document


# Existing PDF directory (kept for backward compatibility with /upload_pdfs/).
UPLOAD_DIR = "./uploaded_pdfs"
# New multi-format upload directory (Section 2).
DOCUMENTS_UPLOAD_DIR = "./uploaded_documents"


def get_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def get_vectorstore():
    return Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=get_embedding_model(),
    )


def _chunk_id(source: str, location_key: str, text: str, index: int) -> str:
    digest = sha1(text.encode("utf-8")).hexdigest()[:10]
    location = location_key if location_key else "doc"
    stem = Path(source).stem
    return f"{stem}::{location}::{index}::{digest}"


def _location_key(metadata: dict) -> str:
    if metadata.get("page") is not None:
        return f"p{metadata['page']}"
    if metadata.get("slide") is not None:
        return f"s{metadata['slide']}"
    if metadata.get("section"):
        return f"sec-{metadata['section'][:20]}"
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


def _records_to_chunks(records: list[dict]) -> tuple[list[Document], list[str]]:
    """Clean each record, split into chunks, attach stable metadata + chunk_id."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks: list[Document] = []
    ids: list[str] = []

    for record in records:
        cleaned = clean_text(record.get("text", ""))
        if not cleaned:
            continue
        base_metadata = record.get("metadata", {}) or {}
        loc_key = _location_key(base_metadata)

        for index, piece in enumerate(splitter.split_text(cleaned)):
            metadata = {
                "source": base_metadata.get("source"),
                "file_name": base_metadata.get("source"),  # back-compat alias
                "document_type": base_metadata.get("document_type"),
                "page": base_metadata.get("page"),
                "slide": base_metadata.get("slide"),
                "section": base_metadata.get("section"),
                "language": base_metadata.get("language"),
                "chunk_id": None,  # filled below
            }
            cid = _chunk_id(
                source=base_metadata.get("source", "doc"),
                location_key=loc_key,
                text=piece,
                index=index,
            )
            metadata["chunk_id"] = cid

            # Drop None values so Chroma doesn't reject the metadata dict.
            metadata = {k: v for k, v in metadata.items() if v is not None}

            chunks.append(Document(page_content=piece, metadata=metadata))
            ids.append(cid)

    return chunks, ids


def _ingest_paths(file_paths: list[str]) -> int:
    all_chunks: list[Document] = []
    all_ids: list[str] = []

    for path in file_paths:
        records = load_document(path)
        chunks, ids = _records_to_chunks(records)
        all_chunks.extend(chunks)
        all_ids.extend(ids)

    if all_chunks:
        Path(CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
        vectorstore = get_vectorstore()
        vectorstore.add_documents(documents=all_chunks, ids=all_ids)

    return len(all_chunks)


def load_vectorstore(uploaded_files) -> int:
    """Legacy PDF-only entry point used by POST /upload_pdfs/.

    Saves PDFs into the original ./uploaded_pdfs directory to preserve
    existing behavior, then ingests them via the new pipeline.
    """
    file_paths = _save_uploaded(uploaded_files, UPLOAD_DIR)
    pdf_paths = [p for p in file_paths if Path(p).suffix.lower() == ".pdf"]
    return _ingest_paths(pdf_paths)


def load_vectorstore_multi(uploaded_files) -> dict:
    """Multi-format entry point used by POST /upload_documents/.

    Accepts PDF/PPTX/DOCX/MD/TXT. Unsupported extensions are skipped and
    reported back in the result.
    """
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

    chunk_count = _ingest_paths(accepted) if accepted else 0
    return {
        "chunks": chunk_count,
        "accepted_files": [Path(p).name for p in accepted],
        "skipped_files": skipped,
        "supported_extensions": sorted(SUPPORTED_EXTENSIONS),
    }
