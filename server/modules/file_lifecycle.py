from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

from fastapi import UploadFile

from modules.config import DOCUMENT_STORAGE_DIR
from modules.load_vectorstore import _read_validated_upload, get_vectorstore, ingest_file_paths
from modules.source_of_truth import (
    content_hash_bytes,
    exams,
    instructor_reviews,
    soft_delete_source,
    source_id_for,
    source_documents,
    upload_batches,
    utc_now,
)


def create_pending_whatsapp_batch(upload: UploadFile, uploaded_by: str = "admin") -> dict[str, Any]:
    del upload, uploaded_by
    raise RuntimeError(
        "Private chat ingestion is permanently disabled; use the consent-based course-only review service."
    )


def confirm_whatsapp_batch(
    batch_id: str,
    *,
    approved: bool = True,
    approved_by: str = "admin",
) -> dict[str, Any]:
    del batch_id, approved, approved_by
    raise RuntimeError(
        "Private chat and instructor-review ingestion is permanently disabled."
    )


def ingest_exam_upload(
    upload: UploadFile,
    *,
    course_code: str = "",
    year: str = "",
    semester: str = "",
    exam_type: str = "",
    uploaded_by: str = "admin",
) -> dict[str, Any]:
    filename = Path(str(upload.filename or "exam.pdf")).name
    if Path(filename).suffix.lower() != ".pdf":
        raise ValueError("Unsupported upload file type")
    content = _read_validated_upload(upload, ".pdf")
    content_hash = content_hash_bytes(content)
    source_id = source_id_for("exam", content_hash)
    storage_key = _save_bytes(content, "exams", f"{content_hash[:16]}-{_safe_filename(filename)}")
    chunks = ingest_file_paths(
        [str(storage_key)],
        document_type="exam",
        created_by=uploaded_by,
        extra_metadata={
            "courseCode": course_code.strip().upper(),
            "year": year.strip(),
            "semester": semester.strip(),
            "examType": exam_type.strip().lower(),
        },
    )
    now = utc_now()
    exam_id = f"exam:{content_hash[:24]}"
    exams.update_one(
        {"examId": exam_id},
        {
            "$set": {
                "examId": exam_id,
                "sourceId": source_id,
                "courseCode": course_code.strip().upper(),
                "year": year.strip(),
                "semester": semester.strip(),
                "examType": exam_type.strip().lower(),
                "fileName": upload.filename or "exam.pdf",
                "storageKey": str(storage_key),
                "contentHash": content_hash,
                "status": "indexed",
                "chunksCreated": chunks,
                "updatedAt": now,
            },
            "$setOnInsert": {"createdAt": now},
        },
        upsert=True,
    )
    return {"examId": exam_id, "sourceId": source_id, "status": "indexed", "chunks": chunks}


def cascade_delete_source(source_id: str, *, hard: bool = False) -> dict[str, Any]:
    source = soft_delete_source(source_id)
    if not source:
        raise ValueError(f"Source document not found: {source_id}")

    storage_key = str(source.get("storageKey") or "")
    if storage_key and Path(storage_key).exists():
        path = Path(storage_key)
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()

    deleted_chunks = 0
    collection = getattr(get_vectorstore(), "_collection", None)
    if collection is not None:
        existing = collection.get(where={"sourceId": source_id}, include=["metadatas"])
        deleted_chunks = len(existing.get("ids") or [])
        collection.delete(where={"sourceId": source_id})

    now = utc_now()
    instructor_reviews.update_many(
        {"sourceId": source_id},
        {"$set": {"status": "deleted", "deletedAt": now, "updatedAt": now}},
    )
    exams.update_many(
        {"sourceId": source_id},
        {"$set": {"status": "deleted", "deletedAt": now, "updatedAt": now}},
    )
    upload_batches.update_many(
        {"sourceId": source_id},
        {"$set": {"status": "deleted", "deletedAt": now, "updatedAt": now}},
    )
    if hard:
        source_documents.delete_one({"sourceId": source_id})

    return {
        "sourceId": source_id,
        "status": "deleted",
        "storageDeleted": bool(storage_key),
        "chunksDeleted": deleted_chunks,
        "hardDeleted": hard,
    }


def _save_bytes(content: bytes, category: str, filename: str) -> Path:
    target_dir = Path(DOCUMENT_STORAGE_DIR) / category
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / filename
    path.write_bytes(content)
    return path


def _safe_filename(filename: str | None) -> str:
    name = Path(filename or "upload.txt").name
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return safe or "upload.txt"
