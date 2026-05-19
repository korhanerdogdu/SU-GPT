from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

from fastapi import UploadFile

from modules.config import DOCUMENT_STORAGE_DIR
from modules.load_vectorstore import get_vectorstore, ingest_file_paths, ingest_text_source
from modules.source_of_truth import (
    content_hash_bytes,
    content_hash_text,
    exams,
    instructor_reviews,
    soft_delete_source,
    source_id_for,
    source_documents,
    upsert_source_document,
    upload_batches,
    utc_now,
)


WHATSAPP_LINE_RE = re.compile(
    r"^(?P<date>\d{1,2}[./]\d{1,2}[./]\d{2,4}),?\s+"
    r"(?P<time>\d{1,2}:\d{2})(?:\s?[AP]M)?\s+-\s+"
    r"(?:(?P<author>[^:]+):\s+)?(?P<body>.*)$"
)
BRACKET_WHATSAPP_LINE_RE = re.compile(
    r"^\[(?P<date>\d{1,2}[./]\d{1,2}[./]\d{2,4})\s+"
    r"(?P<time>\d{1,2}:\d{2})(?::\d{2})?\]\s+"
    r"(?:(?P<author>[^:]+):\s+)?(?P<body>.*)$"
)
PII_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+|\+?\d[\d\s().-]{7,}\d")


def create_pending_whatsapp_batch(upload: UploadFile, uploaded_by: str = "admin") -> dict[str, Any]:
    content = upload.file.read()
    content_hash = content_hash_bytes(content)
    source_id = source_id_for("review", content_hash)
    batch_id = f"whatsapp:{content_hash[:24]}"
    storage_key = _save_bytes(content, "whatsapp", f"{content_hash[:16]}-{_safe_filename(upload.filename)}")
    text = _decode_text(content)
    messages = parse_whatsapp_export(text)
    now = utc_now()

    upsert_source_document(
        source_id=source_id,
        document_type="review",
        file_name=upload.filename or "whatsapp.txt",
        storage_key=str(storage_key),
        content_hash=content_hash,
        status="pending",
        created_by=uploaded_by,
        metadata={"batchId": batch_id, "documentType": "review", "sourceId": source_id},
    )
    upload_batches.update_one(
        {"batchId": batch_id},
        {
            "$set": {
                "batchId": batch_id,
                "sourceId": source_id,
                "type": "whatsapp",
                "status": "pending",
                "storageKey": str(storage_key),
                "fileName": upload.filename or "whatsapp.txt",
                "contentHash": content_hash,
                "uploadedBy": uploaded_by,
                "messageCount": len(messages),
                "preview": [message["text"] for message in messages[:8]],
                "updatedAt": now,
            },
            "$setOnInsert": {"createdAt": now},
        },
        upsert=True,
    )
    return {
        "batchId": batch_id,
        "sourceId": source_id,
        "status": "pending",
        "messageCount": len(messages),
        "preview": [message["text"] for message in messages[:8]],
    }


def confirm_whatsapp_batch(
    batch_id: str,
    *,
    approved: bool = True,
    approved_by: str = "admin",
) -> dict[str, Any]:
    batch = upload_batches.find_one({"batchId": batch_id})
    if not batch:
        raise ValueError(f"Upload batch not found: {batch_id}")

    now = utc_now()
    if not approved:
        upload_batches.update_one(
            {"batchId": batch_id},
            {"$set": {"status": "rejected", "approvedBy": approved_by, "updatedAt": now}},
        )
        source_documents.update_one(
            {"sourceId": batch["sourceId"]},
            {"$set": {"status": "rejected", "updatedAt": now}},
        )
        return {"batchId": batch_id, "sourceId": batch["sourceId"], "status": "rejected"}

    storage_key = str(batch.get("storageKey") or "")
    raw_text = Path(storage_key).read_text(encoding="utf-8", errors="replace")
    messages = parse_whatsapp_export(raw_text)
    normalized_text = "\n".join(message["text"] for message in messages)

    for index, message in enumerate(messages):
        review_hash = content_hash_text(f"{batch['sourceId']}:{index}:{message['text']}")
        instructor_reviews.update_one(
            {"reviewId": f"review:{review_hash[:24]}"},
            {
                "$set": {
                    "reviewId": f"review:{review_hash[:24]}",
                    "sourceId": batch["sourceId"],
                    "uploadBatchId": batch_id,
                    "text": message["text"],
                    "author": message.get("author") or "",
                    "messageDate": message.get("date") or "",
                    "status": "approved",
                    "updatedAt": now,
                },
                "$setOnInsert": {"createdAt": now},
            },
            upsert=True,
        )

    chunks = ingest_text_source(
        text=normalized_text,
        source_name=str(batch.get("fileName") or "whatsapp.txt"),
        document_type="review",
        source_id=batch["sourceId"],
        storage_key=storage_key,
        created_by=approved_by,
        extra_metadata={"uploadBatchId": batch_id, "reviewStatus": "approved"},
    )
    upload_batches.update_one(
        {"batchId": batch_id},
        {
            "$set": {
                "status": "indexed",
                "approvedBy": approved_by,
                "approvedAt": now,
                "chunksCreated": chunks,
                "updatedAt": now,
            }
        },
    )
    return {
        "batchId": batch_id,
        "sourceId": batch["sourceId"],
        "status": "indexed",
        "chunks": chunks,
    }


def ingest_exam_upload(
    upload: UploadFile,
    *,
    course_code: str = "",
    year: str = "",
    semester: str = "",
    exam_type: str = "",
    uploaded_by: str = "admin",
) -> dict[str, Any]:
    content = upload.file.read()
    content_hash = content_hash_bytes(content)
    source_id = source_id_for("exam", content_hash)
    storage_key = _save_bytes(content, "exams", f"{content_hash[:16]}-{_safe_filename(upload.filename)}")
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


def parse_whatsapp_export(raw_text: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = BRACKET_WHATSAPP_LINE_RE.match(line) or WHATSAPP_LINE_RE.match(line)
        if match:
            if current:
                messages.append(current)
            body = _clean_message(match.group("body"))
            current = {
                "date": match.group("date") or "",
                "time": match.group("time") or "",
                "author": _clean_message(match.group("author") or ""),
                "text": body,
            }
            continue
        if current:
            current["text"] = f"{current['text']} {_clean_message(line)}".strip()
        else:
            current = {"date": "", "time": "", "author": "", "text": _clean_message(line)}
    if current:
        messages.append(current)
    return [message for message in messages if message.get("text")]


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


def _decode_text(content: bytes) -> str:
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return content.decode("utf-8", errors="replace")


def _clean_message(text: str) -> str:
    return PII_RE.sub("[redacted]", text or "").strip()
