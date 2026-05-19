from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

from pymongo import ASCENDING, MongoClient

from modules.config import MONGO_DB_NAME, MONGO_URI


mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
sync_db = mongo_client[MONGO_DB_NAME]

source_documents = sync_db["sourceDocuments"]
ingestion_jobs = sync_db["ingestionJobs"]
upload_batches = sync_db["uploadBatches"]
instructor_reviews = sync_db["instructorReviews"]
exams = sync_db["exams"]
embedding_cache = sync_db["embeddingCache"]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def content_hash_bytes(content: bytes) -> str:
    return sha256(content).hexdigest()


def content_hash_text(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def file_content_hash(path: str | Path) -> str:
    return content_hash_bytes(Path(path).read_bytes())


def source_id_for(document_type: str, content_hash: str) -> str:
    return f"{document_type}:{content_hash[:24]}"


def embedding_cache_key(provider: str, model: str, text: str) -> str:
    safe_provider = (provider or "unknown").strip().replace(":", "_")
    safe_model = (model or "unknown").strip().replace(":", "_")
    return f"{safe_provider}_{safe_model}_{content_hash_text(text)}"


def ensure_source_of_truth_indexes() -> None:
    source_documents.create_index([("sourceId", ASCENDING)], unique=True)
    source_documents.create_index([("type", ASCENDING), ("status", ASCENDING)])
    source_documents.create_index([("contentHash", ASCENDING)])
    source_documents.create_index([("storageKey", ASCENDING)])

    ingestion_jobs.create_index([("jobId", ASCENDING)], unique=True)
    ingestion_jobs.create_index([("sourceId", ASCENDING), ("startedAt", ASCENDING)])
    ingestion_jobs.create_index([("status", ASCENDING)])

    upload_batches.create_index([("batchId", ASCENDING)], unique=True)
    upload_batches.create_index([("sourceId", ASCENDING)])
    upload_batches.create_index([("status", ASCENDING)])

    instructor_reviews.create_index([("reviewId", ASCENDING)], unique=True)
    instructor_reviews.create_index([("sourceId", ASCENDING)])
    instructor_reviews.create_index([("instructorName", ASCENDING)])
    instructor_reviews.create_index([("courseCode", ASCENDING)])

    exams.create_index([("examId", ASCENDING)], unique=True)
    exams.create_index([("sourceId", ASCENDING)])
    exams.create_index([("courseCode", ASCENDING)])

    embedding_cache.create_index([("cacheKey", ASCENDING)], unique=True)


def upsert_source_document(
    *,
    source_id: str,
    document_type: str,
    file_name: str = "",
    storage_key: str = "",
    content_hash: str,
    status: str,
    created_by: str = "system",
    metadata: dict[str, Any] | None = None,
    chunks_created: int | None = None,
) -> None:
    now = utc_now()
    update: dict[str, Any] = {
        "sourceId": source_id,
        "type": document_type,
        "fileName": file_name,
        "storageKey": storage_key,
        "contentHash": content_hash,
        "status": status,
        "createdBy": created_by,
        "metadata": metadata or {},
        "updatedAt": now,
    }
    if chunks_created is not None:
        update["chunksCreated"] = chunks_created
    if status == "indexed":
        update["lastIndexedAt"] = now

    source_documents.update_one(
        {"sourceId": source_id},
        {"$set": update, "$setOnInsert": {"createdAt": now}},
        upsert=True,
    )


def start_ingestion_job(source_id: str) -> str:
    now = utc_now()
    job_id = f"{source_id}:{now.strftime('%Y%m%d%H%M%S%f')}"
    ingestion_jobs.update_one(
        {"jobId": job_id},
        {
            "$set": {
                "jobId": job_id,
                "sourceId": source_id,
                "status": "processing",
                "chunksCreated": 0,
                "error": "",
                "startedAt": now,
                "updatedAt": now,
            },
            "$setOnInsert": {"createdAt": now},
        },
        upsert=True,
    )
    source_documents.update_one(
        {"sourceId": source_id},
        {"$set": {"status": "processing", "updatedAt": now}},
    )
    return job_id


def finish_ingestion_job(
    *,
    job_id: str,
    source_id: str,
    status: str,
    chunks_created: int,
    error: str = "",
) -> None:
    now = utc_now()
    ingestion_jobs.update_one(
        {"jobId": job_id},
        {
            "$set": {
                "status": status,
                "chunksCreated": chunks_created,
                "error": error,
                "finishedAt": now,
                "updatedAt": now,
            }
        },
    )
    source_documents.update_one(
        {"sourceId": source_id},
        {
            "$set": {
                "status": status,
                "chunksCreated": chunks_created,
                "updatedAt": now,
                **({"lastIndexedAt": now} if status == "indexed" else {}),
            }
        },
    )


def soft_delete_source(source_id: str) -> dict[str, Any] | None:
    now = utc_now()
    doc = source_documents.find_one({"sourceId": source_id})
    if not doc:
        return None
    source_documents.update_one(
        {"sourceId": source_id},
        {"$set": {"status": "deleted", "deletedAt": now, "updatedAt": now}},
    )
    return doc
