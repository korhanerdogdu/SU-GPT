from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ASCENDING, UpdateOne

from logger import logger
from modules.config import ADMIN_USERNAME, CATALOG_DATA_DIR, MONGO_DB_NAME, MONGO_URI


client = AsyncIOMotorClient(MONGO_URI, serverSelectionTimeoutMS=3000)
db = client[MONGO_DB_NAME]

users = db["users"]
courses = db["courses"]
user_courses = db["user_courses"]
upload_batches = db["uploadBatches"]
source_documents = db["sourceDocuments"]
ingestion_jobs = db["ingestionJobs"]
instructor_reviews = db["instructorReviews"]
exams = db["exams"]
embedding_cache = db["embeddingCache"]


FALLBACK_COURSES = [
    {"code": "CS 300", "subject": "CS", "number": "300", "title": "Data Structures", "su_credits": 3, "ects": 6},
    {"code": "CS 302", "subject": "CS", "number": "302", "title": "Formal Languages and Automata Theory", "su_credits": 3, "ects": 6},
    {"code": "CS 307", "subject": "CS", "number": "307", "title": "Operating Systems", "su_credits": 3, "ects": 6},
    {"code": "CS 310", "subject": "CS", "number": "310", "title": "Mobile Application Development", "su_credits": 3, "ects": 6},
    {"code": "CS 455", "subject": "CS", "number": "455", "title": "Deep Learning", "su_credits": 3, "ects": 6},
    {"code": "CS 555", "subject": "CS", "number": "555", "title": "Advanced Deep Learning", "su_credits": 3, "ects": 6},
]


async def ensure_database() -> None:
    await client.admin.command("ping")
    await users.create_index([("username", ASCENDING)], unique=True)
    await courses.create_index([("code", ASCENDING)], unique=True)
    await courses.create_index([("subject", ASCENDING), ("number", ASCENDING)])
    await user_courses.create_index([("user_id", ASCENDING), ("course_id", ASCENDING)], unique=True)
    await user_courses.create_index([("user_id", ASCENDING)])
    await upload_batches.create_index([("batchId", ASCENDING)], unique=True)
    await upload_batches.create_index([("sourceId", ASCENDING)])
    await upload_batches.create_index([("status", ASCENDING)])
    await source_documents.create_index([("sourceId", ASCENDING)], unique=True)
    await source_documents.create_index([("type", ASCENDING), ("status", ASCENDING)])
    await source_documents.create_index([("contentHash", ASCENDING)])
    await source_documents.create_index([("storageKey", ASCENDING)])
    await ingestion_jobs.create_index([("jobId", ASCENDING)], unique=True)
    await ingestion_jobs.create_index([("sourceId", ASCENDING), ("startedAt", ASCENDING)])
    await instructor_reviews.create_index([("reviewId", ASCENDING)], unique=True)
    await instructor_reviews.create_index([("sourceId", ASCENDING)])
    await instructor_reviews.create_index([("instructorName", ASCENDING)])
    await instructor_reviews.create_index([("courseCode", ASCENDING)])
    await exams.create_index([("examId", ASCENDING)], unique=True)
    await exams.create_index([("sourceId", ASCENDING)])
    await exams.create_index([("courseCode", ASCENDING)])
    await embedding_cache.create_index([("cacheKey", ASCENDING)], unique=True)
    await ensure_user(ADMIN_USERNAME)


async def ensure_user(username: str) -> dict[str, Any]:
    clean_username = username.strip()
    await users.update_one(
        {"username": clean_username},
        {"$setOnInsert": {"username": clean_username, "role": "admin"}},
        upsert=True,
    )
    user = await users.find_one({"username": clean_username})
    if not user:
        raise RuntimeError(f"Could not create or load user: {clean_username}")
    return user


async def seed_courses_from_catalog(data_dir: str = CATALOG_DATA_DIR) -> int:
    if await courses.estimated_document_count() > 0:
        return 0

    path = Path(data_dir).expanduser() / "all_coursepage_info.jsonl"
    course_docs = _read_catalog_courses(path) if path.exists() else FALLBACK_COURSES
    operations = [
        UpdateOne({"code": doc["code"]}, {"$set": doc}, upsert=True)
        for doc in course_docs
        if doc.get("code") and doc.get("title")
    ]
    if not operations:
        return 0
    result = await courses.bulk_write(operations, ordered=False)
    inserted = result.upserted_count + result.modified_count
    logger.info("course seed complete: %d course records touched", inserted)
    return inserted


async def list_courses(search: str = "", limit: int = 200) -> list[dict[str, Any]]:
    query: dict[str, Any] = {}
    if search.strip():
        pattern = re.escape(search.strip())
        query = {
            "$or": [
                {"code": {"$regex": pattern, "$options": "i"}},
                {"title": {"$regex": pattern, "$options": "i"}},
                {"subject": {"$regex": pattern, "$options": "i"}},
            ]
        }
    cursor = courses.find(query).sort([("subject", ASCENDING), ("number", ASCENDING)]).limit(limit)
    return [_serialize_course(doc) async for doc in cursor]


async def get_user_courses(username: str) -> list[dict[str, Any]]:
    user = await ensure_user(username)
    links = [link async for link in user_courses.find({"user_id": user["_id"]})]
    if not links:
        return []
    course_ids = [link["course_id"] for link in links]
    cursor = courses.find({"_id": {"$in": course_ids}}).sort([("subject", ASCENDING), ("number", ASCENDING)])
    return [_serialize_course(doc) async for doc in cursor]


async def set_user_courses(username: str, course_ids: list[str]) -> list[dict[str, Any]]:
    user = await ensure_user(username)
    object_ids = [_to_object_id(value) for value in course_ids]
    object_ids = [value for value in dict.fromkeys(object_ids) if value is not None]

    valid_ids = [
        doc["_id"]
        async for doc in courses.find({"_id": {"$in": object_ids}}, {"_id": 1})
    ]

    await user_courses.delete_many({"user_id": user["_id"]})
    if valid_ids:
        await user_courses.insert_many(
            [{"user_id": user["_id"], "course_id": course_id} for course_id in valid_ids],
            ordered=False,
        )
    return await get_user_courses(username)


async def get_user_course_context(username: str | None) -> str:
    if not username:
        return ""
    try:
        selected = await get_user_courses(username)
    except Exception:
        logger.exception("Could not load user course context")
        return ""
    if not selected:
        return (
            f"User profile: {username}. No completed/taken courses are currently "
            "stored in MongoDB."
        )

    course_lines = []
    total_su_credits = 0.0
    for course in selected:
        su_credits = _su_credit_value(course.get("su_credits"))
        total_su_credits += su_credits
        fields = [
            f"{course['code']} - {course['title']}",
            _maybe(f"SU credits {course.get('su_credits')}", course.get("su_credits") is not None),
            _maybe(f"ECTS {course.get('ects')}", course.get("ects") is not None),
            _maybe(f"engineering ECTS {course.get('engineering_ects')}", course.get("engineering_ects") is not None),
            _maybe(f"basic science ECTS {course.get('basic_science_ects')}", course.get("basic_science_ects") is not None),
        ]
        course_lines.append("; ".join(field for field in fields if field))

    return (
        f"User profile from MongoDB: {username}.\n"
        f"Authoritative completed SU credit total from MongoDB: {_format_credit(total_su_credits)}/125 SU credits.\n"
        f"Authoritative remaining SU credits to 125: {_format_credit(max(125 - total_su_credits, 0))}.\n"
        f"Authoritative completed course count from MongoDB: {len(selected)}.\n"
        "Use the authoritative MongoDB total for final graduation arithmetic. Do not recompute the final x/125 total from memory if this line is present.\n"
        "Important intent guard: this profile contains graduation totals, but use them only when the user explicitly asks for graduation, credit, audit, or degree evaluation. For course recommendation questions, use this profile only as a taken-course exclusion list.\n"
        "Taken/completed courses selected by the user:\n"
        + "\n".join(f"- {line}" for line in course_lines)
        + "\nUse this personal course history when answering eligibility, prerequisite, recommendation, or personalization questions."
    )


def _read_catalog_courses(path: Path) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            subject = _clean(row.get("subj_code") or row.get("parsed_subj_code"))
            number = _clean(row.get("crse_numb") or row.get("parsed_crse_numb"))
            code = _clean(row.get("course_id")) or f"{subject} {number}".strip()
            if code and " " not in code:
                match = re.match(r"^([A-Za-z]+)(\d.+)$", code)
                if match:
                    code = f"{match.group(1).upper()} {match.group(2)}"
            docs.append(
                {
                    "code": code,
                    "subject": subject or code.split(" ", 1)[0],
                    "number": number or (code.split(" ", 1)[1] if " " in code else ""),
                    "title": _clean(row.get("title") or row.get("header_text")),
                    "su_credits": row.get("su_credits"),
                    "ects": row.get("ects"),
                    "engineering_ects": row.get("engineering"),
                    "basic_science_ects": row.get("basic_science"),
                    "description": _clean(row.get("description")),
                    "prerequisites": _clean(row.get("prerequisites")),
                    "corequisites": _clean(row.get("corequisites")),
                    "source_url": _clean(row.get("source_url")),
                }
            )
    return docs


def _serialize_course(doc: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(doc["_id"]),
        "code": doc.get("code", ""),
        "subject": doc.get("subject", ""),
        "number": doc.get("number", ""),
        "title": doc.get("title", ""),
        "su_credits": doc.get("su_credits"),
        "ects": doc.get("ects"),
        "engineering_ects": doc.get("engineering_ects"),
        "basic_science_ects": doc.get("basic_science_ects"),
        "description": doc.get("description", ""),
        "prerequisites": doc.get("prerequisites", ""),
        "corequisites": doc.get("corequisites", ""),
        "source_url": doc.get("source_url", ""),
    }


def _to_object_id(value: str) -> ObjectId | None:
    try:
        return ObjectId(value)
    except Exception:
        return None


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _su_credit_value(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _format_credit(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:.1f}"


def _maybe(text: str, condition: bool) -> str:
    return text if condition else ""
