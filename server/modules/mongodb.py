from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ASCENDING, UpdateOne

from logger import logger
from modules.config import (
    ADMIN_USERNAME,
    CATALOG_DATA_DIR,
    CONVERSATION_RETENTION_DAYS,
    DEGREE_DATA_DIR,
    MONGO_DB_NAME,
    MONGO_URI,
)


client = AsyncIOMotorClient(MONGO_URI, serverSelectionTimeoutMS=3000)
db = client[MONGO_DB_NAME]

users = db["users"]
courses = db["courses"]
user_courses = db["user_courses"]
upload_batches = db["uploadBatches"]
source_documents = db["sourceDocuments"]
ingestion_jobs = db["ingestionJobs"]
instructor_reviews = db["instructorReviews"]
course_reviews = db["courseReviews"]
exams = db["exams"]
embedding_cache = db["embeddingCache"]
conversations = db["conversations"]


class ScheduleRevisionConflict(RuntimeError):
    """Raised when a manual edit was based on a stale saved-schedule revision."""

    def __init__(self, current: dict[str, Any]):
        super().__init__("Saved schedule was updated by another request")
        self.current = current


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
    await conversations.create_index([("sessionId", ASCENDING)], unique=True)
    await conversations.create_index(
        [("updatedAt", ASCENDING)],
        expireAfterSeconds=CONVERSATION_RETENTION_DAYS * 24 * 60 * 60,
        name="conversation_retention_ttl",
    )
    await ensure_user(ADMIN_USERNAME, role="admin")


async def ensure_user(username: str, role: str = "student") -> dict[str, Any]:
    clean_username = username.strip()
    await users.update_one(
        {"username": clean_username},
        {"$setOnInsert": {"username": clean_username, "role": role}},
        upsert=True,
    )
    user = await users.find_one({"username": clean_username})
    if not user:
        raise RuntimeError(f"Could not create or load user: {clean_username}")
    return user


async def seed_courses_from_catalog(data_dir: str = DEGREE_DATA_DIR, force: bool = False) -> int:
    """Upsert the searchable course list into MongoDB (roadmap section 5.3).

    Prefers data/course_catalog/current.jsonl (built from the degree-requirement corpus,
    ~800 courses with SU/ECTS/faculty + engineering/basic-science slots). Falls back to the
    legacy all_coursepage_info.jsonl, then to a tiny hardcoded set. Idempotent."""
    catalog_path = Path(data_dir).expanduser() / "course_catalog" / "current.jsonl"
    if catalog_path.exists():
        course_docs = _read_course_catalog(catalog_path)
    else:
        legacy = Path(CATALOG_DATA_DIR).expanduser() / "all_coursepage_info.jsonl"
        course_docs = _read_catalog_courses(legacy) if legacy.exists() else FALLBACK_COURSES

    # Only skip when Mongo already holds the full catalog (the old guard left 6 fallback rows).
    if not force and await courses.estimated_document_count() >= len(course_docs):
        return 0

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


def _read_course_catalog(path: Path) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            docs.append({
                "code": row.get("course_id"),
                "subject": row.get("subject"),
                "number": row.get("number"),
                "title": row.get("title"),
                "su_credits": row.get("su_credits"),
                "ects": row.get("ects"),
                "engineering_ects": row.get("engineering_ects"),
                "basic_science_ects": row.get("basic_science_ects"),
                "faculty": row.get("faculty"),
            })
    return docs


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


# Course-history status (roadmap section 8). Only these count toward graduation progress.
COURSE_STATUSES = {"completed", "enrolled", "failed", "withdrawn", "transfer", "exempted"}
ELIGIBLE_FOR_CREDIT = {"completed", "transfer", "exempted"}


def _norm_status(value: Any) -> str:
    status = str(value or "completed").strip().lower()
    if status not in COURSE_STATUSES:
        raise ValueError(f"Unsupported course status: {status}")
    return status


async def get_user_courses(username: str) -> list[dict[str, Any]]:
    user = await ensure_user(username)
    links = [link async for link in user_courses.find({"user_id": user["_id"]})]
    if not links:
        return []
    status_by_id = {link["course_id"]: _norm_status(link.get("status")) for link in links}
    cursor = courses.find({"_id": {"$in": list(status_by_id)}}).sort([("subject", ASCENDING), ("number", ASCENDING)])
    result = []
    async for doc in cursor:
        serialized = _serialize_course(doc)
        serialized["status"] = status_by_id.get(doc["_id"], "completed")
        result.append(serialized)
    return result


async def set_user_courses(
    username: str,
    course_ids: list[str],
    statuses: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    user = await ensure_user(username)
    statuses = statuses or {}
    # map requested ids -> status, preserving order and dropping invalid/duplicate ids
    requested: dict[Any, str] = {}
    for value in course_ids:
        oid = _to_object_id(value)
        if oid is not None and oid not in requested:
            requested[oid] = _norm_status(statuses.get(value))

    valid_ids = {
        doc["_id"]
        async for doc in courses.find({"_id": {"$in": list(requested)}}, {"_id": 1})
    }

    await user_courses.delete_many({"user_id": user["_id"]})
    docs = [
        {"user_id": user["_id"], "course_id": oid, "status": status}
        for oid, status in requested.items()
        if oid in valid_ids
    ]
    if docs:
        await user_courses.insert_many(docs, ordered=False)
    return await get_user_courses(username)


async def mutate_user_courses_by_codes(
    username: str,
    course_codes: list[str] | tuple[str, ...],
    *,
    status: str | None = None,
    remove: bool = False,
) -> dict[str, Any]:
    """Apply a chat-originated course-history mutation without replacing other rows.

    The existing PUT endpoint intentionally remains a full replacement operation. Chat
    commands need different semantics: "I completed CS 201" must upsert only CS 201 and
    preserve every course already stored for the student.
    """
    user = await ensure_user(username)
    normalized = list(
        dict.fromkeys(
            re.sub(r"\s+", " ", str(code or "").strip().upper())
            for code in course_codes
            if str(code or "").strip()
        )
    )
    found = [
        doc
        async for doc in courses.find(
            {"code": {"$in": normalized}},
            {
                "_id": 1,
                "code": 1,
                "title": 1,
                "su_credits": 1,
                "ects": 1,
                "engineering_ects": 1,
                "basic_science_ects": 1,
            },
        )
    ]
    by_code = {str(doc.get("code", "")).upper(): doc for doc in found}
    updates: list[dict[str, Any]] = []
    clean_status = _norm_status(status)

    if remove and found:
        await user_courses.delete_many(
            {"user_id": user["_id"], "course_id": {"$in": [doc["_id"] for doc in found]}}
        )
        for doc in found:
            updates.append(
                {
                    "code": doc.get("code"),
                    "title": doc.get("title"),
                    "status": None,
                    "action": "removed",
                }
            )
    elif found:
        operations = [
            UpdateOne(
                {"user_id": user["_id"], "course_id": doc["_id"]},
                {
                    "$set": {"status": clean_status},
                    "$setOnInsert": {"user_id": user["_id"], "course_id": doc["_id"]},
                },
                upsert=True,
            )
            for doc in found
        ]
        await user_courses.bulk_write(operations, ordered=False)
        for doc in found:
            updates.append(
                {
                    "code": doc.get("code"),
                    "title": doc.get("title"),
                    "status": clean_status,
                    "action": "updated",
                }
            )

    selected = await get_user_courses(username)
    credit_courses = [
        course
        for course in selected
        if course.get("status", "completed") in ELIGIBLE_FOR_CREDIT
    ]
    return {
        "updates": sorted(updates, key=lambda row: str(row.get("code") or "")),
        "missing": [code for code in normalized if code not in by_code],
        "courses": selected,
        "course_count": len(selected),
        "credit_eligible_course_count": len(credit_courses),
        "total_su_credits": sum(_su_credit_value(course.get("su_credits")) for course in credit_courses),
    }


async def get_completed_course_codes(username: str) -> list[str]:
    """Course codes with a credit-eligible status (completed/transfer/exempted)."""
    selected = await get_user_courses(username)
    return [c["code"] for c in selected if c.get("status", "completed") in ELIGIBLE_FOR_CREDIT and c.get("code")]


# `current_term` was removed 2026-07-22: it was collected but never read. Retrieval scoping and
# the degree audit both key off `curriculum_term` (the admit term / graduation contract). Any
# value already stored on existing user documents is simply ignored — `set_academic_profile`
# persists only keys present here.
DEFAULT_ACADEMIC_PROFILE = {
    "major": None,
    "degree_code": None,
    "admission_term": None,
    "curriculum_term": None,
    "academic_year": None,
    "minor_codes": [],
    "profile_status": "unset",
}


async def get_academic_profile(username: str) -> dict[str, Any]:
    """Return the student's academic profile (roadmap section 8). Never raises.

    Reads are filtered to the known keys, not merged blindly: user documents written before a
    field was retired (e.g. `current_term`) still carry it, and echoing that back would resurrect
    a field the product no longer has. The stored value is left alone — it is simply not served.
    """
    user = await ensure_user(username)
    stored = user.get("academic_profile") or {}
    return {key: stored.get(key, default) for key, default in DEFAULT_ACADEMIC_PROFILE.items()}


async def set_academic_profile(username: str, profile: dict[str, Any]) -> dict[str, Any]:
    """Store/merge the student's academic profile. Only known keys are persisted."""
    allowed = set(DEFAULT_ACADEMIC_PROFILE)
    clean = {k: v for k, v in (profile or {}).items() if k in allowed}
    if clean.get("major"):
        clean["major"] = str(clean["major"]).strip().upper()
    if clean:
        clean.setdefault("profile_status", "confirmed")
    await ensure_user(username)
    await users.update_one(
        {"username": username.strip()},
        {"$set": {f"academic_profile.{k}": v for k, v in clean.items()}},
    )
    return await get_academic_profile(username)


def _empty_user_schedule() -> dict[str, Any]:
    return {"schedule": None, "revision": 0, "updated_at": None}


async def get_user_schedule(username: str) -> dict[str, Any]:
    """Return the user's editable weekly schedule and optimistic-lock metadata."""
    user = await ensure_user(username)
    stored = user.get("weekly_schedule") or {}
    schedule = stored.get("schedule")
    if not isinstance(schedule, dict):
        return _empty_user_schedule()
    return {
        "schedule": schedule,
        "revision": max(int(stored.get("revision") or 0), 0),
        "updated_at": stored.get("updated_at"),
    }


async def set_user_schedule(
    username: str,
    schedule: dict[str, Any],
    *,
    expected_revision: int | None = None,
) -> dict[str, Any]:
    """Atomically replace one user's schedule.

    ``expected_revision`` is optional for chat-generated replacement. Manual editors should send
    the revision returned by GET; only one concurrent writer can then win and stale writes receive
    a 409 from the API instead of silently overwriting a newer timetable.
    """
    if not isinstance(schedule, dict):
        raise ValueError("schedule must be an object")
    clean_username = str(username or "").strip()
    if not clean_username:
        raise ValueError("username is required")
    if expected_revision is not None and expected_revision < 0:
        raise ValueError("expected_revision cannot be negative")

    await ensure_user(clean_username)
    query: dict[str, Any] = {"username": clean_username}
    if expected_revision == 0:
        query["$or"] = [
            {"weekly_schedule.revision": {"$exists": False}},
            {"weekly_schedule.revision": 0},
        ]
    elif expected_revision is not None:
        query["weekly_schedule.revision"] = expected_revision

    updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    result = await users.update_one(
        query,
        {
            "$set": {
                "weekly_schedule.schedule": schedule,
                "weekly_schedule.updated_at": updated_at,
            },
            "$inc": {"weekly_schedule.revision": 1},
        },
    )
    if result.matched_count != 1:
        raise ScheduleRevisionConflict(await get_user_schedule(clean_username))
    return await get_user_schedule(clean_username)


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
        status = _norm_status(course.get("status"))
        if status in ELIGIBLE_FOR_CREDIT:
            total_su_credits += _su_credit_value(course.get("su_credits"))
        fields = [
            f"{course['code']} - {course['title']}",
            f"status {status}",
            _maybe(f"SU credits {course.get('su_credits')}", course.get("su_credits") is not None),
            _maybe(f"ECTS {course.get('ects')}", course.get("ects") is not None),
            _maybe(f"engineering ECTS {course.get('engineering_ects')}", course.get("engineering_ects") is not None),
            _maybe(f"basic science ECTS {course.get('basic_science_ects')}", course.get("basic_science_ects") is not None),
        ]
        course_lines.append("; ".join(field for field in fields if field))

    total_min_su_credits: float | None = None
    try:
        profile = await get_academic_profile(username)
        program = str(profile.get("major") or "").strip().upper()
        term = str(profile.get("curriculum_term") or "").strip()
        if program and term:
            from modules.degree_audit import load_requirements

            requirements = load_requirements(program, term)
            if requirements and requirements.get("total_min_su_credits") is not None:
                total_min_su_credits = float(requirements["total_min_su_credits"])
    except Exception:
        logger.exception("Could not resolve curriculum credit target")

    eligible_count = sum(
        1 for course in selected if _norm_status(course.get("status")) in ELIGIBLE_FOR_CREDIT
    )
    total_line = f"Authoritative credit-eligible SU total from MongoDB: {_format_credit(total_su_credits)}."
    remaining_line = ""
    if total_min_su_credits is not None:
        total_line = (
            "Authoritative credit-eligible SU total from MongoDB: "
            f"{_format_credit(total_su_credits)}/{_format_credit(total_min_su_credits)} SU credits."
        )
        remaining_line = (
            "\nAuthoritative remaining SU credits to the selected curriculum minimum: "
            f"{_format_credit(max(total_min_su_credits - total_su_credits, 0))}."
        )

    return (
        f"User profile from MongoDB: {username}.\n"
        f"{total_line}{remaining_line}\n"
        f"Authoritative credit-eligible course count from MongoDB: {eligible_count}.\n"
        f"All stored course-history rows, including non-credit statuses: {len(selected)}.\n"
        "Use only completed, transfer, and exempted statuses for graduation arithmetic.\n"
        "Important intent guard: this profile contains graduation totals, but use them only when the user explicitly asks for graduation, credit, audit, or degree evaluation. For course recommendation questions, use this profile only as a taken-course exclusion list.\n"
        "Stored courses selected by the user:\n"
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
        "faculty": doc.get("faculty"),
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
