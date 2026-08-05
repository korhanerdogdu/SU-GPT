from __future__ import annotations

"""
Bounded session memory (roadmap section 19).

The LLM stays mostly stateless, but we keep the last few turns + a small "working context"
(last course code / term) per session so follow-ups like "can I take it next semester?"
resolve to the course from the previous turn. This memory only improves usability — it must
NEVER override official JSON data or the deterministic audit (roadmap section 19.6). Authenticated
memory reads and writes fail closed: a storage outage or ownership mismatch cannot silently turn a
stateful request into a cross-user read or claim an existing session.
"""

import re
from datetime import datetime, timezone
from typing import Any

from logger import logger
from modules.mongodb import db
from modules.config import MAX_TRANSCRIPT_MESSAGES

conversations = db["conversations"]

# `recent_turns` stays capped — it is the prompt's working memory and must not grow unbounded.
# `messages` is the durable transcript the user browses; the two serve different jobs and are
# stored separately on purpose.
MAX_TURNS = 5
TITLE_MAX = 70
_COURSE_RE = re.compile(r"\b([A-Z]{2,5})\s*-?\s*(\d{3}[0-9A-Z]?)\b")
_REFERENCE_RE = re.compile(
    r"\b(this course|that course|the course|same course|it|its|"
    r"bu ders|bu dersi|bu dersin|o ders|o dersi|şu ders|su ders|aynı ders|ayni ders)\b",
    re.IGNORECASE,
)
_FOLLOWUP_RE = re.compile(
    r"^\s*(?:"
    r"yani|peki|tamam|öyleyse|o zaman|then|so|okay|ok|"
    r"ne almam lazım|ne almalıyım|what should i take|what do i need"
    r")\b",
    re.IGNORECASE,
)
_TITLE_FILLER_RE = re.compile(
    r"\b(?:merhaba|selam|lütfen|lutfen|acaba|bana|yardım eder misin|"
    r"yardim eder misin|please|could you|can you|tell me)\b",
    re.IGNORECASE,
)


def extract_course_code(text: str) -> str | None:
    m = _COURSE_RE.search((text or "").upper())
    return f"{m.group(1)} {m.group(2)}" if m else None


def resolve_reference(question: str, working_context: dict[str, Any] | None) -> str:
    """If the question refers to a prior course without naming one, inject that course code."""
    working_context = working_context or {}
    last_course = working_context.get("last_course_id")
    if last_course and _REFERENCE_RE.search(question or "") and not extract_course_code(question):
        return f"{question} (the course being referred to is {last_course})"
    last_intent = working_context.get("last_intent")
    if last_intent and _FOLLOWUP_RE.search(question or ""):
        return f"{question} (conversation context: the previous intent was {last_intent})"
    return question


def _automatic_title(turns: list[dict[str, str]]) -> str:
    """Build a compact topic title from the first few user turns without another LLM call."""
    questions = [str(turn.get("user") or "").strip() for turn in turns[:3]]
    normalized = " ".join(questions).lower()
    graduation = any(term in normalized for term in ("mezun", "kalan kredi", "degree audit"))
    next_courses = any(
        term in normalized
        for term in ("ne alm", "hangi ders", "sonraki dönem", "gelecek dönem", "next semester")
    )
    if graduation and next_courses:
        return "Mezuniyet Durumu ve Gelecek Dönem Planı"
    if graduation:
        return "Ayrıntılı Mezuniyet Durumu"
    if any(term in normalized for term in ("aldım", "aldim", "tamamladım", "completed")):
        codes = list(dict.fromkeys(extract_course_code(q) for q in questions))
        codes = [code for code in codes if code]
        suffix = f": {', '.join(codes[:3])}" if codes else ""
        return f"Ders Geçmişi Güncellemesi{suffix}"[:TITLE_MAX]
    combined = " · ".join(question for question in questions if question)
    combined = _TITLE_FILLER_RE.sub(" ", combined)
    combined = re.sub(r"\([^)]*conversation context:[^)]*\)", "", combined, flags=re.IGNORECASE)
    combined = re.sub(r"\s+", " ", combined).strip(" .,:;!?-")
    if not combined:
        return "New chat"
    if len(combined) <= TITLE_MAX:
        return combined[0].upper() + combined[1:]
    shortened = combined[: TITLE_MAX + 1].rsplit(" ", 1)[0].rstrip(" .,:;-")
    return shortened or combined[:TITLE_MAX]


async def get_working_context(
    session_id: str | None, *, username: str | None
) -> dict[str, Any]:
    if not session_id:
        return {}
    if not username:
        raise RuntimeError("conversation owner is required")
    try:
        doc = await conversations.find_one({"sessionId": session_id, "username": username})
    except Exception:
        logger.exception("conversation memory read failed")
        raise RuntimeError("conversation memory unavailable") from None
    return (doc or {}).get("working_context", {}) if doc else {}


async def recent_turns(
    session_id: str | None, *, username: str | None, limit: int = 3
) -> list[dict[str, str]]:
    if not session_id:
        return []
    if not username:
        raise RuntimeError("conversation owner is required")
    try:
        doc = await conversations.find_one({"sessionId": session_id, "username": username})
    except Exception:
        logger.exception("conversation memory read failed")
        raise RuntimeError("conversation memory unavailable") from None
    turns = (doc or {}).get("recent_turns", []) if doc else []
    return turns[-limit:]


async def append_turn(
    session_id: str | None,
    *,
    username: str | None,
    question: str,
    answer: str,
    intent: str,
    course_id: str | None = None,
    term_code: str | None = None,
    sources: list[str] | None = None,
    working_context_updates: dict[str, Any] | None = None,
) -> None:
    if not session_id:
        return
    if not username:
        raise RuntimeError("conversation owner is required")
    turn = {"user": question[:1000], "assistant": (answer or "")[:1500]}
    now = datetime.now(timezone.utc)
    message_pair = [
        {"role": "user", "content": question, "at": now},
        {"role": "assistant", "content": answer or "", "sources": sources or [], "at": now},
    ]
    title = (question or "").strip().replace("\n", " ")[:TITLE_MAX] or "New chat"
    try:
        context_updates: dict[str, Any] = {
            "sessionId": session_id,
            "username": username,
            "working_context.last_intent": intent,
            "updatedAt": now,
        }
        if course_id:
            context_updates["working_context.last_course_id"] = course_id
        if term_code:
            context_updates["working_context.last_term_code"] = term_code
        for key, value in (working_context_updates or {}).items():
            if value is not None:
                context_updates[f"working_context.{key}"] = value
        await conversations.update_one(
            # The unique sessionId index plus the owner-qualified upsert makes creation atomic:
            # an absent ID is inserted, the same owner can update it, and an existing foreign or
            # legacy-unowned ID fails with a duplicate-key error instead of being claimed.
            {"sessionId": session_id, "username": username},
            {
                "$set": context_updates,
                "$setOnInsert": {
                    "title": title,
                    "titleEdited": False,
                    "pinned": False,
                    "createdAt": now,
                },
                "$push": {
                    "recent_turns": {"$each": [turn], "$slice": -MAX_TURNS},
                    "messages": {
                        "$each": message_pair,
                        "$slice": -MAX_TRANSCRIPT_MESSAGES,
                    },
                },
            },
            upsert=True,
        )
        doc = await conversations.find_one(
            {"sessionId": session_id, "username": username},
            {"recent_turns": 1, "titleEdited": 1},
        )
        turns = (doc or {}).get("recent_turns") or []
        if len(turns) >= 2 and not (doc or {}).get("titleEdited"):
            await conversations.update_one(
                {
                    "sessionId": session_id,
                    "username": username,
                    "titleEdited": {"$ne": True},
                },
                {"$set": {"title": _automatic_title(turns)}},
            )
    except Exception:
        logger.exception("conversation memory write failed")
        raise RuntimeError("conversation memory write unavailable") from None


def _summarise(doc: dict[str, Any]) -> dict[str, Any]:
    updated = doc.get("updatedAt") or doc.get("createdAt")
    return {
        "session_id": doc.get("sessionId"),
        "title": doc.get("title") or "New chat",
        "updated_at": updated.isoformat() if hasattr(updated, "isoformat") else None,
        "message_count": len(doc.get("messages") or []),
        "pinned": bool(doc.get("pinned")),
    }


async def list_conversations(username: str | None, limit: int = 50) -> list[dict[str, Any]]:
    """Newest-first list of a user's chats, for the sidebar. Best-effort like the rest."""
    if not username:
        return []
    try:
        cursor = (
            conversations.find({"username": username, "messages": {"$exists": True, "$ne": []}})
            .sort([("pinned", -1), ("updatedAt", -1)])
            .limit(max(1, min(limit, 100)))
        )
        return [_summarise(doc) async for doc in cursor]
    except Exception:
        logger.exception("conversation list failed")
        return []


async def get_conversation(
    session_id: str, *, username: str | None = None
) -> dict[str, Any] | None:
    """Full transcript for one chat."""
    try:
        query = {"sessionId": session_id}
        if username is not None:
            query["username"] = username
        doc = await conversations.find_one(query)
    except Exception:
        logger.exception("conversation read failed")
        return None
    if not doc:
        return None
    messages = []
    for m in doc.get("messages") or []:
        at = m.get("at")
        messages.append({
            "role": m.get("role"),
            "content": m.get("content") or "",
            "sources": m.get("sources") or [],
            "at": at.isoformat() if hasattr(at, "isoformat") else None,
        })
    return {**_summarise(doc), "messages": messages}


async def conversation_owner(session_id: str | None) -> str | None:
    """Return owner, ``None`` when absent, or ``""`` for a legacy unowned record.

    The distinction prevents an existing legacy conversation from being mistaken for a new
    session and claimed by the first authenticated caller.
    """
    if not session_id:
        return None
    try:
        doc = await conversations.find_one({"sessionId": session_id}, {"username": 1})
    except Exception:
        logger.exception("conversation ownership check failed")
        # Authorization callers must distinguish "not found" from "ownership could not be
        # checked". Propagating this sanitized error makes every route fail closed.
        raise RuntimeError("conversation ownership unavailable") from None
    if doc is None:
        return None
    return str(doc.get("username") or "")


async def delete_conversation(session_id: str, *, username: str | None = None) -> bool:
    try:
        query = {"sessionId": session_id}
        if username is not None:
            query["username"] = username
        result = await conversations.delete_one(query)
        return result.deleted_count > 0
    except Exception:
        logger.exception("conversation delete failed")
        return False


async def rename_conversation(
    session_id: str, title: str, *, username: str | None = None
) -> dict[str, Any] | None:
    clean_title = re.sub(r"\s+", " ", str(title or "")).strip()[:TITLE_MAX]
    if not clean_title:
        return None
    try:
        query = {"sessionId": session_id}
        if username is not None:
            query["username"] = username
        result = await conversations.update_one(
            query,
            {"$set": {"title": clean_title, "titleEdited": True}},
        )
        if result.matched_count == 0:
            return None
        doc = await conversations.find_one(query)
        return _summarise(doc or {})
    except Exception:
        logger.exception("conversation rename failed")
        return None


async def set_conversation_pinned(
    session_id: str, pinned: bool, *, username: str | None = None
) -> dict[str, Any] | None:
    try:
        query = {"sessionId": session_id}
        if username is not None:
            query["username"] = username
        result = await conversations.update_one(
            query,
            {"$set": {"pinned": bool(pinned)}},
        )
        if result.matched_count == 0:
            return None
        doc = await conversations.find_one(query)
        return _summarise(doc or {})
    except Exception:
        logger.exception("conversation pin update failed")
        return None
