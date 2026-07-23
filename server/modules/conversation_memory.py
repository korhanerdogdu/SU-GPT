from __future__ import annotations

"""
Bounded session memory (roadmap section 19).

The LLM stays mostly stateless, but we keep the last few turns + a small "working context"
(last course code / term) per session so follow-ups like "can I take it next semester?"
resolve to the course from the previous turn. This memory only improves usability — it must
NEVER override official JSON data or the deterministic audit (roadmap section 19.6). All calls
are best-effort: if Mongo is down they degrade to no-memory rather than breaking /ask.
"""

import re
from datetime import datetime, timezone
from typing import Any

from logger import logger
from modules.mongodb import db

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


def extract_course_code(text: str) -> str | None:
    m = _COURSE_RE.search((text or "").upper())
    return f"{m.group(1)} {m.group(2)}" if m else None


def resolve_reference(question: str, working_context: dict[str, Any] | None) -> str:
    """If the question refers to a prior course without naming one, inject that course code."""
    working_context = working_context or {}
    last_course = working_context.get("last_course_id")
    if last_course and _REFERENCE_RE.search(question or "") and not extract_course_code(question):
        return f"{question} (the course being referred to is {last_course})"
    return question


async def get_working_context(session_id: str | None) -> dict[str, Any]:
    if not session_id:
        return {}
    try:
        doc = await conversations.find_one({"sessionId": session_id})
    except Exception:
        logger.exception("conversation memory read failed")
        return {}
    return (doc or {}).get("working_context", {}) if doc else {}


async def recent_turns(session_id: str | None, limit: int = 3) -> list[dict[str, str]]:
    if not session_id:
        return []
    try:
        doc = await conversations.find_one({"sessionId": session_id})
    except Exception:
        return []
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
) -> None:
    if not session_id:
        return
    working_context = {"last_intent": intent}
    if course_id:
        working_context["last_course_id"] = course_id
    if term_code:
        working_context["last_term_code"] = term_code
    turn = {"user": question[:1000], "assistant": (answer or "")[:1500]}
    now = datetime.now(timezone.utc)
    message_pair = [
        {"role": "user", "content": question, "at": now},
        {"role": "assistant", "content": answer or "", "sources": sources or [], "at": now},
    ]
    title = (question or "").strip().replace("\n", " ")[:TITLE_MAX] or "New chat"
    try:
        await conversations.update_one(
            {"sessionId": session_id},
            {
                "$set": {
                    "sessionId": session_id,
                    "username": username,
                    "working_context": working_context,
                    "updatedAt": now,
                },
                # Title is the first question asked and never changes afterwards.
                "$setOnInsert": {"title": title, "createdAt": now},
                "$push": {
                    "recent_turns": {"$each": [turn], "$slice": -MAX_TURNS},
                    "messages": {"$each": message_pair},
                },
            },
            upsert=True,
        )
    except Exception:
        logger.exception("conversation memory write failed")


def _summarise(doc: dict[str, Any]) -> dict[str, Any]:
    updated = doc.get("updatedAt") or doc.get("createdAt")
    return {
        "session_id": doc.get("sessionId"),
        "title": doc.get("title") or "New chat",
        "updated_at": updated.isoformat() if hasattr(updated, "isoformat") else None,
        "message_count": len(doc.get("messages") or []),
    }


async def list_conversations(username: str | None, limit: int = 50) -> list[dict[str, Any]]:
    """Newest-first list of a user's chats, for the sidebar. Best-effort like the rest."""
    if not username:
        return []
    try:
        cursor = (
            conversations.find({"username": username, "messages": {"$exists": True, "$ne": []}})
            .sort("updatedAt", -1)
            .limit(max(1, min(limit, 100)))
        )
        return [_summarise(doc) async for doc in cursor]
    except Exception:
        logger.exception("conversation list failed")
        return []


async def get_conversation(session_id: str) -> dict[str, Any] | None:
    """Full transcript for one chat."""
    try:
        doc = await conversations.find_one({"sessionId": session_id})
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


async def delete_conversation(session_id: str) -> bool:
    try:
        result = await conversations.delete_one({"sessionId": session_id})
        return result.deleted_count > 0
    except Exception:
        logger.exception("conversation delete failed")
        return False
