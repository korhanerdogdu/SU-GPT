from __future__ import annotations

import asyncio
import copy
import sys
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pytest
from pymongo.errors import DuplicateKeyError


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.course_review_store import (  # noqa: E402
    AUTHOR_DIGEST_NAMESPACE,
    CourseReviewIndexError,
    CourseReviewPersistencePolicy,
    CourseReviewStorageError,
    CourseReviewStore,
    ModerationConflictError,
)
from modules.course_reviews import DuplicateReviewError, ReviewValidationError  # noqa: E402


SECRET = b"course-review-store-test-secret-32-bytes-minimum"
FIXED_NOW = datetime(2026, 8, 5, 12, 30, tzinfo=timezone.utc)


def _payload(score: int = 4, **overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "course_code": "CS 455",
        "difficulty": score,
        "workload": score,
        "learning_value": score,
        "organization": score,
        "overall_satisfaction": score,
        "consent": True,
    }
    payload.update(overrides)
    return payload


@dataclass
class _WriteResult:
    deleted_count: int = 0


class _Cursor:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    async def to_list(self, *, length: int) -> list[dict[str, Any]]:
        return copy.deepcopy(self._rows[:length])


class FakeCourseReviewCollection:
    """Small stateful adapter that exercises the Motor-facing contract without Mongo."""

    def __init__(self) -> None:
        self.documents: list[dict[str, Any]] = []
        self.index_calls: list[tuple[list[tuple[str, int]], dict[str, Any]]] = []
        self.fail_indexes = False
        self.fail_inserts = False
        self._lock = threading.Lock()

    async def create_index(self, keys: list[tuple[str, int]], **kwargs: Any) -> str:
        if self.fail_indexes:
            raise RuntimeError("backend address and credentials must not escape")
        self.index_calls.append((copy.deepcopy(keys), dict(kwargs)))
        return str(kwargs["name"])

    async def insert_one(self, document: Mapping[str, Any]) -> object:
        if self.fail_inserts:
            raise RuntimeError("mongodb://private-host.invalid/?secret=do-not-leak")
        # Yield once to make concurrent callers race at the database boundary.  The lock models
        # Mongo's unique-index serialization, not an application preflight check.
        await asyncio.sleep(0)
        with self._lock:
            for current in self.documents:
                duplicate_author_course = (
                    current["courseCode"] == document["courseCode"]
                    and current["authorDigest"] == document["authorDigest"]
                )
                if duplicate_author_course or current["reviewId"] == document["reviewId"]:
                    raise DuplicateKeyError("unique index violation")
            self.documents.append(copy.deepcopy(dict(document)))
        return object()

    async def find_one_and_update(
        self,
        query: Mapping[str, Any],
        update: Mapping[str, Any],
        **kwargs: Any,
    ) -> Mapping[str, Any] | None:
        with self._lock:
            for current in self.documents:
                if all(current.get(key) == value for key, value in query.items()):
                    current.update(copy.deepcopy(update.get("$set", {})))
                    for key, value in update.get("$push", {}).items():
                        current.setdefault(key, []).append(copy.deepcopy(value))
                    result = copy.deepcopy(current)
                    for key, include in kwargs.get("projection", {}).items():
                        if not include:
                            result.pop(key, None)
                    return result
        return None

    async def delete_one(self, query: Mapping[str, Any]) -> _WriteResult:
        with self._lock:
            for index, current in enumerate(self.documents):
                if all(current.get(key) == value for key, value in query.items()):
                    self.documents.pop(index)
                    return _WriteResult(deleted_count=1)
        return _WriteResult()

    def aggregate(self, pipeline: list[Mapping[str, Any]]) -> _Cursor:
        match = pipeline[0]["$match"]
        threshold = pipeline[2]["$match"]["reviewCount"]["$gte"]
        eligible = [
            document
            for document in self.documents
            if document.get("courseCode") == match["courseCode"]
            and document.get("moderationState") == match["moderationState"]
        ]
        if len(eligible) < threshold:
            return _Cursor([])
        averages = {
            name: round(sum(document["ratings"][name] for document in eligible) / len(eligible), 2)
            for name in ("difficulty", "workload", "learning_value", "organization", "overall_satisfaction")
        }
        distributions = {
            name: {
                str(score): sum(1 for document in eligible if document["ratings"][name] == score)
                for score in range(1, 6)
            }
            for name in ("difficulty", "workload", "learning_value", "organization", "overall_satisfaction")
        }
        return _Cursor([{
            "reviewCount": len(eligible),
            "averages": averages,
            "distributions": distributions,
        }])


def _store(collection: FakeCourseReviewCollection, **kwargs: Any) -> CourseReviewStore:
    return CourseReviewStore(
        collection=collection,
        hmac_secret=SECRET,
        clock=lambda: FIXED_NOW,
        **kwargs,
    )


def test_required_indexes_are_named_and_fail_closed() -> None:
    async def scenario() -> None:
        collection = FakeCourseReviewCollection()
        names = await _store(collection).ensure_indexes()
        assert names == (
            "course_review_review_id_unique",
            "course_review_author_per_course_unique",
            "course_review_approved_aggregate",
        )
        assert collection.index_calls == [
            ([('reviewId', 1)], {"unique": True, "name": "course_review_review_id_unique"}),
            (
                [('courseCode', 1), ('authorDigest', 1)],
                {"unique": True, "name": "course_review_author_per_course_unique"},
            ),
            (
                [('courseCode', 1), ('moderationState', 1)],
                {"name": "course_review_approved_aggregate"},
            ),
        ]

        collection.fail_indexes = True
        with pytest.raises(CourseReviewIndexError, match="could not be created or verified") as error:
            await _store(collection).ensure_indexes()
        assert "credentials" not in str(error.value)
        assert error.value.__cause__ is None

    asyncio.run(scenario())


def test_atomic_duplicate_mapping_survives_store_restart_and_concurrency() -> None:
    async def scenario() -> None:
        collection = FakeCourseReviewCollection()
        first_process = _store(collection)
        restarted_process = _store(collection)

        await first_process.submit(_payload(), author_token="Student@Example.Invalid")
        with pytest.raises(DuplicateReviewError):
            await restarted_process.submit(_payload(), author_token="student@example.invalid")

        results = await asyncio.gather(
            first_process.submit(_payload(course_code="CS 412"), author_token="racing-user"),
            restarted_process.submit(_payload(course_code="CS412"), author_token="RACING-USER"),
            return_exceptions=True,
        )
        assert sum(not isinstance(result, Exception) for result in results) == 1
        assert sum(isinstance(result, DuplicateReviewError) for result in results) == 1
        assert len(collection.documents) == 2

    asyncio.run(scenario())


def test_private_digest_is_versioned_and_neither_identity_nor_comment_is_public() -> None:
    async def scenario() -> None:
        collection = FakeCourseReviewCollection()
        review = await _store(collection).submit(
            _payload(comment="The weekly assignments were useful."),
            author_token="private-student-identity",
        )

        stored = collection.documents[0]
        assert stored["authorDigest"].startswith(f"{AUTHOR_DIGEST_NAMESPACE}$")
        assert "private-student-identity" not in repr(stored)
        assert stored["consentVersion"] == "course-review-v1"
        assert stored["consentedAt"] == FIXED_NOW.isoformat()
        assert review.moderation_state == "pending"

        public = review.to_public_mapping()
        assert not ({"author", "authorDigest", "digestNamespace", "sanitizedComment", "comment"} & set(public))
        assert "weekly assignments" not in repr(public)
        history = stored["moderationHistory"][0]
        assert set(history) == {
            "event", "reviewId", "courseCode", "fromState", "toState", "reasonCodes", "timestamp"
        }
        assert "authorDigest" not in history and "sanitizedComment" not in history

    asyncio.run(scenario())


def test_persistent_v1_rejects_subgroups_and_stale_consent_policy() -> None:
    async def scenario() -> None:
        collection = FakeCourseReviewCollection()
        store = _store(collection)
        with pytest.raises(ReviewValidationError, match="Subgroups are not supported"):
            await store.submit(_payload(subgroup="first_year"), author_token="one")
        with pytest.raises(ReviewValidationError, match="current course-review consent"):
            await store.submit(_payload(consent_version="course-review-v0"), author_token="two")
        assert collection.documents == []

    asyncio.run(scenario())


def test_ratings_only_auto_approve_but_comments_require_cas_moderation() -> None:
    async def scenario() -> None:
        collection = FakeCourseReviewCollection()
        store = _store(collection, id_factory=iter(("rating", "comment")).__next__)
        rating = await store.submit(_payload(), author_token="rating-author")
        pending = await store.submit(_payload(comment="Useful assignment sequence"), author_token="comment-author")
        assert rating.moderation_state == "approved"
        assert pending.moderation_state == "pending"

        approved = await store.moderate(
            pending.review_id,
            state="approved",
            reason_codes=("human_policy_check_passed",),
        )
        assert approved.moderation_state == "approved"
        assert "authorDigest" not in approved.__dict__
        with pytest.raises(ModerationConflictError):
            await store.moderate(
                pending.review_id,
                state="rejected",
                reason_codes=("not_course_focused",),
            )

        stored = next(document for document in collection.documents if document["reviewId"] == pending.review_id)
        assert [event["event"] for event in stored["moderationHistory"]] == ["submitted", "moderated"]
        assert all("authorDigest" not in event and "sanitizedComment" not in event for event in stored["moderationHistory"])

    asyncio.run(scenario())


def test_privacy_rejected_comment_cannot_be_approved() -> None:
    async def scenario() -> None:
        collection = FakeCourseReviewCollection()
        store = _store(collection, blocked_names={"Ada Example"})
        rejected = await store.submit(
            _payload(comment="Prof Ada Example; contact person@example.com"),
            author_token="author",
        )
        assert rejected.moderation_state == "rejected"
        assert "Ada Example" not in rejected.sanitized_comment
        assert "person@example.com" not in rejected.sanitized_comment
        with pytest.raises(ModerationConflictError):
            await store.moderate(
                rejected.review_id,
                state="approved",
                reason_codes=("human_policy_check_passed",),
            )

    asyncio.run(scenario())


def test_hard_delete_removes_digest_and_resuppresses_aggregate() -> None:
    async def scenario() -> None:
        collection = FakeCourseReviewCollection()
        store = _store(collection)
        for index in range(10):
            await store.submit(_payload(score=5), author_token=f"author-{index}")

        available = await store.aggregate("CS455")
        assert available.available and available.review_count == 10
        assert available.distributions["difficulty"] == {
            "1": 0, "2": 0, "3": 0, "4": 0, "5": 10,
        }
        assert await store.hard_delete_mine("CS 455", author_token="AUTHOR-0") is True
        assert len(collection.documents) == 9
        assert all("author-0" not in repr(document) for document in collection.documents)

        suppressed = await store.aggregate("CS 455")
        assert not suppressed.available
        assert suppressed.review_count is None
        assert suppressed.averages == {}
        assert await store.hard_delete_mine("CS455", author_token="author-0") is False

    asyncio.run(scenario())


def test_aggregate_is_approved_only_and_subgroups_are_never_accepted() -> None:
    async def scenario() -> None:
        collection = FakeCourseReviewCollection()
        store = _store(collection)
        for index in range(9):
            await store.submit(_payload(), author_token=f"approved-{index}")
        await store.submit(_payload(comment="Pending human review"), author_token="pending")
        result = await store.aggregate("CS455")
        assert not result.available and result.review_count is None
        with pytest.raises(ReviewValidationError, match="Subgroup aggregates"):
            await store.aggregate("CS455", subgroup="first_year")

    asyncio.run(scenario())


def test_database_failures_are_wrapped_without_backend_details() -> None:
    async def scenario() -> None:
        collection = FakeCourseReviewCollection()
        collection.fail_inserts = True
        with pytest.raises(CourseReviewStorageError, match="could not be stored safely") as error:
            await _store(collection).submit(_payload(), author_token="student")
        assert "mongodb" not in str(error.value).lower()
        assert "secret" not in str(error.value).lower()
        assert error.value.__cause__ is None

    asyncio.run(scenario())


def test_production_policy_rejects_weak_secrets_and_privacy_thresholds() -> None:
    collection = FakeCourseReviewCollection()
    with pytest.raises(ValueError, match="at least 32 bytes"):
        CourseReviewStore(collection=collection, hmac_secret=b"too-short")
    with pytest.raises(ValueError, match="cannot be lower than 10"):
        _store(collection, policy=CourseReviewPersistencePolicy(minimum_reviews=9))
