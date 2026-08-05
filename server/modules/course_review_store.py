from __future__ import annotations

"""Async persistence boundary for privacy-preserving, course-only reviews.

The store accepts an injected Motor-compatible collection.  It never imports the application's
Mongo module and therefore cannot accidentally bind to the legacy ``instructorReviews`` data.
Duplicate prevention is a single-document insert guarded by a Mongo unique index; no check-then-
insert path exists.
"""

import inspect
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Callable, Iterable, Literal, Mapping, Protocol

from pymongo import ASCENDING, ReturnDocument
from pymongo.errors import DuplicateKeyError

from modules.course_reviews import (
    AUTHOR_DIGEST_NAMESPACE,
    RATING_FIELDS,
    CourseReviewAggregate,
    DuplicateReviewError,
    ModerationState,
    ReviewSubmission,
    ReviewTextPolicy,
    ReviewValidationError,
    StoredCourseReview,
    derive_author_digest,
    normalize_course_code,
)


DEFAULT_CONSENT_POLICY_VERSION = "course-review-v1"
_VERSION_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,79}$")

# Moderators select codes rather than writing notes that could become a second PII channel.
MODERATOR_REASON_CODES = frozenset(
    {
        "human_policy_check_passed",
        "contains_personal_information",
        "instructor_targeted_content",
        "not_course_focused",
        "abusive_content",
        "spam_or_manipulation",
    }
)

PUBLIC_REVIEW_PROJECTION: Mapping[str, int] = MappingProxyType(
    {
        "_id": 0,
        "authorDigest": 0,
        "digestNamespace": 0,
        "moderationHistory": 0,
    }
)


class AsyncCourseReviewCollection(Protocol):
    """Small Motor-compatible surface required by :class:`CourseReviewStore`."""

    async def create_index(self, keys: list[tuple[str, int]], **kwargs: Any) -> str: ...

    async def insert_one(self, document: Mapping[str, Any]) -> Any: ...

    async def find_one_and_update(
        self,
        query: Mapping[str, Any],
        update: Mapping[str, Any],
        **kwargs: Any,
    ) -> Mapping[str, Any] | None: ...

    async def delete_one(self, query: Mapping[str, Any]) -> Any: ...

    def aggregate(self, pipeline: list[Mapping[str, Any]]) -> Any: ...


class CourseReviewStorageError(RuntimeError):
    """Safe failure raised when persistence cannot prove a successful operation."""


class CourseReviewIndexError(CourseReviewStorageError):
    """Required privacy/integrity indexes could not be created or verified."""


class ModerationConflictError(ReviewValidationError):
    """A moderation write lost its compare-and-set race or targeted a terminal review."""


@dataclass(frozen=True)
class CourseReviewPersistencePolicy:
    """Configuration contract that must validate before the feature can be enabled."""

    consent_policy_version: str = DEFAULT_CONSENT_POLICY_VERSION
    digest_namespace: str = AUTHOR_DIGEST_NAMESPACE
    minimum_reviews: int = 10

    def validate(self) -> None:
        consent_version = str(self.consent_policy_version)
        digest_namespace = str(self.digest_namespace)
        if consent_version != consent_version.strip() or not _VERSION_RE.fullmatch(consent_version):
            raise ValueError("A stable lowercase consent policy version is required.")
        if digest_namespace != digest_namespace.strip() or not _VERSION_RE.fullmatch(digest_namespace):
            raise ValueError("A stable lowercase digest namespace is required.")
        if isinstance(self.minimum_reviews, bool) or not isinstance(self.minimum_reviews, int):
            raise ValueError("minimum_reviews must be an integer.")
        if self.minimum_reviews < 10:
            raise ValueError("minimum_reviews cannot be lower than 10.")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp(clock: Callable[[], datetime]) -> str:
    value = clock()
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def _clean_review_id(value: object) -> str:
    review_id = str(value or "").strip()
    if not review_id or len(review_id) > 100 or not re.fullmatch(r"crv_[A-Za-z0-9_-]+", review_id):
        raise ReviewValidationError("A valid course-review ID is required.")
    return review_id


def _moderator_reasons(reason_codes: Iterable[str]) -> tuple[str, ...]:
    values = tuple(dict.fromkeys(str(reason).strip() for reason in reason_codes if str(reason).strip()))
    if not values:
        raise ReviewValidationError("A structured moderation reason is required.")
    invalid = sorted(set(values) - MODERATOR_REASON_CODES)
    if invalid:
        raise ReviewValidationError("Unsupported moderation reason code.")
    return values


def _to_domain_review(document: Mapping[str, Any]) -> StoredCourseReview:
    """Drop all private persistence fields at the storage boundary."""

    review_id = _clean_review_id(document["reviewId"])
    course_code = normalize_course_code(document["courseCode"])
    state_value = str(document["moderationState"])
    if state_value not in {"pending", "approved", "rejected"}:
        raise ValueError("Invalid stored moderation state.")
    state: ModerationState = state_value  # type: ignore[assignment]
    ratings: dict[str, int] = {}
    for name in RATING_FIELDS:
        value = document["ratings"][name]
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5:
            raise ValueError("Invalid stored course-review rating.")
        ratings[name] = value
    return StoredCourseReview(
        review_id=review_id,
        course_code=course_code,
        ratings=MappingProxyType(ratings),
        sanitized_comment=str(document.get("sanitizedComment") or ""),
        moderation_state=state,
        moderation_reasons=tuple(str(value) for value in document.get("moderationReasons", ())),
        consent_version=str(document["consentVersion"]),
        subgroup=None,
        created_at=str(document["createdAt"]),
        updated_at=str(document["updatedAt"]),
        consented_at=str(document["consentedAt"]),
    )


class CourseReviewStore:
    """Mongo-compatible persistence with privacy-safe projections and atomic writes."""

    def __init__(
        self,
        *,
        collection: AsyncCourseReviewCollection,
        hmac_secret: bytes,
        policy: CourseReviewPersistencePolicy | None = None,
        blocked_names: Iterable[str] = (),
        clock: Callable[[], datetime] = _utc_now,
        id_factory: Callable[[], str] = lambda: uuid.uuid4().hex,
    ) -> None:
        if not isinstance(hmac_secret, bytes) or len(hmac_secret) < 32:
            raise ValueError("Persistent course reviews require an HMAC secret of at least 32 bytes.")
        selected_policy = policy or CourseReviewPersistencePolicy()
        selected_policy.validate()
        if collection is None:
            raise ValueError("A dedicated course-review collection is required.")
        self._collection = collection
        self._secret = hmac_secret
        self.policy = selected_policy
        self._text_policy = ReviewTextPolicy(blocked_names)
        self._clock = clock
        self._id_factory = id_factory

    async def ensure_indexes(self) -> tuple[str, str, str]:
        """Create/verify all indexes required before enabling production traffic.

        A caller must treat :class:`CourseReviewIndexError` as a startup-blocking failure whenever
        the feature flag is enabled.
        """

        try:
            review_id = await self._collection.create_index(
                [("reviewId", ASCENDING)],
                unique=True,
                name="course_review_review_id_unique",
            )
            duplicate_guard = await self._collection.create_index(
                [("courseCode", ASCENDING), ("authorDigest", ASCENDING)],
                unique=True,
                name="course_review_author_per_course_unique",
            )
            aggregate_lookup = await self._collection.create_index(
                [("courseCode", ASCENDING), ("moderationState", ASCENDING)],
                name="course_review_approved_aggregate",
            )
        except Exception:
            raise CourseReviewIndexError("Course-review indexes could not be created or verified.") from None
        return str(review_id), str(duplicate_guard), str(aggregate_lookup)

    def _author_digest(self, course_code: str, author_token: str) -> str:
        return derive_author_digest(
            hmac_secret=self._secret,
            course_code=course_code,
            author_token=author_token,
            namespace=self.policy.digest_namespace,
        )

    async def submit(
        self,
        payload: Mapping[str, object],
        *,
        author_token: str,
    ) -> StoredCourseReview:
        submission = ReviewSubmission.from_mapping(payload)
        if submission.subgroup is not None:
            raise ReviewValidationError("Subgroups are not supported by the persistent course-review policy.")
        if submission.consent_version != self.policy.consent_policy_version:
            raise ReviewValidationError("The current course-review consent policy must be accepted.")

        moderation = self._text_policy.inspect(submission.comment)
        now = _timestamp(self._clock)
        review_id = f"crv_{self._id_factory()}"
        _clean_review_id(review_id)
        event = {
            "event": "submitted",
            "reviewId": review_id,
            "courseCode": submission.course_code,
            "fromState": None,
            "toState": moderation.state,
            "reasonCodes": list(moderation.reason_codes),
            "timestamp": now,
        }
        document: dict[str, Any] = {
            "schemaVersion": 1,
            "reviewId": review_id,
            "courseCode": submission.course_code,
            "authorDigest": self._author_digest(submission.course_code, author_token),
            "digestNamespace": self.policy.digest_namespace,
            "ratings": submission.ratings,
            "sanitizedComment": moderation.sanitized_text,
            "moderationState": moderation.state,
            "moderationReasons": list(moderation.reason_codes),
            "consentVersion": self.policy.consent_policy_version,
            "consentedAt": now,
            "createdAt": now,
            "updatedAt": now,
            "moderationHistory": [event],
        }
        try:
            await self._collection.insert_one(document)
        except DuplicateKeyError:
            # The write itself is the duplicate check.  Never add a racy preflight query.
            raise DuplicateReviewError("Duplicate course review.") from None
        except Exception:
            raise CourseReviewStorageError("Course review could not be stored safely.") from None
        return _to_domain_review(document)

    async def moderate(
        self,
        review_id: str,
        *,
        expected_state: Literal["pending"] = "pending",
        state: Literal["approved", "rejected"],
        reason_codes: Iterable[str],
    ) -> StoredCourseReview:
        clean_id = _clean_review_id(review_id)
        if expected_state != "pending":
            raise ReviewValidationError("Persistent moderation may transition only a pending review.")
        if state not in {"approved", "rejected"}:
            raise ReviewValidationError("Moderation state must be approved or rejected.")
        reasons = _moderator_reasons(reason_codes)
        if state == "approved" and reasons != ("human_policy_check_passed",):
            raise ReviewValidationError("Approval requires the human policy-check reason only.")

        now = _timestamp(self._clock)
        event = {
            "event": "moderated",
            "reviewId": clean_id,
            "fromState": "pending",
            "toState": state,
            "reasonCodes": list(reasons),
            "timestamp": now,
        }
        try:
            document = await self._collection.find_one_and_update(
                {"reviewId": clean_id, "moderationState": "pending"},
                {
                    "$set": {
                        "moderationState": state,
                        "moderationReasons": list(reasons),
                        "updatedAt": now,
                    },
                    "$push": {"moderationHistory": event},
                },
                projection=dict(PUBLIC_REVIEW_PROJECTION),
                return_document=ReturnDocument.AFTER,
            )
        except Exception:
            raise CourseReviewStorageError("Course-review moderation could not be stored safely.") from None
        if document is None:
            raise ModerationConflictError("Review is no longer pending or does not exist.")
        try:
            return _to_domain_review(document)
        except Exception:
            raise CourseReviewStorageError("Stored course-review data failed validation.") from None

    async def hard_delete_mine(self, course_code: str, *, author_token: str) -> bool:
        """Delete the entire document, including its private digest and embedded history."""

        course = normalize_course_code(course_code)
        digest = self._author_digest(course, author_token)
        try:
            result = await self._collection.delete_one({"courseCode": course, "authorDigest": digest})
        except Exception:
            raise CourseReviewStorageError("Course review could not be deleted safely.") from None
        return int(getattr(result, "deleted_count", 0)) == 1

    async def aggregate(self, course_code: str, *, subgroup: str | None = None) -> CourseReviewAggregate:
        if subgroup is not None:
            raise ReviewValidationError("Subgroup aggregates are not supported.")
        course = normalize_course_code(course_code)
        average_fields = {name: {"$avg": f"$ratings.{name}"} for name in RATING_FIELDS}
        distribution_fields = {
            f"{name}_{score}": {
                "$sum": {"$cond": [{"$eq": [f"$ratings.{name}", score]}, 1, 0]}
            }
            for name in RATING_FIELDS
            for score in range(1, 6)
        }
        rounded_fields = {name: {"$round": [f"${name}", 2]} for name in RATING_FIELDS}
        distributions = {
            name: {str(score): f"${name}_{score}" for score in range(1, 6)}
            for name in RATING_FIELDS
        }
        pipeline: list[Mapping[str, Any]] = [
            {"$match": {"courseCode": course, "moderationState": "approved"}},
            {
                "$group": {
                    "_id": None,
                    "reviewCount": {"$sum": 1},
                    **average_fields,
                    **distribution_fields,
                }
            },
            {"$match": {"reviewCount": {"$gte": self.policy.minimum_reviews}}},
            {
                "$project": {
                    "_id": 0,
                    "reviewCount": 1,
                    "averages": rounded_fields,
                    "distributions": distributions,
                }
            },
            {"$limit": 1},
        ]
        try:
            cursor = self._collection.aggregate(pipeline)
            if inspect.isawaitable(cursor):
                cursor = await cursor
            rows = await cursor.to_list(length=1)
        except Exception:
            raise CourseReviewStorageError("Course-review aggregate could not be computed safely.") from None
        if not rows:
            return CourseReviewAggregate(
                course_code=course,
                available=False,
                minimum_required=self.policy.minimum_reviews,
                suppression_reason="minimum_reviews_not_met",
            )

        row = rows[0]
        try:
            count = int(row["reviewCount"])
            if count < self.policy.minimum_reviews:
                # Defense in depth for a broken/non-Mongo adapter; never disclose its small count.
                raise CourseReviewStorageError("Course-review aggregate suppression could not be verified.")
            averages = MappingProxyType({name: round(float(row["averages"][name]), 2) for name in RATING_FIELDS})
            distribution_values = MappingProxyType(
                {
                    name: MappingProxyType(
                        {str(score): int(row["distributions"][name][str(score)]) for score in range(1, 6)}
                    )
                    for name in RATING_FIELDS
                }
            )
        except CourseReviewStorageError:
            raise
        except Exception:
            raise CourseReviewStorageError("Stored course-review aggregate failed validation.") from None
        return CourseReviewAggregate(
            course_code=course,
            available=True,
            minimum_required=self.policy.minimum_reviews,
            review_count=count,
            averages=averages,
            distributions=distribution_values,
        )
