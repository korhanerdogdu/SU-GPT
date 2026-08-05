from __future__ import annotations

"""Privacy-preserving, course-centered review domain service.

This module is intentionally storage- and API-agnostic.  It does not read WhatsApp exports or
the existing instructor-review collection.  The default service state is disabled, and the only
review dimensions are course-level attributes.
"""

import hashlib
import hmac
import re
import unicodedata
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Callable, Iterable, Literal, Mapping


RATING_FIELDS = (
    "difficulty",
    "workload",
    "learning_value",
    "organization",
    "overall_satisfaction",
)
ModerationState = Literal["pending", "approved", "rejected"]
AUTHOR_DIGEST_NAMESPACE = "course-review-author-v1"

_COURSE_RE = re.compile(r"^([A-Z]{2,6})\s*-?\s*(\d{3,5}[A-Z]?)$")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", re.I)
_PHONE_RE = re.compile(r"(?<!\w)\+?\d[\d\s().-]{7,}\d(?!\w)")
_STUDENT_ID_RE = re.compile(
    r"\b(?:(?:student\s*id|öğrenci\s*no|ogrenci\s*no)\s*[:#-]?\s*)?\d{7,9}\b",
    re.I,
)
_URL_RE = re.compile(r"\b(?:https?://|www\.)[^\s<>()]+", re.I)
_SOCIAL_HANDLE_RE = re.compile(r"(?<![\w@])@[A-Za-z0-9_][A-Za-z0-9_.-]{1,29}\b")
_INSTRUCTOR_REFERENCE_RE = re.compile(
    r"\b(?:prof(?:essor)?|instructor|teacher|lecturer|hoca(?:m|nın|nin|yi|yı)?|"
    r"öğretim\s+üyesi|ogretim\s+uyesi|doç(?:ent)?|doc(?:ent)?|dr)\b",
    re.I,
)
_LIKELY_FULL_NAME_RE = re.compile(
    r"(?<![.!?]\s)\b[A-ZÇĞİÖŞÜ][a-zçğıöşü]{2,}\s+[A-ZÇĞİÖŞÜ][a-zçğıöşü]{2,}\b"
)
_FORBIDDEN_KEYS = frozenset(
    {
        "instructor_rating", "professor_rating", "teacher_rating", "lecturer_rating",
        "instructor_name", "professor_name", "teacher_name", "author", "author_id",
        "email", "phone", "student_id", "username",
    }
)
_ALLOWED_KEYS = frozenset(
    {"course_code", "ratings", "comment", "consent", "consent_version", "subgroup", *RATING_FIELDS}
)


class CourseReviewError(ValueError):
    """Base class for safe, user-correctable course-review failures."""


class FeatureDisabledError(CourseReviewError):
    pass


class ReviewValidationError(CourseReviewError):
    pass


class DuplicateReviewError(CourseReviewError):
    pass


def derive_author_digest(
    *,
    hmac_secret: bytes,
    course_code: str,
    author_token: str,
    namespace: str = AUTHOR_DIGEST_NAMESPACE,
) -> str:
    """Derive a versioned, course-scoped pseudonymous duplicate key.

    ``author_token`` is normalized only in local memory.  Callers must never persist or log it.
    The namespace is included both in the authenticated message and the stored digest prefix so a
    future key/version migration cannot silently mix incompatible duplicate-key schemes.
    """

    if not isinstance(hmac_secret, bytes) or len(hmac_secret) < 16:
        raise ValueError("hmac_secret must contain at least 16 bytes; use 32 or more in production.")
    if not isinstance(author_token, str):
        raise ReviewValidationError("A non-empty transient author token is required for duplicate prevention.")
    transient = unicodedata.normalize("NFKC", author_token).strip().casefold()
    if not transient:
        raise ReviewValidationError("A non-empty transient author token is required for duplicate prevention.")
    clean_namespace = str(namespace).strip()
    if not clean_namespace or len(clean_namespace) > 80 or not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", clean_namespace):
        raise ValueError("A stable lowercase digest namespace is required.")
    course = normalize_course_code(course_code)
    message = f"{clean_namespace}\x00{course}\x00{transient}".encode("utf-8")
    digest = hmac.new(hmac_secret, message, hashlib.sha256).hexdigest()
    return f"{clean_namespace}${digest}"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def normalize_course_code(value: object) -> str:
    compact = " ".join(str(value or "").strip().upper().split())
    match = _COURSE_RE.fullmatch(compact)
    if not match:
        raise ReviewValidationError("A valid course code is required.")
    return f"{match.group(1)} {match.group(2)}"


def _score(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5:
        raise ReviewValidationError(f"{field_name} must be an integer from 1 to 5.")
    return value


def _clean_text(value: object, *, maximum: int = 1000) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = " ".join(text.replace("\x00", " ").split())
    if len(text) > maximum:
        raise ReviewValidationError(f"Comment must not exceed {maximum} characters.")
    return text


@dataclass(frozen=True)
class ReviewSubmission:
    course_code: str
    difficulty: int
    workload: int
    learning_value: int
    organization: int
    overall_satisfaction: int
    consent: bool
    comment: str = ""
    consent_version: str = "course-review-v1"
    subgroup: str | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "ReviewSubmission":
        keys = {str(key) for key in payload}
        forbidden = sorted(keys & _FORBIDDEN_KEYS)
        unknown = sorted(keys - _ALLOWED_KEYS)
        if forbidden:
            raise ReviewValidationError("Instructor/professor ratings and identity fields are not allowed.")
        if unknown:
            raise ReviewValidationError(f"Unsupported course-review fields: {', '.join(unknown)}")

        nested = payload.get("ratings")
        if nested is not None and not isinstance(nested, Mapping):
            raise ReviewValidationError("ratings must be an object.")
        ratings = dict(nested or {})
        if set(ratings) - set(RATING_FIELDS):
            raise ReviewValidationError("Only the five course-level rating dimensions are allowed.")

        values: dict[str, object] = {}
        for name in RATING_FIELDS:
            if name in ratings and name in payload:
                raise ReviewValidationError(f"Duplicate rating value: {name}")
            values[name] = ratings.get(name, payload.get(name))

        consent = payload.get("consent")
        if consent is not True:
            raise ReviewValidationError("Explicit consent is required.")

        subgroup = payload.get("subgroup")
        clean_subgroup = str(subgroup).strip().lower() if subgroup is not None else None
        return cls(
            course_code=normalize_course_code(payload.get("course_code")),
            **{name: _score(values[name], name) for name in RATING_FIELDS},
            consent=True,
            comment=_clean_text(payload.get("comment")),
            consent_version=_clean_text(payload.get("consent_version") or "course-review-v1", maximum=80),
            subgroup=clean_subgroup or None,
        )

    @property
    def ratings(self) -> dict[str, int]:
        return {name: int(getattr(self, name)) for name in RATING_FIELDS}


@dataclass(frozen=True)
class TextModerationResult:
    sanitized_text: str
    state: ModerationState
    reason_codes: tuple[str, ...] = ()


class ReviewTextPolicy:
    """Conservative filter; safe-looking comments still require human approval.

    Known names should be injected from an authorized roster.  The raw comment is never returned
    when PII, an instructor reference, or a likely full name is found.
    """

    def __init__(self, blocked_names: Iterable[str] = ()) -> None:
        names = sorted({" ".join(str(name).split()) for name in blocked_names if str(name).strip()}, key=len, reverse=True)
        self._name_patterns = tuple(re.compile(rf"(?<!\w){re.escape(name)}(?!\w)", re.I) for name in names)

    def inspect(self, comment: str) -> TextModerationResult:
        text = _clean_text(comment)
        if not text:
            return TextModerationResult("", "approved")

        reasons: list[str] = []
        sanitized = text
        for pattern, reason in (
            (_EMAIL_RE, "email_address"),
            (_STUDENT_ID_RE, "student_identifier"),
            (_PHONE_RE, "phone_number"),
            (_URL_RE, "url"),
            (_SOCIAL_HANDLE_RE, "social_handle"),
        ):
            if pattern.search(sanitized):
                reasons.append(reason)
                sanitized = pattern.sub("[redacted]", sanitized)

        for pattern in self._name_patterns:
            if pattern.search(sanitized):
                reasons.append("known_person_name")
                sanitized = pattern.sub("[redacted]", sanitized)

        if _INSTRUCTOR_REFERENCE_RE.search(sanitized):
            reasons.append("instructor_targeted_content")
            sanitized = _INSTRUCTOR_REFERENCE_RE.sub("[redacted role]", sanitized)
        if _LIKELY_FULL_NAME_RE.search(sanitized):
            reasons.append("possible_person_name")
            sanitized = _LIKELY_FULL_NAME_RE.sub("[redacted name]", sanitized)

        if reasons:
            # The sanitized text may support a moderator decision, but rejected content never
            # contributes to aggregates and the original text is never retained by this service.
            return TextModerationResult(sanitized, "rejected", tuple(dict.fromkeys(reasons)))
        return TextModerationResult(sanitized, "pending", ("human_review_required",))


@dataclass(frozen=True)
class StoredCourseReview:
    review_id: str
    course_code: str
    ratings: Mapping[str, int]
    sanitized_comment: str
    moderation_state: ModerationState
    moderation_reasons: tuple[str, ...]
    consent_version: str
    subgroup: str | None
    created_at: str
    updated_at: str
    consented_at: str | None = None

    def to_public_mapping(self) -> dict[str, object]:
        """Serialize only fields suitable for an unprivileged API response.

        Comments and moderation reasons deliberately require a separate privileged moderation
        projection.  Raw author identity and the private HMAC digest are not members of this
        domain object at all.
        """

        return {
            "reviewId": self.review_id,
            "courseCode": self.course_code,
            "ratings": dict(self.ratings),
            "moderationState": self.moderation_state,
            "consentVersion": self.consent_version,
            "consentedAt": self.consented_at,
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
        }


@dataclass(frozen=True)
class AuditDecision:
    event: str
    review_id: str
    course_code: str
    moderation_state: ModerationState
    reason_codes: tuple[str, ...]
    timestamp: str


@dataclass(frozen=True)
class CourseReviewAggregate:
    course_code: str
    available: bool
    minimum_required: int
    review_count: int | None = None
    averages: Mapping[str, float] = field(default_factory=dict)
    distributions: Mapping[str, Mapping[str, int]] = field(default_factory=dict)
    subgroup: str | None = None
    suppression_reason: str | None = None


class CourseReviewService:
    """Pure in-memory domain service; callers may add persistence behind this contract later."""

    def __init__(
        self,
        *,
        hmac_secret: bytes,
        enabled: bool = False,
        minimum_reviews: int = 10,
        minimum_subgroup_size: int = 10,
        allowed_subgroups: Iterable[str] = (),
        blocked_names: Iterable[str] = (),
        clock: Callable[[], datetime] = _utc_now,
        id_factory: Callable[[], str] = lambda: uuid.uuid4().hex,
    ) -> None:
        if not isinstance(hmac_secret, bytes) or len(hmac_secret) < 16:
            raise ValueError("hmac_secret must contain at least 16 bytes; use 32 or more in production.")
        if minimum_reviews < 10:
            raise ValueError("minimum_reviews cannot be lower than 10.")
        if minimum_subgroup_size < 10:
            raise ValueError("minimum_subgroup_size cannot be lower than 10.")
        self.enabled = bool(enabled)
        self.minimum_reviews = int(minimum_reviews)
        self.minimum_subgroup_size = int(minimum_subgroup_size)
        self._secret = hmac_secret
        self._allowed_subgroups = {str(value).strip().lower() for value in allowed_subgroups if str(value).strip()}
        self._text_policy = ReviewTextPolicy(blocked_names)
        self._clock = clock
        self._id_factory = id_factory
        self._reviews: dict[str, StoredCourseReview] = {}
        self._dedupe_digests: set[str] = set()
        self._audit: list[AuditDecision] = []

    def _require_enabled(self) -> None:
        if not self.enabled:
            raise FeatureDisabledError("Course reviews are disabled.")

    def _dedupe_digest(self, course_code: str, author_token: str) -> str:
        return derive_author_digest(
            hmac_secret=self._secret,
            course_code=course_code,
            author_token=author_token,
        )

    def _timestamp(self) -> str:
        value = self._clock()
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()

    def _audit_decision(self, review: StoredCourseReview, event: str) -> None:
        self._audit.append(
            AuditDecision(
                event=event,
                review_id=review.review_id,
                course_code=review.course_code,
                moderation_state=review.moderation_state,
                reason_codes=review.moderation_reasons,
                timestamp=review.updated_at,
            )
        )

    def submit(self, payload: Mapping[str, object], *, author_token: str) -> StoredCourseReview:
        self._require_enabled()
        submission = ReviewSubmission.from_mapping(payload)
        if submission.subgroup and submission.subgroup not in self._allowed_subgroups:
            raise ReviewValidationError("Unknown or disallowed subgroup.")

        digest = self._dedupe_digest(submission.course_code, author_token)
        if digest in self._dedupe_digests:
            raise DuplicateReviewError("Duplicate course review.")

        moderation = self._text_policy.inspect(submission.comment)
        now = self._timestamp()
        review_id = f"crv_{self._id_factory()}"
        if review_id in self._reviews:
            raise RuntimeError("Course-review ID collision.")
        review = StoredCourseReview(
            review_id=review_id,
            course_code=submission.course_code,
            ratings=MappingProxyType(submission.ratings),
            sanitized_comment=moderation.sanitized_text,
            moderation_state=moderation.state,
            moderation_reasons=moderation.reason_codes,
            consent_version=submission.consent_version,
            subgroup=submission.subgroup,
            created_at=now,
            updated_at=now,
            consented_at=now,
        )
        self._reviews[review.review_id] = review
        self._dedupe_digests.add(digest)
        self._audit_decision(review, "submitted")
        return review

    def moderate(
        self,
        review_id: str,
        *,
        state: Literal["approved", "rejected"],
        reason_codes: Iterable[str] = (),
    ) -> StoredCourseReview:
        self._require_enabled()
        if state not in {"approved", "rejected"}:
            raise ReviewValidationError("Moderation state must be approved or rejected.")
        current = self._reviews.get(review_id)
        if current is None:
            raise ReviewValidationError("Unknown review.")
        if current.moderation_state == "rejected" and state == "approved":
            raise ReviewValidationError("Privacy-rejected content cannot be approved.")
        updated = replace(
            current,
            moderation_state=state,
            moderation_reasons=tuple(dict.fromkeys(str(reason).strip() for reason in reason_codes if str(reason).strip())),
            updated_at=self._timestamp(),
        )
        self._reviews[review_id] = updated
        self._audit_decision(updated, "moderated")
        return updated

    def aggregate(self, course_code: str, *, subgroup: str | None = None) -> CourseReviewAggregate:
        self._require_enabled()
        course = normalize_course_code(course_code)
        group = str(subgroup).strip().lower() if subgroup is not None else None
        if group and group not in self._allowed_subgroups:
            raise ReviewValidationError("Unknown or disallowed subgroup.")
        eligible = [
            review
            for review in self._reviews.values()
            if review.course_code == course
            and review.moderation_state == "approved"
            and (group is None or review.subgroup == group)
        ]
        threshold = max(self.minimum_reviews, self.minimum_subgroup_size) if group else self.minimum_reviews
        if len(eligible) < threshold:
            # Do not reveal the exact small count; this prevents differencing attacks on subgroups.
            return CourseReviewAggregate(
                course_code=course,
                available=False,
                minimum_required=threshold,
                subgroup=group,
                suppression_reason="minimum_group_size_not_met" if group else "minimum_reviews_not_met",
            )
        averages = MappingProxyType(
            {
                name: round(sum(review.ratings[name] for review in eligible) / len(eligible), 2)
                for name in RATING_FIELDS
            }
        )
        distributions = MappingProxyType(
            {
                name: MappingProxyType(
                    {
                        str(score): sum(1 for review in eligible if review.ratings[name] == score)
                        for score in range(1, 6)
                    }
                )
                for name in RATING_FIELDS
            }
        )
        return CourseReviewAggregate(
            course_code=course,
            available=True,
            minimum_required=threshold,
            review_count=len(eligible),
            averages=averages,
            distributions=distributions,
            subgroup=group,
        )

    def audit_log(self) -> tuple[AuditDecision, ...]:
        """Identity-free decisions only; author tokens and HMAC digests are never exposed."""

        return tuple(self._audit)
