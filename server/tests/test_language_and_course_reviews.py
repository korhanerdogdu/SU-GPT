from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.course_reviews import (  # noqa: E402
    CourseReviewService,
    DuplicateReviewError,
    FeatureDisabledError,
    ReviewSubmission,
    ReviewValidationError,
)
from modules.language import analyze_language, classify_language, detect_language  # noqa: E402
from modules.localization import message  # noqa: E402


SECRET = b"test-only-course-review-hmac-key"


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


def _service(**kwargs: object) -> CourseReviewService:
    serial = iter(range(1000))
    return CourseReviewService(
        hmac_secret=SECRET,
        enabled=True,
        clock=lambda: datetime(2026, 8, 4, tzinfo=timezone.utc),
        id_factory=lambda: str(next(serial)),
        **kwargs,
    )


def test_language_detection_is_course_code_and_acronym_resistant():
    assert detect_language("CS 412 NLP ECTS") == "en"
    assert detect_language("CS 412 dersi için program öner") == "tr"
    assert detect_language("Can you recommend CS 412 for next term?") == "en"
    assert detect_language("DERS ÖNER CS 412 NLP") == "tr"
    assert detect_language("WHAT COURSE SHOULD I TAKE CS 412?") == "en"


def test_language_analysis_exposes_mixed_without_breaking_response_contract():
    result = analyze_language("Can you recommend bir ders for my program?")
    assert result.classification == "mixed"
    assert result.response_language in {"tr", "en"}
    assert classify_language("What is bu ders için prerequisite?") == "mixed"


def test_language_defaults_safely_for_empty_or_code_only_input():
    assert detect_language("") == "en"
    assert detect_language("```python\nprint('x')\n``` CS 201 API") == "en"


def test_localization_resources_are_bilingual_and_formatted():
    assert "10" in message("course_reviews.aggregate_suppressed", "en", minimum=10)
    assert "10" in message("course_reviews.aggregate_suppressed", "tr", minimum=10)
    assert message("course_reviews.disabled", "tr") != message("course_reviews.disabled", "en")
    assert message("confidence.insufficient_evidence", "tr") != message(
        "confidence.insufficient_evidence", "en"
    )
    assert message("recommendation.ask_interest", "tr") != message(
        "recommendation.ask_interest", "en"
    )


def test_course_review_feature_is_disabled_by_default():
    service = CourseReviewService(hmac_secret=SECRET)
    with pytest.raises(FeatureDisabledError):
        service.submit(_payload(), author_token="opaque-session-token")


@pytest.mark.parametrize("bad", [0, 6, 3.5, "4", True, None])
def test_all_scores_must_be_integer_one_to_five(bad: object):
    with pytest.raises(ReviewValidationError):
        ReviewSubmission.from_mapping(_payload(difficulty=bad))


def test_explicit_consent_is_required():
    with pytest.raises(ReviewValidationError):
        ReviewSubmission.from_mapping(_payload(consent=False))
    with pytest.raises(ReviewValidationError):
        ReviewSubmission.from_mapping({key: value for key, value in _payload().items() if key != "consent"})


def test_instructor_rating_and_identity_fields_are_rejected():
    for forbidden in ("instructor_rating", "professor_rating", "instructor_name", "author_id", "email"):
        with pytest.raises(ReviewValidationError):
            ReviewSubmission.from_mapping(_payload(**{forbidden: 5}))


def test_duplicate_control_uses_transient_identity_without_storing_it():
    service = _service()
    review = service.submit(_payload(), author_token="student@example.invalid")
    with pytest.raises(DuplicateReviewError):
        service.submit(_payload(), author_token="STUDENT@example.invalid")
    assert "student" not in repr(review).lower()
    assert all("digest" not in event.__dict__ and "author" not in event.__dict__ for event in service.audit_log())
    with pytest.raises(TypeError):
        review.ratings["difficulty"] = 1


def test_same_author_can_review_different_courses_without_raw_identifier_storage():
    service = _service()
    service.submit(_payload(course_code="CS 455"), author_token="opaque-1")
    service.submit(_payload(course_code="CS 412"), author_token="opaque-1")
    assert len(service.audit_log()) == 2


def test_comments_are_pending_and_privacy_or_instructor_content_is_rejected_and_redacted():
    service = _service(blocked_names={"Ada Example"})
    pending = service.submit(_payload(comment="The weekly assignments were useful."), author_token="one")
    assert pending.moderation_state == "pending"

    rejected = service.submit(
        _payload(comment="Prof Ada Example was great; email me at user@example.com"),
        author_token="two",
    )
    assert rejected.moderation_state == "rejected"
    assert "Ada Example" not in rejected.sanitized_comment
    assert "user@example.com" not in rejected.sanitized_comment
    assert {"known_person_name", "email_address", "instructor_targeted_content"} <= set(rejected.moderation_reasons)

    other_pii = service.submit(
        _payload(comment="See https://private.example/path, @private_handle, ID 12345678"),
        author_token="three",
    )
    assert other_pii.moderation_state == "rejected"
    assert "private.example" not in other_pii.sanitized_comment
    assert "private_handle" not in other_pii.sanitized_comment
    assert "12345678" not in other_pii.sanitized_comment
    assert {"url", "social_handle", "student_identifier"} <= set(other_pii.moderation_reasons)
    with pytest.raises(ReviewValidationError):
        service.moderate(rejected.review_id, state="approved")


def test_only_human_approved_reviews_contribute_to_aggregate_and_small_counts_are_hidden():
    service = _service()
    for index in range(9):
        service.submit(_payload(score=5), author_token=f"author-{index}")
    suppressed = service.aggregate("CS455")
    assert not suppressed.available
    assert suppressed.review_count is None
    assert suppressed.minimum_required == 10

    tenth = service.submit(_payload(score=3, comment="The project structure was clear."), author_token="author-9")
    assert not service.aggregate("CS 455").available
    service.moderate(tenth.review_id, state="approved", reason_codes=("human_policy_check_passed",))
    aggregate = service.aggregate("CS 455")
    assert aggregate.available and aggregate.review_count == 10
    assert aggregate.averages["difficulty"] == 4.8


def test_subgroup_is_allowlisted_and_suppressed_below_ten_without_revealing_count():
    service = _service(allowed_subgroups={"first_year", "upper_year"})
    for index in range(10):
        subgroup = "first_year" if index < 9 else "upper_year"
        service.submit(_payload(subgroup=subgroup), author_token=f"student-{index}")
    result = service.aggregate("CS 455", subgroup="first_year")
    assert not result.available and result.review_count is None
    with pytest.raises(ReviewValidationError):
        service.submit(_payload(subgroup="tiny-private-club"), author_token="outsider")


def test_audit_decisions_contain_no_comment_subgroup_or_identity():
    service = _service(allowed_subgroups={"first_year"})
    service.submit(
        _payload(comment="Useful course structure", subgroup="first_year"),
        author_token="private-identity",
    )
    event = service.audit_log()[0]
    assert set(event.__dict__) == {
        "event", "review_id", "course_code", "moderation_state", "reason_codes", "timestamp"
    }
