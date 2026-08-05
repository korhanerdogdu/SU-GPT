from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main as main_module
from modules.auth import issue_token
from modules.course_review_store import CourseReviewStorageError
from modules.rate_limit_backends import BackendUnavailableError


VALID_REVIEW = {
    "course_code": "CS 201",
    "difficulty": 3,
    "workload": 4,
    "learning_value": 5,
    "organization": 4,
    "overall_satisfaction": 5,
    "consent": True,
    "consent_version": "course-review-v1",
}


def _headers(username: str, role: str = "student") -> dict[str, str]:
    token, _ = issue_token(username, role)
    return {"Authorization": f"Bearer {token}"}


class _Review:
    moderation_state = "approved"

    @staticmethod
    def to_public_mapping():
        return {
            "reviewId": "crv_public",
            "courseCode": "CS 201",
            "ratings": {name: score for name, score in (
                ("difficulty", 3), ("workload", 4), ("learning_value", 5),
                ("organization", 4), ("overall_satisfaction", 5),
            )},
            "moderationState": "approved",
        }


class _Store:
    def __init__(self):
        self.author_tokens: list[str] = []

    async def submit(self, payload, *, author_token):
        assert payload == VALID_REVIEW
        self.author_tokens.append(author_token)
        return _Review()

    async def hard_delete_mine(self, course_code, *, author_token):
        self.author_tokens.append(author_token)
        return course_code == "CS 201" and author_token == "student-a"

    async def aggregate(self, course_code):
        assert course_code == "CS 201"
        return SimpleNamespace(available=False, minimum_required=10)

    async def moderate(self, review_id, *, state, reason_codes):
        assert review_id == "crv_public"
        assert state == "approved" and reason_codes == ["human_policy_check_passed"]
        return _Review()


def test_feature_flag_off_is_visible_and_real_routes_fail_closed(monkeypatch):
    monkeypatch.setattr(main_module, "COURSE_REVIEWS_ENABLED", False)
    client = TestClient(main_module.app)
    policy = client.get("/course-reviews/policy?language=tr")
    assert policy.status_code == 200 and policy.json()["enabled"] is False
    response = client.post(
        "/course-reviews?language=tr",
        json=VALID_REVIEW,
        headers=_headers("student-a"),
    )
    assert response.status_code == 503
    assert "etkin değil" in response.json()["detail"]


def test_authenticated_submit_delete_and_suppressed_aggregate_do_not_expose_identity(monkeypatch):
    store = _Store()
    monkeypatch.setattr(main_module, "COURSE_REVIEWS_ENABLED", True)
    monkeypatch.setattr(main_module, "_course_review_store", lambda: store)
    client = TestClient(main_module.app)

    submitted = client.post(
        "/course-reviews",
        json=VALID_REVIEW,
        headers=_headers("student-a"),
    )
    assert submitted.status_code == 200
    serialized = str(submitted.json()).casefold()
    assert "student-a" not in serialized and "digest" not in serialized and "comment" not in serialized

    suppressed = client.get(
        "/course-reviews/CS%20201/aggregate",
        headers=_headers("student-b"),
    )
    assert suppressed.status_code == 200
    assert suppressed.json()["available"] is False
    assert "reviewCount" not in suppressed.json()

    forbidden_delete = client.delete(
        "/course-reviews/CS%20201",
        headers=_headers("student-b"),
    )
    assert forbidden_delete.status_code == 404
    own_delete = client.delete(
        "/course-reviews/CS%20201",
        headers=_headers("student-a"),
    )
    assert own_delete.status_code == 200
    assert store.author_tokens == ["student-a", "student-b", "student-a"]


def test_moderation_is_admin_only_and_storage_errors_are_sanitized(monkeypatch):
    store = _Store()
    monkeypatch.setattr(main_module, "COURSE_REVIEWS_ENABLED", True)
    monkeypatch.setattr(main_module, "_course_review_store", lambda: store)
    client = TestClient(main_module.app)
    payload = {"state": "approved", "reason_codes": ["human_policy_check_passed"]}

    denied = client.post(
        "/course-reviews/crv_public/moderation",
        json=payload,
        headers=_headers("student-a"),
    )
    assert denied.status_code == 403
    allowed = client.post(
        "/course-reviews/crv_public/moderation",
        json=payload,
        headers=_headers("admin-a", "admin"),
    )
    assert allowed.status_code == 200

    async def unavailable(*_args, **_kwargs):
        raise CourseReviewStorageError("backend-topology-secret")

    store.submit = unavailable
    failed = client.post(
        "/course-reviews",
        json=VALID_REVIEW,
        headers=_headers("student-a"),
    )
    assert failed.status_code == 503
    assert "topology" not in str(failed.json()).casefold()


def test_course_review_route_has_independent_rate_limit_and_retry_after(monkeypatch):
    store = _Store()
    main_module.resource_controller.reset_for_tests()
    monkeypatch.setattr(main_module, "COURSE_REVIEWS_ENABLED", True)
    monkeypatch.setattr(main_module, "COURSE_REVIEW_REQUESTS_PER_MINUTE", 1)
    monkeypatch.setattr(main_module, "_course_review_store", lambda: store)
    client = TestClient(main_module.app)
    headers = _headers("rate-student")
    assert client.post("/course-reviews", json=VALID_REVIEW, headers=headers).status_code == 200
    limited = client.post("/course-reviews", json=VALID_REVIEW, headers=headers)
    assert limited.status_code == 429
    assert limited.headers["Retry-After"] == "60"


def test_shared_backend_unavailable_is_safe_503_not_local_fallback(monkeypatch):
    class _Unavailable:
        def check_rate_limits(self, **_kwargs):
            raise BackendUnavailableError("private backend topology")

    original_backend = main_module.resource_controller.backend
    monkeypatch.setattr(main_module.resource_controller, "backend", _Unavailable())
    monkeypatch.setattr(main_module, "COURSE_REVIEWS_ENABLED", True)
    response = TestClient(main_module.app).post(
        "/course-reviews",
        json=VALID_REVIEW,
        headers=_headers("student-a"),
    )
    assert response.status_code == 503
    assert "topology" not in str(response.json()).casefold()
    monkeypatch.setattr(main_module.resource_controller, "backend", original_backend)
