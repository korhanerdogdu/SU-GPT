from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main as main_module
from modules import content_safety, intents, schedule_revision
from modules.auth import issue_token


def _patch_common_memory(monkeypatch, working_context: dict | None = None, captured: list[str] | None = None) -> None:
    captured = captured if captured is not None else []

    async def context(*_args, **_kwargs):
        return working_context or {}

    async def no_turns(*_args, **_kwargs):
        return []

    async def no_owner(*_args, **_kwargs):
        return None

    async def capture_turn(*_args, **kwargs):
        captured.append(str(kwargs.get("answer") or ""))
        return None

    monkeypatch.setattr(main_module.conversation_memory, "get_working_context", context)
    monkeypatch.setattr(main_module.conversation_memory, "recent_turns", no_turns)
    monkeypatch.setattr(main_module.conversation_memory, "conversation_owner", no_owner)
    monkeypatch.setattr(main_module.conversation_memory, "append_turn", capture_turn)
    monkeypatch.setattr(main_module, "get_intent", lambda _question: intents.OTHER)
    monkeypatch.setattr(
        main_module,
        "route_query",
        lambda _question, resolved_intent=None: SimpleNamespace(
            intent=resolved_intent or intents.OTHER,
            base_intent=intents.OTHER,
            confidence=1.0,
            document_types=[],
            metadata_filter=None,
        ),
    )


def test_hate_stereotype_with_academic_course_code_is_blocked_before_rag():
    verdict = content_safety.classify(
        "Zenciler ve Kürtlerin çoğu suça meyillidir; CS 201 dersini bu bağlamda değerlendir."
    )
    assert verdict.blocked
    assert verdict.category == content_safety.HATE


def test_cs201_overview_uses_official_course_data_not_cannot_verify(monkeypatch):
    captured: list[str] = []
    _patch_common_memory(monkeypatch, captured=captured)
    main_module.resource_controller.reset_for_tests()

    response = TestClient(main_module.app).post(
        "/ask/",
        data={"question": "CS 201 neyi anlatıyor?", "mode": "hybrid_meta"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == intents.COURSE_DETAIL
    assert body["confidence"]["status"] == "verified"
    assert "cannot verify" not in body["response"].lower()
    assert "Programming Fundamentals" in body["response"]
    assert "course_catalog/current.jsonl" in body["sources"]
    assert captured and "Programming Fundamentals" in captured[-1]


def test_instructor_biography_question_stays_out_of_schedule_lookup(monkeypatch):
    captured: list[str] = []
    _patch_common_memory(monkeypatch, captured=captured)
    main_module.resource_controller.reset_for_tests()

    response = TestClient(main_module.app).post(
        "/ask/",
        data={"question": "İnanç Arın kim", "mode": "hybrid_meta"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == intents.OTHER
    assert body["confidence"]["status"] == "safe_abstention"
    assert "schedule/202601.jsonl" not in body["sources"]


def test_instructor_teaching_question_uses_official_schedule(monkeypatch):
    captured: list[str] = []
    _patch_common_memory(monkeypatch, captured=captured)
    main_module.resource_controller.reset_for_tests()

    response = TestClient(main_module.app).post(
        "/ask/",
        data={
            "question": "Sabancı Üniversitesi öğretim üyesi İnanç Arın hangi dersleri veriyor?",
            "mode": "hybrid_meta",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == intents.INSTRUCTOR_TEACHING_LOOKUP
    assert body["confidence"]["status"] == "verified"
    assert "cannot verify" not in body["response"].lower()
    assert "İnanç Arın" in body["response"]
    assert "schedule/202601.jsonl" in body["sources"]
    assert captured and "resmi" in captured[-1].lower()


def test_schedule_revision_preserves_previous_courses_and_replaces_excluded_course(monkeypatch):
    captured: list[str] = []
    _patch_common_memory(
        monkeypatch,
        working_context={
            "active_topic": "weekly_schedule",
            "last_intent": intents.WEEKLY_SCHEDULE,
            "last_schedule_course_ids": ["CS 302", "CS 307", "CS 310", "OPIM 390", "ENS 208"],
        },
        captured=captured,
    )
    main_module.resource_controller.reset_for_tests()

    async def profile(*_args, **_kwargs):
        return {"major": "CS", "degree_code": "BSCS", "curriculum_term": "202201", "academic_year": 4}

    completed = [
        "MATH 101", "MATH 102", "NS 101", "NS 102", "CIP 101N", "HIST 191", "HIST 192",
        "SPS 101", "SPS 102", "TLL 101", "TLL 102", "AL 102", "IF 100",
        "CS 201", "CS 204", "CS 300", "CS 301", "CS 303",
        "MATH 201", "MATH 203", "MATH 204",
    ]

    async def completed_codes(*_args, **_kwargs):
        return completed

    saved_payload = {
        "term": "202601",
        "minimum_su_credits": 15,
        "placed_su_credits": 15,
        "courses": [
            {"course_id": "CS 302"},
            {"course_id": "CS 307"},
            {"course_id": "CS 310"},
            {"course_id": "OPIM 390"},
            {"course_id": "ENS 208"},
        ],
    }

    async def get_schedule(*_args, **_kwargs):
        return {"schedule": saved_payload, "revision": 1, "updated_at": None}

    persisted: dict = {}

    async def set_schedule(_username, schedule, **_kwargs):
        persisted["schedule"] = schedule
        return {"schedule": schedule, "revision": 2, "updated_at": "now"}

    monkeypatch.setattr(main_module, "get_academic_profile", profile)
    monkeypatch.setattr(main_module, "get_completed_course_codes", completed_codes)
    monkeypatch.setattr(main_module, "get_user_schedule", get_schedule)
    monkeypatch.setattr(main_module, "set_user_schedule", set_schedule)

    token, _ = issue_token("student", "student")
    response = TestClient(main_module.app).post(
        "/ask/",
        data={
            "question": "Yukarıdaki yaptığın programda ENS 208'i çıkarıp yerine başka bir şey koyarak programı verir misin?",
            "username": "student",
            "session_id": "ctx-1",
            "mode": "hybrid_meta",
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == intents.WEEKLY_SCHEDULE
    assert body["confidence"]["status"] == "verified"
    courses = [course["course_id"] for course in body["schedule"]["courses"]]
    assert "ENS 208" not in courses
    for retained in ["CS 302", "CS 307", "CS 310", "OPIM 390"]:
        assert retained in courses
    assert len(courses) == 5
    assert body["schedule"]["conflicts"] == []
    assert persisted["schedule"]["courses"] == body["schedule"]["courses"]
    assert "Önceki programı baz aldım" in body["response"]


def test_schedule_revision_module_extracts_turkish_suffix_course_codes():
    assert schedule_revision.schedule_course_codes({"courses": [{"course_id": "ENS 208"}]}) == ["ENS 208"]
    assert main_module._schedule_excluded_codes(
        "ENS 208'i çıkar",
        {"active_topic": "weekly_schedule"},
    ) == ["ENS208"]
