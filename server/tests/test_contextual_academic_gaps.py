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

    async def no_preferences(*_args, **_kwargs):
        return {"excluded_courses": [], "preferred_topics": [], "schedule_preferences": {}}

    monkeypatch.setattr(main_module.conversation_memory, "get_working_context", context)
    monkeypatch.setattr(main_module.conversation_memory, "recent_turns", no_turns)
    monkeypatch.setattr(main_module.conversation_memory, "conversation_owner", no_owner)
    monkeypatch.setattr(main_module.conversation_memory, "append_turn", capture_turn)
    monkeypatch.setattr(main_module, "get_user_preferences", no_preferences)
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


def test_instructor_biography_question_uses_official_profile_sources(monkeypatch):
    captured: list[str] = []
    _patch_common_memory(monkeypatch, captured=captured)
    main_module.resource_controller.reset_for_tests()

    response = TestClient(main_module.app).post(
        "/ask/",
        data={"question": "İnanç Arın kim", "mode": "hybrid_meta"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == intents.INSTRUCTOR_PROFILE_LOOKUP
    assert body["confidence"]["status"] == "verified"
    assert "İnanç Arın" in body["response"]
    assert "faculty_profiles/current.jsonl" in body["sources"]
    assert "Data Analytics" in body["response"] or "Natural Language Processing" in body["response"]


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


def test_schedule_revision_removes_cs307_and_persists_negative_preference(monkeypatch):
    captured: list[str] = []
    updates: list[dict] = []

    async def capture_turn(*_args, **kwargs):
        captured.append(str(kwargs.get("answer") or ""))
        updates.append(kwargs.get("working_context_updates") or {})
        return None

    _patch_common_memory(
        monkeypatch,
        working_context={
            "active_topic": "weekly_schedule",
            "last_intent": intents.WEEKLY_SCHEDULE,
            "last_schedule_course_ids": ["CS 302", "CS 307", "CS 310", "OPIM 390", "ENS 208"],
        },
        captured=captured,
    )
    monkeypatch.setattr(main_module.conversation_memory, "append_turn", capture_turn)
    main_module.resource_controller.reset_for_tests()

    async def profile(*_args, **_kwargs):
        return {"major": "CS", "degree_code": "BSCS", "curriculum_term": "202201", "academic_year": 4}

    async def completed_codes(*_args, **_kwargs):
        return [
            "MATH 101", "MATH 102", "NS 101", "NS 102", "CIP 101N", "HIST 191", "HIST 192",
            "SPS 101", "SPS 102", "TLL 101", "TLL 102", "AL 102", "IF 100",
            "CS 201", "CS 204", "CS 300", "CS 301", "CS 303",
            "MATH 201", "MATH 203", "MATH 204",
        ]

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

    async def set_schedule(_username, schedule, **_kwargs):
        return {"schedule": schedule, "revision": 2, "updated_at": "now"}

    async def add_exclusions(_username, codes):
        return {"excluded_courses": [main_module.course_planner.display_code(code) for code in codes]}

    monkeypatch.setattr(main_module, "get_academic_profile", profile)
    monkeypatch.setattr(main_module, "get_completed_course_codes", completed_codes)
    monkeypatch.setattr(main_module, "get_user_schedule", get_schedule)
    monkeypatch.setattr(main_module, "set_user_schedule", set_schedule)
    monkeypatch.setattr(main_module, "add_user_excluded_courses", add_exclusions)

    token, _ = issue_token("student", "student")
    response = TestClient(main_module.app).post(
        "/ask/",
        data={
            "question": "I don't want CS 307 in this schedule; can you suggest another course instead?",
            "username": "student",
            "session_id": "ctx-cs307",
            "mode": "hybrid_meta",
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    body = response.json()
    courses = [course["course_id"] for course in body["schedule"]["courses"]]
    assert "CS 307" not in courses
    for retained in ["CS 302", "CS 310", "OPIM 390", "ENS 208"]:
        assert retained in courses
    assert updates[-1]["persistent_excluded_codes"] == ["CS 307"]


def test_latest_schedule_crn_and_conflict_followups_do_not_rebuild_schedule(monkeypatch):
    captured: list[str] = []
    _patch_common_memory(
        monkeypatch,
        working_context={
            "active_topic": "weekly_schedule",
            "last_intent": intents.WEEKLY_SCHEDULE,
            "persistent_excluded_codes": ["CS 307"],
        },
        captured=captured,
    )
    main_module.resource_controller.reset_for_tests()

    async def profile(*_args, **_kwargs):
        return {"major": "CS", "degree_code": "BSCS", "curriculum_term": "202201", "academic_year": 4}

    async def completed_codes(*_args, **_kwargs):
        return []

    saved_payload = {
        "term": "202601",
        "crns": ["11111", "22222", "33333"],
        "conflicts": [],
        "courses": [
            {
                "course_id": "CS 302",
                "sections": [{"crn": "11111", "meetings": []}],
            },
            {
                "course_id": "OPIM 390",
                "sections": [{"crn": "22222", "meetings": []}],
            },
            {
                "course_id": "ENS 208",
                "sections": [{"crn": "33333", "meetings": []}],
            },
        ],
    }

    async def get_schedule(*_args, **_kwargs):
        return {"schedule": saved_payload, "revision": 3, "updated_at": "now"}

    def fail_if_rebuilt(*_args, **_kwargs):
        raise AssertionError("CRN/conflict follow-up must use the latest saved schedule")

    monkeypatch.setattr(main_module, "get_academic_profile", profile)
    monkeypatch.setattr(main_module, "get_completed_course_codes", completed_codes)
    monkeypatch.setattr(main_module, "get_user_schedule", get_schedule)
    monkeypatch.setattr(main_module.course_planner, "build_plan", fail_if_rebuilt)

    token, _ = issue_token("student", "student")
    client = TestClient(main_module.app)
    crn_response = client.post(
        "/ask/",
        data={
            "question": "Can you give me only the CRNs for the final schedule?",
            "username": "student",
            "session_id": "ctx-crns",
            "mode": "hybrid_meta",
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    assert crn_response.status_code == 200
    crn_body = crn_response.json()
    assert crn_body["structured_content"]["crns"] == ["11111", "22222", "33333"]
    assert "CS 307" not in crn_body["response"]

    conflict_response = client.post(
        "/ask/",
        data={
            "question": "Does the final schedule have any time conflicts? Please check.",
            "username": "student",
            "session_id": "ctx-crns",
            "mode": "hybrid_meta",
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    assert conflict_response.status_code == 200
    conflict_body = conflict_response.json()
    assert conflict_body["structured_content"]["conflicts"] == []
    assert "no time conflicts" in conflict_body["response"].lower()


def test_persistent_schedule_exclusion_is_applied_to_future_rebuilds(monkeypatch):
    assert main_module._resolve_intent(
        "I don't want CS 307",
        intents.OTHER,
        {"active_topic": "weekly_schedule", "last_intent": intents.WEEKLY_SCHEDULE},
    ) == intents.WEEKLY_SCHEDULE
    assert main_module._schedule_excluded_codes(
        "I don't want CS 307",
        {"active_topic": "weekly_schedule", "last_intent": intents.WEEKLY_SCHEDULE},
    ) == ["CS307"]
    assert main_module._merge_excluded_codes(
        set(),
        {"persistent_excluded_codes": ["CS 307"], "last_schedule_excluded_codes": ["ENS 208"]},
    ) == {"CS307", "ENS208"}


def test_interest_followup_after_weekly_schedule_rebuilds_weekly_schedule(monkeypatch):
    captured: list[str] = []
    updates: list[dict] = []

    async def capture_turn(*_args, **kwargs):
        captured.append(str(kwargs.get("answer") or ""))
        updates.append(kwargs.get("working_context_updates") or {})
        return None

    _patch_common_memory(
        monkeypatch,
        working_context={
            "active_topic": "weekly_schedule",
            "last_intent": intents.WEEKLY_SCHEDULE,
            "persistent_excluded_codes": ["CS 302", "CS 307"],
        },
        captured=captured,
    )
    monkeypatch.setattr(main_module.conversation_memory, "append_turn", capture_turn)
    main_module.resource_controller.reset_for_tests()

    async def profile(*_args, **_kwargs):
        return {"major": "CS", "degree_code": "BSCS", "curriculum_term": "202201", "academic_year": 4}

    async def completed_codes(*_args, **_kwargs):
        return [
            "AL 102", "CIP 101N", "HIST 191", "HIST 192", "HUM 202", "IF 100",
            "MATH 101", "MATH 102", "NS 101", "NS 102", "PROJ 201", "SPS 101",
            "SPS 102", "SPS 303", "TLL 101", "TLL 102",
            "CS 201", "CS 204", "CS 300", "CS 301", "CS 303",
            "MATH 201", "MATH 203", "MATH 204",
            "CS 210", "CS 306", "CS 308", "CS 404", "CS 412", "CS 445",
            "MATH 306", "ECON 201", "ECON 204", "IE 303",
        ]

    persisted: dict = {}

    async def set_schedule(_username, schedule, **_kwargs):
        persisted["schedule"] = schedule
        return {"schedule": schedule, "revision": 4, "updated_at": "now"}

    monkeypatch.setattr(main_module, "get_academic_profile", profile)
    monkeypatch.setattr(main_module, "get_completed_course_codes", completed_codes)
    monkeypatch.setattr(main_module, "set_user_schedule", set_schedule)

    token, _ = issue_token("student", "student")
    response = TestClient(main_module.app).post(
        "/ask/",
        data={
            "question": "I am interested in AI and NLP. Can you prioritize electives around that?",
            "username": "student",
            "session_id": "ctx-interest-schedule",
            "mode": "hybrid_meta",
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == intents.WEEKLY_SCHEDULE
    assert body["structured_content"]["kind"] == "course_schedule"
    courses = [course["course_id"] for course in body["schedule"]["courses"]]
    assert "CS 302" not in courses
    assert "CS 307" not in courses
    assert any(code in courses for code in ["CS 415", "DSA 440", "ECON 494"])
    assert updates[-1]["last_interest_key"] in {"ai", "nlp"}
    assert persisted["schedule"]["excluded_codes"] == ["CS 302", "CS 307"]
