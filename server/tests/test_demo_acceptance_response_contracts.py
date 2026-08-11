from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

import main as main_module
from modules import intents
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


def _auth_headers(username: str = "student") -> dict[str, str]:
    token, _ = issue_token(username, "student")
    return {"Authorization": f"Bearer {token}"}


def _completed_demo_courses() -> list[str]:
    return [
        "AL 102", "CIP 101N", "HIST 191", "HIST 192", "HUM 202", "IF 100",
        "MATH 101", "MATH 102", "NS 101", "NS 102", "PROJ 201", "SPS 101",
        "SPS 102", "SPS 303", "TLL 101", "TLL 102",
        "CS 201", "CS 204", "CS 300", "CS 301", "CS 303",
        "MATH 201", "MATH 203", "MATH 204",
        "CS 210", "CS 306", "CS 308", "CS 404", "CS 412", "CS 445",
        "MATH 306", "ECON 201", "ECON 204", "IE 303",
    ]


def _saved_revised_schedule() -> dict:
    return {
        "term": "202601",
        "term_label": "Fall 2026-2027 (Güz)",
        "minimum_su_credits": 15,
        "placed_su_credits": 15,
        "conflicts": [],
        "crns": ["10116", "10118", "10160", "10649", "10651", "10374", "10555"],
        "courses": [
            {"course_id": "CS 302", "crn": "10116", "extras": [{"crn": "10118"}]},
            {"course_id": "CS 310", "crn": "10160", "extras": []},
            {"course_id": "OPIM 390", "crn": "10649", "extras": [{"crn": "10651"}]},
            {"course_id": "ENS 208", "crn": "10374", "extras": []},
            {"course_id": "CS 412", "crn": "10555", "extras": []},
        ],
    }


def test_basic_science_remaining_ects_is_scoped_to_requested_number(monkeypatch):
    _patch_common_memory(monkeypatch)
    main_module.resource_controller.reset_for_tests()

    async def profile(*_args, **_kwargs):
        return {"major": "CS", "degree_code": "BSCS", "curriculum_term": "202201", "academic_year": 4}

    async def completed(*_args, **_kwargs):
        return _completed_demo_courses()

    monkeypatch.setattr(main_module, "get_academic_profile", profile)
    monkeypatch.setattr(main_module, "get_completed_course_codes", completed)

    response = TestClient(main_module.app).post(
        "/ask/",
        data={
            "question": "How many basic science credits do I have left to graduate?",
            "username": "student",
            "session_id": "basic-science-1",
            "mode": "hybrid_meta",
        },
        headers=_auth_headers(),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == intents.GRADUATION_STATUS
    assert "3 ECTS" in body["response"]
    assert "Degree Audit by Category" not in body["response"]
    assert "University courses" not in body["response"]
    assert body.get("structured_content") is None


def test_exact_demo_basic_science_wording_never_hits_llm(monkeypatch):
    _patch_common_memory(monkeypatch)
    main_module.resource_controller.reset_for_tests()

    async def profile(*_args, **_kwargs):
        return {"major": "CS", "degree_code": "BSCS", "curriculum_term": "202201", "academic_year": 4}

    async def completed(*_args, **_kwargs):
        return _completed_demo_courses()

    def fail_provider(*_args, **_kwargs):
        raise AssertionError("basic-science scoped answer must be deterministic")

    monkeypatch.setattr(main_module, "get_academic_profile", profile)
    monkeypatch.setattr(main_module, "get_completed_course_codes", completed)
    monkeypatch.setattr(main_module, "answer_without_context_with_telemetry", fail_provider)

    response = TestClient(main_module.app).post(
        "/ask/",
        data={
            "question": "Only tell me how many basic science ECTS I still need.",
            "username": "student",
            "session_id": "basic-science-exact",
            "mode": "hybrid_meta",
        },
        headers=_auth_headers(),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["response"] == "3 ECTS"
    assert body["confidence"]["status"] == "verified"


def test_mandatory_or_elective_wording_routes_to_requirement_lookup(monkeypatch):
    _patch_common_memory(monkeypatch)
    main_module.resource_controller.reset_for_tests()

    async def profile(*_args, **_kwargs):
        return {"major": "CS", "degree_code": "BSCS", "curriculum_term": "202201", "academic_year": 4}

    monkeypatch.setattr(main_module, "get_academic_profile", profile)

    response = TestClient(main_module.app).post(
        "/ask/",
        data={
            "question": "For my admit term, is CS 306 mandatory or elective?",
            "username": "student",
            "session_id": "mandatory-elective-1",
            "mode": "hybrid_meta",
        },
        headers=_auth_headers(),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == intents.COURSE_REQUIREMENT_LOOKUP
    assert "CS 306" in body["response"]
    assert "not in the mandatory-course category" in body["response"].lower()
    assert "core elective" in body["response"].lower()


def test_course_learning_outcomes_and_summary_wording_use_official_lookup(monkeypatch):
    _patch_common_memory(monkeypatch)
    main_module.resource_controller.reset_for_tests()
    client = TestClient(main_module.app)

    learning = client.post(
        "/ask/",
        data={
            "question": "Okay, then what are the learning outcomes of CS 201?",
            "username": "student",
            "session_id": "learning-outcomes-1",
            "mode": "hybrid_meta",
        },
        headers=_auth_headers(),
    ).json()
    assert learning["intent"] == intents.COURSE_DETAIL
    assert "Programming Fundamentals" in learning["response"]
    assert "learning" in learning["response"].lower() or "course coverage" in learning["response"].lower()

    turkish = client.post(
        "/ask/",
        data={
            "question": "Şimdi Türkçe olarak CS 302’nin kısa özetini ver.",
            "username": "student",
            "session_id": "turkish-summary-1",
            "mode": "hybrid_meta",
        },
        headers=_auth_headers(),
    ).json()
    assert turkish["intent"] == intents.COURSE_DETAIL
    assert "CS 302" in turkish["response"]
    assert "Dil modeli" not in turkish["response"]


def test_non_academic_recommendation_redirects_instead_of_asking_interest(monkeypatch):
    _patch_common_memory(monkeypatch)
    main_module.resource_controller.reset_for_tests()

    response = TestClient(main_module.app).post(
        "/ask/",
        data={
            "question": "Can you recommend a movie for tonight?",
            "username": "student",
            "session_id": "movie-1",
            "mode": "hybrid_meta",
        },
        headers=_auth_headers(),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == intents.OTHER
    assert "Which area interests you" not in body["response"]
    assert "academic" in body["response"].lower() or "akademik" in body["response"].lower()


def test_only_crns_uses_latest_saved_schedule_without_regenerating(monkeypatch):
    _patch_common_memory(
        monkeypatch,
        working_context={"active_topic": "weekly_schedule", "last_intent": intents.WEEKLY_SCHEDULE},
    )
    main_module.resource_controller.reset_for_tests()

    async def profile(*_args, **_kwargs):
        return {"major": "CS", "degree_code": "BSCS", "curriculum_term": "202201", "academic_year": 4}

    async def saved(*_args, **_kwargs):
        return {"schedule": _saved_revised_schedule(), "revision": 2, "updated_at": "now"}

    monkeypatch.setattr(main_module, "get_academic_profile", profile)
    monkeypatch.setattr(main_module, "get_user_schedule", saved)

    response = TestClient(main_module.app).post(
        "/ask/",
        data={
            "question": "Can you give me only the CRNs for the final schedule?",
            "username": "student",
            "session_id": "crn-1",
            "mode": "hybrid_meta",
        },
        headers=_auth_headers(),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == intents.SCHEDULE_CRN_LOOKUP
    assert body["response"] == "10116 10118 10160 10649 10651 10374 10555"
    assert "10135" not in body["response"]  # old CS 307 lecture CRN must not resurrect
    assert "10147" not in body["response"]  # old CS 307 recitation CRN must not resurrect
    assert "Weekly Schedule" not in body["response"]
    assert "CS 302" not in body["response"]


def test_conflict_check_answers_status_only_from_latest_schedule(monkeypatch):
    _patch_common_memory(
        monkeypatch,
        working_context={"active_topic": "weekly_schedule", "last_intent": intents.WEEKLY_SCHEDULE},
    )
    main_module.resource_controller.reset_for_tests()

    async def profile(*_args, **_kwargs):
        return {"major": "CS", "degree_code": "BSCS", "curriculum_term": "202201", "academic_year": 4}

    async def saved(*_args, **_kwargs):
        return {"schedule": _saved_revised_schedule(), "revision": 2, "updated_at": "now"}

    monkeypatch.setattr(main_module, "get_academic_profile", profile)
    monkeypatch.setattr(main_module, "get_user_schedule", saved)

    response = TestClient(main_module.app).post(
        "/ask/",
        data={
            "question": "Does the final schedule have any time conflicts?",
            "username": "student",
            "session_id": "conflict-1",
            "mode": "hybrid_meta",
        },
        headers=_auth_headers(),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == intents.SCHEDULE_CONFLICT_LOOKUP
    assert "No conflict" in body["response"]
    assert "Your conflict-free" not in body["response"]
    assert "CRNs for registration" not in body["response"]


def test_ai_nlp_priority_followup_rebuilds_weekly_schedule_from_context(monkeypatch):
    captured: list[str] = []
    _patch_common_memory(
        monkeypatch,
        working_context={
            "active_topic": "weekly_schedule",
            "last_intent": intents.WEEKLY_SCHEDULE,
            "persistent_excluded_codes": ["CS 302", "CS 307"],
        },
        captured=captured,
    )
    main_module.resource_controller.reset_for_tests()

    async def profile(*_args, **_kwargs):
        return {"major": "CS", "degree_code": "BSCS", "curriculum_term": "202201", "academic_year": 4}

    async def completed(*_args, **_kwargs):
        return _completed_demo_courses()

    persisted: dict = {}

    async def set_schedule(_username, schedule, **_kwargs):
        persisted["schedule"] = schedule
        return {"schedule": schedule, "revision": 3, "updated_at": "now"}

    monkeypatch.setattr(main_module, "get_academic_profile", profile)
    monkeypatch.setattr(main_module, "get_completed_course_codes", completed)
    monkeypatch.setattr(main_module, "set_user_schedule", set_schedule)

    response = TestClient(main_module.app).post(
        "/ask/",
        data={
            "question": "I am interested in AI and NLP. Can you prioritize electives around that?",
            "username": "student",
            "session_id": "ai-nlp-schedule-1",
            "mode": "hybrid_meta",
        },
        headers=_auth_headers(),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == intents.WEEKLY_SCHEDULE
    assert body["confidence"]["status"] == "verified"
    assert "Your conflict-free, balanced weekly schedule" in body["response"]
    assert "CRNs for registration" in body["response"]
    assert "CS 415" in body["response"]
    assert "Interest alignment (AI / NLP)" in body["response"]
    assert "Current aligned course" in body["response"]
    courses = [course["course_id"] for course in body["schedule"]["courses"]]
    assert "CS 302" not in courses
    assert "CS 307" not in courses
    assert body["schedule"]["conflicts"] == []
    assert persisted["schedule"]["courses"] == body["schedule"]["courses"]
    assert captured and "CS 415" in captured[-1]


def test_cs307_prerequisite_query_uses_deterministic_prerequisite_rules(monkeypatch):
    _patch_common_memory(monkeypatch)
    main_module.resource_controller.reset_for_tests()

    response = TestClient(main_module.app).post(
        "/ask/",
        data={"question": "What are the prerequisites for CS 307?", "mode": "hybrid_meta"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == intents.COURSE_PREREQUISITE_LOOKUP
    assert "CS 204" in body["response"]
    assert "cannot verify" not in body["response"].lower()


def test_cs302_offering_query_uses_official_schedule(monkeypatch):
    _patch_common_memory(monkeypatch)
    main_module.resource_controller.reset_for_tests()

    response = TestClient(main_module.app).post(
        "/ask/",
        data={"question": "Is CS 302 offered in Fall 2026-2027?", "mode": "hybrid_meta"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == intents.COURSE_OFFERING_LOOKUP
    assert "offered" in body["response"].lower()
    assert "10116" in body["response"]
    assert "schedule/202601.jsonl" in body["sources"]


def test_schedule_replacement_followup_answers_shortly_from_memory(monkeypatch):
    _patch_common_memory(
        monkeypatch,
        working_context={
            "active_topic": "weekly_schedule",
            "last_intent": intents.WEEKLY_SCHEDULE,
            "last_schedule_replacement": {
                "removed_codes": ["CS 307"],
                "added_codes": ["DSA 440"],
            },
        },
    )
    main_module.resource_controller.reset_for_tests()

    async def profile(*_args, **_kwargs):
        return {"major": "CS", "degree_code": "BSCS", "curriculum_term": "202201", "academic_year": 4}

    async def saved(*_args, **_kwargs):
        return {"schedule": _saved_revised_schedule(), "revision": 3, "updated_at": "now"}

    monkeypatch.setattr(main_module, "get_academic_profile", profile)
    monkeypatch.setattr(main_module, "get_user_schedule", saved)

    response = TestClient(main_module.app).post(
        "/ask/",
        data={
            "question": "Based on the schedule we revised earlier, remind me which course replaced CS 307.",
            "username": "student",
            "session_id": "replacement-1",
            "mode": "hybrid_meta",
        },
        headers=_auth_headers(),
    )
    assert response.status_code == 200
    body = response.json()
    assert "DSA 440" in body["response"]
    assert "CS 307" in body["response"]
    assert "Weekly Schedule" not in body["response"]
    assert "CRNs for registration" not in body["response"]
    assert not body.get("summary")


def test_short_final_schedule_summary_does_not_rebuild_or_dump_table(monkeypatch):
    _patch_common_memory(
        monkeypatch,
        working_context={"active_topic": "weekly_schedule", "last_intent": intents.WEEKLY_SCHEDULE},
    )
    main_module.resource_controller.reset_for_tests()

    async def profile(*_args, **_kwargs):
        return {"major": "CS", "degree_code": "BSCS", "curriculum_term": "202201", "academic_year": 4}

    saved_payload = _saved_revised_schedule()
    saved_payload["excluded_codes"] = ["CS 302", "CS 307"]

    async def saved(*_args, **_kwargs):
        return {"schedule": saved_payload, "revision": 3, "updated_at": "now"}

    def fail_if_rebuilt(*_args, **_kwargs):
        raise AssertionError("short schedule summary must not rebuild the weekly schedule")

    monkeypatch.setattr(main_module, "get_academic_profile", profile)
    monkeypatch.setattr(main_module, "get_user_schedule", saved)
    monkeypatch.setattr(main_module.course_planner, "build_plan", fail_if_rebuilt)

    response = TestClient(main_module.app).post(
        "/ask/",
        data={
            "question": "Give me a short final summary of my recommended semester plan.",
            "username": "student",
            "session_id": "summary-1",
            "mode": "hybrid_meta",
        },
        headers=_auth_headers(),
    )
    assert response.status_code == 200
    body = response.json()
    assert "Final plan:" in body["response"]
    assert "CS 310" in body["response"]
    assert "10116" in body["response"]
    assert "Weekly Schedule" not in body["response"]
    assert "I could not find enough official evidence" not in body["response"]
