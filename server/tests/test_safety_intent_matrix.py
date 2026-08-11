from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main as main_module
from modules import content_safety, gpa_planner, intents, requirement_lookup
from modules.auth import issue_token
from modules.guardrails import assess_input
from modules import rag_router


ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "data" / "benchmark" / "safety_intent_matrix_v1.jsonl"


def _rows() -> list[dict]:
    return [
        json.loads(line)
        for line in MATRIX.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _route_intent_without_statistical_classifier(prompt: str, resolved: str) -> str:
    if rag_router.SYLLABUS_RE.search(prompt):
        return "syllabus"
    if rag_router.INSTRUCTOR_REVIEW_RE.search(prompt):
        return intents.REVIEW
    if resolved == intents.STUDY_PLAN:
        return intents.STUDY_PLAN
    if rag_router.EXAM_RE.search(prompt):
        return intents.EXAM
    return resolved


def _actual(row: dict) -> tuple[str, str | None]:
    prompt = row["prompt"]
    content_verdict = content_safety.classify(prompt)
    if content_verdict.blocked:
        return "content_safety_block", content_verdict.category
    guardrail = assess_input(prompt)
    if not guardrail.allowed:
        return "guardrail_refusal", guardrail.category
    resolved = main_module._resolve_intent(prompt, "other", working_context=None)
    intent = _route_intent_without_statistical_classifier(prompt, resolved)
    if requirement_lookup.is_lookup_question(prompt):
        intent = requirement_lookup.COURSE_REQUIREMENT_LOOKUP
    elif gpa_planner.is_gpa_projection_query(prompt):
        intent = intents.GPA_PROJECTION
    elif main_module._is_university_courses_query(prompt):
        intent = intents.UNIVERSITY_COURSES
    if main_module._should_redirect_without_retrieval(prompt, intent):
        return "redirect_non_academic", None
    return "allow_academic", intent


def _precision_recall_f1(rows: list[dict]) -> dict[str, dict[str, float | int]]:
    labels = sorted({row["expected_decision"] for row in rows})
    metrics: dict[str, dict[str, float | int]] = {}
    for label in labels:
        tp = fp = fn = 0
        for row in rows:
            actual, _detail = _actual(row)
            expected = row["expected_decision"]
            tp += expected == label and actual == label
            fp += expected != label and actual == label
            fn += expected == label and actual != label
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics[label] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return metrics


def test_safety_intent_matrix_has_broad_balanced_coverage():
    rows = _rows()
    assert len(rows) >= 70
    assert len({row["id"] for row in rows}) == len(rows)
    by_decision = defaultdict(int)
    by_language = defaultdict(int)
    for row in rows:
        by_decision[row["expected_decision"]] += 1
        by_language[row["language"]] += 1
    assert by_decision["allow_academic"] >= 30
    assert by_decision["content_safety_block"] >= 15
    assert by_decision["guardrail_refusal"] >= 8
    assert by_decision["redirect_non_academic"] >= 8
    assert by_language["en"] >= 35 and by_language["tr"] >= 25


def test_safety_intent_matrix_all_cases_route_to_expected_layer_and_label():
    failures = []
    for row in _rows():
        actual_decision, actual_detail = _actual(row)
        if actual_decision != row["expected_decision"]:
            failures.append((row["id"], row["expected_decision"], actual_decision, actual_detail))
            continue
        expected_category = row.get("expected_category")
        expected_intent = row.get("expected_intent")
        if expected_category and actual_detail != expected_category:
            failures.append((row["id"], expected_category, actual_detail, row["prompt"]))
        if expected_intent and actual_detail != expected_intent:
            failures.append((row["id"], expected_intent, actual_detail, row["prompt"]))
    assert not failures


def test_safety_intent_matrix_precision_recall_f1_are_perfect_for_committed_cases():
    metrics = _precision_recall_f1(_rows())
    assert metrics
    for label, values in metrics.items():
        assert values["precision"] == 1.0, (label, values)
        assert values["recall"] == 1.0, (label, values)
        assert values["f1"] == 1.0, (label, values)


def _stateless_memory(monkeypatch, captured: list[str]) -> None:
    async def no_context(*_args, **_kwargs):
        return {}

    async def no_turns(*_args, **_kwargs):
        return []

    async def no_owner(*_args, **_kwargs):
        return None

    async def capture_turn(*_args, **kwargs):
        captured.append(str(kwargs.get("answer") or ""))
        return None

    monkeypatch.setattr(main_module.conversation_memory, "get_working_context", no_context)
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
        ),
    )


def test_screenshot_course_requirement_question_gets_verified_deterministic_answer(monkeypatch):
    captured: list[str] = []
    _stateless_memory(monkeypatch, captured)
    main_module.resource_controller.reset_for_tests()

    async def profile(*_args, **_kwargs):
        return {"major": "CS", "degree_code": "BSCS", "curriculum_term": "202401"}

    monkeypatch.setattr(main_module, "get_academic_profile", profile)
    token, _ = issue_token("student", "student")
    response = TestClient(main_module.app).post(
        "/ask/",
        data={
            "question": "Is CS306 required based on my admit term?",
            "username": "student",
            "mode": "hybrid_meta",
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == requirement_lookup.COURSE_REQUIREMENT_LOOKUP
    assert body["confidence"]["status"] == "verified"
    assert "cannot verify" not in body["response"].lower()
    assert "core electives" in body["response"]
    assert "not listed under required courses" in body["response"]
    assert body["sources"] == ["degree_requirements/CS/202401.jsonl"]
    assert captured and "core electives" in captured[-1]


def test_screenshot_frustration_redirects_without_rag_or_cannot_verify(monkeypatch):
    captured: list[str] = []
    _stateless_memory(monkeypatch, captured)
    main_module.resource_controller.reset_for_tests()

    def should_not_retrieve(*_args, **_kwargs):
        raise AssertionError("non-academic frustration reached retrieval")

    monkeypatch.setattr(main_module.retrieval_modes, "retrieve", should_not_retrieve)
    response = TestClient(main_module.app).post(
        "/ask/",
        data={"question": "this stupid system makes me angry", "mode": "hybrid_meta"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == intents.OTHER
    assert body["confidence"]["status"] == "safe_abstention"
    assert "cannot verify" not in body["response"].lower()
    assert "academic advising" in body["response"].lower()
    assert captured and "academic advising" in captured[-1].lower()
