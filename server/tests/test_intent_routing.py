from __future__ import annotations

"""Regression: every prompt the frontend's own main-menu starter chips send must resolve to its
intended deterministic intent, not fall through to the generic "other" RAG/LLM path.

This is a contract test, not an exhaustive intent-classifier test: these exact strings are the
literal starter texts in frontend/src/localization/resources.ts (the `chat.starters` /
EmptyState list) -- keep both lists in sync by hand, since TS resources can't be imported here.
A starter chip is the very first thing a new user (or a grader watching the demo) clicks, so a
routing gap here is maximally visible and was previously invisible only because every answer used
to abstain regardless of intent.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

import main as main_module
from modules import intents

STARTER_INTENTS = [
    # Turkish starters (chat.starters, locale === "tr")
    ("Mezuniyet durumumu hesapla", intents.GRADUATION_STATUS),
    ("Bu dönem hangi dersleri alayım?", intents.WEEKLY_SCHEDULE),
    ("Mezun olana kadar hangi dersleri almalıyım?", intents.COURSE_RECOMMENDATION),
    # English starters
    ("Calculate my degree progress", intents.GRADUATION_STATUS),
    ("Which courses should I take this term?", intents.WEEKLY_SCHEDULE),
    ("What should I take until graduation?", intents.COURSE_RECOMMENDATION),
]


def test_every_frontend_starter_chip_resolves_to_its_deterministic_intent():
    for question, expected_intent in STARTER_INTENTS:
        # detected_intent="other" simulates the ML classifier being unconfident (as it genuinely
        # is for several of these short, natural phrasings) -- the deterministic regex chain in
        # _resolve_intent must still recover the right intent regardless of classifier confidence.
        resolved = main_module._resolve_intent(question, "other", working_context=None)
        assert resolved == expected_intent, (
            f"starter chip {question!r} resolved to {resolved!r}, expected {expected_intent!r}"
        )


def test_route_query_trusts_an_already_resolved_intent_over_its_own_bare_keyword_guard():
    # Regression: rag_router.route_query() had its own independent GRADUATION_RE ("mezun\w*")
    # that fired before checking whether a caller had already resolved a *more specific* intent,
    # silently discarding main.py's correct COURSE_RECOMMENDATION resolution for exactly this
    # question (it contains "mezun") and turning it back into GRADUATION_STATUS.
    from modules.rag_router import route_query

    question = "Mezun olana kadar hangi dersleri almalıyım?"
    resolved = main_module._resolve_intent(question, "other", working_context=None)
    assert resolved == intents.COURSE_RECOMMENDATION
    route = route_query(question, resolved)
    assert route.intent == intents.COURSE_RECOMMENDATION


def test_gpa_projection_starters_are_recognised_independently_of_intent_routing():
    # These two go through gpa_planner.is_gpa_projection_query(), a separate check earlier in the
    # pipeline than _resolve_intent -- covered here so the full starter list has one home.
    from modules import gpa_planner

    assert gpa_planner.is_gpa_projection_query("Bu dönem GPA'mı ne kadar yükseltebilirim?")
    assert gpa_planner.is_gpa_projection_query("How much could I raise my GPA this term?")


def test_interest_area_survives_a_later_turn_that_does_not_restate_it():
    # Regression (section 1.7 of the course-advising spec): conversation_memory only keeps a
    # 5-turn rolling window of raw text, and working_context previously never persisted a stated
    # interest area at all -- so a student who said "NLP" early on and kept chatting would find
    # a later, unrelated-sounding recommendation question silently forget it. _resolve_interest_key
    # falls back to working_context["last_interest_key"], which main.py's recommendation branch
    # now writes whenever a fresh interest area is stated.
    resolved = main_module._resolve_interest_key(
        "hangi dersleri önerirsin?", {"last_interest_key": "nlp"}
    )
    assert resolved == "nlp"


def test_interest_area_freshly_stated_this_turn_overrides_the_persisted_one():
    resolved = main_module._resolve_interest_key(
        "artık security ile ilgileniyorum, ne önerirsin?", {"last_interest_key": "nlp"}
    )
    assert resolved == "security"


def test_interest_area_resolution_is_none_when_never_stated():
    assert main_module._resolve_interest_key("hangi dersleri önerirsin?", {}) is None
    assert main_module._resolve_interest_key("hangi dersleri önerirsin?", None) is None


def test_schedule_edit_followup_stays_in_context_instead_of_becoming_graduation_status():
    # Regression: "15 krediye tamamla, 4 kodluları da yazabilirsin" (a follow-up asking to fill
    # out an already-shown schedule to 15 credits) contains "krediye", which GRADUATION_INTENT_RE
    # also matches -- Turkish inflection makes a trailing \b useless here, since "krediye" always
    # contains "kredi" as a substring. Without a context-aware guard this follow-up silently
    # became a graduation-status request instead of continuing the schedule/recommendation.
    for previous, question in (
        (intents.WEEKLY_SCHEDULE, "15 krediye tamamla, 4 kodluları da yazabilirsin"),
        (intents.COURSE_RECOMMENDATION, "bunun yerine AI dersi ekle"),
    ):
        resolved = main_module._resolve_intent(
            question, "other", working_context={"last_intent": previous}
        )
        assert resolved == previous, f"{question!r} after {previous!r} resolved to {resolved!r}"

    # The same phrase asked cold (no prior schedule/recommendation turn) still reads as a
    # graduation question -- the guard only applies when it is genuinely a continuation.
    assert main_module._resolve_intent(
        "15 krediye tamamla", "other", working_context=None
    ) == intents.GRADUATION_STATUS


def test_instructor_opinion_question_gets_a_warm_redirect_not_a_dry_disabled_message():
    # Regression (section 1.6 of the course-advising spec): a gossip/opinion question about an
    # instructor used to surface "Course reviews are not enabled." -- an internal, feature-flag
    # sounding message. It must instead be a warm, human redirect, and never mention the old
    # internal-sounding wording.
    response = TestClient(main_module.app).post(
        "/ask/", data={"question": "Yücel Saygın nasıl biri, zor mu?", "mode": "hybrid_meta"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "review_disabled"
    assert "not enabled" not in body["response"]
    assert "etkin değil" not in body["response"]
    assert body["response"].count("doğrulayabilirim") <= 1
