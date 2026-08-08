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


def test_minor_intent_matches_turkish_possessive_inflection():
    # Regression: MINOR_INTENT_RE's trailing \b required an immediate word boundary right after
    # "yandal", which a Turkish possessive suffix never leaves -- "yandalımı seçmek istiyorum"
    # ("I want to choose my minor") never matched at all.
    assert main_module.MINOR_INTENT_RE.search("Yandalımı seçmek istiyorum")
    assert main_module.MINOR_INTENT_RE.search("yandal başvurusu ne zaman?")
    assert main_module.MINOR_INTENT_RE.search("minor requirements?")


def test_heavy_negation_matches_natural_word_order_not_only_direct_adjacency():
    # Regression: HEAVY_NEGATION_RE required the heavy-word and its negation to sit immediately
    # next to each other. Natural phrasing almost always has a noun phrase between them --
    # "ağır bir program istemiyorum" (heavy-word before negation) and "I do not want a heavy
    # load" (negation before heavy-word) both failed to match, so a student explicitly asking
    # for a *lighter* term was read as requesting a heavier one (the opposite of what they said).
    negated = (
        "ağır bir program istemiyorum",
        "yoğun bir dönem istemiyorum",
        "I do not want a heavy load",
        "not a heavy term please",
        "don't give me an intensive schedule",
    )
    for text in negated:
        assert main_module.HEAVY_LOAD_RE.search(text), text
        assert main_module.HEAVY_NEGATION_RE.search(text), text

    not_negated = ("bu dönem yoğun bir program istiyorum", "I want a heavy course load")
    for text in not_negated:
        assert main_module.HEAVY_LOAD_RE.search(text), text
        assert not main_module.HEAVY_NEGATION_RE.search(text), text


def test_study_plan_major_selection_specialization_have_real_english_coverage():
    # Regression: these three previously had only literal Turkish phrasing plus one or two
    # hardcoded English fragments ("study plan", "major seç") -- an ordinary English question
    # fell through to the generic "other" RAG/LLM path instead of the deterministic intent,
    # an asymmetry with Turkish's much broader coverage (CLAUDE.md requires TR/EN symmetry).
    assert main_module.STUDY_PLAN_INTENT_RE.search("How should I study for the final?")
    assert main_module.STUDY_PLAN_INTENT_RE.search("I need to prepare for the midterm")
    assert main_module.MAJOR_SELECTION_INTENT_RE.search("Which major should I pick?")
    assert main_module.MAJOR_SELECTION_INTENT_RE.search("I want to choose my major")
    assert main_module.SPECIALIZATION_INTENT_RE.search(
        "which specialization should I pick, NLP or CV?"
    )
    assert main_module.SPECIALIZATION_INTENT_RE.search("I want to specialize in AI")


def test_course_detail_recognises_who_teaches_in_english():
    # "kim veriyor" ("who teaches it") had no English counterpart. Masked when a course code is
    # present (COURSE_CODE_RE's fallback in _is_course_detail_like already catches "who teaches
    # CS 306?"), but a real gap for a course referenced by name only.
    assert main_module.COURSE_DETAIL_INTENT_RE.search("Who teaches Database Systems?")
    assert main_module.COURSE_DETAIL_INTENT_RE.search("Who is the instructor for CS 306?")


def test_recommendation_intent_recognises_english_difficulty_cues():
    # kolay/rahat/zor/ağır/yoğun (easy/light/hard/heavy/intense) had no English counterpart.
    assert main_module.RECOMMENDATION_INTENT_RE.search("I want easy courses this term")
    assert main_module.RECOMMENDATION_INTENT_RE.search("give me a light course load")


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
