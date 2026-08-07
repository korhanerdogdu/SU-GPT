from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("MONGO_URI", "mongodb://localhost:27017")

from modules import gpa_planner as gp


def test_specific_scenario_turkish_grade_after_code():
    result = gp.parse_specific_scenarios("CS 455'ten B+ alsam, ENS 211'den C alsam ne olur?")
    assert result == [("CS 455", "B+"), ("ENS 211", "C")]


def test_specific_scenario_english_grade_before_code():
    result = gp.parse_specific_scenarios("What if I get a B+ in CS 455?")
    assert result == [("CS 455", "B+")]


def test_specific_scenario_does_not_confuse_if_with_f_grade():
    # "if" contains a lowercase "f"; case-sensitive grade matching must not read it as an F.
    assert gp.parse_specific_scenarios("What if I take CS 455?") == []


def test_specific_scenario_plus_minus_suffix_is_not_truncated():
    # Regression: a naive trailing `\b` breaks on "B+"/"A-" because +/- are non-word characters.
    assert gp.parse_specific_scenarios("CS 303'ten A- alırsam ne olur?") == [("CS 303", "A-")]


def test_bulk_scenario_turkish():
    assert gp.parse_bulk_scenario("6 ders alıp hepsinden A alırsam GPA'm kaça çıkar?") == (6, "A")


def test_bulk_scenario_does_not_confuse_hepsinden_with_d_grade():
    # Regression: case-insensitive matching let a bare "D" grade match the lowercase "d" inside
    # "hepsinden" itself, before ever reaching the real "A" later in the sentence.
    count, grade = gp.parse_bulk_scenario("6 ders alıp hepsinden A alırsam GPA'm kaça çıkar?")
    assert grade == "A"


def test_bulk_scenario_english():
    assert gp.parse_bulk_scenario("5 course all A") == (5, "A")


def test_is_gpa_projection_query_true_for_keyword_and_scenario_forms():
    for q in (
        "Bu dönem GPA'mı ne kadar yükseltebilirim?",
        "What if I get a B+ in CS 455?",  # no "GPA" keyword at all
        "6 ders alıp hepsinden A alırsam GPA'm kaça çıkar?",
    ):
        assert gp.is_gpa_projection_query(q), q


def test_is_gpa_projection_query_false_for_unrelated_questions():
    for q in (
        "Mezuniyet durumumu hesapla",
        "Bu dönem hangi dersleri alayım?",
        "CS 306 zorunlu mu?",
        "Merhaba nasılsın",
        "What courses should I take this term?",
    ):
        assert not gp.is_gpa_projection_query(q), q


def test_project_gpa_credit_weighted():
    # current: 3.0 GPA over 30 credits (90 points). Add one A (4.0) at 3 credits.
    projected = gp.project_gpa(current_points=90.0, current_credit=30.0, additions=[(3.0, "A")])
    assert projected == round((90.0 + 4.0 * 3) / 33.0, 2)


def test_project_gpa_with_no_credits_returns_none():
    assert gp.project_gpa(current_points=0.0, current_credit=0.0, additions=[]) is None


def test_render_projection_specific_scenario_shows_each_course_and_new_gpa():
    body = gp.render_projection(
        current_gpa=3.40,
        current_points=90.0,
        current_credit=30.0,
        specific=[("CS 455", "B+", 3.0), ("ENS 211", "C", 3.0)],
        bulk=None,
        default_scenarios=False,
        language="tr",
    )
    # each course is its own markdown table row, not a "CODE: GRADE" bullet
    assert "| CS 455 | B+ | 3 |" in body
    assert "| ENS 211 | C | 3 |" in body
    assert "3.40" in body  # current GPA is echoed back


def test_render_projection_bulk_scenario():
    body = gp.render_projection(
        current_gpa=3.0, current_points=90.0, current_credit=30.0,
        specific=[], bulk=(6, "A"), default_scenarios=False, language="en",
    )
    assert "6 courses" in body
    assert "| 6 × A |" in body


def test_render_projection_default_scenarios_when_nothing_specific_parsed():
    body = gp.render_projection(
        current_gpa=3.0, current_points=90.0, current_credit=30.0,
        specific=[], bulk=None, default_scenarios=True, language="tr",
    )
    # a spread of common grades, not a request for clarification
    assert "GPA" in body
    for grade in ("A", "B+"):
        assert grade in body


def test_render_projection_display_credit_shows_true_total_not_the_adjusted_one():
    # A caller may pull an already-completed course's old grade back out of current_credit before
    # projecting (a re-grade replaces, it doesn't stack) -- but the "current GPA" line must still
    # show the student's real, unadjusted credit total, not the intermediate adjusted figure used
    # only for the arithmetic.
    body = gp.render_projection(
        current_gpa=3.44,
        current_points=430.0 - 4.0 * 3,   # adjusted: one 3-SU 'D' course pulled back out
        current_credit=125.0 - 3.0,
        specific=[("CS 455", "B+", 3.0)],
        bulk=None,
        default_scenarios=False,
        language="tr",
        display_credit=125.0,             # the true, unadjusted total
    )
    assert "125 SU" in body
    assert "122 SU" not in body


def test_render_projection_handles_no_current_gpa():
    body = gp.render_projection(
        current_gpa=None, current_points=0.0, current_credit=0.0,
        specific=[], bulk=(5, "A"), default_scenarios=False, language="tr",
    )
    assert "yok" in body  # "no GPA yet" rendered instead of crashing on None
