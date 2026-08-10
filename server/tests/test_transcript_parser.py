from __future__ import annotations

"""
Unit tests for modules.transcript_parser, built from the exact layout quirks a real Sabanci
"Academic Records Summary" PDF exposed once run through pypdf's text extraction: a title glued
directly to the LEVEL token with no space, a title that wraps onto its own line before the
LEVEL/GRADE/CREDIT/ECTS tail, a repeated course whose earlier grade must NOT count, and a
withdrawn course that was never retaken.

No real transcript (a student's name/ID/grades) is checked into the repository -- this is
synthetic text in the same shape pypdf produces, not a real person's academic record. The parser
was separately validated against a real transcript during development (55/55 attempts, 45/45
courses, and a computed GPA matching the reported CGPA exactly); these tests pin the underlying
regex/dedup behaviour so that validation can't silently regress.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("MONGO_URI", "mongodb://localhost:27017")

from modules.transcript_parser import (
    _latest_program_hint,
    _parse_attempts,
    _term_blocks,
    parse_transcript_text,
    term_sort_key,
)


def _pages(*page_texts: str) -> list[str]:
    return list(page_texts)


def test_term_sort_key_orders_fall_before_spring_before_summer_same_year():
    assert term_sort_key("Fall 2022-2023") < term_sort_key("Spring 2022-2023")
    assert term_sort_key("Spring 2022-2023") < term_sort_key("Summer 2022-2023")
    assert term_sort_key("Summer 2022-2023") < term_sort_key("Fall 2023-2024")


def test_row_with_title_glued_to_level_token():
    pages = _pages(
        "Fall 2022-2023 Status : Active / Level : Undergraduate\n"
        "Program : Computer Science and Engineering (FENS)\n"
        "COURSE CODE COURSE TITLE LEVEL GRADE CREDIT ECTS STATUS\n"
        "CS 201 Introduction to ComputingUG A- 3.00 6.00\n"
        "Standing: Satisfactory\n"
    )
    blocks = _term_blocks(pages)
    attempts = _parse_attempts(blocks)
    assert len(attempts) == 1
    a = attempts[0]
    assert a.course_code == "CS 201"
    assert a.title == "Introduction to Computing"
    assert a.grade == "A-"
    assert a.credit == 3.0
    assert a.ects == 6.0
    assert a.term == "Fall 2022-2023"
    assert a.status is None


def test_row_with_title_wrapped_onto_its_own_line():
    pages = _pages(
        "Spring 2021-2022 Status : Active / Level : Foundation Development Year\n"
        "Program : Programs of Management (SBS)\n"
        "COURSE CODE COURSE TITLE LEVEL GRADE CREDIT ECTS STATUS\n"
        "ELAE 000 Eng. Language Assessment\n"
        "Exam\n"
        "FDY EL 0.00 0.00 Excluded\n"
        "Standing: Satisfactory\n"
    )
    attempts = _parse_attempts(_term_blocks(pages))
    assert len(attempts) == 1
    a = attempts[0]
    assert a.course_code == "ELAE 000"
    assert a.title == "Eng. Language Assessment Exam"
    assert a.grade == "EL"
    assert a.status == "Excluded"


def test_term_table_continues_across_a_page_break_without_a_new_header():
    # Sabanci's export splits a long term's course table mid-page; the continuation page has no
    # repeated "COURSE CODE..." header and no repeated term line -- rows must still attach to the
    # term that was active when the page ended.
    page_a = (
        "Spring 2025-2026 Status : Active / Level : Undergraduate\n"
        "Program : Computer Science and Engineering (FENS)\n"
        "COURSE CODE COURSE TITLE LEVEL GRADE CREDIT ECTS STATUS\n"
        "CS 310 Mobile Application\n"
        "Development\n"
        "UG B- 3.00 6.00\n"
    )
    page_b = (
        "CS 455 LLMs: Foundations and\n"
        "Practice\n"
        "UG D 3.00 6.00\n"
        "Standing: Satisfactory\n"
    )
    attempts = _parse_attempts(_term_blocks(_pages(page_a, page_b)))
    codes = [a.course_code for a in attempts]
    assert codes == ["CS 310", "CS 455"]
    assert all(a.term == "Spring 2025-2026" for a in attempts)
    assert attempts[1].title == "LLMs: Foundations and Practice"


def test_repeated_course_keeps_only_the_latest_grade():
    pages = _pages(
        "Spring 2023-2024 Status : Active / Level : Undergraduate\n"
        "Program : Computer Science and Engineering (FENS)\n"
        "COURSE CODE COURSE TITLE LEVEL GRADE CREDIT ECTS STATUS\n"
        "CS 301 Algorithms UG F 3.00 6.00 Repeated\n"
        "Standing: Satisfactory\n"
        "Fall 2024-2025 Status : Active / Level : Undergraduate\n"
        "Program : Computer Science and Engineering (FENS)\n"
        "COURSE CODE COURSE TITLE LEVEL GRADE CREDIT ECTS STATUS\n"
        "CS 301 Algorithms UG B- 3.00 6.00\n"
        "Standing: Satisfactory\n"
    )
    result = parse_transcript_text(pages)
    assert len(result.attempts) == 2  # both attempts are preserved in history
    assert len(result.courses) == 1  # only one current record per course
    assert result.courses[0].grade == "B-"  # the later grade wins, not the F


def test_withdrawn_course_never_retaken_is_dropped_from_completed_courses():
    pages = _pages(
        "Fall 2025-2026 Status : Active / Level : Undergraduate\n"
        "Program : Computer Science and Engineering (FENS)\n"
        "COURSE CODE COURSE TITLE LEVEL GRADE CREDIT ECTS STATUS\n"
        "CS 419 Digital Image & Video Analysis UG W 3.00 0.00\n"
        "Standing: Satisfactory\n"
    )
    result = parse_transcript_text(pages)
    assert len(result.attempts) == 1
    assert result.courses == []  # withdrawn, never retaken -> not a completed course


def test_satisfactory_graded_course_counts_as_completed_but_not_toward_gpa():
    pages = _pages(
        "Summer 2024-2025 Status : Active / Level : Undergraduate\n"
        "Program : Computer Science and Engineering (FENS)\n"
        "COURSE CODE COURSE TITLE LEVEL GRADE CREDIT ECTS STATUS\n"
        "CS 395 Internship Project UG S 0.00 5.00\n"
        "Standing: Satisfactory\n"
    )
    result = parse_transcript_text(pages)
    assert len(result.courses) == 1
    assert result.courses[0].grade == "S"
    assert result.courses[0].counts_in_gpa is False


def test_gpa_is_credit_weighted_average_of_only_gpa_eligible_courses():
    pages = _pages(
        "Fall 2022-2023 Status : Active / Level : Undergraduate\n"
        "Program : Computer Science and Engineering (FENS)\n"
        "COURSE CODE COURSE TITLE LEVEL GRADE CREDIT ECTS STATUS\n"
        "CS 201 Introduction to Computing UG A 3.00 6.00\n"
        "PROJ 201 Undergraduate Project Course UG B 1.00 1.00\n"
        "CIP 101N Civic Involvement Projects I-N UG S 0.00 1.00\n"
        "Standing: Satisfactory\n"
    )
    result = parse_transcript_text(pages)
    # (4.0*3 + 3.0*1) / (3+1) = 3.75; the 0-credit S course must not enter the average at all.
    assert result.gpa == 3.75


def test_program_hint_is_the_last_program_line_not_the_first():
    # Regression: an internal-transfer student's early terms show their OLD major first; only
    # the most recently printed "Program :" line (terms are chronological) is the current one.
    pages = _pages(
        "Fall 2021-2022 Status : Active / Level : Foundation Development Year\n"
        "Program : Programs of Management (SBS)\n"
        "Standing: Satisfactory\n"
        "Summer 2022-2023 Status : Active / Level : Undergraduate\n"
        "Program : Computer Science and Engineering (FENS)\n"
        "COURSE CODE COURSE TITLE LEVEL GRADE CREDIT ECTS STATUS\n"
        "MATH 203 Introduction to Probability UG A 3.00 6.00\n"
        "Standing: Satisfactory\n"
    )
    assert _latest_program_hint(pages) == "Computer Science and Engineering (FENS)"
    result = parse_transcript_text(pages)
    assert result.program_hint == "Computer Science and Engineering (FENS)"


def test_program_hint_is_none_when_no_term_block_is_present():
    assert _latest_program_hint(_pages("Just some unrelated preamble text.\n")) is None

