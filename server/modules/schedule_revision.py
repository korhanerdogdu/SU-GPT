from __future__ import annotations

"""Deterministic edits to a previously generated weekly schedule.

The normal schedule builder answers a fresh "this term what should I take?" request.  A follow-up
like "remove ENS 208 from the schedule above and put something else" is a different task: keep the
existing context, preserve the still-valid courses, and only search for a replacement.
"""

from dataclasses import dataclass
from typing import Any

from modules import course_planner, schedule_planner


@dataclass(frozen=True)
class RevisionResult:
    body: str
    summary: str
    timetable: schedule_planner.TimetableResult
    schedule_payload: dict[str, Any]
    removed_codes: list[str]
    added_codes: list[str]
    sources: list[str]


def schedule_course_codes(schedule: dict[str, Any]) -> list[str]:
    codes: list[str] = []
    for course in schedule.get("courses") or []:
        code = course_planner.display_code(str(course.get("course_id") or ""))
        if code and code not in codes:
            codes.append(code)
    return codes


def revise_saved_schedule(
    *,
    saved_schedule: dict[str, Any],
    excluded_codes: set[str],
    program: str,
    curriculum_term: str,
    completed_codes: list[str],
    interest_codes: list[str] | None = None,
    academic_year: int | None = None,
    language: str = "tr",
) -> RevisionResult | None:
    current_codes = schedule_course_codes(saved_schedule)
    if not current_codes or not excluded_codes:
        return None

    excluded_normalized = {course_planner.normalize_code(code) for code in excluded_codes}
    removed = [code for code in current_codes if course_planner.normalize_code(code) in excluded_normalized]
    if not removed:
        return None

    retained = [code for code in current_codes if code not in removed]
    target_courses = len(current_codes)
    minimum_su = int(saved_schedule.get("minimum_su_credits") or saved_schedule.get("placed_su_credits") or 15)
    term = str(saved_schedule.get("term") or schedule_planner.latest_term() or "202601")

    plan = course_planner.build_plan(
        program,
        curriculum_term,
        completed_codes,
        interest_codes or [],
        target=max(target_courses + 10, 14),
        minimum_su_credits=minimum_su,
        exact_course_count=False,
        max_courses=24,
        balance_by_subject=False,
        academic_year=academic_year,
    )
    if not plan.has_official_data:
        return None

    blocked = {course_planner.normalize_code(code) for code in [*retained, *removed]}
    replacement_candidates = [
        course_planner.display_code(item.code)
        for item in plan.recommended
        if course_planner.normalize_code(item.code) not in blocked
    ]
    if not replacement_candidates:
        return None

    candidate_codes = retained + replacement_candidates
    timetable = schedule_planner.build_timetable_for_load(
        candidate_codes,
        target_courses=target_courses,
        minimum_su_credits=minimum_su,
        term=term,
        mandatory_codes=retained,
    )
    placed_codes = [
        course_planner.display_code(section.course_id)
        for section, _extras in timetable.placed
    ]
    if any(course_planner.normalize_code(code) in excluded_normalized for code in placed_codes):
        return None

    added = [code for code in placed_codes if code not in retained]
    schedule_payload = schedule_planner.timetable_payload(timetable, language=language)
    schedule_payload["removed_codes"] = removed
    schedule_payload["added_codes"] = added
    schedule_payload["excluded_codes"] = [
        course_planner.display_code(code) for code in sorted(excluded_normalized)
    ]
    base_body, base_summary = schedule_planner.render_timetable(timetable, language=language)

    if language == "en":
        prefix = (
            f"I removed {', '.join(removed)} from the previous schedule"
            + (f" and added {', '.join(added)}." if added else " but could not find a conflict-free replacement.")
        )
        summary = (
            f"Removed {', '.join(removed)}"
            + (f"; added {', '.join(added)}." if added else "; no replacement found.")
        )
    else:
        prefix = (
            f"Önceki programı baz aldım: {', '.join(removed)} dersini çıkardım"
            + (f" ve yerine {', '.join(added)} ekledim." if added else " ama çakışmasız uygun alternatif bulamadım.")
        )
        summary = (
            f"{', '.join(removed)} çıkarıldı"
            + (f"; {', '.join(added)} eklendi." if added else "; alternatif bulunamadı.")
        )

    return RevisionResult(
        body=f"{prefix}\n\n{base_body}",
        summary=f"{summary} {base_summary}",
        timetable=timetable,
        schedule_payload=schedule_payload,
        removed_codes=removed,
        added_codes=added,
        sources=[
            "Sabancı SUIS course schedule (official)",
            f"degree_requirements/{program}/{curriculum_term}.jsonl",
        ],
    )
