from __future__ import annotations

"""
Deterministic weekly-timetable builder (schedule option, "Haftalık ders programı yap").

Distinct from the times-less recommendation ("Hangi dersleri alayım"): this takes the balanced,
prerequisite-eligible course set from ``course_planner.build_plan`` and turns it into a REAL,
conflict-free weekly schedule using the official SUIS section/time data
(data/schedule/<TERM>.jsonl, one CRN per row).

Everything is computed in code:
  * one lecture (Primary) section per course is chosen so that no two meetings overlap
    (back-tracking search; falls back to a best-effort partial timetable if a course cannot be
    placed without a clash),
  * a matching recitation / lab / discussion component is added conflict-free where the course has
    one (preferring the same section-letter group as the chosen lecture),
  * TBA meetings (internship / project / thesis with no announced time) never conflict and are
    kept and clearly flagged.

The result carries the chosen CRNs so the UI can offer "copy CRNs" and a CRN-bearing Excel export.
Instructor/time data is shown exactly as SUIS published it; nothing here is invented.
"""

import json
import re
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from itertools import combinations
from pathlib import Path

from modules.config import DEGREE_DATA_DIR
from modules import course_planner


_SECONDARY_COMPONENTS = {"Recitation", "Laboratory", "Discussion", "Lab", "Recitation/Discussion"}
_DAY_ORDER = {"M": 0, "T": 1, "W": 2, "R": 3, "F": 4, "S": 5, "U": 6}
_DAY_TR = {"M": "Pzt", "T": "Salı", "W": "Çar", "R": "Perş", "F": "Cuma", "S": "Cmt", "U": "Paz"}
_DAY_EN = {"M": "Mon", "T": "Tue", "W": "Wed", "R": "Thu", "F": "Fri", "S": "Sat", "U": "Sun"}


def _schedule_dir() -> Path:
    return Path(DEGREE_DATA_DIR) / "schedule"


@lru_cache(maxsize=1)
def latest_term() -> str | None:
    """The current schedule snapshot's term, from the latest.json pointer."""
    pointer = _schedule_dir() / "latest.json"
    if not pointer.exists():
        return None
    try:
        return json.loads(pointer.read_text(encoding="utf-8")).get("term")
    except Exception:
        return None


def _to_min(hhmm: str | None) -> int | None:
    if not hhmm or ":" not in str(hhmm):
        return None
    h, m = str(hhmm).split(":")[:2]
    try:
        return int(h) * 60 + int(m)
    except ValueError:
        return None


@dataclass(frozen=True)
class Slot:
    day: str          # Banner day code M/T/W/R/F
    start: int        # minutes since midnight
    end: int


@dataclass(frozen=True)
class Meeting:
    """One official SUIS meeting, kept separately for calendar-grid consumers."""

    day_codes: tuple[str, ...] = ()
    day_names_tr: tuple[str, ...] = ()
    day_names_en: tuple[str, ...] = ()
    start_time: str | None = None
    end_time: str | None = None
    location: str = ""
    instructors: str = ""
    status: str = "tba"
    start_date: str | None = None
    end_date: str | None = None


@dataclass
class Section:
    course_id: str
    title: str
    crn: str
    section: str
    component: str
    instructors: str
    locations: str
    section_title: str = ""
    component_code: str = ""
    meetings: list[Meeting] = field(default_factory=list)
    slots: list[Slot] = field(default_factory=list)   # scheduled meetings only (TBA excluded)
    has_time: bool = True
    term_label: str = ""
    data_version: str = ""
    scraped_at: str = ""
    source_authority: str = "Sabanci University SUIS (BannerWeb)"
    source_url: str = ""

    @property
    def group(self) -> str:
        m = re.match(r"[A-Za-z]+", self.section or "")
        return m.group(0).upper() if m else ""

    def conflicts_with(self, other: "Section") -> bool:
        for a in self.slots:
            for b in other.slots:
                if a.day == b.day and a.start < b.end and b.start < a.end:
                    return True
        return False

    def conflicts_any(self, chosen: list["Section"]) -> bool:
        return any(self.conflicts_with(c) for c in chosen)


@lru_cache(maxsize=4)
def _sections_by_course(term: str) -> dict[str, list[Section]]:
    path = _schedule_dir() / f"{term}.jsonl"
    out: dict[str, list[Section]] = {}
    if not path.exists():
        return out
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            # One malformed source row must not make the entire published catalog unavailable.
            continue
        code = course_planner.normalize_code(row.get("course_id", ""))
        if not code:
            continue
        slots: list[Slot] = []
        meetings: list[Meeting] = []
        for meeting in row.get("meetings", []) or []:
            day_codes = tuple(str(day).upper() for day in (meeting.get("day_codes") or []) if day)
            start, end = _to_min(meeting.get("start_time")), _to_min(meeting.get("end_time"))
            if start is not None and end is not None and start < end:
                for day in day_codes:
                    slots.append(Slot(day, start, end))
            meetings.append(Meeting(
                day_codes=day_codes,
                day_names_tr=tuple(str(day) for day in (meeting.get("day_names_tr") or []) if day),
                day_names_en=tuple(str(day) for day in (meeting.get("day_names_en") or []) if day),
                start_time=str(meeting.get("start_time")) if meeting.get("start_time") else None,
                end_time=str(meeting.get("end_time")) if meeting.get("end_time") else None,
                location=str(meeting.get("where") or ""),
                instructors=str(meeting.get("instructors") or row.get("instructors") or ""),
                status=str(meeting.get("status") or ("scheduled" if start is not None else "tba")),
                start_date=str(meeting.get("start_date")) if meeting.get("start_date") else None,
                end_date=str(meeting.get("end_date")) if meeting.get("end_date") else None,
            ))
        out.setdefault(code, []).append(Section(
            course_id=row.get("course_id", ""),
            title=row.get("title") or row.get("section_title") or "",
            crn=str(row.get("crn", "")),
            section=str(row.get("section", "")),
            component=row.get("component") or "Primary",
            instructors=row.get("instructors") or "",
            locations=row.get("locations") or "",
            section_title=row.get("section_title") or row.get("title") or "",
            component_code=row.get("component_code") or "",
            meetings=meetings,
            slots=slots,
            has_time=bool(slots),
            term_label=row.get("term_label") or term,
            data_version=row.get("data_version") or "",
            scraped_at=row.get("scraped_at") or "",
            source_authority=row.get("source_authority") or "Sabanci University SUIS (BannerWeb)",
            source_url=row.get("source_url") or "",
        ))
    return out


@dataclass
class TimetableResult:
    term: str
    term_label: str
    placed: list[tuple[Section, list[Section]]] = field(default_factory=list)   # (lecture, [extras])
    not_offered: list[str] = field(default_factory=list)       # recommended but no section this term
    unplaced: list[str] = field(default_factory=list)          # offered but no conflict-free slot
    minimum_su_credits: int = 0

    @property
    def all_sections(self) -> list[Section]:
        out: list[Section] = []
        for lecture, extras in self.placed:
            out.append(lecture)
            out.extend(extras)
        return out

    @property
    def crns(self) -> list[str]:
        return [s.crn for s in self.all_sections if s.crn]

    @property
    def placed_su_credits(self) -> int:
        return sum(
            int((course_planner.resolve(lecture.course_id) or {}).get("su_credits") or 0)
            for lecture, _ in self.placed
        )

    @property
    def credit_shortfall(self) -> int:
        return max(0, int(self.minimum_su_credits) - self.placed_su_credits)


def _primaries(sections: list[Section]) -> list[Section]:
    prim = [s for s in sections if s.component.lower() == "primary"]
    return prim or sections


def _secondaries(sections: list[Section]) -> list[Section]:
    return [s for s in sections if s.component in _SECONDARY_COMPONENTS]


# A course's candidate "block": one lecture + an optional matching recitation/lab, already
# internally conflict-free. The whole timetable is a set of mutually conflict-free blocks.
Block = tuple[Section, list[Section]]


def _candidate_blocks(sections: list[Section]) -> list[Block]:
    """Every (lecture, [recitation?]) combination that is internally conflict-free."""
    primaries, secondaries = _primaries(sections), _secondaries(sections)
    blocks: list[Block] = []
    for lecture in primaries:
        if not secondaries:
            blocks.append((lecture, []))
            continue
        # Recitations belonging to this lecture's section group (A -> A1/A2); fall back to all
        # when the lecture has no letter (single "0" primary).
        group = [s for s in secondaries if s.group == lecture.group] or secondaries
        fitting = [s for s in group if not s.conflicts_with(lecture)]
        if fitting:
            blocks.extend((lecture, [s]) for s in fitting)
        # If an official course has a secondary component, silently dropping it creates an
        # invalid schedule.  No complete block means this lecture choice is unavailable.
    return blocks


def _block_sections(block: Block) -> list[Section]:
    return [block[0], *block[1]]


def _block_conflicts(block: Block, placed: list[Section]) -> bool:
    return any(s.conflicts_with(p) for s in _block_sections(block) for p in placed)


def build_timetable(recommended_codes: list[str], term: str | None = None) -> TimetableResult:
    term = term or latest_term() or "202601"
    catalog = _sections_by_course(term)
    codes = [course_planner.normalize_code(c) for c in recommended_codes if c]

    course_blocks: dict[str, list[Block]] = {}
    not_offered: list[str] = []
    term_label = term
    for code in codes:
        secs = catalog.get(code, [])
        if not secs:
            not_offered.append(code)
            continue
        term_label = secs[0].term_label or term_label
        course_blocks[code] = _candidate_blocks(secs)

    # Joint back-tracking over whole blocks: lectures AND recitations/labs are placed together so
    # a later lecture can never overlap an earlier course's lab (the bug a two-phase pass hides).
    order = sorted(course_blocks, key=lambda c: len(course_blocks[c]))

    def backtrack(i: int, chosen: dict[str, Block], placed: list[Section]) -> dict[str, Block] | None:
        if i == len(order):
            return dict(chosen)
        code = order[i]
        for block in course_blocks[code]:
            if not _block_conflicts(block, placed):
                chosen[code] = block
                got = backtrack(i + 1, chosen, placed + _block_sections(block))
                if got is not None:
                    return got
                del chosen[code]
        return None

    assignment = backtrack(0, {}, [])
    unplaced: list[str] = []
    if assignment is None:
        # No fully conflict-free timetable: greedily place what fits, report the rest.
        assignment = {}
        placed: list[Section] = []
        for code in order:
            block = next((b for b in course_blocks[code] if not _block_conflicts(b, placed)), None)
            if block is not None:
                assignment[code] = block
                placed += _block_sections(block)
            else:
                unplaced.append(code)

    result = TimetableResult(term=term, term_label=term_label, not_offered=not_offered, unplaced=unplaced)
    for code in codes:  # preserve the requested course order in the output
        if code in assignment:
            lecture, extras = assignment[code]
            result.placed.append((lecture, extras))
    return result


def build_timetable_for_load(
    candidate_codes: list[str],
    *,
    target_courses: int = 5,
    minimum_su_credits: int = 15,
    term: str | None = None,
) -> TimetableResult:
    """Choose the smallest conflict-free candidate subset that satisfies the load target.

    A course-level plan can meet 18 SU while a particular six-course combination has no complete
    section assignment. Trying bounded subsets from the deterministic candidate pool prevents the
    weekly page from silently dropping below the credit promise after timetable placement.
    """
    codes = list(dict.fromkeys(course_planner.normalize_code(code) for code in candidate_codes if code))
    if not codes:
        result = build_timetable([], term=term)
        result.minimum_su_credits = max(0, int(minimum_su_credits))
        return result

    target_courses = min(max(1, int(target_courses)), len(codes))
    minimum_su_credits = max(0, int(minimum_su_credits))

    def course_su(code: str) -> int:
        record = course_planner.resolve(code) or {}
        value = record.get("su_credits")
        return int(value) if isinstance(value, (int, float)) else 0

    def placed_score(result: TimetableResult) -> tuple[int, int]:
        placed_codes = [course_planner.normalize_code(lecture.course_id) for lecture, _ in result.placed]
        return sum(course_su(code) for code in placed_codes), len(placed_codes)

    best = build_timetable(codes[:target_courses], term=term)
    for size in range(target_courses, len(codes) + 1):
        for chosen in combinations(codes, size):
            if sum(course_su(code) for code in chosen) < minimum_su_credits:
                continue
            result = build_timetable(list(chosen), term=term)
            if placed_score(result) > placed_score(best):
                best = result
            if (
                len(result.placed) == size
                and not result.not_offered
                and not result.unplaced
                and placed_score(result)[0] >= minimum_su_credits
            ):
                result.minimum_su_credits = minimum_su_credits
                return result
    best.minimum_su_credits = minimum_su_credits
    return best


# ---------------------------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------------------------

_COMPONENT_TR = {
    "Primary": "Ders", "Recitation": "Problem Saati", "Laboratory": "Laboratuvar",
    "Discussion": "Tartışma", "Lab": "Laboratuvar",
}


def _slots_text(section: Section, language: str = "tr") -> str:
    if not section.slots:
        return "TBA (saat açıklanmadı)" if language == "tr" else "TBA (time not announced)"
    day_map = _DAY_TR if language == "tr" else _DAY_EN
    parts = []
    for slot in sorted(section.slots, key=lambda s: (_DAY_ORDER.get(s.day, 9), s.start)):
        hh = f"{slot.start // 60:02d}:{slot.start % 60:02d}-{slot.end // 60:02d}:{slot.end % 60:02d}"
        parts.append(f"{day_map.get(slot.day, slot.day)} {hh}")
    return ", ".join(parts)


def _component_label(component: str, language: str) -> str:
    if language == "tr":
        return _COMPONENT_TR.get(component, component)
    return component


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _meeting_payload(meeting: Meeting) -> dict:
    """Stable, locale-independent meeting shape consumed by the weekly calendar grid."""
    return {
        "day_codes": list(meeting.day_codes),
        "day_names_tr": list(meeting.day_names_tr),
        "day_names_en": list(meeting.day_names_en),
        "start_time": meeting.start_time,
        "end_time": meeting.end_time,
        "location": meeting.location,
        "instructors": meeting.instructors,
        "status": meeting.status,
        "start_date": meeting.start_date,
        "end_date": meeting.end_date,
    }


def section_payload(section: Section, *, language: str = "tr") -> dict:
    """Serialize one official section without losing per-meeting rooms or TBA state."""
    return {
        "course_id": course_planner.display_code(section.course_id),
        "title": section.title,
        "section_title": section.section_title,
        "crn": section.crn,
        "section": section.section,
        "component": section.component,
        "component_code": section.component_code,
        "component_label": _component_label(section.component, language),
        "instructors": section.instructors,
        "locations": section.locations,
        "tba": not section.has_time,
        "meetings": [_meeting_payload(meeting) for meeting in section.meetings],
        "source_url": section.source_url,
    }


def timetable_payload(
    result: TimetableResult,
    *,
    language: str = "tr",
    generated_at: str | None = None,
) -> dict:
    """Canonical schedule contract used by chat, persistence, and the schedule page.

    This object comes directly from the deterministic section assignment. It is deliberately
    separate from ``render_timetable`` so clients never need to regex-parse assistant prose.
    """
    courses: list[dict] = []
    for lecture, extras in result.placed:
        sections = [lecture, *extras]
        courses.append({
            "course_id": course_planner.display_code(lecture.course_id),
            "title": lecture.title,
            "sections": [section_payload(section, language=language) for section in sections],
        })

    first = result.all_sections[0] if result.all_sections else None
    return {
        "schema_version": 1,
        "term": result.term,
        "term_label": result.term_label,
        "generated_at": generated_at or _utc_now_iso(),
        "origin": "chatbot",
        "source": {
            "name": first.source_authority if first else "Sabanci University SUIS (BannerWeb)",
            "authority": "official",
            "snapshot_date": first.data_version if first else "",
            "scraped_at": first.scraped_at if first else "",
        },
        "courses": courses,
        "crns": result.crns,
        "not_offered": [course_planner.display_code(code) for code in result.not_offered],
        "unplaced": [course_planner.display_code(code) for code in result.unplaced],
        "placed_su_credits": result.placed_su_credits,
        "minimum_su_credits": result.minimum_su_credits,
        "credit_shortfall": result.credit_shortfall,
        # A non-empty list is reserved for future manual-edit validation. Auto-generated results
        # are conflict-free by construction; TBA sections do not occupy a time interval.
        "conflicts": [],
    }


def _validated_term(term: str | None) -> str:
    selected = str(term or latest_term() or "202601").strip()
    if not re.fullmatch(r"\d{6}", selected):
        raise ValueError("term must be a six-digit SUIS term code")
    return selected


def _catalog_sections(term: str) -> list[Section]:
    catalog = _sections_by_course(term)
    sections = [section for course_sections in catalog.values() for section in course_sections]
    return sorted(
        sections,
        key=lambda section: (
            course_planner.normalize_code(section.course_id),
            section.component.lower() != "primary",
            section.section,
            section.crn,
        ),
    )


def _matches_search(section: Section, search: str) -> bool:
    query = " ".join(str(search or "").strip().casefold().split())
    if not query:
        return True
    haystack = " ".join(
        part.casefold()
        for part in (
            section.course_id,
            section.title,
            section.section_title,
            section.crn,
            section.component,
            section.instructors,
        )
        if part
    )
    if query in haystack:
        return True
    # Users commonly type CS204 while the official source stores CS 204.
    compact_query = re.sub(r"[^a-z0-9]", "", query)
    compact_code = re.sub(r"[^a-z0-9]", "", section.course_id.casefold())
    return bool(compact_query and compact_query in compact_code)


def search_sections_payload(
    *,
    term: str | None = None,
    search: str = "",
    limit: int = 80,
    language: str = "tr",
) -> dict:
    """Search the immutable official snapshot with a bounded response size."""
    selected_term = _validated_term(term)
    bounded_limit = min(max(int(limit), 1), 200)
    sections = _catalog_sections(selected_term)
    if not sections and not (_schedule_dir() / f"{selected_term}.jsonl").exists():
        raise FileNotFoundError(f"No schedule snapshot for term {selected_term}")
    matches = [section for section in sections if _matches_search(section, search)]
    term_label = matches[0].term_label if matches else (sections[0].term_label if sections else selected_term)
    return {
        "term": selected_term,
        "term_label": term_label,
        "total": len(matches),
        "limit": bounded_limit,
        "has_more": len(matches) > bounded_limit,
        "sections": [section_payload(section, language=language) for section in matches[:bounded_limit]],
    }


def course_sections_payload(
    course_code: str,
    *,
    term: str | None = None,
    language: str = "tr",
) -> dict:
    """All official primary/secondary options for one exact course selection."""
    selected_term = _validated_term(term)
    normalized = course_planner.normalize_code(course_code)
    if not normalized:
        raise ValueError("course_code is required")
    catalog = _sections_by_course(selected_term)
    if not catalog and not (_schedule_dir() / f"{selected_term}.jsonl").exists():
        raise FileNotFoundError(f"No schedule snapshot for term {selected_term}")
    sections = catalog.get(normalized, [])
    if not sections:
        raise LookupError(f"Course {course_planner.display_code(normalized)} is not offered in {selected_term}")
    ordered = sorted(
        sections,
        key=lambda section: (section.component.lower() != "primary", section.section, section.crn),
    )
    return {
        "term": selected_term,
        "term_label": ordered[0].term_label or selected_term,
        "course_id": course_planner.display_code(ordered[0].course_id),
        "title": ordered[0].title,
        "sections": [section_payload(section, language=language) for section in ordered],
    }


def validate_schedule_payload(payload: dict) -> dict:
    """Validate and canonicalize a manually edited schedule before persistence.

    CRNs, TBA flags, and conflicts are recomputed from nested sections/meetings so clients cannot
    accidentally leave stale derived values after an edit.
    """
    clean = deepcopy(payload)
    term = str(clean.get("term") or "").strip()
    if not re.fullmatch(r"\d{6}", term):
        raise ValueError("schedule.term must be a six-digit SUIS term code")
    clean["term"] = term
    if isinstance(clean.get("source"), str):
        clean["origin"] = str(clean.get("source") or "manual")
        clean["source"] = {
            "name": "Sabanci University SUIS (BannerWeb)",
            "authority": "official",
            "snapshot_date": "",
            "scraped_at": "",
        }
    clean.setdefault("origin", "manual")

    allowed_days = set(_DAY_ORDER)
    crns: list[str] = []
    seen_crns: set[str] = set()
    events: list[dict] = []
    for course in clean.get("courses") or []:
        course_id = course_planner.display_code(str(course.get("course_id") or ""))
        if not course_id:
            raise ValueError("Every schedule course must have course_id")
        course["course_id"] = course_id
        for section_index, section in enumerate(course.get("sections") or []):
            section["course_id"] = course_id
            crn = str(section.get("crn") or "").strip()
            section["crn"] = crn
            if crn:
                if crn in seen_crns:
                    raise ValueError(f"Duplicate CRN in schedule: {crn}")
                seen_crns.add(crn)
                crns.append(crn)
            event_id = crn or f"{course_id}:{section.get('section') or section_index}"
            scheduled = False
            for meeting in section.get("meetings") or []:
                days = [str(day).upper() for day in (meeting.get("day_codes") or [])]
                if any(day not in allowed_days for day in days):
                    raise ValueError(f"Invalid day code for CRN {crn}")
                meeting["day_codes"] = list(dict.fromkeys(days))
                start_text, end_text = meeting.get("start_time"), meeting.get("end_time")
                start, end = _to_min(start_text), _to_min(end_text)
                if (start is None) != (end is None):
                    raise ValueError(f"CRN {crn} meeting must have both start_time and end_time")
                if start is None:
                    continue
                if start >= end or not days:
                    raise ValueError(f"Invalid timed meeting for CRN {crn}")
                scheduled = True
                for day in days:
                    events.append({
                        "course_id": course_id,
                        "crn": crn,
                        "event_id": event_id,
                        "day_code": day,
                        "start": start,
                        "end": end,
                    })
            section["tba"] = not scheduled

    conflicts: list[dict] = []
    for index, first in enumerate(events):
        for second in events[index + 1:]:
            if first["event_id"] == second["event_id"] or first["day_code"] != second["day_code"]:
                continue
            if first["start"] < second["end"] and second["start"] < first["end"]:
                overlap_start = max(first["start"], second["start"])
                overlap_end = min(first["end"], second["end"])
                conflicts.append({
                    "first_course_id": first["course_id"],
                    "first_crn": first["crn"],
                    "second_course_id": second["course_id"],
                    "second_crn": second["crn"],
                    "day_code": first["day_code"],
                    "start_time": f"{overlap_start // 60:02d}:{overlap_start % 60:02d}",
                    "end_time": f"{overlap_end // 60:02d}:{overlap_end % 60:02d}",
                })

    clean["schema_version"] = 1
    clean["crns"] = crns
    clean["conflicts"] = conflicts
    if not clean.get("generated_at"):
        clean["generated_at"] = _utc_now_iso()
    clean.setdefault("not_offered", [])
    clean.setdefault("unplaced", [])
    return clean


def timetable_structured_content(result: TimetableResult, *, language: str = "tr") -> dict | None:
    if not result.placed:
        return None
    rows = []
    for lecture, extras in result.placed:
        for section in [lecture, *extras]:
            rec = course_planner.resolve(section.course_id) or {}
            rows.append({
                "course": rec.get("display_code", course_planner.display_code(section.course_id)),
                "title": section.title or rec.get("official_name", ""),
                "component": _component_label(section.component, language),
                "crn": section.crn,
                "section": section.section,
                "schedule": _slots_text(section, language),
                "location": section.locations,
                "instructor": section.instructors,
            })
    title = (f"Haftalık Program — {result.term_label}" if language == "tr"
             else f"Weekly Schedule — {result.term_label}")
    return {
        "kind": "course_schedule",
        "term": result.term,
        "crns": result.crns,
        "tables": [{
            "id": "weekly-schedule",
            "title": title,
            "columns": [
                {"key": "course", "label": "Ders" if language == "tr" else "Course"},
                {"key": "component", "label": "Tür" if language == "tr" else "Type"},
                {"key": "crn", "label": "CRN"},
                {"key": "section", "label": "Section"},
                {"key": "schedule", "label": "Gün / Saat" if language == "tr" else "Day / Time"},
                {"key": "location", "label": "Yer" if language == "tr" else "Location"},
                {"key": "instructor", "label": "Öğretim üyesi" if language == "tr" else "Instructor"},
            ],
            "rows": rows,
            "exportable": True,
        }],
    }


def render_timetable(result: TimetableResult, *, language: str = "tr") -> tuple[str, str]:
    tr = language == "tr"
    if not result.placed:
        msg = (
            f"{result.term_label} için önerilen derslerin açılış/saat bilgisini bulamadım."
            if tr else
            f"I could not find offered sections for the recommended courses in {result.term_label}."
        )
        return msg, msg

    lines: list[str] = []
    lines.append(
        f"**{result.term_label}** için çakışmasız, dengeli haftalık ders programın:" if tr else
        f"Your conflict-free, balanced weekly schedule for **{result.term_label}**:"
    )
    lines.append("")
    for lecture, extras in result.placed:
        head = (
            f"- **{course_planner._fmt(lecture.course_id)}** — CRN {lecture.crn} · Section "
            f"{lecture.section} · {_slots_text(lecture, language)}"
        )
        if lecture.instructors:
            head += f" · {lecture.instructors}"
        if lecture.locations:
            head += f" · {lecture.locations}"
        lines.append(head)
        for extra in extras:
            lines.append(
                f"   - {_component_label(extra.component, language)}: CRN {extra.crn} · Section "
                f"{extra.section} · {_slots_text(extra, language)}"
                + (f" · {extra.locations}" if extra.locations else "")
            )

    crns = result.crns
    lines.append("")
    lines.append(
        f"Toplam {len(result.placed)} ders, {len(crns)} CRN." if tr else
        f"{len(result.placed)} courses, {len(crns)} CRNs."
    )
    lines.append(("Kayıt için CRN'ler: " if tr else "CRNs for registration: ") + " ".join(crns))

    if result.not_offered:
        names = ", ".join(course_planner.display_code(c) for c in result.not_offered)
        lines.append("")
        lines.append(
            f"Bu dönem açılmayan (program dışı bırakılan) dersler: {names}." if tr else
            f"Not offered this term (left out of the schedule): {names}."
        )
    if result.unplaced:
        names = ", ".join(course_planner.display_code(c) for c in result.unplaced)
        lines.append(
            f"Çakışma nedeniyle otomatik yerleştiremediğim dersler (alternatif section seçmelisin): {names}."
            if tr else
            f"Could not auto-place without a clash (pick an alternative section): {names}."
        )

    # The 15/18-SU floor is a product invariant (see course_planner.build_plan's docstring), but
    # a section-level timetable conflict can still make it unreachable even when enough eligible
    # *courses* exist. Say so explicitly rather than silently handing back a lighter load — the
    # same shortfall the deterministic plan itself would report if the timetable step weren't
    # involved at all.
    if result.credit_shortfall:
        lines.append("")
        lines.append(
            f"Çakışmasız yerleştirebildiğim toplam {result.placed_su_credits} SU — hedeflenen "
            f"{result.minimum_su_credits} SU'nun {result.credit_shortfall} SU altında. Bu dönem uygun "
            f"ve çakışmasız başka bir aday bulunamadı; danışmanınla ek seçenekleri değerlendir."
            if tr else
            f"I could only place a conflict-free {result.placed_su_credits} SU — {result.credit_shortfall} "
            f"SU short of the {result.minimum_su_credits} SU target. No other eligible, conflict-free "
            f"candidate was available this term; check additional options with your advisor."
        )

    summary = (
        f"{result.term_label} için {len(result.placed)} derslik çakışmasız program hazırladım "
        f"({len(crns)} CRN)."
        if tr else
        f"I built a conflict-free {len(result.placed)}-course schedule for {result.term_label} "
        f"({len(crns)} CRNs)."
    )
    return "\n".join(lines), summary
