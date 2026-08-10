from __future__ import annotations

"""Convert Sabanci recommended-program PDFs into deterministic RAG JSONL.

The input PDFs are spreadsheet exports.  Their text layer is not consistently ordered,
so this script reads ruled table cells with ``pdfplumber`` instead of trying to split
plain text on whitespace.  It emits two retrieval granularities:

* one ``suggested_program_semester`` chunk per semester/summer period;
* one ``suggested_program_course`` chunk per course/elective slot;
* one ``suggested_program_track`` chunk per non-binding concentration suggestion.

Every row keeps the original file, page and SHA-256.  A missing value is ``null``; the
converter never guesses a prerequisite, credit or source URL.  The two small text-layer
repairs in ``_repair_source_artifacts`` are backed by the visible PDF table and are tested.

Usage (from the repository root):

    python server/scripts/ingest_suggested_programs.py \
        --input-dir C:/Users/mehme/Downloads/suggested_programs

The script requires pdfplumber (``python -m pip install pdfplumber``).  The generated
files are written under ``data/suggested_programs/<PROGRAM>/`` by default.
"""

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SERVER_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SERVER_ROOT.parent


@dataclass(frozen=True, slots=True)
class SourceSpec:
    filename: str
    program: str
    degree_code: str
    program_name: str
    output_stem: str
    plan_type: str = "standard"
    track: str = "standard"


SOURCE_SPECS: tuple[SourceSpec, ...] = (
    SourceSpec(
        "bio_courseplan.pdf",
        "BIO",
        "BSBIO",
        "Molecular Biology, Genetics and Bioengineering",
        "standard",
    ),
    SourceSpec(
        "cs_sampleprograms_v2.xlsx_-_fast_track_1_0.pdf",
        "CS",
        "BSCS",
        "Computer Science and Engineering",
        "fast_track_1",
        "fast_track",
        "fast_track_1",
    ),
    SourceSpec(
        "cs_sampleprograms_v2.xlsx_-_fast_track_2_1.pdf",
        "CS",
        "BSCS",
        "Computer Science and Engineering",
        "fast_track_2",
        "fast_track",
        "fast_track_2",
    ),
    SourceSpec(
        "cs_sampleprograms_v2.xlsx_-_fast_track_3.pdf",
        "CS",
        "BSCS",
        "Computer Science and Engineering",
        "fast_track_3",
        "fast_track",
        "fast_track_3",
    ),
    SourceSpec(
        "cs_sampleprograms_v2.xlsx_-_standard_track_0.pdf",
        "CS",
        "BSCS",
        "Computer Science and Engineering",
        "standard",
    ),
    SourceSpec(
        "dsa-suggested-course-plan_0.pdf",
        "DSA",
        "BSDSA",
        "Data Science and Analytics",
        "standard",
    ),
    SourceSpec("ECONBA_1.pdf", "ECON", "BAECON", "Economics", "standard"),
    SourceSpec(
        "ee_v2.xlsx_-_courseplan.pdf",
        "EE",
        "BSEE",
        "Electronics Engineering",
        "standard",
    ),
    SourceSpec(
        "ie-coursetable-v4.xlsx_-_ie_course_plan_0.pdf",
        "IE",
        "BSIE",
        "Industrial Engineering",
        "standard",
    ),
    SourceSpec(
        "mat_courseplan.pdf",
        "MAT",
        "BSMAT",
        "Materials Science and Nano Engineering",
        "standard",
    ),
    SourceSpec(
        "me_v2.xlsx_-_sheet1.pdf",
        "ME",
        "BSME",
        "Mechatronics Engineering",
        "standard",
    ),
    SourceSpec(
        "PSIR BA (2).pdf",
        "PSIR",
        "BAPSIR",
        "Political Science and International Relations",
        "standard",
    ),
    SourceSpec("psyrecommended.pdf", "PSY", "BAPSY", "Psychology", "standard"),
)


REQUIRED_KEYS = {
    "data_role",
    "document_type",
    "authority_level",
    "binding_status",
    "program",
    "degree_code",
    "program_name",
    "plan_type",
    "track",
    "track_name",
    "study_year",
    "semester",
    "term_kind",
    "term_label",
    "course_code",
    "course_title",
    "course_type",
    "su_credits",
    "ects",
    "prerequisites",
    "source_authority",
    "source_url",
    "source_file",
    "source_document",
    "source_page",
    "source_sha256",
    "text",
    "chunk_id",
}


def _clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\x00", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _slug(value: str) -> str:
    value = value.casefold().replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "unspecified"


def _number(value: Any) -> int | float | None:
    text = _clean(value)
    direct = re.fullmatch(r"\d+(?:\.\d+)?", text)
    if direct:
        number = float(text)
        return int(number) if number.is_integer() else number
    # In the ME spreadsheet export, a column heading is interleaved into a
    # handful of numeric cells: ``(To6tal)`` and ``Credit5s (ECTS)``.  The
    # rendered cell unambiguously shows the embedded number.
    artifact = re.fullmatch(r"\(?To(\d+)tal\)?|Credit(\d+)s(?: \(ECTS\))?", text, re.I)
    if not artifact:
        return None
    number = float(next(group for group in artifact.groups() if group is not None))
    return int(number) if number.is_integer() else number


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_id(prefix: str, identity: dict[str, Any]) -> str:
    material = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{digest}"


def _parse_period(value: Any) -> dict[str, Any] | None:
    text = _clean(value)
    if not text:
        return None
    normal = text.replace("–", "-").replace("—", "-")
    year_match = re.search(r"year\s*-?\s*(\d)", normal, re.I)
    semester_match = re.search(r"semester\s*-?\s*(\d)", normal, re.I)
    if year_match and semester_match:
        year, semester = int(year_match.group(1)), int(semester_match.group(1))
        return {
            "study_year": year,
            "semester": semester,
            "term_kind": "semester",
            "term_label": f"Year {year}, Semester {semester}",
        }
    if year_match and re.search(r"summer", normal, re.I):
        year = int(year_match.group(1))
        return {
            "study_year": year,
            "semester": None,
            "term_kind": "summer",
            "term_label": f"Summer after Year {year}",
        }
    return None


def _fallback_periods(positions: list[int], next_semester: int) -> tuple[dict[int, dict[str, Any]], int]:
    contexts: dict[int, dict[str, Any]] = {}
    for position in positions:
        semester = next_semester
        year = (semester + 1) // 2
        contexts[position] = {
            "study_year": year,
            "semester": semester,
            "term_kind": "semester",
            "term_label": f"Year {year}, Semester {semester}",
        }
        next_semester += 1
    return contexts, next_semester


def _is_period_heading(row: list[Any], next_row: list[Any] | None) -> bool:
    text = " ".join(_clean(cell) for cell in row)
    nonempty = [_clean(cell) for cell in row if _clean(cell)]
    if re.search(r"\byear\b", text, re.I) and re.search(r"semester|summer", text, re.I):
        return True
    if re.search(r"summer", text, re.I) and len(nonempty) <= 2:
        return True
    # Several Excel exports interleave letters (for example
    # "YeaSre -m1 eFrsetesrh-m1an").  It is still a heading when followed by
    # a course-header row and has cells beginning with "Yea".
    if next_row and "course code" in " ".join(_clean(c).casefold() for c in next_row):
        if re.search(r"\byear\b", text, re.I):
            return True
        if any(_clean(cell).casefold().startswith("yea") for cell in row):
            return True
        # Badly interleaved Excel text can turn "Year - 3 / Semester - 5" into
        # "YeSaerm - e3s Jtuern-i5or".  Its structural signature remains: one
        # or two short, numbered cells immediately before the repeated header.
        return bool(nonempty and len(nonempty) <= 2 and all(re.search(r"\d", x) for x in nonempty))
    return False


def _header_positions(row: list[Any]) -> list[int]:
    return [index for index, cell in enumerate(row) if "course code" in _clean(cell).casefold()]


def _nearest_context(position: int, contexts: dict[int, dict[str, Any]]) -> dict[str, Any] | None:
    if not contexts:
        return None
    key = min(contexts, key=lambda item: abs(item - position))
    return dict(contexts[key])


def _column_map(row: list[Any], start: int, end: int) -> dict[str, int]:
    mapping: dict[str, int] = {"course_code": start}
    ects_candidates: list[int] = []
    for index in range(start, end):
        label = _clean(row[index]).casefold()
        if "course name" in label:
            mapping["course_title"] = index
        elif "course type" in label:
            mapping["course_type"] = index
        elif "prereq" in label:
            mapping["prerequisites"] = index
        elif "suggested term" in label:
            mapping["suggested_term"] = index
        elif "su" in label and "credit" in label:
            mapping["su_credits"] = index
        elif "engineering" in label or "enginering" in label or "endgitins" in label:
            mapping["engineering_ects"] = index
        elif "basic" in label and ("science" in label or "sci" in label):
            mapping["basic_science_ects"] = index
        elif "ects" in label or "tocttasl" in label:
            ects_candidates.append(index)
    if ects_candidates:
        mapping["ects"] = ects_candidates[-1]
    return mapping


_PLACEHOLDER_CODES = {"CORE", "AREA", "FREE", "ELECTIVE", "CORE/AREA/FREE"}


def _normalise_course_code(value: Any) -> str | None:
    code = _clean(value).replace("**", "").replace("*", "").strip()
    code = code.replace("//", "/")
    if not code:
        return None
    compact = code.replace(" ", "")
    if compact.upper() == "CORE/AREA/FREE":
        return "CORE/AREA/FREE"
    match = re.fullmatch(r"([A-Za-z]+)(\d+[A-Za-z]*)", compact)
    if match:
        return f"{match.group(1).upper()} {match.group(2).upper()}"
    return re.sub(r"\s+", " ", code).upper()


def _repair_source_artifacts(
    *, source_file: str, course_code: str | None, course_title: str | None
) -> tuple[str | None, str | None]:
    """Repair only extraction artifacts confirmed against the rendered PDF."""

    code = course_code
    title = course_title.lstrip("* ").strip() if course_title else course_title
    if source_file == "cs_sampleprograms_v2.xlsx_-_fast_track_2_1.pdf":
        if code and code.startswith("HUM") and "201/202/20" in code and title == "Major Works":
            code = "HUM 201/202/207"
        elif code and code.startswith("7CS "):
            code = code[1:]
    if source_file.startswith("cs_sampleprograms"):
        if code and code.startswith("HUM") and "201/202/20" in code:
            code = "HUM 201/202/207"
        if code and code.startswith("7CS "):
            code = code[1:]
    if source_file == "ee_v2.xlsx_-_courseplan.pdf":
        if code == "ESTEUE R 3MM 9M5 E- R":
            code, title = "EE 395", "Summer Term"
        elif code in {"EEELE TCRTAIVCEK", "EE ELE TCRTAIVCEK"}:
            code, title = None, "EE Track Elective"
        elif code == "EAE E LREE T CAR T A IV C E K OR":
            code, title = None, "EE Track or Area Elective"
        elif code in {"AELREECAT IVE", "AELREECAT IVE"}:
            code, title = None, "Area Elective"
    if title == "Law and Ethics" and code is None:
        # The visible PSIR table contains SPS 303; its text layer drops the first cell.
        code = "SPS 303"
    if source_file == "psyrecommended.pdf" and title == "Mind and Behavior" and code is None:
        # PSY 201 is visible in the first cell; pdfplumber drops that cell only.
        code = "PSY 201"
    if code == "HIST 191" and title in {
        "Principles of Atatürk and the History of the",
        "Principles of Atatürk and the History of the Turkish",
    }:
        title = "Principles of Atatürk and the History of the Turkish Revolution I"
    if code == "HIST 192" and title in {
        "Principles of Atatürk and the History of the",
        "Principles of Atatürk and the History of the Turkish",
    }:
        title = "Principles of Atatürk and the History of the Turkish Revolution II"
    if source_file == "dsa-suggested-course-plan_0.pdf":
        title = {
            "IF 100": "Computational Approaches to Problem Solving",
            "MATH 203": "Introduction to Probability",
            "DSA 210": "Introduction to Data Science",
            "MATH 306": "Statistical Modelling",
            "OPIM 390": "Introduction to Business Analytics",
            "DSA 301": "Data Visualization",
            "DSA 492": "Graduation Project",
        }.get(code or "", title)
    return code, title


def _placeholder_title(code: str | None, title: str | None) -> str | None:
    if title:
        return title
    return {
        "CORE": "Core Elective",
        "AREA": "Area Elective",
        "FREE": "Free Elective",
        "ELECTIVE": "Elective",
        "CORE/AREA/FREE": "Core, Area, or Free Elective",
    }.get(code or "")


def _is_placeholder(code: str | None, title: str | None) -> bool:
    if code in _PLACEHOLDER_CODES:
        return True
    if code and "ELECTIVE" in code:
        return True
    text = (title or "").casefold()
    return bool(text and "elective" in text and not re.search(r"[a-z]{2,}\s*\d", text))


def _track_heading(value: Any, program_name: str) -> tuple[str, str] | None:
    text = _clean(value)
    lower = text.casefold()
    if "suggested courses" not in lower and "suggested for elective courses" not in lower:
        return None
    if lower == "suggested for elective courses":
        name = "400-level elective courses"
    else:
        name = re.sub(r"^\d+\.\s*", "", text)
        name = re.sub(r"^Suggested Courses(?: for)?\s*", "", name, flags=re.I)
        quoted = re.search(r'["“]([^"”]+)["”]', name)
        if quoted:
            name = quoted.group(1)
        else:
            name = re.sub(re.escape(program_name), "", name, flags=re.I)
            name = re.sub(r"^\s*(?:and\s+)?", "", name, flags=re.I)
            name = re.sub(r"\s+Track(?:-\d+)?\)?\s*$", "", name, flags=re.I)
            name = name.strip(" -,:()")
    return _slug(name), name or "Unspecified elective track"


def _row_value(row: list[Any], mapping: dict[str, int], key: str) -> str:
    index = mapping.get(key)
    return _clean(row[index]) if index is not None and index < len(row) else ""


def _parse_course_row(
    row: list[Any],
    mapping: dict[str, int],
    *,
    source_file: str,
) -> dict[str, Any] | None:
    raw_code = _row_value(row, mapping, "course_code")
    raw_title = _row_value(row, mapping, "course_title")
    lower_code = raw_code.casefold()
    known_ee_artifact = source_file == "ee_v2.xlsx_-_courseplan.pdf" and any(
        marker in lower_code for marker in ("3mm 9m5", "lree t car")
    )
    if (
        "total" in lower_code
        or "total credits" in raw_title.casefold()
        or lower_code.endswith(".")
        or " is " in lower_code
        or "semester no" in lower_code
        or lower_code.startswith(("the students", "students are", "description", "minimum "))
        or (len(raw_code.split()) >= 5 and not known_ee_artifact)
    ):
        return None
    code = _normalise_course_code(raw_code)
    title = raw_title or None
    code, title = _repair_source_artifacts(
        source_file=source_file, course_code=code, course_title=title
    )
    suggested_from_code: str | None = None
    if code and (match := re.fullmatch(r"([A-Z]+\s+\d+)_S(\d)", code)):
        code = match.group(1)
        suggested_from_code = f"Semester {match.group(2)}"
    title = _placeholder_title(code, title)
    if not code and not title:
        return None
    if code and any(marker in code for marker in ("COURSE CODE", "TERM TOTAL", "SU CREDITS TOTAL")):
        return None
    if title and title.casefold() in {"course name", "total credits"}:
        return None

    placeholder = _is_placeholder(code, title)
    course_group = code if code in _PLACEHOLDER_CODES or (code and "ELECTIVE" in code) else None
    if placeholder and (code in _PLACEHOLDER_CODES or (code and "ELECTIVE" in code)):
        if not title:
            title = {
                "FREE ELECTIVE": "Free Elective",
                "CORE ELECTIVE": "Core Elective",
                "AREA ELECTIVE": "Area Elective",
            }.get(code, "Elective")
        code = None
    prerequisites = _row_value(row, mapping, "prerequisites") or None
    if source_file == "me_v2.xlsx_-_sheet1.pdf":
        if prerequisites == "Courses":
            prerequisites = None
        elif prerequisites == "CENouS r2s0e6s":
            prerequisites = "ENS 206"
    return {
        "course_code": code,
        "course_title": title,
        "course_group": course_group,
        "course_type": (
            "Elective"
            if _row_value(row, mapping, "course_type") == "Elecve"
            else (_row_value(row, mapping, "course_type") or None)
        ),
        "su_credits": _number(_row_value(row, mapping, "su_credits")),
        "ects": _number(_row_value(row, mapping, "ects")),
        "engineering_ects": _number(_row_value(row, mapping, "engineering_ects")),
        "basic_science_ects": _number(_row_value(row, mapping, "basic_science_ects")),
        "prerequisites": prerequisites,
        "suggested_term": _row_value(row, mapping, "suggested_term") or suggested_from_code,
        "is_elective_placeholder": placeholder,
    }


def _extract_totals(text: str) -> tuple[int | None, int | None]:
    flat = _clean(text)
    su_patterns = (
        r"SU CREDITS? TOTAL\s*:?\s*(\d+)",
        r"MINIMUM SU CREDITS\s*:?\s*(\d+)",
    )
    ects_patterns = (
        r"ECTS(?: CREDITS?)? TOTAL\s*:?\s*(\d+)",
        r"MINIMUM ECTS CREDITS\s*:?\s*(\d+)",
    )

    def match(patterns: Iterable[str]) -> int | None:
        for pattern in patterns:
            found = re.search(pattern, flat, re.I)
            if found:
                return int(found.group(1))
        return None

    return match(su_patterns), match(ects_patterns)


def _extract_notes(text: str) -> list[str]:
    flat = _clean(text)
    patterns = (
        r"The students have to take \*?Undergraduate Project Course \(PROJ 201\)[^.]{0,120}\.",
        r"Undergraduate Project Course \(PROJ 201\)[^.]{0,120}may also be taken[^.]{0,120}\.",
        r"PROJ 201 is displayed in multiple semesters[^.]{0,260}\.",
        r"PHYS 113 can be taken[^.]{0,160}\.",
        r"DSA 301 is only offered[^.]{0,120}\.",
        r"ECONOMICS BA students are advised to take[^.]{0,180}\.",
    )
    candidates = [match.group(0) for pattern in patterns for match in re.finditer(pattern, flat, re.I)]
    notes: list[str] = []
    for candidate in candidates:
        note = _clean(candidate).lstrip("* ")
        if note and note not in notes:
            notes.append(note)
    return notes


def _extract_pdf(pdf_path: Path, spec: SourceSpec) -> dict[str, Any]:
    try:
        import pdfplumber
    except ImportError as exc:  # pragma: no cover - environment-specific guidance
        raise RuntimeError(
            "pdfplumber is required; install it with `python -m pip install pdfplumber`."
        ) from exc

    courses: list[dict[str, Any]] = []
    track_courses: list[dict[str, Any]] = []
    term_totals: dict[tuple[int, int | None, str], dict[str, Any]] = {}
    full_text: list[str] = []
    next_regular_semester = 1
    mode = "main"
    pending_periods: dict[int, dict[str, Any]] = {}
    pending_tracks: dict[int, dict[str, Any]] = {}
    last_maps: dict[int, dict[str, int]] = {}
    active_blocks: list[dict[str, Any]] = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text() or ""
            full_text.append(page_text)
            for table_index, table in enumerate(page.extract_tables()):
                for row_index, raw_row in enumerate(table):
                    row = list(raw_row)
                    joined = " ".join(_clean(cell) for cell in row)
                    lower = joined.casefold()
                    next_row = list(table[row_index + 1]) if row_index + 1 < len(table) else None

                    if "minor programs" in lower:
                        mode = "ignore"
                        pending_tracks = {}
                        active_blocks = []
                        continue
                    if "track elective courses" in lower:
                        mode = "track"
                        pending_periods = {}
                        active_blocks = []
                        continue

                    if mode == "main" and _is_period_heading(row, next_row):
                        positions = [index for index, cell in enumerate(row) if _clean(cell)]
                        parsed = {
                            index: period
                            for index, cell in enumerate(row)
                            if (period := _parse_period(cell)) is not None
                        }
                        if parsed:
                            pending_periods = parsed
                            regular = [p["semester"] for p in parsed.values() if p["semester"]]
                            if regular:
                                next_regular_semester = max(next_regular_semester, max(regular) + 1)
                        elif "summer" in lower:
                            year = max(1, (next_regular_semester - 1) // 2)
                            position = positions[0] if positions else 0
                            pending_periods = {
                                position: {
                                    "study_year": year,
                                    "semester": None,
                                    "term_kind": "summer",
                                    "term_label": f"Summer after Year {year}",
                                }
                            }
                        else:
                            pending_periods, next_regular_semester = _fallback_periods(
                                positions, next_regular_semester
                            )
                        active_blocks = []
                        continue

                    if mode == "track":
                        if "ee4xx courses" in lower:
                            position = next(
                                (index for index, cell in enumerate(row) if _clean(cell)), 0
                            )
                            pending_tracks = {
                                position: {
                                    "track": "400_level_elective_courses",
                                    "track_name": "400-level elective courses",
                                }
                            }
                            active_blocks = []
                            continue
                        headings = {
                            index: {"track": found[0], "track_name": found[1]}
                            for index, cell in enumerate(row)
                            if (found := _track_heading(cell, spec.program_name)) is not None
                        }
                        if headings:
                            pending_tracks = headings
                            active_blocks = []
                            continue

                    starts = _header_positions(row)
                    if mode == "track" and pending_tracks:
                        # The ME PDF loses the first "Course Code" header cell, but keeps
                        # "Course Name" one column later.  The track heading preserves start.
                        for position in pending_tracks:
                            if (
                                position not in starts
                                and position + 1 < len(row)
                                and "course name" in _clean(row[position + 1]).casefold()
                            ):
                                starts.append(position)
                        starts.sort()
                    if starts:
                        active_blocks = []
                        for block_index, start in enumerate(starts):
                            end = starts[block_index + 1] if block_index + 1 < len(starts) else len(row)
                            mapping = _column_map(row, start, end)
                            last_maps[start] = mapping
                            if mode == "main":
                                context = _nearest_context(start, pending_periods)
                            elif mode == "track":
                                context = _nearest_context(start, pending_tracks)
                            else:
                                context = None
                            if context:
                                active_blocks.append(
                                    {"start": start, "end": end, "mapping": mapping, "context": context}
                                )
                        continue

                    if mode == "ignore":
                        continue
                    if not active_blocks and mode == "main" and pending_periods and last_maps:
                        # Summer tables in a few exports omit a repeated header.
                        for position, context in pending_periods.items():
                            nearest_start = min(last_maps, key=lambda item: abs(item - position))
                            active_blocks.append(
                                {
                                    "start": nearest_start,
                                    "end": len(row),
                                    "mapping": last_maps[nearest_start],
                                    "context": dict(context),
                                }
                            )

                    for block in active_blocks:
                        mapping = block["mapping"]
                        raw_code = _row_value(row, mapping, "course_code")
                        if "term total" in raw_code.casefold():
                            if mode == "main":
                                context = block["context"]
                                key = (
                                    context["study_year"],
                                    context["semester"],
                                    context["term_kind"],
                                )
                                term_totals[key] = {
                                    "term_su_credits": _number(
                                        _row_value(row, mapping, "su_credits")
                                    ),
                                    "term_ects": _number(_row_value(row, mapping, "ects")),
                                }
                            continue
                        parsed_course = _parse_course_row(
                            row, mapping, source_file=spec.filename
                        )
                        if not parsed_course:
                            continue
                        record = {
                            **block["context"],
                            **parsed_course,
                            "source_page": page_number,
                            "source_table": table_index,
                            "source_row": row_index,
                        }
                        if mode == "main":
                            title = (record.get("course_title") or "").casefold()
                            code = record.get("course_code") or ""
                            if (
                                (code.endswith(" 395") and "internship" in title)
                                or "summer semester" in title
                                or (code == "EE 395" and "summer" in title)
                            ):
                                year = record["study_year"]
                                record.update(
                                    {
                                        "semester": None,
                                        "term_kind": "summer",
                                        "term_label": f"Summer after Year {year}",
                                    }
                                )
                            courses.append(record)
                        else:
                            track_courses.append(record)

    raw_text = "\n".join(full_text)
    plan_su, plan_ects = _extract_totals(raw_text)
    notes = _extract_notes(raw_text)
    return {
        "courses": courses,
        "track_courses": track_courses,
        "term_totals": term_totals,
        "plan_total_su_credits": plan_su,
        "plan_total_ects": plan_ects,
        "notes": notes,
    }


def _common(
    spec: SourceSpec,
    *,
    source_sha256: str,
    source_document: str,
    source_page: int,
    context: dict[str, Any],
) -> dict[str, Any]:
    return {
        "data_role": "suggested_program",
        "authority_level": "official_advisory",
        "binding_status": "non_binding_recommended_plan",
        "program": spec.program,
        "degree_code": spec.degree_code,
        "program_name": spec.program_name,
        "plan_type": spec.plan_type,
        "track": context.get("track", spec.track),
        "track_name": context.get("track_name"),
        "study_year": context.get("study_year"),
        "semester": context.get("semester"),
        "term_kind": context.get("term_kind"),
        "term_label": context.get("term_label"),
        "source_authority": "Sabanci University",
        "source_url": None,
        "source_file": spec.filename,
        "source_document": source_document,
        "corpus_document": f"suggested_programs/{spec.program}/{spec.output_stem}.jsonl",
        "source_page": source_page,
        "source_sha256": source_sha256,
    }


def _credit_text(course: dict[str, Any]) -> str:
    values: list[str] = []
    if course.get("su_credits") is not None:
        values.append(f"{course['su_credits']} SU")
    if course.get("ects") is not None:
        values.append(f"{course['ects']} ECTS")
    return ", ".join(values) if values else "credit not stated"


def _course_label(course: dict[str, Any]) -> str:
    if course.get("course_code"):
        return f"{course['course_code']} - {course.get('course_title') or 'title not stated'}"
    return course.get("course_title") or course.get("course_group") or "unspecified elective slot"


def _build_rows(pdf_path: Path, spec: SourceSpec, extracted: dict[str, Any]) -> list[dict[str, Any]]:
    source_hash = _sha256(pdf_path)
    source_document = f"suggested_programs/sources/{spec.filename}"
    rows: list[dict[str, Any]] = []
    plan_label = spec.track.replace("_", " ")

    overview_common = _common(
        spec,
        source_sha256=source_hash,
        source_document=source_document,
        source_page=1,
        context={},
    )
    track_names = sorted(
        {course["track_name"] for course in extracted["track_courses"] if course.get("track_name")}
    )
    totals: list[str] = []
    if extracted["plan_total_su_credits"] is not None:
        totals.append(f"{extracted['plan_total_su_credits']} SU credits")
    if extracted["plan_total_ects"] is not None:
        totals.append(f"{extracted['plan_total_ects']} ECTS")
    overview_text = (
        f"{spec.program_name} ({spec.program}) için {plan_label} resmi önerilen ders planı. "
        f"This is a non-binding recommended sequence, not a graduation-requirement rule. "
        + (f"Plan total shown in the source: {', '.join(totals)}. " if totals else "")
        + (
            "The PDF also lists non-official concentration suggestions: "
            + ", ".join(track_names)
            + ". "
            if track_names
            else ""
        )
        + "Always evaluate binding graduation obligations from degree_requirements data."
    )
    overview = {
        **overview_common,
        "document_type": "suggested_program_profile",
        "course_code": None,
        "course_title": None,
        "course_type": None,
        "course_group": None,
        "su_credits": None,
        "ects": None,
        "engineering_ects": None,
        "basic_science_ects": None,
        "prerequisites": None,
        "suggested_term": None,
        "is_elective_placeholder": False,
        "is_one_time_flexible_placement": False,
        "plan_total_su_credits": extracted["plan_total_su_credits"],
        "plan_total_ects": extracted["plan_total_ects"],
        "notes": extracted["notes"],
        "available_suggestion_tracks": track_names,
        "text": overview_text,
    }
    overview["chunk_id"] = _stable_id(
        f"suggested_program:{spec.program}:{spec.track}:overview",
        {"source_sha256": source_hash, "document_type": overview["document_type"]},
    )
    rows.append(overview)

    grouped: dict[tuple[int, int | None, str], list[dict[str, Any]]] = defaultdict(list)
    for course in extracted["courses"]:
        grouped[(course["study_year"], course["semester"], course["term_kind"])].append(course)

    for key in sorted(grouped, key=lambda item: (item[0], item[1] if item[1] is not None else 99)):
        period_courses = sorted(
            grouped[key], key=lambda item: (item["source_page"], item["source_table"], item["source_row"])
        )
        context = period_courses[0]
        totals_for_period = extracted["term_totals"].get(key, {})
        term_su_credits = totals_for_period.get("term_su_credits")
        term_ects = totals_for_period.get("term_ects")
        term_total_basis = "source_term_total"
        if term_su_credits is None and all(
            course.get("su_credits") is not None for course in period_courses
        ):
            term_su_credits = sum(course["su_credits"] for course in period_courses)
            term_total_basis = "sum_of_listed_courses"
        if term_ects is None and all(course.get("ects") is not None for course in period_courses):
            term_ects = sum(course["ects"] for course in period_courses)
            if totals_for_period.get("term_su_credits") is None:
                term_total_basis = "sum_of_listed_courses"
        labels = [
            f"{_course_label(course)} ({_credit_text(course)})" for course in period_courses
        ]
        summary_common = _common(
            spec,
            source_sha256=source_hash,
            source_document=source_document,
            source_page=min(course["source_page"] for course in period_courses),
            context=context,
        )
        summary_text = (
            f"{spec.program_name} ({spec.program}) {plan_label} önerilen programı, "
            f"{context['term_label']}: "
            + "; ".join(labels)
            + ". "
            + (
                f"Dönem toplamı/term total: {term_su_credits} SU, {term_ects} ECTS. "
                if term_su_credits is not None and term_ects is not None
                else ""
            )
            + "This sequence is advisory, not a binding degree requirement."
        )
        summary_courses = [
            {
                "course_code": course.get("course_code"),
                "course_title": course.get("course_title"),
                "course_type": course.get("course_type"),
                "course_group": course.get("course_group"),
                "su_credits": course.get("su_credits"),
                "ects": course.get("ects"),
                "prerequisites": course.get("prerequisites"),
                "is_elective_placeholder": course.get("is_elective_placeholder", False),
                "is_one_time_flexible_placement": course.get("course_code") == "PROJ 201",
            }
            for course in period_courses
        ]
        summary = {
            **summary_common,
            "document_type": "suggested_program_semester",
            "course_code": None,
            "course_title": None,
            "course_type": None,
            "course_group": None,
            "su_credits": None,
            "ects": None,
            "engineering_ects": None,
            "basic_science_ects": None,
            "prerequisites": None,
            "suggested_term": None,
            "is_elective_placeholder": False,
            "is_one_time_flexible_placement": False,
            "term_su_credits": term_su_credits,
            "term_ects": term_ects,
            "term_total_basis": term_total_basis,
            "courses": summary_courses,
            "text": summary_text,
        }
        summary["chunk_id"] = _stable_id(
            f"suggested_program:{spec.program}:{spec.track}:semester_plan",
            {
                "source_sha256": source_hash,
                "study_year": context["study_year"],
                "semester": context["semester"],
                "term_kind": context["term_kind"],
            },
        )
        rows.append(summary)

        occurrences: Counter[str] = Counter()
        for course in period_courses:
            label = course.get("course_code") or course.get("course_title") or "elective"
            occurrences[label] += 1
            common = _common(
                spec,
                source_sha256=source_hash,
                source_document=source_document,
                source_page=course["source_page"],
                context=course,
            )
            flexible_proj = course.get("course_code") == "PROJ 201"
            text = (
                f"{spec.program_name} ({spec.program}) {plan_label} önerilen programında "
                f"{course['term_label']} için {_course_label(course)}. "
                f"Ders türü/course type: {course.get('course_type') or 'not stated'}. "
                f"Kredi/credits: {_credit_text(course)}. "
                f"Önkoşul/prerequisite: {course.get('prerequisites') or 'not stated'}. "
            )
            if flexible_proj:
                text += (
                    "PROJ 201 is a one-time course shown in multiple possible enrollment periods; "
                    "do not count or recommend it more than once. "
                )
            text += "This is advisory sequencing, not a binding graduation requirement."
            row = {
                **common,
                "document_type": "suggested_program_course",
                "course_code": course.get("course_code"),
                "course_title": course.get("course_title"),
                "course_type": course.get("course_type"),
                "course_group": course.get("course_group"),
                "su_credits": course.get("su_credits"),
                "ects": course.get("ects"),
                "engineering_ects": course.get("engineering_ects"),
                "basic_science_ects": course.get("basic_science_ects"),
                "prerequisites": course.get("prerequisites"),
                "suggested_term": course.get("suggested_term"),
                "is_elective_placeholder": course.get("is_elective_placeholder", False),
                "is_one_time_flexible_placement": flexible_proj,
                "text": text,
            }
            row["chunk_id"] = _stable_id(
                f"suggested_program:{spec.program}:{spec.track}:course",
                {
                    "source_sha256": source_hash,
                    "study_year": course["study_year"],
                    "semester": course["semester"],
                    "term_kind": course["term_kind"],
                    "course": label,
                    "occurrence": occurrences[label],
                },
            )
            rows.append(row)

    track_occurrences: Counter[tuple[str, str]] = Counter()
    for course in sorted(
        extracted["track_courses"],
        key=lambda item: (
            item.get("track") or "",
            item["source_page"],
            item["source_table"],
            item["source_row"],
        ),
    ):
        label = course.get("course_code") or course.get("course_title") or "elective"
        track_occurrences[(course["track"], label)] += 1
        common = _common(
            spec,
            source_sha256=source_hash,
            source_document=source_document,
            source_page=course["source_page"],
            context=course,
        )
        text = (
            f"{spec.program_name} ({spec.program}) PDF'sindeki bağlayıcı olmayan "
            f"{course['track_name']} odak alanı önerisi: {_course_label(course)}. "
            f"Kredi/credits: {_credit_text(course)}. "
            f"Önerilen dönem: {course.get('suggested_term') or 'not assigned'}. "
            f"Önkoşul/prerequisite: {course.get('prerequisites') or 'not stated'}. "
            "The source explicitly describes these tracks as suggestions, not official specializations."
        )
        row = {
            **common,
            "document_type": "suggested_program_track",
            "binding_status": "non_binding_concentration_suggestion",
            "course_code": course.get("course_code"),
            "course_title": course.get("course_title"),
            "course_type": course.get("course_type"),
            "course_group": course.get("course_group"),
            "su_credits": course.get("su_credits"),
            "ects": course.get("ects"),
            "engineering_ects": course.get("engineering_ects"),
            "basic_science_ects": course.get("basic_science_ects"),
            "prerequisites": course.get("prerequisites"),
            "suggested_term": course.get("suggested_term"),
            "is_elective_placeholder": course.get("is_elective_placeholder", False),
            "is_one_time_flexible_placement": False,
            "text": text,
        }
        row["chunk_id"] = _stable_id(
            f"suggested_program:{spec.program}:{course['track']}:track_course",
            {
                "source_sha256": source_hash,
                "course": label,
                "occurrence": track_occurrences[(course["track"], label)],
            },
        )
        rows.append(row)
    return rows


def validate_rows(rows: list[dict[str, Any]], *, source_document: str = "") -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    for line_number, row in enumerate(rows, start=1):
        missing = REQUIRED_KEYS - row.keys()
        if missing:
            errors.append(f"{source_document}:{line_number}: missing keys {sorted(missing)}")
        chunk_id = row.get("chunk_id")
        if not isinstance(chunk_id, str) or not re.search(r":[0-9a-f]{16}$", chunk_id):
            errors.append(f"{source_document}:{line_number}: invalid chunk_id {chunk_id!r}")
        elif chunk_id in seen:
            errors.append(f"{source_document}:{line_number}: duplicate chunk_id {chunk_id}")
        seen.add(str(chunk_id))
        if row.get("data_role") != "suggested_program":
            errors.append(f"{source_document}:{line_number}: invalid data_role")
        if row.get("semester") is not None and row["semester"] not in range(1, 9):
            errors.append(f"{source_document}:{line_number}: semester outside 1..8")
        if row.get("study_year") is not None and row["study_year"] not in range(1, 5):
            errors.append(f"{source_document}:{line_number}: study_year outside 1..4")
        for key in ("su_credits", "ects"):
            value = row.get(key)
            if value is not None and not isinstance(value, (int, float)):
                errors.append(f"{source_document}:{line_number}: {key} must be numeric or null")
        if not isinstance(row.get("text"), str) or not row["text"].strip():
            errors.append(f"{source_document}:{line_number}: empty text")
        if row.get("source_url") is not None:
            errors.append(
                f"{source_document}:{line_number}: source_url must remain null unless supplied by source"
            )
        if row.get("document_type") in {"suggested_program_course", "suggested_program_track"}:
            if not row.get("course_code") and not row.get("course_title"):
                errors.append(f"{source_document}:{line_number}: course chunk has no code/title")
    return errors


def convert(input_dir: Path, output_dir: Path) -> dict[str, int]:
    missing = [spec.filename for spec in SOURCE_SPECS if not (input_dir / spec.filename).is_file()]
    if missing:
        raise FileNotFoundError("Missing expected source PDFs: " + ", ".join(missing))
    output_counts: dict[str, int] = {}
    all_ids: set[str] = set()
    manifest_sources: list[dict[str, Any]] = []
    for spec in SOURCE_SPECS:
        pdf_path = input_dir / spec.filename
        extracted = _extract_pdf(pdf_path, spec)
        rows = _build_rows(pdf_path, spec, extracted)
        relative = f"suggested_programs/{spec.program}/{spec.output_stem}.jsonl"
        errors = validate_rows(rows, source_document=relative)
        semesters = {
            row["semester"]
            for row in rows
            if row.get("document_type") == "suggested_program_semester"
            and row.get("term_kind") == "semester"
        }
        if semesters != set(range(1, 9)):
            errors.append(
                f"{relative}: regular semester coverage must be exactly 1..8, got {sorted(semesters)}"
            )
        duplicates = [row["chunk_id"] for row in rows if row["chunk_id"] in all_ids]
        if duplicates:
            errors.append(f"{relative}: chunk ids collide with another source: {duplicates[:3]}")
        if errors:
            raise ValueError("\n".join(errors))
        all_ids.update(row["chunk_id"] for row in rows)
        destination = output_dir / spec.program / f"{spec.output_stem}.jsonl"
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = "\n".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in rows
        )
        destination.write_text(payload + "\n", encoding="utf-8")
        output_counts[relative] = len(rows)
        manifest_sources.append(
            {
                "source_file": spec.filename,
                "source_document": f"suggested_programs/sources/{spec.filename}",
                "source_sha256": _sha256(pdf_path),
                "source_pages": max(row["source_page"] for row in rows),
                "program": spec.program,
                "degree_code": spec.degree_code,
                "program_name": spec.program_name,
                "plan_type": spec.plan_type,
                "track": spec.track,
                "output_document": relative,
                "row_count": len(rows),
            }
        )
    manifest = {
        "schema_version": 1,
        "data_role": "suggested_program",
        "source_authority": "Sabanci University",
        "source_url_policy": "null_when_not_supplied_by_source",
        "source_count": len(manifest_sources),
        "validation": {
            "status": "passed",
            "strict_schema": True,
            "unique_chunk_ids": True,
            "regular_semester_coverage": "exactly_1_through_8_per_plan",
            "summer_semester_value": None,
        },
        "sources": manifest_sources,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return output_counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "suggested_programs" / "sources",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "suggested_programs",
    )
    args = parser.parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    counts = convert(input_dir, output_dir)
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"PDFs converted: {len(counts)}")
    print(f"JSONL rows: {sum(counts.values())}")
    for path, count in sorted(counts.items()):
        print(f"  {path}: {count}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
