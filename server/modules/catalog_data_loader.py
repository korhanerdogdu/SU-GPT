from __future__ import annotations

import json
import re
from collections import defaultdict
from hashlib import sha1
from pathlib import Path
from typing import Any, Iterable, Iterator

from langchain_core.documents import Document


COURSES_PER_AGGREGATE_CARD = 70
SECTIONS_PER_AGGREGATE_CARD = 20
HISTORY_TERMS_PER_CARD = 30

TERM_SEASONS = {
    "01": ("Fall", "Guz"),
    "02": ("Spring", "Bahar"),
    "03": ("Summer", "Yaz"),
}

DAY_NAMES = {
    "M": "Monday",
    "T": "Tuesday",
    "W": "Wednesday",
    "R": "Thursday",
    "F": "Friday",
    "S": "Saturday",
    "U": "Sunday",
}

REQUIREMENT_LABELS = {
    "required": "required course / zorunlu ders",
    "core": "core elective / cekirdek secmeli",
    "area": "area elective / alan secmeli",
    "free": "free elective / serbest secmeli",
    "university": "university course / universite dersi",
    "unknown": "unknown requirement category",
}


def iter_catalog_documents(data_dir: str | Path) -> Iterator[Document]:
    root = Path(data_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Catalog data directory does not exist: {root}")

    for path in _existing_files(
        [
            root / "all_coursepage_info.jsonl",
            root / "basic_science_credits.jsonl",
            root / "graduation_requirements.jsonl",
            root / "terms.jsonl",
        ]
    ):
        yield from _dispatch_root_file(path, root)

    degree_requirements_dir = root / "degree_requirements"
    if degree_requirements_dir.exists():
        for path in sorted(degree_requirements_dir.glob("*.jsonl")):
            yield from _iter_degree_requirement_source_file(path, root)

    for term_dir in sorted(p for p in root.iterdir() if p.is_dir() and _is_term_code(p.name)):
        for path in sorted(term_dir.glob("*.jsonl")):
            yield from _iter_requirement_file(
                path=path,
                root=root,
                program=path.stem,
                program_type="major",
                term_code=term_dir.name,
            )

    minors_dir = root / "minors"
    if minors_dir.exists():
        for path in sorted(minors_dir.glob("*.jsonl")):
            yield from _iter_requirement_file(
                path=path,
                root=root,
                program=path.stem,
                program_type="minor",
                term_code=None,
            )
        for term_dir in sorted(p for p in minors_dir.iterdir() if p.is_dir() and _is_term_code(p.name)):
            for path in sorted(term_dir.glob("*.jsonl")):
                yield from _iter_requirement_file(
                    path=path,
                    root=root,
                    program=path.stem,
                    program_type="minor",
                    term_code=term_dir.name,
                )

    schedule_dir = root / "schedule"
    if schedule_dir.exists():
        for path in sorted(schedule_dir.glob("*.jsonl")):
            yield from _iter_schedule_file(path, root)


def _dispatch_root_file(path: Path, root: Path) -> Iterator[Document]:
    if path.name == "all_coursepage_info.jsonl":
        yield from _iter_course_catalog_file(path, root)
    elif path.name == "basic_science_credits.jsonl":
        yield from _iter_basic_science_file(path, root)
    elif path.name == "graduation_requirements.jsonl":
        yield from _iter_graduation_requirements_file(path, root)
    elif path.name == "terms.jsonl":
        yield from _iter_terms_file(path, root)


def _iter_requirement_file(
    path: Path,
    root: Path,
    program: str,
    program_type: str,
    term_code: str | None,
) -> Iterator[Document]:
    rows = list(_read_jsonl(path))
    groups: dict[str, list[tuple[str, dict[str, Any], int]]] = defaultdict(list)

    for line_no, row in rows:
        course_id = _course_id(row.get("Major"), row.get("Code"))
        requirement_type = _clean_value(row.get("EL_Type")) or "unknown"
        groups[requirement_type].append((course_id, row, line_no))

        term_name = _term_name(term_code)
        title = _clean_value(row.get("Course_Name"))
        text = _lines(
            "Record type: program curriculum requirement course eligibility.",
            f"Program: {program} ({program_type}).",
            _maybe(f"Term: {term_code} - {term_name}.", term_code and term_name),
            _maybe("Term: current or non-term-specific minor catalog.", term_code is None and program_type == "minor"),
            f"Course: {course_id} - {title}.",
            f"Requirement category: {requirement_type} ({REQUIREMENT_LABELS.get(requirement_type, requirement_type)}).",
            "Interpretation: this course can count in this requirement category for the listed program and term.",
            _field("SU credit", row.get("SU_credit")),
            _field("ECTS", row.get("ECTS")),
            _field("Engineering ECTS", row.get("Engineering")),
            _field("Basic science ECTS", row.get("Basic_Science")),
            _field("Faculty", row.get("Faculty")),
            _field("Faculty course flag", row.get("Faculty_Course")),
            f"Search aliases: {_course_aliases(course_id)}.",
            _source_line(path, root, line_no),
        )
        yield _document(
            kind="program_requirement",
            source=f"{program} {program_type} {term_code or 'current'} {requirement_type} {course_id}",
            source_path=path.relative_to(root).as_posix(),
            line_no=line_no,
            text=text,
            metadata={
                "program": program,
                "program_type": program_type,
                "term_code": term_code,
                "term_name": term_name,
                "requirement_type": requirement_type,
                "course_id": course_id,
                "course_subject": _clean_value(row.get("Major")),
                "course_number": _clean_value(row.get("Code")),
                "course_title": title,
                "faculty": _clean_value(row.get("Faculty")),
            },
        )

    for requirement_type, entries in sorted(groups.items()):
        sorted_entries = sorted(entries, key=lambda item: item[0])
        for part, chunk in enumerate(_chunks(sorted_entries, COURSES_PER_AGGREGATE_CARD), start=1):
            course_list = "; ".join(
                f"{course_id} - {_clean_value(row.get('Course_Name'))} "
                f"(SU {_clean_value(row.get('SU_credit'))}, ECTS {_clean_value(row.get('ECTS'))})"
                for course_id, row, _line_no in chunk
            )
            term_name = _term_name(term_code)
            text = _lines(
                "Record type: aggregate course list for a program requirement category.",
                f"Program: {program} ({program_type}).",
                _maybe(f"Term: {term_code} - {term_name}.", term_code and term_name),
                _maybe("Term: current or non-term-specific minor catalog.", term_code is None and program_type == "minor"),
                f"Requirement category: {requirement_type} ({REQUIREMENT_LABELS.get(requirement_type, requirement_type)}).",
                f"This card lists {len(chunk)} of {len(sorted_entries)} courses in this category.",
                f"Courses: {course_list}.",
                f"Search aliases: {program} {requirement_type} courses, {program} curriculum, {program} electives.",
                f"Source data: {path.relative_to(root).as_posix()}.",
            )
            yield _document(
                kind="program_requirement_list",
                source=f"{program} {program_type} {term_code or 'current'} {requirement_type} list part {part}",
                source_path=path.relative_to(root).as_posix(),
                line_no=0,
                text=text,
                metadata={
                    "program": program,
                    "program_type": program_type,
                    "term_code": term_code,
                    "term_name": term_name,
                    "requirement_type": requirement_type,
                    "aggregate": True,
                    "part": part,
                    "course_count": len(chunk),
                },
            )


def _iter_course_catalog_file(path: Path, root: Path) -> Iterator[Document]:
    for line_no, row in _read_jsonl(path):
        course_id = _clean_value(row.get("course_id")) or _course_id(row.get("subj_code"), row.get("crse_numb"))
        title = _clean_value(row.get("title")) or _clean_value(row.get("header_text"))
        description = _normalize_text(row.get("description"))
        recent_terms = _format_offered_terms(row.get("last_offered_terms"), limit=12)
        text = _lines(
            "Record type: official course catalog overview.",
            f"Course: {course_id} - {title}.",
            _field("SU credits", row.get("su_credits")),
            _field("ECTS", row.get("ects")),
            _field("Engineering ECTS", row.get("engineering")),
            _field("Basic science ECTS", row.get("basic_science")),
            _field("Prerequisites", row.get("prerequisites")),
            _field("Corequisites", row.get("corequisites")),
            _maybe(f"Description: {description}", bool(description)),
            _maybe(f"Recent offered terms: {recent_terms}.", bool(recent_terms)),
            _field("Source URL", row.get("source_url")),
            _field("Scraped at", row.get("scraped_at")),
            f"Search aliases: {_course_aliases(course_id)}.",
            _source_line(path, root, line_no),
        )
        yield _document(
            kind="course_catalog_overview",
            source=f"course catalog {course_id}",
            source_path=path.relative_to(root).as_posix(),
            line_no=line_no,
            text=text,
            metadata={
                "course_id": course_id,
                "course_subject": _clean_value(row.get("subj_code")) or _subject_from_course_id(course_id),
                "course_number": _clean_value(row.get("crse_numb")) or _number_from_course_id(course_id),
                "course_title": title,
                "source_url": _clean_value(row.get("source_url")),
            },
        )

        offered_terms = row.get("last_offered_terms") or []
        if isinstance(offered_terms, list):
            for part, chunk in enumerate(_chunks(offered_terms, HISTORY_TERMS_PER_CARD), start=1):
                history = "; ".join(
                    " - ".join(
                        str(value)
                        for value in [
                            _clean_value(item.get("term")),
                            _clean_value(item.get("course_name")),
                            _maybe(f"SU credit {item.get('su_credit')}", item.get("su_credit") is not None),
                        ]
                        if value
                    )
                    for item in chunk
                    if isinstance(item, dict)
                )
                if not history:
                    continue
                text = _lines(
                    "Record type: course offering history.",
                    f"Course: {course_id} - {title}.",
                    f"Offering history part {part}: {history}.",
                    _source_line(path, root, line_no),
                )
                yield _document(
                    kind="course_offering_history",
                    source=f"course offering history {course_id} part {part}",
                    source_path=path.relative_to(root).as_posix(),
                    line_no=line_no,
                    text=text,
                    metadata={
                        "course_id": course_id,
                        "course_subject": _subject_from_course_id(course_id),
                        "course_number": _number_from_course_id(course_id),
                        "course_title": title,
                        "part": part,
                    },
                )


def _iter_basic_science_file(path: Path, root: Path) -> Iterator[Document]:
    for line_no, row in _read_jsonl(path):
        course_id = _clean_value(row.get("course_id")) or _course_id(row.get("subj_code"), row.get("crse_numb"))
        text = _lines(
            "Record type: engineering and basic science credit breakdown.",
            f"Course: {course_id}.",
            _field("ECTS", row.get("ects")),
            _field("Engineering ECTS", row.get("engineering")),
            _field("Basic science ECTS", row.get("basic_science")),
            _field("Breakdown present on source page", row.get("breakdown_present")),
            _field("Scrape OK", row.get("scrape_ok")),
            _field("Scrape error", row.get("scrape_error")),
            _field("Source URL", row.get("source_url")),
            _field("Scraped at", row.get("scraped_at")),
            f"Search aliases: {_course_aliases(course_id)}, engineering credits, basic science credits.",
            _source_line(path, root, line_no),
        )
        yield _document(
            kind="basic_science_credit",
            source=f"credit breakdown {course_id}",
            source_path=path.relative_to(root).as_posix(),
            line_no=line_no,
            text=text,
            metadata={
                "course_id": course_id,
                "course_subject": _clean_value(row.get("subj_code")) or _subject_from_course_id(course_id),
                "course_number": _clean_value(row.get("crse_numb")) or _number_from_course_id(course_id),
                "source_url": _clean_value(row.get("source_url")),
            },
            )


def _iter_degree_requirement_source_file(path: Path, root: Path) -> Iterator[Document]:
    for line_no, row in _read_jsonl(path):
        record_type = _clean_value(row.get("record_type")) or "degree_requirement_record"
        program = _clean_value(row.get("program")) or path.stem
        degree_code = _clean_value(row.get("degree_code")) or program
        program_name = _clean_value(row.get("program_name_tr") or row.get("program_name_en"))
        source_title = _clean_value(row.get("source_title")) or path.stem

        if record_type == "degree_requirement_profile":
            categories = _format_requirement_minima(row.get("category_minima"))
            rules = _format_list(row.get("rules"))
            text = _lines(
                "Record type: official degree requirement profile.",
                f"Program: {program} / {degree_code} - {program_name}.",
                _field("Program requirements term", row.get("program_requirements_term_label") or row.get("program_requirements_term")),
                _field("Minimum total SU credits", row.get("total_min_su_credits")),
                _field("Minimum total ECTS", row.get("total_min_ects")),
                _field("Minimum program GPA", row.get("min_program_gpa")),
                _field("Minimum cumulative GPA", row.get("min_cumulative_gpa")),
                _maybe(f"Category minima: {categories}.", bool(categories)),
                _maybe(f"Rules: {rules}.", bool(rules)),
                f"Source title: {source_title}.",
                _source_line(path, root, line_no),
            )
            yield _document(
                kind="degree_requirement_profile",
                source=f"{program} degree requirements profile {degree_code}",
                source_path=path.relative_to(root).as_posix(),
                line_no=line_no,
                text=text,
                metadata={
                    "program": program,
                    "degree_code": degree_code,
                    "program_name": program_name,
                    "program_requirements_term": _clean_value(row.get("program_requirements_term")),
                    "source_authority": "official_degree_requirements",
                    "aggregate": True,
                },
            )
            continue

        if record_type == "degree_requirement_category_pool":
            category = _clean_value(row.get("category")) or "unknown"
            requirement_type = _canonical_requirement_type(category)
            courses = row.get("courses") or row.get("courses_representative") or []
            course_list = _format_course_entries(courses)
            choice_pool_text = _format_choice_pools(row.get("choice_pools"))
            text = _lines(
                "Record type: official degree requirement category pool.",
                f"Program: {program} / {degree_code} - {program_name}.",
                f"Category: {category} - {_clean_value(row.get('category_label_tr'))}.",
                _field("Minimum SU credits", row.get("min_su_credits")),
                _field("Minimum course count", row.get("min_courses")),
                _field("Selection rule", row.get("selection_rule")),
                _maybe(f"Courses in this pool: {course_list}.", bool(course_list)),
                _maybe(f"Choice pools: {choice_pool_text}.", bool(choice_pool_text)),
                _field("Full pool note", row.get("full_pool_note")),
                f"Source title: {source_title}.",
                _source_line(path, root, line_no),
            )
            yield _document(
                kind="degree_requirement_category_pool",
                source=f"{program} degree requirements {category} pool",
                source_path=path.relative_to(root).as_posix(),
                line_no=line_no,
                text=text,
                metadata={
                    "program": program,
                    "degree_code": degree_code,
                    "category": category,
                    "requirement_type": requirement_type,
                    "source_authority": "official_degree_requirements",
                    "aggregate": True,
                },
            )
            for course in _course_entries(courses, row.get("choice_pools")):
                course_id = _clean_value(course.get("code"))
                if not course_id:
                    continue
                course_text = _lines(
                    "Record type: official degree requirement pool course.",
                    f"Program: {program} / {degree_code}.",
                    f"Category: {category} - {_clean_value(row.get('category_label_tr'))}.",
                    f"Course: {course_id} - {_clean_value(course.get('title_tr') or course.get('title_en'))}.",
                    _field("SU credit", course.get("su_credits")),
                    _field("Equivalent to", course.get("equivalent_to")),
                    _field("Selection rule", row.get("selection_rule")),
                    _source_line(path, root, line_no),
                )
                yield _document(
                    kind="degree_requirement_pool_course",
                    source=f"{program} degree requirements {category} {course_id}",
                    source_path=path.relative_to(root).as_posix(),
                    line_no=line_no,
                    text=course_text,
                    metadata={
                        "program": program,
                        "degree_code": degree_code,
                        "category": category,
                        "requirement_type": requirement_type,
                        "course_id": course_id,
                        "course_subject": _subject_from_course_id(course_id),
                        "course_number": _number_from_course_id(course_id),
                        "source_authority": "official_degree_requirements",
                    },
                )
            continue

        if record_type == "official_degree_evaluation_projection":
            categories = row.get("categories") or []
            category_summary = _format_evaluation_categories(categories)
            text = _lines(
                "Record type: official degree evaluation projection.",
                f"Program: {program} / {degree_code} - {program_name}.",
                _field("Evaluation term", row.get("evaluation_term_label") or row.get("evaluation_term")),
                _field("Program requirements term", row.get("program_requirements_term")),
                _field("Source authority", row.get("source_authority")),
                _field("Applies to", row.get("applies_to")),
                _field("Result", row.get("result")),
                _field("Completed total SU credits", row.get("completed_total_su_credits")),
                _field("Completed total ECTS", row.get("completed_total_ects")),
                _field("Program GPA", row.get("program_gpa")),
                _field("Cumulative GPA", row.get("cumulative_gpa")),
                _maybe(f"Official category allocation summary: {category_summary}.", bool(category_summary)),
                f"Source title: {source_title}.",
                _source_line(path, root, line_no),
            )
            yield _document(
                kind="official_degree_evaluation_summary",
                source=f"{program} official degree evaluation summary {degree_code}",
                source_path=path.relative_to(root).as_posix(),
                line_no=line_no,
                text=text,
                metadata={
                    "program": program,
                    "degree_code": degree_code,
                    "program_requirements_term": _clean_value(row.get("program_requirements_term")),
                    "evaluation_term": _clean_value(row.get("evaluation_term")),
                    "source_authority": "official_degree_evaluation",
                    "aggregate": True,
                },
            )

            for category_entry in categories if isinstance(categories, list) else []:
                if not isinstance(category_entry, dict):
                    continue
                category = _clean_value(category_entry.get("category")) or "unknown"
                requirement_type = _canonical_requirement_type(category)
                courses = category_entry.get("courses") or []
                course_list = _format_course_entries(courses)
                category_text = _lines(
                    "Record type: official degree evaluation category allocation.",
                    f"Program: {program} / {degree_code}.",
                    f"Category: {category} - {_clean_value(category_entry.get('label_tr'))}.",
                    _field("Minimum SU credits", category_entry.get("min_su_credits")),
                    _field("Completed SU credits", category_entry.get("completed_su_credits")),
                    _field("Completed ECTS", category_entry.get("completed_ects")),
                    _field("Completed course count", category_entry.get("course_count")),
                    f"Courses officially allocated to this category: {course_list}.",
                    "Interpretation: for this official Degree Evaluation, these completed courses count in this exact category; do not move them to another category unless a newer official evaluation source says so.",
                    _source_line(path, root, line_no),
                )
                yield _document(
                    kind="official_degree_evaluation_category",
                    source=f"{program} official degree evaluation {category}",
                    source_path=path.relative_to(root).as_posix(),
                    line_no=line_no,
                    text=category_text,
                    metadata={
                        "program": program,
                        "degree_code": degree_code,
                        "category": category,
                        "requirement_type": requirement_type,
                        "source_authority": "official_degree_evaluation",
                        "aggregate": True,
                    },
                )
                for course in courses if isinstance(courses, list) else []:
                    if not isinstance(course, dict):
                        continue
                    course_id = _clean_value(course.get("code"))
                    if not course_id:
                        continue
                    course_text = _lines(
                        "Record type: official degree evaluation course assignment.",
                        f"Program: {program} / {degree_code}.",
                        f"Course: {course_id}.",
                        f"Officially allocated category: {category} - {_clean_value(category_entry.get('label_tr'))}.",
                        _field("SU credit counted", course.get("su_credits")),
                        "Interpretation: this course is already counted in the listed category in the official Degree Evaluation projection.",
                        _source_line(path, root, line_no),
                    )
                    yield _document(
                        kind="official_degree_evaluation_course_assignment",
                        source=f"{program} official degree evaluation {category} {course_id}",
                        source_path=path.relative_to(root).as_posix(),
                        line_no=line_no,
                        text=course_text,
                        metadata={
                            "program": program,
                            "degree_code": degree_code,
                            "category": category,
                            "requirement_type": requirement_type,
                            "course_id": course_id,
                            "course_subject": _subject_from_course_id(course_id),
                            "course_number": _number_from_course_id(course_id),
                            "source_authority": "official_degree_evaluation",
                        },
                    )
            continue

        text = _lines(
            "Record type: degree requirement source record.",
            f"Program: {program} / {degree_code} - {program_name}.",
            f"Record type field: {record_type}.",
            f"Raw JSON: {json.dumps(row, ensure_ascii=False, sort_keys=True)}.",
            _source_line(path, root, line_no),
        )
        yield _document(
            kind="degree_requirement_record",
            source=f"{program} degree requirement record {record_type}",
            source_path=path.relative_to(root).as_posix(),
            line_no=line_no,
            text=text,
            metadata={
                "program": program,
                "degree_code": degree_code,
                "source_authority": "official_degree_requirements",
            },
        )


def _iter_graduation_requirements_file(path: Path, root: Path) -> Iterator[Document]:
    category_rows: dict[str, list[tuple[dict[str, Any], int]]] = defaultdict(list)

    for line_no, row in _read_jsonl(path):
        if row.get("tip") == "MEZUNİYET_KOŞULLARI_ÖZETI":
            categories = row.get("kategoriler") or {}
            summary_lines = [
                "Record type: graduation requirements summary.",
                _field("Description", row.get("aciklama")),
                _field("Minimum total SU credits", row.get("toplam_min_su_kredi")),
            ]
            if isinstance(categories, dict):
                for name, details in categories.items():
                    if not isinstance(details, dict):
                        continue
                    notes = "; ".join(map(str, details.get("notlar") or []))
                    summary_lines.append(
                        _lines(
                            f"Category {name}:",
                            _field("minimum SU credits", details.get("min_su_kredi")),
                            _field("minimum course count", details.get("min_ders_sayisi")),
                            _field("minimum ECTS", details.get("min_akts")),
                            _maybe(f"notes: {notes}", bool(notes)),
                        )
                    )
                    yield _document(
                        kind="graduation_requirement_category",
                        source=f"graduation requirement category {name}",
                        source_path=path.relative_to(root).as_posix(),
                        line_no=line_no,
                        text=_lines(
                            "Record type: graduation requirement category.",
                            _field("Program description", row.get("aciklama")),
                            f"Category: {name}.",
                            _field("Minimum SU credits", details.get("min_su_kredi")),
                            _field("Minimum course count", details.get("min_ders_sayisi")),
                            _field("Minimum ECTS", details.get("min_akts")),
                            _maybe(f"Notes: {notes}.", bool(notes)),
                            _source_line(path, root, line_no),
                        ),
                        metadata={"category": name},
                    )
            yield _document(
                kind="graduation_requirement_summary",
                source="graduation requirements summary",
                source_path=path.relative_to(root).as_posix(),
                line_no=line_no,
                text=_lines(*summary_lines, _source_line(path, root, line_no)),
                metadata={"program": "BSCS"},
            )
            continue

        category = _clean_value(row.get("kategori")) or "unknown"
        category_rows[category].append((row, line_no))
        course_id = _clean_value(row.get("kod"))
        text = _lines(
            "Record type: graduation requirement course rule.",
            f"Category: {category}.",
            _field("Requirement status", row.get("zorunluluk")),
            _field("Course", f"{course_id} - {_clean_value(row.get('ad'))}" if course_id else row.get("ad")),
            _field("SU credit", row.get("su_kredi")),
            _field("Faculty", row.get("fakulte")),
            _field("Note", row.get("not")),
            _source_line(path, root, line_no),
        )
        yield _document(
            kind="graduation_requirement_course",
            source=f"graduation requirement {category} {course_id}",
            source_path=path.relative_to(root).as_posix(),
            line_no=line_no,
            text=text,
            metadata={"category": category, "course_id": course_id, "program": "BSCS"},
        )

    for category, entries in sorted(category_rows.items()):
        for part, chunk in enumerate(_chunks(entries, COURSES_PER_AGGREGATE_CARD), start=1):
            courses = "; ".join(
                f"{_clean_value(row.get('kod'))} - {_clean_value(row.get('ad'))} "
                f"({_clean_value(row.get('zorunluluk'))}, SU {_clean_value(row.get('su_kredi'))})"
                for row, _line_no in chunk
            )
            text = _lines(
                "Record type: aggregate graduation requirement course list.",
                f"Category: {category}.",
                f"This card lists {len(chunk)} of {len(entries)} course rules in this category.",
                f"Courses: {courses}.",
                f"Source data: {path.relative_to(root).as_posix()}.",
            )
            yield _document(
                kind="graduation_requirement_course_list",
                source=f"graduation requirement {category} list part {part}",
                source_path=path.relative_to(root).as_posix(),
                line_no=0,
                text=text,
                metadata={"category": category, "program": "BSCS", "aggregate": True, "part": part},
            )


def _iter_terms_file(path: Path, root: Path) -> Iterator[Document]:
    all_terms: list[str] = []
    for line_no, row in _read_jsonl(path):
        term_code = _clean_value(row.get("term"))
        majors = row.get("majors") or []
        majors_text = ", ".join(map(str, majors)) if isinstance(majors, list) else str(majors)
        all_terms.append(f"{term_code} ({_term_name(term_code)})")
        text = _lines(
            "Record type: available data term.",
            f"Term: {term_code} - {_term_name(term_code)}.",
            f"Programs with data in this term: {majors_text}.",
            _source_line(path, root, line_no),
        )
        yield _document(
            kind="available_term",
            source=f"available term {term_code}",
            source_path=path.relative_to(root).as_posix(),
            line_no=line_no,
            text=text,
            metadata={"term_code": term_code, "term_name": _term_name(term_code)},
        )

    if all_terms:
        text = _lines(
            "Record type: all available terms summary.",
            f"Available term codes and names: {'; '.join(all_terms)}.",
            f"Source data: {path.relative_to(root).as_posix()}.",
        )
        yield _document(
            kind="available_terms_summary",
            source="available terms summary",
            source_path=path.relative_to(root).as_posix(),
            line_no=0,
            text=text,
            metadata={"aggregate": True, "term_count": len(all_terms)},
        )


def _iter_schedule_file(path: Path, root: Path) -> Iterator[Document]:
    rows = list(_read_jsonl(path))
    by_course: dict[tuple[str, str], list[tuple[dict[str, Any], int]]] = defaultdict(list)

    for line_no, row in rows:
        term_code = _clean_value(row.get("term")) or path.stem
        course_id = _spaced_course_id(_clean_value(row.get("course_id")))
        by_course[(term_code, course_id)].append((row, line_no))
        meetings = _format_meetings(row.get("meetings"))
        text = _lines(
            "Record type: scheduled course section.",
            f"Term: {term_code} - {_term_name(term_code)}.",
            f"Course: {course_id} - {_clean_value(row.get('title'))}.",
            _field("CRN", row.get("crn")),
            _field("Section", row.get("section")),
            _field("Component", row.get("component")),
            _field("Credits", row.get("credits")),
            _maybe(f"Meetings: {meetings}.", bool(meetings)),
            _field("Subject", row.get("subject")),
            _field("Source URL", row.get("source_url")),
            f"Search aliases: {_course_aliases(course_id)}, CRN {_clean_value(row.get('crn'))}.",
            _source_line(path, root, line_no),
        )
        yield _document(
            kind="schedule_section",
            source=f"schedule {term_code} {course_id} section {_clean_value(row.get('section'))} CRN {_clean_value(row.get('crn'))}",
            source_path=path.relative_to(root).as_posix(),
            line_no=line_no,
            text=text,
            metadata={
                "term_code": term_code,
                "term_name": _term_name(term_code),
                "course_id": course_id,
                "course_subject": _clean_value(row.get("subject")) or _subject_from_course_id(course_id),
                "course_title": _clean_value(row.get("title")),
                "crn": _clean_value(row.get("crn")),
                "section": _clean_value(row.get("section")),
                "component": _clean_value(row.get("component")),
                "source_url": _clean_value(row.get("source_url")),
            },
        )

    for (term_code, course_id), entries in sorted(by_course.items()):
        for part, chunk in enumerate(_chunks(entries, SECTIONS_PER_AGGREGATE_CARD), start=1):
            sections = "; ".join(
                _lines(
                    f"CRN {_clean_value(row.get('crn'))}",
                    f"section {_clean_value(row.get('section'))}",
                    f"component {_clean_value(row.get('component'))}",
                    f"credits {_clean_value(row.get('credits'))}",
                    _format_meetings(row.get("meetings")),
                )
                for row, _line_no in chunk
            )
            title = _clean_value(chunk[0][0].get("title")) if chunk else ""
            text = _lines(
                "Record type: aggregate schedule for a course in a term.",
                f"Term: {term_code} - {_term_name(term_code)}.",
                f"Course: {course_id} - {title}.",
                f"This card lists {len(chunk)} of {len(entries)} scheduled sections for this course.",
                f"Sections: {sections}.",
                f"Source data: {path.relative_to(root).as_posix()}.",
            )
            yield _document(
                kind="schedule_course_list",
                source=f"schedule {term_code} {course_id} list part {part}",
                source_path=path.relative_to(root).as_posix(),
                line_no=0,
                text=text,
                metadata={
                    "term_code": term_code,
                    "term_name": _term_name(term_code),
                    "course_id": course_id,
                    "course_title": title,
                    "aggregate": True,
                    "part": part,
                    "section_count": len(chunk),
                },
            )


def _read_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            yield line_no, json.loads(line)


def _document(
    *,
    kind: str,
    source: str,
    source_path: str,
    line_no: int,
    text: str,
    metadata: dict[str, Any],
) -> Document:
    doc_id = _stable_id(kind, source_path, line_no, source, text[:160])
    base = {
        "source": source,
        "source_path": source_path,
        "document_type": kind,
        "record_line": line_no,
        "chunk_id": doc_id,
    }
    base.update(metadata)
    return Document(page_content=text, metadata=_clean_metadata(base))


def _stable_id(*parts: Any) -> str:
    raw = "\x1f".join("" if part is None else str(part) for part in parts)
    return "catalog::" + sha1(raw.encode("utf-8")).hexdigest()


def _clean_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    clean: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            clean[key] = value
        else:
            clean[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return clean


def _canonical_requirement_type(category: str) -> str:
    normalized = (category or "").lower()
    if "university" in normalized or "universite" in normalized:
        return "university"
    if "required" in normalized or "zorunlu" in normalized:
        return "required"
    if "core" in normalized or "cekirdek" in normalized:
        return "core"
    if "area" in normalized or "alan" in normalized:
        return "area"
    if "free" in normalized or "serbest" in normalized:
        return "free"
    return normalized or "unknown"


def _format_list(value: Any) -> str:
    if not isinstance(value, list):
        return _clean_value(value)
    return "; ".join(_clean_value(item) for item in value if _clean_value(item))


def _format_requirement_minima(value: Any) -> str:
    if not isinstance(value, list):
        return ""
    parts = []
    for item in value:
        if not isinstance(item, dict):
            continue
        label = _clean_value(item.get("label_tr") or item.get("category"))
        bits = [
            _maybe(f"min SU {_clean_value(item.get('min_su_credits'))}", item.get("min_su_credits") is not None),
            _maybe(f"min ECTS {_clean_value(item.get('min_ects'))}", item.get("min_ects") is not None),
            _maybe(f"min courses {_clean_value(item.get('min_courses'))}", item.get("min_courses") is not None),
        ]
        parts.append(f"{label} ({', '.join(bit for bit in bits if bit)})")
    return "; ".join(parts)


def _format_course_entries(value: Any) -> str:
    if not isinstance(value, list):
        return ""
    parts = []
    for item in value:
        if not isinstance(item, dict):
            continue
        code = _clean_value(item.get("code"))
        if not code:
            continue
        title = _clean_value(item.get("title_tr") or item.get("title_en"))
        credit = _clean_value(item.get("su_credits"))
        equivalent = _clean_value(item.get("equivalent_to"))
        details = ", ".join(
            detail
            for detail in [
                _maybe(f"SU {credit}", bool(credit)),
                _maybe(f"equivalent to {equivalent}", bool(equivalent)),
            ]
            if detail
        )
        label = " - ".join(part for part in [code, title] if part)
        parts.append(f"{label} ({details})" if details else label)
    return "; ".join(parts)


def _format_choice_pools(value: Any) -> str:
    if not isinstance(value, list):
        return ""
    parts = []
    for pool in value:
        if not isinstance(pool, dict):
            continue
        name = _clean_value(pool.get("name"))
        choose = _clean_value(pool.get("choose"))
        courses = _format_course_entries(pool.get("courses"))
        parts.append(
            _lines(
                _maybe(f"{name}:", bool(name)),
                _maybe(f"choose {choose}", bool(choose)),
                _maybe(f"courses {courses}", bool(courses)),
            )
        )
    return "; ".join(part for part in parts if part)


def _course_entries(courses: Any, choice_pools: Any) -> Iterator[dict[str, Any]]:
    if isinstance(courses, list):
        for course in courses:
            if isinstance(course, dict):
                yield course
    if isinstance(choice_pools, list):
        for pool in choice_pools:
            if not isinstance(pool, dict):
                continue
            pool_courses = pool.get("courses")
            if not isinstance(pool_courses, list):
                continue
            for course in pool_courses:
                if isinstance(course, dict):
                    yield course


def _format_evaluation_categories(value: Any) -> str:
    if not isinstance(value, list):
        return ""
    parts = []
    for item in value:
        if not isinstance(item, dict):
            continue
        label = _clean_value(item.get("label_tr") or item.get("category"))
        min_su = _clean_value(item.get("min_su_credits"))
        completed_su = _clean_value(item.get("completed_su_credits"))
        courses = _format_course_entries(item.get("courses"))
        parts.append(f"{label}: {completed_su}/{min_su} SU; courses: {courses}")
    return " | ".join(parts)


def _readable_days(days: str) -> str:
    compact = re.sub(r"[^A-Za-z]", "", days or "")
    names = [DAY_NAMES.get(ch.upper(), ch) for ch in compact]
    return "/".join(names) if names else days


def _format_meetings(meetings: Any) -> str:
    if not isinstance(meetings, list):
        return ""
    parts = []
    for meeting in meetings:
        if not isinstance(meeting, dict):
            continue
        days = _readable_days(_clean_value(meeting.get("days")))
        bits = [
            _maybe(days, bool(days)),
            _clean_value(meeting.get("time")),
            _maybe(f"at {_clean_value(meeting.get('where'))}", bool(_clean_value(meeting.get("where")))),
            _maybe(f"date range {_clean_value(meeting.get('date_range'))}", bool(_clean_value(meeting.get("date_range")))),
            _maybe(f"instructor {_clean_value(meeting.get('instructors'))}", bool(_clean_value(meeting.get("instructors")))),
        ]
        line = ", ".join(bit for bit in bits if bit)
        if line:
            parts.append(line)
    return " | ".join(parts)


def _format_offered_terms(value: Any, limit: int) -> str:
    if not isinstance(value, list):
        return ""
    parts = []
    for item in value[:limit]:
        if not isinstance(item, dict):
            continue
        term = _clean_value(item.get("term"))
        name = _clean_value(item.get("course_name"))
        credit = _clean_value(item.get("su_credit"))
        parts.append(" - ".join(part for part in [term, name, f"SU credit {credit}" if credit else ""] if part))
    suffix = f"; plus {len(value) - limit} older entries" if len(value) > limit else ""
    return "; ".join(parts + ([suffix] if suffix else []))


def _course_id(subject: Any, number: Any) -> str:
    subject_text = _clean_value(subject)
    number_text = _clean_value(number)
    return " ".join(part for part in [subject_text, number_text] if part).strip()


def _spaced_course_id(course_id: str) -> str:
    match = re.match(r"^([A-Za-z]+)\s*([0-9].*)$", course_id or "")
    if not match:
        return course_id
    return f"{match.group(1).upper()} {match.group(2).strip()}"


def _course_aliases(course_id: str) -> str:
    spaced = _spaced_course_id(course_id)
    compact = spaced.replace(" ", "")
    return ", ".join(dict.fromkeys([spaced, compact, spaced.lower(), compact.lower()]))


def _subject_from_course_id(course_id: str) -> str:
    return (course_id or "").split(" ", 1)[0]


def _number_from_course_id(course_id: str) -> str:
    parts = (course_id or "").split(" ", 1)
    return parts[1] if len(parts) > 1 else ""


def _term_name(term_code: str | None) -> str:
    if not term_code or not _is_term_code(term_code):
        return ""
    academic_year = f"{term_code[:4]}-{int(term_code[:4]) + 1}"
    season, tr_alias = TERM_SEASONS.get(term_code[-2:], ("Unknown", "Unknown"))
    return f"{season} {academic_year} ({tr_alias})"


def _is_term_code(value: str) -> bool:
    return bool(re.fullmatch(r"\d{6}", value or ""))


def _existing_files(paths: Iterable[Path]) -> Iterator[Path]:
    for path in paths:
        if path.exists():
            yield path


def _chunks(items: list[Any], size: int) -> Iterator[list[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _clean_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return _normalize_text(str(value))


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n+", " ", text)
    return text.strip()


def _field(label: str, value: Any) -> str:
    clean = _clean_value(value)
    return f"{label}: {clean}." if clean else ""


def _maybe(text: str, condition: bool) -> str:
    return text if condition else ""


def _source_line(path: Path, root: Path, line_no: int) -> str:
    if line_no:
        return f"Source data: {path.relative_to(root).as_posix()} line {line_no}."
    return f"Source data: {path.relative_to(root).as_posix()}."


def _lines(*parts: str) -> str:
    return "\n".join(part.strip() for part in parts if part and part.strip())
