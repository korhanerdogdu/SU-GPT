from __future__ import annotations

"""Fetch the public SUIS course schedule and write RAG-ready JSONL records.

The output is intentionally limited to source data.  It does not insert records into
Chroma, MongoDB, or any other retrieval/indexing layer.

Usage examples:
    python server/scripts/scrape_course_schedule.py --term 202601
    python server/scripts/scrape_course_schedule.py --term 202601 --output-dir data/schedule
"""

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import requests
from bs4 import BeautifulSoup, Tag
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


TERM_ENDPOINT = "https://suis.sabanciuniv.edu/prod/bwckgens.p_proc_term_date"
SCHEDULE_ENDPOINT = "https://suis.sabanciuniv.edu/prod/bwckschd.p_get_crse_unsec"
DETAIL_ENDPOINT = "https://suis.sabanciuniv.edu/prod/bwckschd.p_disp_detail_sched"
SOURCE_AUTHORITY = "Sabanci University SUIS (BannerWeb)"
FORMAT_VERSION = 1

DAY_NAMES = {
    "M": ("Monday", "Pazartesi"),
    "T": ("Tuesday", "Salı"),
    "W": ("Wednesday", "Çarşamba"),
    "R": ("Thursday", "Perşembe"),
    "F": ("Friday", "Cuma"),
    "S": ("Saturday", "Cumartesi"),
    "U": ("Sunday", "Pazar"),
}

COMPONENTS = {
    "R": "Recitation",
    "L": "Laboratory",
    "D": "Discussion",
}

SEASONS = {
    "01": ("Fall", "Güz"),
    "02": ("Spring", "Bahar"),
    "03": ("Summer", "Yaz"),
}


@dataclass(frozen=True)
class CourseHeader:
    title: str
    section_title: str
    crn: str
    source_course_code: str
    course_id: str
    subject: str
    course_number: str
    section: str
    component_code: str
    component: str


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def term_metadata(term: str) -> dict[str, str]:
    if not re.fullmatch(r"\d{6}", term):
        raise ValueError(f"Term must be a six-digit SU term code, got: {term!r}")
    season_code = term[-2:]
    if season_code not in SEASONS:
        raise ValueError(f"Unsupported SU term season in code: {term!r}")
    season_en, season_tr = SEASONS[season_code]
    academic_year = f"{term[:4]}-{int(term[:4]) + 1}"
    return {
        "term": term,
        "term_label": f"{season_en} {academic_year} ({season_tr})",
        "term_season_en": season_en,
        "term_season_tr": season_tr,
        "academic_year": academic_year,
    }


def build_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"POST"}),
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.headers.update(
        {
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "AdviSU-course-schedule-importer/1.0",
        }
    )
    return session


def fetch_subjects(session: requests.Session, term: str, timeout: float) -> list[str]:
    response = session.post(
        TERM_ENDPOINT,
        data={"p_calling_proc": "bwckschd.p_disp_dyn_sched", "p_term": term},
        timeout=timeout,
    )
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "html.parser")
    subjects = sorted(
        {
            value
            for option in soup.find_all("option")
            if (value := _clean_text(option.get("value"))).isalpha() and value.isupper()
        }
    )
    if not subjects:
        raise RuntimeError(f"SUIS returned no subjects for term {term}; refusing to write an empty snapshot")
    return subjects


def fetch_schedule_html(
    session: requests.Session,
    term: str,
    subjects: Iterable[str],
    timeout: float,
) -> bytes:
    payload: dict[str, Any] = {
        "term_in": term,
        "sel_subj": ["dummy", *subjects],
        "sel_day": "dummy",
        "sel_schd": "dummy",
        "sel_insm": "dummy",
        "sel_camp": "dummy",
        "sel_levl": "dummy",
        "sel_sess": "dummy",
        "sel_instr": "dummy",
        "sel_ptrm": "dummy",
        "sel_attr": "dummy",
        "sel_crse": "",
        "sel_title": "",
        "sel_from_cred": "",
        "sel_to_cred": "",
        "begin_hh": "0",
        "begin_mi": "0",
        "begin_ap": "a",
        "end_hh": "0",
        "end_mi": "0",
        "end_ap": "a",
    }
    response = session.post(SCHEDULE_ENDPOINT, data=payload, timeout=timeout)
    response.raise_for_status()
    return response.content


def parse_course_header(value: str) -> CourseHeader:
    raw = _clean_text(value)
    parts = raw.rsplit(" - ", 3)
    if len(parts) != 4:
        raise ValueError(f"Unexpected SUIS course heading: {raw!r}")

    section_title, crn, source_course_code, section = (_clean_text(part) for part in parts)
    section = re.sub(r"\s*\[\s*Syllabus\s*\]\s*$", "", section, flags=re.IGNORECASE).strip()
    if not crn.isdigit():
        raise ValueError(f"Unexpected CRN in SUIS course heading: {raw!r}")

    code_match = re.fullmatch(r"([A-Z]+)\s+(\d+[A-Z]?)", source_course_code)
    if not code_match:
        raise ValueError(f"Unexpected course code in SUIS course heading: {raw!r}")
    subject, source_number = code_match.groups()

    component_code = source_number[-1] if source_number[-1] in COMPONENTS else ""
    course_number = source_number[:-1] if component_code else source_number
    course_id = f"{subject} {course_number}"
    component = COMPONENTS.get(component_code, "Primary")
    title = _base_course_title(section_title, component_code)

    return CourseHeader(
        title=title,
        section_title=section_title,
        crn=crn,
        source_course_code=source_course_code,
        course_id=course_id,
        subject=subject,
        course_number=course_number,
        section=section,
        component_code=component_code,
        component=component,
    )


def _base_course_title(title: str, component_code: str) -> str:
    suffix = COMPONENTS.get(component_code)
    if not suffix:
        return title
    pattern = rf"\s*[-\u2013\u2014]?\s*{re.escape(suffix)}\s*$"
    cleaned = re.sub(pattern, "", title, flags=re.IGNORECASE).strip(" -\u2013\u2014")
    if component_code == "L":
        cleaned = re.sub(r"\s*[-\u2013\u2014]?\s*Lab\s*$", "", title, flags=re.IGNORECASE).strip(
            " -\u2013\u2014"
        )
    return cleaned or title


def _parse_clock(value: str) -> str | None:
    value = _clean_text(value)
    if not value:
        return None
    parsed = datetime.strptime(value.lower(), "%I:%M %p")
    return parsed.strftime("%H:%M")


def parse_time_range(value: str) -> tuple[str | None, str | None, int | None]:
    value = _clean_text(value)
    if not value or value.upper() == "TBA":
        return None, None, None
    parts = value.split(" - ")
    if len(parts) != 2:
        raise ValueError(f"Unexpected SUIS meeting time: {value!r}")
    start = _parse_clock(parts[0])
    end = _parse_clock(parts[1])
    if start is None or end is None:
        return start, end, None
    start_dt = datetime.strptime(start, "%H:%M")
    end_dt = datetime.strptime(end, "%H:%M")
    duration = int((end_dt - start_dt).total_seconds() // 60)
    if duration < 0:
        duration += 24 * 60
    return start, end, duration


def _parse_date_range(value: str) -> tuple[str | None, str | None]:
    value = _clean_text(value)
    if not value:
        return None, None
    parts = value.split(" - ")
    if len(parts) != 2:
        return None, None
    try:
        start = datetime.strptime(parts[0], "%b %d, %Y").date().isoformat()
        end = datetime.strptime(parts[1], "%b %d, %Y").date().isoformat()
    except ValueError:
        return None, None
    return start, end


def _day_metadata(value: str) -> tuple[list[str], list[str], list[str]]:
    normalized = _clean_text(value).upper()
    if normalized in {"", "TBA", "ARR", "ARRANGED"}:
        return [], [], []
    codes = [character for character in normalized if character in DAY_NAMES]
    names_en = [DAY_NAMES[code][0] for code in codes]
    names_tr = [DAY_NAMES[code][1] for code in codes]
    return codes, names_en, names_tr


def _normalize_instructors(value: str) -> tuple[str, list[str]]:
    display = _clean_text(value)
    display = re.sub(r"\(\s*([A-Za-z]+)\s*\)", r"(\1)", display)
    if not display:
        return "", []
    names = []
    for part in re.split(r"\s*,\s*", display):
        name = re.sub(r"\s*\([A-Za-z]+\)\s*$", "", part).strip()
        if name and name not in names:
            names.append(name)
    return display, names


def parse_meeting_row(cells: list[str]) -> dict[str, Any]:
    if len(cells) != 7:
        raise ValueError(f"Expected 7 SUIS meeting columns, found {len(cells)}: {cells!r}")
    meeting_type, time_raw, days_raw, location, date_range, schedule_type, instructors_raw = map(
        _clean_text, cells
    )
    start_time, end_time, duration_minutes = parse_time_range(time_raw)
    day_codes, day_names_en, day_names_tr = _day_metadata(days_raw)
    start_date, end_date = _parse_date_range(date_range)
    instructors, instructor_names = _normalize_instructors(instructors_raw)
    scheduled = bool(day_codes and start_time and end_time)

    return {
        "meeting_type": meeting_type,
        "status": "scheduled" if scheduled else "TBA",
        "time": time_raw or "TBA",
        "time_raw": time_raw,
        "start_time": start_time,
        "end_time": end_time,
        "duration_minutes": duration_minutes,
        "days": days_raw,
        "day_codes": day_codes,
        "day_names_en": day_names_en,
        "day_names_tr": day_names_tr,
        "where": location or "TBA",
        "date_range": date_range,
        "start_date": start_date,
        "end_date": end_date,
        "schedule_type": schedule_type,
        "instructors": instructors,
        "instructor_names": instructor_names,
    }


def _meeting_table(header_tag: Tag) -> Tag | None:
    parent_row = header_tag.parent
    if not isinstance(parent_row, Tag):
        return None
    details_row = parent_row.find_next_sibling("tr")
    if not isinstance(details_row, Tag):
        return None
    table = details_row.find("table")
    return table if isinstance(table, Tag) else None


def parse_schedule(html: bytes, term: str, scraped_at: str) -> list[dict[str, Any]]:
    term_info = term_metadata(term)
    soup = BeautifulSoup(html, "html.parser")
    records: list[dict[str, Any]] = []

    for header_tag in soup.find_all("th", attrs={"class": "ddlabel"}):
        header = parse_course_header(header_tag.get_text(" ", strip=True))
        table = _meeting_table(header_tag)
        meetings: list[dict[str, Any]] = []
        if table is not None:
            for row in table.find_all("tr"):
                cells = [cell.get_text(" ", strip=True) for cell in row.find_all("td")]
                if cells:
                    meetings.append(parse_meeting_row(cells))

        source_url = f"{DETAIL_ENDPOINT}?term_in={term}&crn_in={header.crn}"
        record = {
            "data_role": "course_schedule",
            "authority_level": "official",
            "document_type": "course_schedule_section",
            **term_info,
            "course_id": header.course_id,
            "source_course_code": header.source_course_code,
            "subject": header.subject,
            "course_number": header.course_number,
            "title": header.title,
            "section_title": header.section_title,
            "crn": header.crn,
            "section": header.section,
            "component_code": header.component_code,
            "component": header.component,
            "meetings": meetings,
            "meeting_days": _unique_join(
                day
                for meeting in meetings
                for day in meeting.get("day_names_en", [])
            ),
            "meeting_times": _unique_join(
                f"{meeting['start_time']}-{meeting['end_time']}"
                for meeting in meetings
                if meeting.get("start_time") and meeting.get("end_time")
            ),
            "locations": _unique_join(
                meeting.get("where", "")
                for meeting in meetings
                if meeting.get("where") and meeting.get("where") != "TBA"
            ),
            "instructors": _unique_join(
                name
                for meeting in meetings
                for name in meeting.get("instructor_names", [])
            ),
            "source_authority": SOURCE_AUTHORITY,
            "source_url": source_url,
            "source_endpoint": SCHEDULE_ENDPOINT,
            "scraped_at": scraped_at,
            "data_version": scraped_at[:10],
            "chunk_id": f"course_schedule:{term}:{header.crn}",
        }
        record["text"] = build_rag_text(record)
        records.append(record)

    if not records:
        raise RuntimeError(f"SUIS returned no course sections for term {term}; refusing to write an empty snapshot")
    records.sort(
        key=lambda row: (
            row["subject"],
            _natural_number_key(row["course_number"]),
            row["component_code"],
            row["section"],
            row["crn"],
        )
    )
    return records


def _natural_number_key(value: str) -> tuple[int, str]:
    match = re.match(r"(\d+)(.*)", value)
    return (int(match.group(1)), match.group(2)) if match else (10**9, value)


def _unique_join(values: Iterable[str]) -> str:
    unique: list[str] = []
    for value in values:
        cleaned = _clean_text(value)
        if cleaned and cleaned not in unique:
            unique.append(cleaned)
    return " | ".join(unique)


def _meeting_text(meeting: dict[str, Any]) -> str:
    if meeting.get("status") != "scheduled":
        parts = ["gün ve saat TBA/belirtilmemiş"]
    else:
        day_en = "/".join(meeting.get("day_names_en", []))
        day_tr = "/".join(meeting.get("day_names_tr", []))
        parts = [
            f"{day_tr}/{day_en}" if day_tr and day_en else day_tr or day_en,
            f"{meeting.get('start_time')}-{meeting.get('end_time')}",
        ]
    location = meeting.get("where")
    if location and location != "TBA":
        parts.append(f"yer/location {location}")
    instructors = meeting.get("instructors")
    if instructors:
        parts.append(f"öğretim elemanı/instructor {instructors}")
    date_range = meeting.get("date_range")
    if date_range:
        parts.append(f"tarih aralığı/date range {date_range}")
    schedule_type = meeting.get("schedule_type")
    if schedule_type:
        parts.append(f"schedule type {schedule_type}")
    return ", ".join(part for part in parts if part)


def build_rag_text(record: dict[str, Any]) -> str:
    meetings = record.get("meetings") or []
    meeting_text = " | ".join(_meeting_text(meeting) for meeting in meetings)
    if not meeting_text:
        meeting_text = "toplanti bilgisi yok/no meeting information"
    component = record.get("component") or "Primary"
    return (
        "Ders programı kaydı / course schedule record. "
        f"Dönem/term: {record['term']} - {record['term_label']}. "
        f"Ders/course: {record['course_id']} - {record['title']}. "
        f"CRN: {record['crn']}. Section: {record['section']}. "
        f"Bileşen/component: {component}. "
        f"Toplantılar/meetings: {meeting_text}. "
        f"Resmî kaynak/official source: {record['source_url']}"
    )


def write_snapshot(
    records: list[dict[str, Any]],
    term: str,
    subjects: list[str],
    output_dir: Path,
    scraped_at: str,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{term}.jsonl"
    payload = "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records)
    payload_bytes = payload.encode("utf-8")
    # Write bytes so the checksum and JSONL use the same LF line endings on every OS.
    output_path.write_bytes(payload_bytes)

    meeting_count = sum(len(record.get("meetings") or []) for record in records)
    scheduled_count = sum(
        1
        for record in records
        for meeting in record.get("meetings") or []
        if meeting.get("status") == "scheduled"
    )
    digest = hashlib.sha256(payload_bytes).hexdigest()
    manifest = {
        "format_version": FORMAT_VERSION,
        **term_metadata(term),
        "source_authority": SOURCE_AUTHORITY,
        "source_endpoints": [TERM_ENDPOINT, SCHEDULE_ENDPOINT],
        "scraped_at": scraped_at,
        "subject_count": len(subjects),
        "subjects": subjects,
        "distinct_course_count": len({record["course_id"] for record in records}),
        "section_count": len(records),
        "meeting_count": meeting_count,
        "scheduled_meeting_count": scheduled_count,
        "tba_meeting_count": meeting_count - scheduled_count,
        "jsonl_file": output_path.name,
        "jsonl_sha256": digest,
    }
    manifest_path = output_dir / f"{term}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    latest_path = output_dir / "latest.json"
    latest_path.write_text(
        json.dumps(
            {
                "term": term,
                "term_label": manifest["term_label"],
                "jsonl_file": output_path.name,
                "manifest_file": manifest_path.name,
                "scraped_at": scraped_at,
                "jsonl_sha256": digest,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return output_path, manifest_path, latest_path


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--term",
        required=True,
        help="Six-digit SU term code (for example 202601)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(project_root / "data" / "schedule"),
        help="Directory for JSONL, manifest, and latest pointer files",
    )
    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout in seconds")
    args = parser.parse_args()

    term = args.term.strip()
    term_metadata(term)
    output_dir = Path(args.output_dir).expanduser().resolve()
    scraped_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    with build_session() as session:
        subjects = fetch_subjects(session, term, args.timeout)
        html = fetch_schedule_html(session, term, subjects, args.timeout)
    records = parse_schedule(html, term, scraped_at)
    output_path, manifest_path, latest_path = write_snapshot(
        records=records,
        term=term,
        subjects=subjects,
        output_dir=output_dir,
        scraped_at=scraped_at,
    )

    meeting_count = sum(len(record["meetings"]) for record in records)
    print(f"Wrote {len(records)} course sections and {meeting_count} meetings to {output_path}")
    print(f"Manifest: {manifest_path}")
    print(f"Latest pointer: {latest_path}")


if __name__ == "__main__":
    main()
