from __future__ import annotations

"""
Deterministic parser for a Sabanci University "Academic Records Summary" (transcript) PDF.

The PDF is a rendered HTML table; `pypdf`'s text extraction preserves row order but not always
column spacing -- a course title is sometimes glued directly to the LEVEL token with no space
("Civic Involvement Projects I-NUG"), and long titles wrap onto one or two extra lines before the
LEVEL/GRADE/CREDIT/ECTS/STATUS tail appears. Both are handled by accumulating each term's course
table into one flattened string and matching whole rows with a single regex, rather than parsing
line by line.

Everything here is deterministic text extraction + arithmetic -- no model call, no invented
courses or grades. `parse_transcript` is the single entry point.
"""

import io
import re
from dataclasses import dataclass, field

from pypdf import PdfReader

# ---- vocab, straight from the transcript's own "ACADEMIC RECORDS GUIDE" pages ------------------

GRADE_POINTS: dict[str, float] = {
    "A": 4.0, "A-": 3.7, "B+": 3.3, "B": 3.0, "B-": 2.7,
    "C+": 2.3, "C": 2.0, "C-": 1.7, "D+": 1.3, "D": 1.0, "F": 0.0,
}
# The transcript's full grading vocabulary (letter grades + the non-GPA administrative codes from
# its own "OTHER GRADING CODES" table), shared with modules.mongodb for manual grade-entry
# validation so the two never drift apart.
ALL_GRADES: frozenset[str] = frozenset(
    {*GRADE_POINTS, "IP", "NP", "NA", "SL", "UL", "EL", "S", "I", "P", "T", "U", "W"}
)
# Longest/most-specific alternatives first so the regex alternation cannot short-match
# ("A" before "A-" would swallow the "-" as a separate, unmatched character).
_GRADE_ALTERNATION = "|".join(re.escape(g) for g in sorted(ALL_GRADES, key=len, reverse=True))
_LEVEL_ALTERNATION = "FDY|UG|MA|DR|SP"
_SEASON_RANK = {"Fall": 0, "Spring": 1, "Summer": 2}

_TERM_HEADER_RE = re.compile(r"^(?P<term>(Fall|Spring|Summer)\s+(\d{4})-(\d{4}))\s*Status\s*:")
_ROW_RE = re.compile(
    r"(?P<code>[A-Z]{2,6}\s?\d{3,5}[A-Z]?)\s*"
    r"(?P<title>.+?)\s?"
    rf"(?P<level>{_LEVEL_ALTERNATION})\s+"
    rf"(?P<grade>{_GRADE_ALTERNATION})\s+"
    r"(?P<credit>\d+\.\d{2})\s+"
    r"(?P<ects>\d+\.\d{2})"
    r"(?:\s+(?P<status>Excluded|Repeated))?"
)
_STOP_RE = re.compile(r"^(\*\s*The Term GPA calculation|SABANCI UNIVERSITY ACADEMIC RECORDS GUIDE)")
_NORMALIZE_CODE_RE = re.compile(r"\s+")


def _normalize_code(code: str) -> str:
    return _NORMALIZE_CODE_RE.sub(" ", code.strip()).upper()


def term_sort_key(term: str) -> tuple[int, int]:
    """"Fall 2021-2022" -> (2021, 0); "Spring 2021-2022" -> (2021, 1); "Summer" -> (2021, 2).

    Matches transcript chronological order within one academic year (Fall, then Spring, then
    an optional Summer), so a plain tuple comparison tells which of two attempts is more recent.
    """
    m = re.match(r"(Fall|Spring|Summer)\s+(\d{4})-\d{4}", term)
    if not m:
        return (0, 0)
    return (int(m.group(2)), _SEASON_RANK[m.group(1)])


@dataclass
class TranscriptAttempt:
    course_code: str
    title: str
    grade: str
    credit: float
    ects: float
    term: str
    status: str | None = None  # "Excluded" | "Repeated" | None


@dataclass
class TranscriptCourse:
    course_code: str
    title: str
    grade: str
    su_credits: float
    ects: float
    term: str
    counts_in_gpa: bool


@dataclass
class ParsedTranscript:
    attempts: list[TranscriptAttempt] = field(default_factory=list)
    courses: list[TranscriptCourse] = field(default_factory=list)
    gpa: float | None = None
    total_su_credits: float | None = None
    total_ects: float | None = None
    warnings: list[str] = field(default_factory=list)


def _extract_pages_text(data: bytes) -> list[str]:
    reader = PdfReader(io.BytesIO(data))
    return [page.extract_text() or "" for page in reader.pages]


def _term_blocks(pages_text: list[str]) -> list[tuple[str, str]]:
    """Flatten the document into (term, accumulated_row_text) blocks.

    Every non-boilerplate line after a term header is appended to that term's buffer (joined with
    spaces so a wrapped title's continuation line rejoins its row). Summary/legend lines never
    match the row regex, so they are harmless to include rather than needing an exact filter.
    """
    blocks: list[tuple[str, str]] = []
    current_term: str | None = None
    buffer: list[str] = []
    stopped = False

    def flush() -> None:
        if current_term is not None and buffer:
            blocks.append((current_term, " ".join(buffer)))

    for page_text in pages_text:
        if stopped:
            break
        for raw_line in page_text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if _STOP_RE.match(line):
                stopped = True
                break
            header_match = _TERM_HEADER_RE.match(line)
            if header_match:
                flush()
                current_term = header_match.group("term")
                buffer = []
                continue
            if current_term is None:
                continue  # preamble (name/student number table) before the first term
            if line == "COURSE CODE COURSE TITLE LEVEL GRADE CREDIT ECTS STATUS":
                continue
            if line.startswith(("Program :", "Standing:", "Dean's List:", "GPA SU Credits")):
                continue
            if line in ("Term *", "Cumulative", "Term"):
                continue
            buffer.append(line)
    flush()
    return blocks


def _parse_attempts(blocks: list[tuple[str, str]]) -> list[TranscriptAttempt]:
    attempts: list[TranscriptAttempt] = []
    for term, text in blocks:
        for m in _ROW_RE.finditer(text):
            title = re.sub(r"\s+", " ", m.group("title")).strip(" -")
            attempts.append(TranscriptAttempt(
                course_code=_normalize_code(m.group("code")),
                title=title,
                grade=m.group("grade"),
                credit=float(m.group("credit")),
                ects=float(m.group("ects")),
                term=term,
                status=m.group("status"),
            ))
    return attempts


def _cumulative_gpa_from_summary(pages_text: list[str]) -> float | None:
    joined = " ".join(pages_text)
    m = re.search(r"CGPA\s*\*\*\s*[\r\n]?\s*[\d.]+\s+[\d.]+\s+([\d.]+)", joined)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def _totals_from_summary(pages_text: list[str]) -> tuple[float | None, float | None]:
    joined = " ".join(pages_text)
    m = re.search(r"Total Earned SU Credits.*?Total Earned ECTS.*?CGPA.*?([\d.]+)\s+([\d.]+)\s+([\d.]+)", joined, re.DOTALL)
    if m:
        try:
            return float(m.group(1)), float(m.group(2))
        except ValueError:
            return None, None
    return None, None


def _build_result(attempts: list[TranscriptAttempt]) -> ParsedTranscript:
    """Attempts -> de-duplicated completed courses + credit-weighted GPA.

    De-duplication keeps only the most recent (by term) non-"Excluded" attempt per course code --
    matching the transcript's own "only the latest grade received is included" rule -- and drops a
    course entirely if that latest attempt is a Withdrawal (never completed). Shared by
    ``parse_transcript`` and the unit tests so the two can never drift apart.
    """
    result = ParsedTranscript(attempts=attempts)
    if not attempts:
        result.warnings.append("No course rows were recognised in this PDF.")
        return result

    latest_by_code: dict[str, TranscriptAttempt] = {}
    for attempt in attempts:
        if attempt.status == "Excluded":
            continue
        existing = latest_by_code.get(attempt.course_code)
        if existing is None or term_sort_key(attempt.term) >= term_sort_key(existing.term):
            latest_by_code[attempt.course_code] = attempt

    for code, attempt in sorted(latest_by_code.items()):
        if attempt.grade == "W":
            continue  # withdrawn and never retaken -> not completed
        result.courses.append(TranscriptCourse(
            course_code=code,
            title=attempt.title,
            grade=attempt.grade,
            su_credits=attempt.credit,
            ects=attempt.ects,
            term=attempt.term,
            counts_in_gpa=attempt.grade in GRADE_POINTS,
        ))

    gpa_courses = [c for c in result.courses if c.counts_in_gpa]
    if gpa_courses:
        total_points = sum(GRADE_POINTS[c.grade] * c.su_credits for c in gpa_courses)
        total_credit = sum(c.su_credits for c in gpa_courses)
        result.gpa = round(total_points / total_credit, 2) if total_credit else None

    return result


def parse_transcript_text(pages_text: list[str]) -> ParsedTranscript:
    """Same pipeline as ``parse_transcript``, from already-extracted page text. Exists so tests
    (and any caller that already has text, e.g. from a different extraction backend) never need
    to construct or store a PDF file."""
    attempts = _parse_attempts(_term_blocks(pages_text))
    result = _build_result(attempts)
    result.total_su_credits, result.total_ects = _totals_from_summary(pages_text)
    reported_gpa = _cumulative_gpa_from_summary(pages_text)
    if reported_gpa is not None and result.gpa is not None and abs(reported_gpa - result.gpa) > 0.02:
        result.warnings.append(
            f"Computed GPA {result.gpa} does not match the transcript's reported CGPA {reported_gpa}."
        )
    return result


def parse_transcript(data: bytes) -> ParsedTranscript:
    """Parse a Sabanci "Academic Records Summary" PDF into attempts + de-duplicated courses."""
    return parse_transcript_text(_extract_pages_text(data))
