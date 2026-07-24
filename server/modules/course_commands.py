from __future__ import annotations

"""Parse explicit course-history mutations from natural-language chat messages."""

import re
from dataclasses import dataclass, field


COURSE_RE = re.compile(r"\b([A-Z]{2,5})\s*-?\s*(\d{3,5}[A-Z]?)\b", re.IGNORECASE)
SHORTHAND_COURSE_RE = re.compile(
    r"\b([A-Z]{2,5})\s*-?\s*(\d{3}[A-Z]?)(?P<tail>(?:\s*[,/]?\s*\d{3}[A-Z]?){1,12})",
    re.IGNORECASE,
)

_REMOVE_RE = re.compile(
    r"\b(sil|çıkar|cikar|kaldır|kaldir|remove|delete|almadım|almadim|yanlış ekledim|yanlis ekledim)\b",
    re.IGNORECASE,
)
_STATUS_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("withdrawn", re.compile(r"\b(withdrawn|withdraw|çekildim|cekildim|bıraktım|biraktim)\b", re.IGNORECASE)),
    ("failed", re.compile(r"\b(failed|fail|kaldım|kaldim|başarısız|basarisiz)\b", re.IGNORECASE)),
    ("enrolled", re.compile(r"\b(enrolled|taking|alıyorum|aliyorum|kayıtlıyım|kayitliyim)\b", re.IGNORECASE)),
    ("transfer", re.compile(r"\b(transfer|saydırdım|saydirdim)\b", re.IGNORECASE)),
    ("exempted", re.compile(r"\b(exempted|exempt|muafım|muafim|muaf oldum)\b", re.IGNORECASE)),
    (
        "completed",
        re.compile(
            r"\b(aldım|aldim|tamamladım|tamamladim|geçtim|gectim|completed|finished|passed|bitirdim)\b",
            re.IGNORECASE,
        ),
    ),
)
_QUERY_TERMS_RE = re.compile(
    r"\b(mezun|kredi|credit|audit|kaç kaldı|kac kaldi|ne kaldı|ne kaldi|"
    r"ne almalıyım|ne almaliyim|what should|how many|do I need|öner|oner|recommend)\b",
    re.IGNORECASE,
)


def normalize_course_code(subject: str, number: str) -> str:
    return f"{subject.upper()} {number.upper()}"


def extract_course_codes(text: str) -> tuple[str, ...]:
    """Extract explicit and repeated-subject forms such as ``CS 445 412 404``."""
    raw = text or ""
    found: list[tuple[int, str]] = [
        (match.start(), normalize_course_code(match.group(1), match.group(2)))
        for match in COURSE_RE.finditer(raw)
    ]
    for match in SHORTHAND_COURSE_RE.finditer(raw):
        subject = match.group(1)
        for number_match in re.finditer(r"\d{3}[A-Z]?", match.group("tail"), re.IGNORECASE):
            found.append(
                (
                    match.start("tail") + number_match.start(),
                    normalize_course_code(subject, number_match.group()),
                )
            )
    return tuple(dict.fromkeys(code for _position, code in sorted(found)))


@dataclass(frozen=True)
class CourseHistoryCommand:
    course_codes: tuple[str, ...]
    action: str
    status: str | None = None
    standalone: bool = True
    matched_text: str = ""

    @property
    def is_remove(self) -> bool:
        return self.action == "remove"


def parse_course_history_command(text: str) -> CourseHistoryCommand | None:
    raw = text or ""
    codes = extract_course_codes(raw)
    if not codes:
        return None

    if _REMOVE_RE.search(raw):
        return CourseHistoryCommand(
            course_codes=codes,
            action="remove",
            standalone=not bool(_QUERY_TERMS_RE.search(raw)),
            matched_text=raw,
        )

    status = None
    for candidate, pattern in _STATUS_PATTERNS:
        if pattern.search(raw):
            status = candidate
            break
    if status is None:
        return None
    return CourseHistoryCommand(
        course_codes=codes,
        action="upsert",
        status=status,
        standalone=not bool(_QUERY_TERMS_RE.search(raw)),
        matched_text=raw,
    )
