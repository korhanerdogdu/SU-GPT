from __future__ import annotations

"""Pure parsing and artifact helpers for the public Sabanci syllabus corpus.

Network crawling lives in ``server/scripts/scrape_syllabi.py``.  Keeping the
parsers here makes source HTML and downloaded files independently testable and
lets the validator reuse the exact MIME/hash rules used by the crawler.
"""

import csv
import hashlib
import io
import json
import mimetypes
import re
import unicodedata
import zipfile
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import parse_qs, unquote, urljoin, urlparse

from bs4 import BeautifulSoup, Tag


LEGACY_SYLLABUS_URL = "https://www.sabanciuniv.edu/syllabus/?crn={crn}&term={term}"
NEW_SYLLABUS_URL = (
    "https://apps.sabanciuniv.edu/courses/syllabus/view.php"
    "?cn={number}&sc={subject}&section={section}&term={term}&view=public"
)
SOURCE_AUTHORITY = "Sabanci University public syllabus systems"
FORMAT_VERSION = 1

ALLOWED_PAGE_HOSTS = {
    "www.sabanciuniv.edu",
    "sabanciuniv.edu",
    "apps.sabanciuniv.edu",
}
ALLOWED_ATTACHMENT_HOSTS = {
    "www.sabanciuniv.edu",
    "sabanciuniv.edu",
    "apps.sabanciuniv.edu",
    "sucourse.sabanciuniv.edu",
}

_SPACE_RE = re.compile(r"\s+")
_COURSE_RE = re.compile(r"\b([A-Z]{2,})\s*[- ]?\s*(\d+[A-Z]*)\b")
_TERM_RE = re.compile(r"\b(20\d{4})\b")
_DOWNLOAD_EXTENSIONS = {
    ".pdf", ".doc", ".docx", ".txt", ".csv", ".xlsx", ".pptx", ".html", ".htm"
}
_SEMANTIC_LABELS = {
    "catalog description",
    "course description",
    "learning outcome",
    "learning outcomes",
    "course objective",
    "objective",
    "objectives",
    "assessment",
    "grading",
    "attendance",
    "course policy",
    "course policies",
    "academic integrity",
    "resources",
    "technology requirements",
    "weekly schedule",
    "weekly topics",
    "sdg",
    "sustainable development goals",
}


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return _SPACE_RE.sub(" ", unicodedata.normalize("NFC", str(value))).strip()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def term_labels(term: str) -> tuple[str, ...]:
    if not re.fullmatch(r"\d{6}", term):
        return (term,)
    year = int(term[:4])
    season = {"01": ("Fall", "Güz", "Guz"), "02": ("Spring", "Bahar"), "03": ("Summer", "Yaz")}.get(
        term[-2:], ("",)
    )
    academic_year = f"{year}-{year + 1}"
    return tuple(value for value in (term, *(f"{name} {academic_year}" for name in season if name)) if value)


def canonical_course_id(value: str) -> str:
    match = _COURSE_RE.search(clean_text(value).upper())
    return f"{match.group(1)} {match.group(2)}" if match else ""


def safe_filename(value: str, fallback: str = "attachment") -> str:
    value = unicodedata.normalize("NFC", unquote(value or ""))
    value = Path(value.replace("\\", "/")).name
    value = re.sub(r"[<>:\"/\\|?*\x00-\x1f]", "_", value).strip(" .")
    return value[:180] or fallback


def legacy_url(term: str, crn: str) -> str:
    return LEGACY_SYLLABUS_URL.format(term=term, crn=crn)


def new_app_url(term: str, subject: str, number: str, section: str) -> str:
    from urllib.parse import urlencode

    query = urlencode(
        {
            "cn": number,
            "sc": subject,
            "section": section,
            "term": term,
            "view": "public",
        }
    )
    return "https://apps.sabanciuniv.edu/courses/syllabus/view.php?" + query


def host_is_allowed(url: str, allowed: set[str]) -> bool:
    try:
        return urlparse(url).scheme == "https" and (urlparse(url).hostname or "").lower() in allowed
    except ValueError:
        return False


def _visible_root(soup: BeautifulSoup) -> Tag:
    root = soup.select_one("main, [role=main], #content, #main-content, .main-content") or soup.body or soup
    for tag in root.select("script, style, noscript, nav, footer"):
        tag.decompose()
    return root


def _course_and_title(text: str) -> tuple[str, str]:
    match = _COURSE_RE.search(text.upper())
    if not match:
        return "", ""
    course_id = f"{match.group(1)} {match.group(2)}"
    tail = clean_text(text[match.end():]).lstrip("-:, ")
    title = re.split(r"\b20\d{4}\b|\b(?:Fall|Spring|Summer|Güz|Guz|Bahar|Yaz)\b", tail, 1, flags=re.I)[0]
    return course_id, clean_text(title).strip("-, ")


def _identity(
    page_text: str,
    expected_term: str,
    expected_course_id: str,
    expected_section: str,
    fields: dict[str, str],
    *,
    require_section: bool = False,
) -> dict[str, Any]:
    page_upper = page_text.upper()
    found_course = canonical_course_id(page_text)
    course_match = bool(found_course and found_course == canonical_course_id(expected_course_id))
    term_match = any(label.casefold() in page_text.casefold() for label in term_labels(expected_term))
    found_terms = _TERM_RE.findall(page_text)
    if found_terms:
        term_match = expected_term in found_terms

    found_section = clean_text(
        fields.get("section")
        or fields.get("course section")
        or fields.get("ders şubesi")
        or fields.get("sube")
        or fields.get("şube")
    )
    if not found_section:
        title_section = re.search(
            r"\bSYLLABUS[- ]?[A-Z]{2,}\s*\d+[A-Z]*-([A-Z0-9.-]+)-20\d{4}\b",
            page_text,
            re.I,
        )
        if title_section:
            found_section = clean_text(title_section.group(1))
    if clean_text(expected_section) in {"", "0"}:
        found_section = ""
    section_match: bool | None = None
    if found_section:
        section_match = found_section.upper() == clean_text(expected_section).upper()
    elif require_section and clean_text(expected_section):
        section_match = False

    return {
        "identity_verified": bool(course_match and term_match and section_match is not False),
        "identity_course_match": course_match,
        "identity_term_match": term_match,
        "identity_section_match": section_match,
        "page_course_id": found_course,
        "page_term_codes": sorted(set(found_terms)),
        "page_section": found_section,
        "page_contains_expected_course_literal": canonical_course_id(expected_course_id).replace(" ", "") in page_upper.replace(" ", ""),
    }


def _endpoint_target_verified(
    base_url: str,
    expected_term: str,
    expected_course_id: str,
    expected_section: str,
    expected_crn: str,
) -> bool:
    query = parse_qs(urlparse(base_url).query)
    if (query.get("term") or query.get("term_in") or [""])[0] != expected_term:
        return False
    if query.get("crn"):
        return bool(expected_crn and query["crn"][0] == expected_crn)
    expected = canonical_course_id(expected_course_id).split()
    if len(expected) != 2:
        return False
    return bool(
        (query.get("sc") or [""])[0].upper() == expected[0]
        and (query.get("cn") or [""])[0].upper() == expected[1]
        and (query.get("section") or [""])[0].upper() == clean_text(expected_section).upper()
    )


def _key_value_fields(root: Tag) -> dict[str, str]:
    fields: dict[str, str] = {}
    for row in root.find_all("tr"):
        cells = row.find_all(["th", "td"], recursive=False)
        if len(cells) < 2:
            continue
        key = clean_text(cells[0].get_text(" ", strip=True)).rstrip(":").casefold()
        value = clean_text(" ".join(cell.get_text(" ", strip=True) for cell in cells[1:]))
        if key and value and len(key) <= 90:
            fields.setdefault(key, value)
    for dt in root.find_all("dt"):
        dd = dt.find_next_sibling("dd")
        if dd:
            key = clean_text(dt.get_text(" ", strip=True)).rstrip(":").casefold()
            value = clean_text(dd.get_text(" ", strip=True))
            if key and value:
                fields.setdefault(key, value)
    return fields


def _email_and_instructors(root: Tag, fields: dict[str, str]) -> tuple[list[str], list[str]]:
    emails: list[str] = []
    names: list[str] = []
    for anchor in root.select("a[href^='mailto:']"):
        email = clean_text(anchor.get("href", "").split(":", 1)[-1].split("?", 1)[0]).lower()
        if email and email not in emails:
            emails.append(email)
        parent_text = clean_text(anchor.parent.get_text(" ", strip=True) if anchor.parent else "")
        candidate = clean_text(parent_text.replace(anchor.get_text(" ", strip=True), "").replace(email, ""))
        candidate = re.sub(r"(?i)\binstructors?\b\s*:??", "", candidate).strip(" ,:-")
        if candidate.casefold() in {"email", "e-mail", "e-posta", "mail"}:
            candidate = ""
        if not candidate:
            heading = anchor.find_previous(["h6", "h5", "h4"])
            heading_text = clean_text(heading.get_text(" ", strip=True) if heading else "")
            if heading_text.casefold() not in {
                "instructor",
                "instructors",
                "instructor(s) information",
                "öğretim elemanı",
            }:
                candidate = heading_text
        if candidate and "@" not in candidate and len(candidate) <= 120 and candidate not in names:
            names.append(candidate)
    for key, value in fields.items():
        if "instructor" in key or "öğretim elemanı" in key or "ogretim elemani" in key:
            for part in re.split(r"\s*[;|]\s*|\s{2,}", value):
                part = clean_text(re.sub(r"[\w.+-]+@[\w.-]+", "", part)).strip(" ,:-")
                if part and part not in names:
                    names.append(part)
    return names, emails


def _attachment_links(root: Tag, base_url: str) -> list[dict[str, str]]:
    links: list[dict[str, str]] = []
    seen: set[str] = set()
    for anchor in root.find_all("a", href=True):
        raw_href = clean_text(anchor.get("href"))
        url = urljoin(base_url, raw_href)
        parsed = urlparse(url)
        decoded_path = unquote(parsed.path)
        query_lower = unquote(parsed.query).lower()
        ext = Path(decoded_path).suffix.lower()
        looks_downloadable = (
            "syllabusdownload.php" in decoded_path.lower()
            or "filename=" in query_lower
            or ext in _DOWNLOAD_EXTENSIONS
        )
        if not looks_downloadable or url in seen or not host_is_allowed(url, ALLOWED_ATTACHMENT_HOSTS):
            continue
        seen.add(url)
        query_filename = re.search(r"(?:^|&)filename=([^&]+)", parsed.query, re.I)
        candidate_name = unquote(query_filename.group(1)) if query_filename else Path(decoded_path).name
        label = clean_text(anchor.get_text(" ", strip=True))
        filename = safe_filename(label if Path(label).suffix else candidate_name)
        links.append({"url": url, "raw_href": raw_href, "label": label, "filename": filename})
    return links


def _document_order_sections(root: Tag) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    current = "Overview"
    for node in root.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "tr"]):
        if node.name in {"p", "li", "tr"} and node.find_parent(["p", "li", "tr"]):
            continue
        text = clean_text(node.get_text(" ", strip=True))
        if not text:
            continue
        if node.name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            current = text[:160]
            sections.setdefault(current, [])
        else:
            bucket = sections.setdefault(current, [])
            if text not in bucket:
                bucket.append(text)
    return {key: "\n".join(values) for key, values in sections.items() if values}


def _course_and_title_from_root(root: Tag, fallback_text: str) -> tuple[str, str]:
    for heading in root.find_all(["h1", "h2", "h3", "h4"]):
        text = clean_text(heading.get_text(" ", strip=True))
        match = _COURSE_RE.search(text.upper())
        if not match:
            continue
        course_id = f"{match.group(1)} {match.group(2)}"
        tail = clean_text(text[match.end():]).lstrip("-:, ")
        tail = re.sub(r"^(?:section|şube|sube)\s+[A-Z0-9.-]+\s*", "", tail, flags=re.I)
        tail = re.split(
            r"\b20\d{4}\b|\b(?:Fall|Spring|Summer|Güz|Guz|Bahar|Yaz)\b",
            tail,
            1,
            flags=re.I,
        )[0]
        return course_id, clean_text(tail).strip("-, ")
    return _course_and_title(fallback_text)


def parse_new_app_html(
    html: bytes | str,
    *,
    base_url: str,
    expected_term: str,
    expected_course_id: str,
    expected_section: str,
    expected_crn: str = "",
) -> dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")
    root = _visible_root(soup)
    page_text = clean_text(root.get_text("\n", strip=True))
    document_title = clean_text(soup.title.get_text(" ", strip=True) if soup.title else "")
    identity_text = clean_text(f"{document_title} {page_text}")
    fields = _key_value_fields(root)
    sections = _document_order_sections(root)
    instructors, emails = _email_and_instructors(root, fields)
    attachments = _attachment_links(root, base_url)
    endpoint_target_verified = _endpoint_target_verified(
        base_url, expected_term, expected_course_id, expected_section, expected_crn
    )
    identity = _identity(
        identity_text,
        expected_term,
        expected_course_id,
        expected_section,
        fields,
        require_section=clean_text(expected_section) not in {"", "0"},
    )
    course_id, page_title = _course_and_title_from_root(root, identity_text)

    semantic_sections = {
        heading: body
        for heading, body in sections.items()
        if any(label in heading.casefold() for label in _SEMANTIC_LABELS)
    }
    semantic_fields = {
        key: value
        for key, value in fields.items()
        if any(label in key for label in _SEMANTIC_LABELS)
    }
    meaningful = bool(semantic_sections or semantic_fields or attachments)
    login_only = bool(re.search(r"\b(login|sign in|giriş yap)\b", page_text, re.I) and len(page_text) < 700)
    missing_message = bool(
        re.search(r"\b(no syllabus|syllabus (?:is )?not|not found|bulunamadı|yüklenmemiş)\b", page_text, re.I)
    )
    stale_term_response = bool(
        endpoint_target_verified
        and identity.get("identity_course_match")
        and not identity.get("identity_term_match")
        and meaningful
    )
    if identity["identity_verified"] and meaningful and not login_only:
        page_state = "published"
    elif (missing_message or stale_term_response) and endpoint_target_verified and not login_only:
        page_state = "not_published_candidate"
    elif identity["identity_verified"] and not meaningful and not login_only:
        page_state = "not_published_candidate"
    else:
        page_state = "indeterminate"

    return {
        "source_system": "new_app",
        "page_state": page_state,
        "course_id": course_id,
        "course_title": page_title,
        "instructor_names": instructors,
        "instructor_emails": emails,
        "fields": fields,
        "sections": sections,
        "inline_text": "\n\n".join(f"{heading}\n{body}" for heading, body in sections.items()),
        "attachments": attachments,
        "endpoint_target_verified": endpoint_target_verified,
        "stale_term_response": stale_term_response,
        **identity,
    }


def _legacy_inline(root: Tag) -> str:
    collecting = False
    parts: list[str] = []
    for node in root.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "strong", "b", "p", "li", "tr"]):
        text = clean_text(node.get_text(" ", strip=True))
        if not text:
            continue
        label = text.rstrip(":").casefold()
        if label == "syllabus":
            collecting = True
            continue
        if collecting and label == "attachments":
            break
        if collecting and text not in parts:
            parts.append(text)
    return "\n".join(parts)


def parse_legacy_html(
    html: bytes | str,
    *,
    base_url: str,
    expected_term: str,
    expected_course_id: str,
    expected_section: str,
    expected_crn: str = "",
) -> dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")
    root = _visible_root(soup)
    page_text = clean_text(root.get_text("\n", strip=True))
    document_title = clean_text(soup.title.get_text(" ", strip=True) if soup.title else "")
    identity_text = clean_text(f"{document_title} {page_text}")
    fields = _key_value_fields(root)
    instructors, emails = _email_and_instructors(root, fields)
    attachments = _attachment_links(root, base_url)
    endpoint_target_verified = _endpoint_target_verified(
        base_url, expected_term, expected_course_id, expected_section, expected_crn
    )
    inline_text = _legacy_inline(root)
    identity = _identity(identity_text, expected_term, expected_course_id, expected_section, fields)
    course_id, page_title = _course_and_title_from_root(root, identity_text)
    missing_message = bool(re.search(r"\b(no syllabus|syllabus (?:is )?not|not found|bulunamadı|yüklenmemiş)\b", page_text, re.I))
    meaningful_inline = len(clean_text(inline_text)) >= 40
    if identity["identity_verified"] and (meaningful_inline or attachments):
        page_state = "published"
    elif identity["identity_verified"] and not meaningful_inline and not attachments:
        page_state = "not_published_candidate"
    elif missing_message and (
        endpoint_target_verified
        or (identity.get("identity_course_match") and identity.get("identity_term_match"))
    ):
        page_state = "not_published_candidate"
    else:
        page_state = "indeterminate"
    return {
        "source_system": "legacy",
        "page_state": page_state,
        "course_id": course_id,
        "course_title": page_title,
        "instructor_names": instructors,
        "instructor_emails": emails,
        "fields": fields,
        "sections": {"Syllabus": inline_text} if inline_text else {},
        "inline_text": inline_text,
        "attachments": attachments,
        "endpoint_target_verified": endpoint_target_verified,
        **identity,
    }


def detect_attachment_format(payload: bytes, filename: str, declared_mime: str = "") -> dict[str, Any]:
    """Return a magic-byte based format; never trust URL/Content-Type alone."""
    declared = clean_text(declared_mime).split(";", 1)[0].lower()
    result = {"kind": "unknown", "mime": "application/octet-stream", "extension": "", "valid": False}
    if payload.startswith(b"%PDF-"):
        result.update(kind="pdf", mime="application/pdf", extension=".pdf", valid=True)
        return result
    if payload.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"):
        result.update(kind="doc", mime="application/msword", extension=".doc", valid=True)
        return result
    if payload.startswith(b"PK\x03\x04"):
        try:
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                names = set(archive.namelist())
                bad_member = archive.testzip()
                if bad_member:
                    return {**result, "error": f"ZIP CRC failed for {bad_member}"}
                if "word/document.xml" in names:
                    result.update(
                        kind="docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        extension=".docx",
                        valid=True,
                    )
                elif "xl/workbook.xml" in names:
                    result.update(
                        kind="xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        extension=".xlsx",
                        valid=True,
                    )
                elif "ppt/presentation.xml" in names:
                    result.update(
                        kind="pptx",
                        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                        extension=".pptx",
                        valid=True,
                    )
                else:
                    result["error"] = "ZIP is not a supported Office Open XML document"
                return result
        except (zipfile.BadZipFile, OSError) as exc:
            return {**result, "error": f"Invalid ZIP container: {exc}"}

    head = payload[:2048].lstrip().lower()
    if head.startswith(b"<!doctype html") or head.startswith(b"<html") or b"<html" in head[:500]:
        result.update(kind="html", mime="text/html", extension=".html", valid=False, error="download returned HTML")
        return result
    guessed = mimetypes.guess_type(filename)[0] or declared
    if guessed.startswith("text/") or Path(filename).suffix.lower() in {".txt", ".csv"}:
        result.update(
            kind="csv" if Path(filename).suffix.lower() == ".csv" else "text",
            mime=guessed or "text/plain",
            extension=Path(filename).suffix.lower() or ".txt",
            valid=bool(payload),
        )
    return result


def _paragraph_text(paragraph: Any) -> str:
    return clean_text(getattr(paragraph, "text", ""))


def _table_markdown(table: Any) -> str:
    rows: list[list[str]] = []
    for row in table.rows:
        values = [clean_text(cell.text).replace("|", "\\|") for cell in row.cells]
        if any(values):
            rows.append(values)
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    header = rows[0]
    body = rows[1:] or [[""] * width]
    return "\n".join(
        ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * width) + " |"]
        + ["| " + " | ".join(row) + " |" for row in body]
    )


def extract_docx(path: Path) -> dict[str, Any]:
    from docx import Document
    from docx.table import Table

    document = Document(path)
    units: list[dict[str, str]] = []
    current_heading = "Document"
    buffer: list[str] = []

    def flush() -> None:
        nonlocal buffer
        text = "\n".join(buffer).strip()
        if text:
            units.append({"locator_type": "section", "locator": current_heading, "text": text})
        buffer = []

    iterator = document.iter_inner_content() if hasattr(document, "iter_inner_content") else document.paragraphs
    for block in iterator:
        if isinstance(block, Table):
            table_text = _table_markdown(block)
            if table_text:
                buffer.append(table_text)
            continue
        text = _paragraph_text(block)
        if not text:
            continue
        style = clean_text(getattr(getattr(block, "style", None), "name", "")).casefold()
        if style.startswith("heading"):
            flush()
            current_heading = text
        else:
            buffer.append(text)
    flush()

    extras: list[str] = []
    for section in document.sections:
        for container_name, container in (("Header", section.header), ("Footer", section.footer)):
            text = "\n".join(clean_text(p.text) for p in container.paragraphs if clean_text(p.text))
            if text and text not in extras:
                extras.append(text)
                units.append({"locator_type": "section", "locator": container_name, "text": text})
    textbox_text = clean_text(" ".join(document.element.xpath(".//w:txbxContent//w:t/text()")))
    if textbox_text and all(textbox_text not in unit["text"] for unit in units):
        units.append({"locator_type": "section", "locator": "Text boxes", "text": textbox_text})
    return {"parse_state": "parsed" if units else "empty", "units": units, "parser": "python-docx-structured"}


def extract_pdf(path: Path) -> dict[str, Any]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception as exc:
            return {"parse_state": "failed", "units": [], "parser": "pypdf", "error": f"encrypted PDF: {exc}"}
    units: list[dict[str, str]] = []
    empty_pages = 0
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            try:
                text = clean_text(page.extract_text(extraction_mode="layout"))
            except TypeError:
                text = clean_text(page.extract_text())
        except Exception as exc:
            text = ""
            units.append({"locator_type": "page", "locator": str(page_number), "text": "", "error": str(exc)})
        if text:
            units.append({"locator_type": "page", "locator": str(page_number), "text": text})
        else:
            empty_pages += 1
    state = "parsed" if any(unit.get("text") for unit in units) else "ocr_required"
    return {
        "parse_state": state,
        "units": units,
        "parser": "pypdf-layout",
        "page_count": len(reader.pages),
        "empty_page_count": empty_pages,
    }


def extract_text_file(path: Path, kind: str) -> dict[str, Any]:
    payload = path.read_bytes()
    text = ""
    for encoding in ("utf-8-sig", "utf-8", "cp1254", "latin-1"):
        try:
            text = payload.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if kind == "html":
        soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text("\n", strip=True)
    elif kind == "csv":
        rows = list(csv.reader(io.StringIO(text)))
        text = "\n".join(" | ".join(clean_text(cell) for cell in row) for row in rows)
    text = clean_text(text)
    return {
        "parse_state": "parsed" if text else "empty",
        "units": [{"locator_type": "section", "locator": "Document", "text": text}] if text else [],
        "parser": f"builtin-{kind}",
    }


def extract_xlsx(path: Path) -> dict[str, Any]:
    from openpyxl import load_workbook

    workbook = load_workbook(path, read_only=True, data_only=True)
    units: list[dict[str, str]] = []
    for sheet in workbook.worksheets:
        lines: list[str] = []
        for row in sheet.iter_rows(values_only=True):
            values = [clean_text(value).replace("|", "\\|") for value in row]
            if any(values):
                lines.append(" | ".join(values))
        text = "\n".join(lines).strip()
        if text:
            units.append({"locator_type": "sheet", "locator": sheet.title, "text": text})
    workbook.close()
    return {"parse_state": "parsed" if units else "empty", "units": units, "parser": "openpyxl"}


def extract_pptx(path: Path) -> dict[str, Any]:
    from pptx import Presentation

    presentation = Presentation(path)
    units: list[dict[str, str]] = []
    for slide_number, slide in enumerate(presentation.slides, start=1):
        parts: list[str] = []
        for shape in slide.shapes:
            if getattr(shape, "has_text_frame", False):
                text = clean_text(getattr(shape, "text", ""))
                if text:
                    parts.append(text)
            if getattr(shape, "has_table", False):
                rows = [
                    " | ".join(clean_text(cell.text).replace("|", "\\|") for cell in row.cells)
                    for row in shape.table.rows
                ]
                if any(clean_text(row) for row in rows):
                    parts.append("\n".join(rows))
        if parts:
            units.append({"locator_type": "slide", "locator": str(slide_number), "text": "\n".join(parts)})
    return {"parse_state": "parsed" if units else "empty", "units": units, "parser": "python-pptx"}


def extract_attachment(path: Path, kind: str) -> dict[str, Any]:
    try:
        if kind == "pdf":
            return extract_pdf(path)
        if kind == "docx":
            return extract_docx(path)
        if kind in {"text", "csv", "html"}:
            return extract_text_file(path, kind)
        if kind == "xlsx":
            return extract_xlsx(path)
        if kind == "pptx":
            return extract_pptx(path)
        return {"parse_state": "unsupported", "units": [], "parser": "none"}
    except Exception as exc:
        return {"parse_state": "failed", "units": [], "parser": "none", "error": str(exc)}


def split_text(text: str, max_chars: int = 1400, overlap: int = 180) -> list[str]:
    text = clean_text(text)
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            boundary = max(text.rfind(". ", start + max_chars // 2, end), text.rfind("\n", start + max_chars // 2, end))
            if boundary > start:
                end = boundary + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks


def make_rag_chunks(record: dict[str, Any], attachment_parses: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    if record.get("publication_state") != "published":
        return []
    term = str(record["term"])
    crn = str(record["crn"])
    course_id = clean_text(record.get("course_id"))
    title = clean_text(record.get("course_title"))
    base = {
        "format_version": FORMAT_VERSION,
        "data_role": "course_syllabus",
        "authority_level": "official",
        "documentType": "course",
        "term": term,
        "term_code": term,
        "curriculum_term": term,
        "course_id": course_id,
        "course_title": title,
        "subject": clean_text(record.get("subject")),
        "course_number": clean_text(record.get("course_number")),
        "section": clean_text(record.get("section")),
        "component": clean_text(record.get("component")),
        "crn": crn,
        "instructors": " | ".join(record.get("instructor_names") or []),
        "instructor_emails": " | ".join(record.get("instructor_emails") or []),
        "publication_state": record.get("publication_state", "indeterminate"),
        "source_authority": SOURCE_AUTHORITY,
    }
    units: list[dict[str, Any]] = []
    for source in record.get("sources") or []:
        inline = clean_text(source.get("inline_text"))
        if inline:
            units.append(
                {
                    "source_kind": source.get("source_system", "page"),
                    "source_url": source.get("requested_url") or source.get("final_url"),
                    "source_document": source.get("raw_html_path", ""),
                    "locator_type": "section",
                    "locator": "Public syllabus page",
                    "text": inline,
                    "attachment_sha256": "",
                }
            )
    for attachment in record.get("attachments") or []:
        digest = str(attachment.get("sha256") or "")
        parsed = attachment_parses.get(digest) or {}
        for unit in parsed.get("units") or []:
            if not clean_text(unit.get("text")):
                continue
            units.append(
                {
                    "source_kind": "attachment",
                    "source_url": attachment.get("url", ""),
                    "source_document": attachment.get("local_path", ""),
                    "locator_type": unit.get("locator_type", "section"),
                    "locator": unit.get("locator", "Document"),
                    "text": unit.get("text", ""),
                    "attachment_sha256": digest,
                }
            )

    chunks: list[dict[str, Any]] = []
    for unit_index, unit in enumerate(units):
        for part_index, piece in enumerate(split_text(unit["text"])):
            prelude = (
                f"Official syllabus. Term: {term}. Course: {course_id} - {title}. "
                f"Section: {base['section']}. CRN: {crn}. "
                f"Instructor(s): {base['instructors'] or 'not listed'}. "
            )
            text = prelude + piece
            digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
            chunk_id = f"course_syllabus:{term}:{crn}:{unit_index}:{part_index}:{digest}"
            chunks.append(
                {
                    **base,
                    "document_type": "syllabus_attachment_chunk" if unit["source_kind"] == "attachment" else "syllabus_page_chunk",
                    "source_kind": unit["source_kind"],
                    "source_url": unit["source_url"],
                    "source_document": unit["source_document"],
                    "locator_type": unit["locator_type"],
                    "locator": str(unit["locator"]),
                    "attachment_sha256": unit["attachment_sha256"],
                    "text": text,
                    "chunk_id": chunk_id,
                }
            )
    return chunks


def build_record_text(record: dict[str, Any]) -> str:
    source_texts = [clean_text(source.get("inline_text")) for source in record.get("sources") or []]
    source_texts = [text for text in source_texts if text]
    attachments = record.get("attachments") or []
    return "\n".join(
        part
        for part in [
            "Official Sabanci University syllabus record.",
            f"Term: {record.get('term')}.",
            f"Course: {record.get('course_id')} - {record.get('course_title')}.",
            f"Section: {record.get('section')}. CRN: {record.get('crn')}.",
            f"Instructor(s): {' | '.join(record.get('instructor_names') or []) or 'not listed'}.",
            f"Instructor email(s): {' | '.join(record.get('instructor_emails') or []) or 'not listed'}.",
            f"Publication state: {record.get('publication_state')}. Capture state: {record.get('capture_state')}.",
            f"Attachments: {'; '.join(str(item.get('filename')) for item in attachments) or 'none'}.",
            *source_texts,
        ]
        if part
    )


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)
