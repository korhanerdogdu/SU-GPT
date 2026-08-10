from __future__ import annotations

"""Build a complete, provenance-rich public syllabus corpus for SUIS terms.

The crawler accounts for every schedule CRN, checks both the legacy CRN page and
the newer public syllabus app, downloads content-addressed attachments, extracts
RAG text, and writes deterministic JSONL plus manifests.  Empty syllabi are
confirmed with an independent second fetch; transport/parser failures are never
mislabelled as "not published".

Examples:
    python server/scripts/scrape_syllabi.py --terms 202601 202502
    python server/scripts/scrape_syllabi.py --terms 202601 --resume
"""

import argparse
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.syllabus_pipeline import (  # noqa: E402
    ALLOWED_ATTACHMENT_HOSTS,
    ALLOWED_PAGE_HOSTS,
    FORMAT_VERSION,
    SOURCE_AUTHORITY,
    build_record_text,
    clean_text,
    detect_attachment_format,
    extract_attachment,
    host_is_allowed,
    legacy_url,
    make_rag_chunks,
    new_app_url,
    parse_legacy_html,
    parse_new_app_html,
    safe_filename,
    sha256_bytes,
)

from scripts import scrape_course_schedule as schedule_scraper  # noqa: E402


CRAWLER_VERSION = "1.0.0"
DEFAULT_TERMS = ("202601", "202502")
PAGE_MAX_BYTES = 8 * 1024 * 1024
ATTACHMENT_MAX_BYTES = 64 * 1024 * 1024
_WRITE_LOCK = threading.Lock()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "POST"}),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=4)
    session.mount("https://", adapter)
    session.headers.update(
        {
            "User-Agent": "AdviSU-public-syllabus-corpus/1.0 (+academic RAG; respectful crawl)",
            "Accept": "text/html,application/xhtml+xml,application/pdf,application/octet-stream;q=0.9,*/*;q=0.8",
        }
    )
    return session


def _read_limited(response: requests.Response, maximum: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    for chunk in response.iter_content(128 * 1024):
        if not chunk:
            continue
        total += len(chunk)
        if total > maximum:
            raise ValueError(f"response exceeds maximum allowed size ({maximum} bytes)")
        chunks.append(chunk)
    return b"".join(chunks)


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def fetch_page(
    session: requests.Session,
    *,
    url: str,
    timeout: float,
    raw_path: Path,
    output_root: Path,
    source_system: str,
    schedule_row: dict[str, Any],
) -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "source_system": source_system,
        "requested_url": url,
        "fetched_at": utc_now(),
        "page_state": "indeterminate",
    }
    try:
        response = session.get(url, timeout=timeout, allow_redirects=True, stream=True)
        payload = _read_limited(response, PAGE_MAX_BYTES)
        evidence.update(
            {
                "http_status": response.status_code,
                "final_url": response.url,
                "declared_content_type": response.headers.get("Content-Type", ""),
                "response_bytes": len(payload),
                "page_sha256": sha256_bytes(payload),
            }
        )
        if not host_is_allowed(response.url, ALLOWED_PAGE_HOSTS):
            evidence["error"] = "redirected to a non-allowlisted host"
            return evidence
        if response.status_code != 200:
            evidence["error"] = f"HTTP {response.status_code}"
            return evidence
        content_type = response.headers.get("Content-Type", "").lower()
        if "html" not in content_type and b"<html" not in payload[:2000].lower():
            evidence["error"] = "syllabus endpoint did not return HTML"
            return evidence

        _atomic_write(raw_path, payload)
        evidence["raw_html_path"] = _relative(raw_path, output_root)
        kwargs = {
            "base_url": response.url,
            "expected_term": str(schedule_row["term"]),
            "expected_course_id": str(schedule_row["course_id"]),
            "expected_section": str(schedule_row.get("section") or ""),
            "expected_crn": str(schedule_row.get("crn") or ""),
        }
        parsed = (
            parse_legacy_html(payload, **kwargs)
            if source_system == "legacy"
            else parse_new_app_html(payload, **kwargs)
        )
        evidence.update(parsed)
        return evidence
    except Exception as exc:
        evidence["error"] = f"{type(exc).__name__}: {exc}"
        return evidence


def _confirmation_agrees(first: dict[str, Any], second: dict[str, Any]) -> bool:
    return bool(
        first.get("page_state") == "not_published_candidate"
        and second.get("page_state") == "not_published_candidate"
        and (first.get("identity_verified") or first.get("endpoint_target_verified"))
        and (second.get("identity_verified") or second.get("endpoint_target_verified"))
        and first.get("page_sha256")
        and second.get("page_sha256")
    )


def download_attachment(
    session: requests.Session,
    link: dict[str, Any],
    *,
    timeout: float,
    output_root: Path,
    max_bytes: int,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    item = dict(link)
    item.update(download_state="failed", fetched_at=utc_now())
    url = str(link.get("url") or "")
    if not host_is_allowed(url, ALLOWED_ATTACHMENT_HOSTS):
        item["error"] = "attachment URL host is not allowlisted"
        return item, None
    try:
        response = session.get(url, timeout=timeout, allow_redirects=True, stream=True)
        payload = _read_limited(response, max_bytes)
        item.update(
            http_status=response.status_code,
            final_url=response.url,
            declared_mime=clean_text(response.headers.get("Content-Type", "")),
            byte_size=len(payload),
        )
        if not host_is_allowed(response.url, ALLOWED_ATTACHMENT_HOSTS):
            item["error"] = "attachment redirected to a non-allowlisted host"
            return item, None
        if response.status_code != 200:
            item["error"] = f"HTTP {response.status_code}"
            return item, None
        detected = detect_attachment_format(payload, str(link.get("filename") or ""), item["declared_mime"])
        item.update(
            detected_kind=detected["kind"],
            detected_mime=detected["mime"],
            extension=detected["extension"],
            integrity_valid=bool(detected["valid"]),
        )
        if not detected["valid"]:
            item["error"] = detected.get("error") or "unsupported or invalid attachment format"
            return item, None

        digest = sha256_bytes(payload)
        extension = detected["extension"] or Path(str(link.get("filename") or "")).suffix.lower()
        target = output_root / "attachments" / digest[:2] / f"{digest}{extension}"
        with _WRITE_LOCK:
            if not target.exists():
                _atomic_write(target, payload)
            elif sha256_bytes(target.read_bytes()) != digest:
                raise ValueError("content-addressed target exists with a different hash")
        item.update(
            download_state="downloaded",
            sha256=digest,
            local_path=_relative(target, output_root),
            original_filename=safe_filename(str(link.get("filename") or "attachment")),
        )
        parsed_path = output_root / "parsed" / digest[:2] / f"{digest}.json"
        with _WRITE_LOCK:
            if parsed_path.exists():
                parsed = json.loads(parsed_path.read_text(encoding="utf-8"))
            else:
                parsed = extract_attachment(target, str(detected["kind"]))
                parsed.update(
                    attachment_sha256=digest,
                    source_document=item["local_path"],
                    detected_kind=detected["kind"],
                )
                _atomic_write(
                    parsed_path,
                    (json.dumps(parsed, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
                )
        item.update(
            parse_state=parsed.get("parse_state", "failed"),
            parser=parsed.get("parser", "none"),
            parsed_path=_relative(parsed_path, output_root),
            parsed_unit_count=len(parsed.get("units") or []),
            page_count=parsed.get("page_count"),
            empty_page_count=parsed.get("empty_page_count"),
        )
        if parsed.get("error"):
            item["parse_error"] = parsed["error"]
        return item, parsed
    except Exception as exc:
        item["error"] = f"{type(exc).__name__}: {exc}"
        return item, None


def _unique(values: Any) -> list[str]:
    out: list[str] = []
    for value in values:
        cleaned = clean_text(value)
        if cleaned and cleaned not in out:
            out.append(cleaned)
    return out


def crawl_one(
    schedule_row: dict[str, Any],
    *,
    output_root: Path,
    timeout: float,
    delay: float,
    confirm_empty: bool,
    max_attachment_bytes: int,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    session = build_session()
    term = str(schedule_row["term"])
    crn = str(schedule_row["crn"])
    source_specs = [
        ("legacy", legacy_url(term, crn)),
        (
            "new_app",
            new_app_url(
                term,
                str(schedule_row.get("subject") or ""),
                str(schedule_row.get("course_number") or ""),
                str(schedule_row.get("section") or "0"),
            ),
        ),
    ]
    sources: list[dict[str, Any]] = []
    for source_system, url in source_specs:
        raw_path = output_root / "raw" / term / crn / f"{source_system}.html"
        sources.append(
            fetch_page(
                session,
                url=url,
                timeout=timeout,
                raw_path=raw_path,
                output_root=output_root,
                source_system=source_system,
                schedule_row=schedule_row,
            )
        )
        if delay:
            time.sleep(delay + random.random() * min(delay, 0.25))

    if confirm_empty and not any(source.get("page_state") == "published" for source in sources):
        for source, (source_system, url) in zip(sources, source_specs):
            confirm_path = output_root / "raw" / term / crn / f"{source_system}.confirm.html"
            second = fetch_page(
                session,
                url=url,
                timeout=timeout,
                raw_path=confirm_path,
                output_root=output_root,
                source_system=source_system,
                schedule_row=schedule_row,
            )
            source["confirmation"] = {
                key: value
                for key, value in second.items()
                if key not in {"fields", "sections", "inline_text", "attachments"}
            }
            source["empty_confirmation_agrees"] = _confirmation_agrees(source, second)
            if second.get("page_state") == "published":
                source.update(second)
            if delay:
                time.sleep(delay + random.random() * min(delay, 0.25))

    page_states = [str(source.get("page_state") or "indeterminate") for source in sources]
    if "published" in page_states:
        publication_state = "published"
        status_reason = "published content found on at least one official public syllabus surface"
    elif all(source.get("empty_confirmation_agrees") for source in sources):
        publication_state = "not_published_confirmed"
        status_reason = "both official public syllabus surfaces were identity-verified empty twice"
    else:
        publication_state = "indeterminate"
        status_reason = "no published content was found, but empty state could not be safely confirmed"

    links: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for source in sources:
        for link in source.get("attachments") or []:
            if link.get("url") not in seen_urls:
                seen_urls.add(str(link.get("url")))
                links.append({**link, "source_system": source.get("source_system")})

    attachments: list[dict[str, Any]] = []
    attachment_parses: dict[str, dict[str, Any]] = {}
    for link in links:
        item, parsed = download_attachment(
            session,
            link,
            timeout=timeout,
            output_root=output_root,
            max_bytes=max_attachment_bytes,
        )
        attachments.append(item)
        if parsed and item.get("sha256"):
            attachment_parses[str(item["sha256"])] = parsed
        if delay:
            time.sleep(delay + random.random() * min(delay, 0.25))

    names = _unique(
        name for source in sources for name in (source.get("instructor_names") or [])
    )
    emails = _unique(
        email for source in sources for email in (source.get("instructor_emails") or [])
    )
    schedule_instructors = str(schedule_row.get("instructors") or "")
    if schedule_instructors:
        names = _unique([*names, *schedule_instructors.split("|")])

    if publication_state == "not_published_confirmed":
        capture_state = "not_applicable"
    elif publication_state == "indeterminate":
        capture_state = "failed"
    else:
        failed_download = any(item.get("download_state") != "downloaded" for item in attachments)
        weak_parse = any(
            item.get("download_state") == "downloaded"
            and item.get("parse_state") not in {"parsed"}
            for item in attachments
        )
        has_inline = any(clean_text(source.get("inline_text")) for source in sources)
        has_parsed_attachment = any(item.get("parse_state") == "parsed" for item in attachments)
        if failed_download or weak_parse or not (has_inline or has_parsed_attachment):
            capture_state = "partial"
        else:
            capture_state = "complete"

    record: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "crawler_version": CRAWLER_VERSION,
        "data_role": "course_syllabus",
        "authority_level": "official",
        "document_type": "syllabus_overview",
        "documentType": "course",
        "term": term,
        "term_code": term,
        "curriculum_term": term,
        "course_id": schedule_row.get("course_id"),
        "course_title": schedule_row.get("title"),
        "subject": schedule_row.get("subject"),
        "course_number": schedule_row.get("course_number"),
        "source_course_code": schedule_row.get("source_course_code"),
        "section": schedule_row.get("section"),
        "component": schedule_row.get("component"),
        "component_code": schedule_row.get("component_code"),
        "crn": crn,
        "instructor_names": names,
        "instructor_emails": emails,
        "publication_state": publication_state,
        "capture_state": capture_state,
        "status_reason": status_reason,
        "sources": sources,
        "attachments": attachments,
        "source_authority": SOURCE_AUTHORITY,
        "schedule_source_url": schedule_row.get("source_url"),
        "scraped_at": utc_now(),
        "syllabus_id": f"syllabus:{term}:{crn}",
        "chunk_id": f"course_syllabus:{term}:{crn}:overview",
    }
    record["text"] = build_record_text(record)
    chunks = make_rag_chunks(record, attachment_parses)
    return record, attachment_parses, chunks


def _load_schedule(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    rows.sort(key=lambda row: str(row.get("crn") or ""))
    return rows


def ensure_schedule(term: str, schedule_dir: Path, timeout: float, fetch_missing: bool) -> Path:
    path = schedule_dir / f"{term}.jsonl"
    if path.exists():
        return path
    if not fetch_missing:
        raise FileNotFoundError(f"Missing schedule census: {path}")
    latest_path = schedule_dir / "latest.json"
    previous_latest = latest_path.read_bytes() if latest_path.exists() else None
    scraped_at = utc_now()
    with schedule_scraper.build_session() as session:
        subjects = schedule_scraper.fetch_subjects(session, term, timeout)
        html = schedule_scraper.fetch_schedule_html(session, term, subjects, timeout)
    records = schedule_scraper.parse_schedule(html, term, scraped_at)
    schedule_scraper.write_snapshot(records, term, subjects, schedule_dir, scraped_at)
    # Discovering an older syllabus term must not make the product's schedule UI
    # point at that older term.
    if previous_latest is not None:
        _atomic_write(latest_path, previous_latest)
    return path


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows).encode("utf-8")
    _atomic_write(path, payload)
    return sha256_bytes(payload)


def _checkpoint_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[str(row.get("crn"))] = row
    return rows


def _append_checkpoint(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _WRITE_LOCK, path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def crawl_term(args: argparse.Namespace, term: str, output_root: Path, schedule_dir: Path) -> dict[str, Any]:
    schedule_path = ensure_schedule(term, schedule_dir, args.timeout, args.fetch_missing_schedules)
    schedule_rows = _load_schedule(schedule_path)
    checkpoint = output_root / "checkpoints" / f"{term}.jsonl"
    checkpoint_rows = _checkpoint_rows(checkpoint) if args.resume else {}
    schedule_crns = {str(row.get("crn") or "") for row in schedule_rows}
    completed = {
        crn: row
        for crn, row in checkpoint_rows.items()
        if crn in schedule_crns
        and (
            (
                row.get("publication_state") == "published"
                and row.get("capture_state") == "complete"
            )
            or (
                row.get("publication_state") == "not_published_confirmed"
                and row.get("capture_state") == "not_applicable"
            )
        )
    }
    if not args.resume and checkpoint.exists():
        _atomic_write(checkpoint, b"")
    work = [row for row in schedule_rows if str(row["crn"]) not in completed]
    print(f"{term}: schedule CRNs={len(schedule_rows)}, resumed={len(completed)}, remaining={len(work)}")

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                crawl_one,
                row,
                output_root=output_root,
                timeout=args.timeout,
                delay=args.delay,
                confirm_empty=not args.no_confirm_empty,
                max_attachment_bytes=args.max_attachment_bytes,
            ): row
            for row in work
        }
        for completed_count, future in enumerate(as_completed(futures), start=1):
            row = futures[future]
            try:
                record, _parses, _chunks = future.result()
            except Exception as exc:
                record = {
                    "format_version": FORMAT_VERSION,
                    "crawler_version": CRAWLER_VERSION,
                    "data_role": "course_syllabus",
                    "term": term,
                    "term_code": term,
                    "course_id": row.get("course_id"),
                    "course_title": row.get("title"),
                    "subject": row.get("subject"),
                    "course_number": row.get("course_number"),
                    "section": row.get("section"),
                    "component": row.get("component"),
                    "crn": str(row.get("crn")),
                    "publication_state": "indeterminate",
                    "capture_state": "failed",
                    "status_reason": f"worker failed: {type(exc).__name__}: {exc}",
                    "sources": [],
                    "attachments": [],
                    "scraped_at": utc_now(),
                    "syllabus_id": f"syllabus:{term}:{row.get('crn')}",
                    "chunk_id": f"course_syllabus:{term}:{row.get('crn')}:overview",
                }
                record["text"] = build_record_text(record)
            completed[str(record["crn"])] = record
            _append_checkpoint(checkpoint, record)
            if completed_count % 25 == 0 or completed_count == len(work):
                print(f"{term}: processed {completed_count}/{len(work)} new CRNs")

    records = sorted(completed.values(), key=lambda row: str(row.get("crn") or ""))
    all_chunks: list[dict[str, Any]] = []
    for record in records:
        parses: dict[str, dict[str, Any]] = {}
        for attachment in record.get("attachments") or []:
            parsed_rel = attachment.get("parsed_path")
            digest = str(attachment.get("sha256") or "")
            if parsed_rel and digest:
                parsed_path = output_root / str(parsed_rel)
                if parsed_path.exists():
                    parses[digest] = json.loads(parsed_path.read_text(encoding="utf-8"))
        all_chunks.extend(make_rag_chunks(record, parses))

    ledger_path = output_root / f"{term}.jsonl"
    chunks_path = output_root / f"{term}.chunks.jsonl"
    ledger_sha = _write_jsonl(ledger_path, records)
    chunks_sha = _write_jsonl(chunks_path, all_chunks)
    states: dict[str, int] = {}
    captures: dict[str, int] = {}
    for record in records:
        states[record["publication_state"]] = states.get(record["publication_state"], 0) + 1
        captures[record["capture_state"]] = captures.get(record["capture_state"], 0) + 1
    attachments = [item for record in records for item in (record.get("attachments") or [])]
    manifest = {
        "format_version": FORMAT_VERSION,
        "crawler_version": CRAWLER_VERSION,
        "term": term,
        "source_authority": SOURCE_AUTHORITY,
        "source_endpoints": [
            schedule_scraper.TERM_ENDPOINT,
            schedule_scraper.SCHEDULE_ENDPOINT,
            "https://www.sabanciuniv.edu/syllabus/",
            "https://apps.sabanciuniv.edu/courses/syllabus/view.php",
            "https://sucourse.sabanciuniv.edu/plus/syllabusdownload.php",
        ],
        "scraped_at": utc_now(),
        "schedule_crn_count": len(schedule_rows),
        "ledger_record_count": len(records),
        "publication_states": states,
        "capture_states": captures,
        "attachment_link_count": len(attachments),
        "attachment_downloaded_count": sum(item.get("download_state") == "downloaded" for item in attachments),
        "attachment_parsed_count": sum(item.get("parse_state") == "parsed" for item in attachments),
        "rag_chunk_count": len(all_chunks),
        "ledger_file": ledger_path.name,
        "ledger_sha256": ledger_sha,
        "chunks_file": chunks_path.name,
        "chunks_sha256": chunks_sha,
        "schedule_file": schedule_path.relative_to(schedule_dir.parent).as_posix(),
    }
    manifest_path = output_root / f"{term}.manifest.json"
    _atomic_write(manifest_path, (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    return manifest


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terms", nargs="+", default=list(DEFAULT_TERMS))
    parser.add_argument("--output-dir", default=str(project_root / "data" / "syllabi"))
    parser.add_argument("--schedule-dir", default=str(project_root / "data" / "schedule"))
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--delay", type=float, default=0.35, help="Per-request polite delay plus small jitter")
    parser.add_argument("--max-attachment-bytes", type=int, default=ATTACHMENT_MAX_BYTES)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-confirm-empty", action="store_true")
    parser.add_argument(
        "--no-fetch-missing-schedules",
        dest="fetch_missing_schedules",
        action="store_false",
        help="Fail instead of fetching a missing term schedule census",
    )
    parser.set_defaults(fetch_missing_schedules=True)
    args = parser.parse_args()
    if args.max_workers < 1 or args.max_workers > 8:
        parser.error("--max-workers must be between 1 and 8")
    return args


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_dir).expanduser().resolve()
    schedule_dir = Path(args.schedule_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifests = [crawl_term(args, term.strip(), output_root, schedule_dir) for term in args.terms]
    latest_term = str(args.terms[0]).strip()
    latest = {
        "terms": [str(term).strip() for term in args.terms],
        "latest_term": latest_term,
        "manifests": {manifest["term"]: f"{manifest['term']}.manifest.json" for manifest in manifests},
        "scraped_at": utc_now(),
    }
    _atomic_write(output_root / "latest.json", (json.dumps(latest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    for manifest in manifests:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
