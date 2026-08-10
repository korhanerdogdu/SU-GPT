from __future__ import annotations

"""Validate and score the two-term Sabanci syllabus corpus (0-100).

The score is deliberately paired with hard gates: many good records cannot hide
one missing CRN, a silently skipped attachment, or an indeterminate fetch.  Use
``--strict`` in CI/corpus production to return non-zero unless score >= 90 and
every gate passes.
"""

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urljoin, urlparse

from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.syllabus_pipeline import (  # noqa: E402
    detect_attachment_format,
    make_rag_chunks,
    parse_legacy_html,
    parse_new_app_html,
)


DEFAULT_TERMS = ("202601", "202502")
VALID_PUBLICATION_STATES = {"published", "not_published_confirmed", "indeterminate"}
VALID_CAPTURE_STATES = {"complete", "partial", "failed", "not_applicable"}
REQUIRED_KEYS = {
    "term",
    "term_code",
    "course_id",
    "course_title",
    "section",
    "component",
    "crn",
    "instructor_names",
    "instructor_emails",
    "publication_state",
    "capture_state",
    "status_reason",
    "sources",
    "attachments",
    "source_authority",
    "scraped_at",
    "syllabus_id",
    "chunk_id",
    "text",
}
DOWNLOAD_EXTENSIONS = {".pdf", ".doc", ".docx", ".txt", ".csv", ".xlsx", ".pptx", ".html", ".htm"}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: row is not an object")
            rows.append(row)
    return rows


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ratio(numerator: int | float, denominator: int | float, *, empty: float = 1.0) -> float:
    return float(numerator) / float(denominator) if denominator else empty


def _f1(expected: set[str], actual: set[str]) -> float:
    if not expected and not actual:
        return 1.0
    intersection = len(expected & actual)
    precision = _ratio(intersection, len(actual), empty=0.0)
    recall = _ratio(intersection, len(expected), empty=0.0)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _inside(root: Path, relative: str) -> Path | None:
    try:
        candidate = (root / relative).resolve()
        candidate.relative_to(root.resolve())
        return candidate
    except (OSError, ValueError):
        return None


def _tokens(value: str) -> set[str]:
    return set(re.findall(r"[\w@.-]+", (value or "").casefold(), flags=re.UNICODE))


def _broad_attachment_urls(payload: bytes, base_url: str) -> set[str]:
    soup = BeautifulSoup(payload, "html.parser")
    urls: set[str] = set()
    for anchor in soup.find_all("a", href=True):
        url = urljoin(base_url, str(anchor.get("href") or ""))
        parsed = urlparse(url)
        path = unquote(parsed.path).lower()
        query = unquote(parsed.query).lower()
        if (
            "syllabusdownload.php" in path
            or "filename=" in query
            or Path(path).suffix in DOWNLOAD_EXTENSIONS
        ):
            urls.add(url)
    return urls


def _read_and_verify_raw(data_root: Path, evidence: dict[str, Any]) -> tuple[bytes | None, bool]:
    path = _inside(data_root, str(evidence.get("raw_html_path") or ""))
    if path is None or not path.is_file():
        return None, False
    payload = path.read_bytes()
    return payload, bool(evidence.get("page_sha256") == hashlib.sha256(payload).hexdigest())


def _synthetic_retrieval(chunks: list[dict[str, Any]], sample_limit: int = 300) -> dict[str, float | int]:
    """Exact-source smoke benchmark over persisted RAG chunks.

    Queries contain the facts a real syllabus request normally carries (course,
    term, CRN, instructor, section heading).  This is a corpus/retrieval integrity
    gate, not a substitute for the separately curated answer-faithfulness set.
    """
    candidates = [row for row in chunks if row.get("text") and row.get("crn")]
    if not candidates:
        return {"query_count": 0, "recall_at_5": 0.0, "mrr_at_10": 0.0, "ndcg_at_10": 0.0}
    ordered = sorted(candidates, key=lambda row: str(row.get("chunk_id")))
    if len(ordered) > sample_limit:
        step = len(ordered) / sample_limit
        queries = [ordered[math.floor(index * step)] for index in range(sample_limit)]
    else:
        queries = ordered
    doc_tokens = [_tokens(str(row.get("text") or "")) for row in candidates]
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []
    for gold in queries:
        content_tokens = [
            token
            for token in re.findall(r"[\w@.-]+", str(gold.get("text") or ""), flags=re.UNICODE)
            if len(token) >= 4
            and token.casefold() not in {"official", "syllabus", "term", "course", "section", "instructor"}
            and token not in {str(gold.get("term") or ""), str(gold.get("crn") or "")}
        ]
        query = " ".join(
            [str(gold.get("course_id") or ""), str(gold.get("locator") or ""), *content_tokens[-18:]]
        )
        qtokens = _tokens(query)
        gold_probe = set(token.casefold() for token in content_tokens[-18:])
        scored: list[tuple[float, int]] = []
        for index, (row, tokens) in enumerate(zip(candidates, doc_tokens)):
            score = float(len(qtokens & tokens))
            if str(row.get("course_id")) == str(gold.get("course_id")):
                score += 4.0
            if str(row.get("term")) == str(gold.get("term")):
                score += 2.0
            scored.append((score, index))
        ranking = sorted(scored, key=lambda item: (-item[0], str(candidates[item[1]].get("chunk_id"))))
        rank = next(
            (
                position
                for position, (_score, index) in enumerate(ranking, start=1)
                if str(candidates[index].get("term")) == str(gold.get("term"))
                and str(candidates[index].get("course_id")) == str(gold.get("course_id"))
                and str(candidates[index].get("locator")) == str(gold.get("locator"))
                and gold_probe.issubset(doc_tokens[index])
            ),
            None,
        )
        recalls.append(1.0 if rank is not None and rank <= 5 else 0.0)
        reciprocal_ranks.append(1.0 / rank if rank is not None and rank <= 10 else 0.0)
        ndcgs.append(1.0 / math.log2(rank + 1) if rank is not None and rank <= 10 else 0.0)
    return {
        "query_count": len(queries),
        "recall_at_5": sum(recalls) / len(recalls),
        "mrr_at_10": sum(reciprocal_ranks) / len(reciprocal_ranks),
        "ndcg_at_10": sum(ndcgs) / len(ndcgs),
    }


def validate_term(term: str, data_root: Path, schedule_root: Path) -> dict[str, Any]:
    schedule_path = schedule_root / f"{term}.jsonl"
    ledger_path = data_root / f"{term}.jsonl"
    chunks_path = data_root / f"{term}.chunks.jsonl"
    manifest_path = data_root / f"{term}.manifest.json"
    blockers: list[str] = []
    if not schedule_path.exists():
        blockers.append(f"missing schedule census: {schedule_path}")
    if not ledger_path.exists():
        blockers.append(f"missing syllabus ledger: {ledger_path}")
    if not chunks_path.exists():
        blockers.append(f"missing syllabus chunks: {chunks_path}")
    if not manifest_path.exists():
        blockers.append(f"missing syllabus manifest: {manifest_path}")
    if blockers:
        return {"term": term, "score": 0.0, "passed": False, "blockers": blockers}

    try:
        schedule = _read_jsonl(schedule_path)
        ledger = _read_jsonl(ledger_path)
        chunks = _read_jsonl(chunks_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "term": term,
            "score": 0.0,
            "passed": False,
            "blockers": [f"could not parse corpus files: {type(exc).__name__}: {exc}"],
        }

    schedule_counts = Counter(str(row.get("crn")) for row in schedule)
    ledger_counts = Counter(str(row.get("crn")) for row in ledger)
    schedule_keys = set(schedule_counts)
    ledger_keys = set(ledger_counts)
    duplicate_schedule = sorted(key for key, count in schedule_counts.items() if count > 1)
    duplicate_ledger = sorted(key for key, count in ledger_counts.items() if count > 1)
    missing = sorted(schedule_keys - ledger_keys)
    extra = sorted(ledger_keys - schedule_keys)
    if duplicate_schedule:
        blockers.append(f"duplicate CRNs in schedule census: {duplicate_schedule[:10]}")
    if duplicate_ledger:
        blockers.append(f"duplicate CRNs in syllabus ledger: {duplicate_ledger[:10]}")
    if missing:
        blockers.append(f"schedule CRNs missing from ledger: {missing[:10]} (total {len(missing)})")
    if extra:
        blockers.append(f"ledger CRNs absent from schedule: {extra[:10]} (total {len(extra)})")

    schedule_by_crn = {str(row.get("crn")): row for row in schedule}
    identity_matches = 0
    metadata_keys_ok = 0
    metadata_match = 0
    publication_semantics_ok = 0
    evidence_ok = 0
    source_count = 0
    discovered_links = 0
    merged_links = 0
    attachment_downloads_ok = 0
    attachment_integrity_ok = 0
    attachment_parse_ok = 0
    attachments_total = 0
    published_crns: set[str] = set()
    attachment_details: list[dict[str, Any]] = []
    invalid_states: list[str] = []
    bad_source_rows: list[str] = []
    raw_evidence_failures: list[str] = []
    bad_state_pairs: list[str] = []
    identity_failures: list[str] = []
    metadata_mismatches: list[str] = []
    expected_chunks: list[dict[str, Any]] = []

    for row in ledger:
        crn = str(row.get("crn"))
        schedule_row = schedule_by_crn.get(crn) or {}
        if set(row).issuperset(REQUIRED_KEYS):
            metadata_keys_ok += 1
        same = all(
            str(row.get(key) or "") == str(schedule_row.get(source_key) or "")
            for key, source_key in [
                ("term", "term"),
                ("course_id", "course_id"),
                ("course_title", "title"),
                ("section", "section"),
                ("component", "component"),
            ]
        )
        if same:
            metadata_match += 1
            identity_matches += 1
        else:
            metadata_mismatches.append(crn)

        publication = row.get("publication_state")
        capture = row.get("capture_state")
        if publication not in VALID_PUBLICATION_STATES or capture not in VALID_CAPTURE_STATES:
            invalid_states.append(crn)
        sources = row.get("sources") or []
        source_count += len(sources)
        if len(sources) != 2 or {source.get("source_system") for source in sources} != {"legacy", "new_app"}:
            bad_source_rows.append(crn)
        raw_discovered_urls: set[str] = set()
        raw_ok = True
        reparsed_sources: list[dict[str, Any]] = []
        for source in sources:
            payload, hash_ok = _read_and_verify_raw(data_root, source)
            raw_ok = raw_ok and hash_ok
            if payload is None:
                continue
            raw_discovered_urls.update(
                _broad_attachment_urls(payload, str(source.get("final_url") or source.get("requested_url") or ""))
            )
            parser = parse_legacy_html if source.get("source_system") == "legacy" else parse_new_app_html
            try:
                reparsed_sources.append(
                    parser(
                        payload,
                        base_url=str(source.get("final_url") or source.get("requested_url") or ""),
                        expected_term=term,
                        expected_course_id=str(row.get("course_id") or ""),
                        expected_section=str(row.get("section") or ""),
                        expected_crn=crn,
                    )
                )
            except Exception:
                raw_ok = False
        source_evidence = bool(sources) and len(sources) == 2 and raw_ok and all(
            source.get("http_status") == 200
            and source.get("fetched_at")
            and source.get("final_url")
            and source.get("page_sha256")
            for source in sources
        )
        if source_evidence:
            evidence_ok += 1
        else:
            raw_evidence_failures.append(crn)

        if publication == "published":
            published_crns.add(crn)
            semantic_ok = any(
                source.get("page_state") == "published" and source.get("identity_verified")
                for source in sources
            ) and any(
                source.get("page_state") == "published" and source.get("identity_verified")
                for source in reparsed_sources
            )
            if capture != "complete":
                bad_state_pairs.append(crn)
        elif publication == "not_published_confirmed":
            confirmations_ok = True
            for source in sources:
                confirmation = source.get("confirmation") or {}
                confirmation_payload, confirmation_hash_ok = _read_and_verify_raw(data_root, confirmation)
                confirmation_reparsed: dict[str, Any] = {}
                if confirmation_payload is not None:
                    parser = parse_legacy_html if source.get("source_system") == "legacy" else parse_new_app_html
                    try:
                        confirmation_reparsed = parser(
                            confirmation_payload,
                            base_url=str(confirmation.get("final_url") or confirmation.get("requested_url") or ""),
                            expected_term=term,
                            expected_course_id=str(row.get("course_id") or ""),
                            expected_section=str(row.get("section") or ""),
                            expected_crn=crn,
                        )
                    except Exception:
                        confirmation_reparsed = {}
                confirmations_ok = bool(
                    confirmations_ok
                    and source.get("page_state") == "not_published_candidate"
                    and source.get("empty_confirmation_agrees")
                    and confirmation.get("http_status") == 200
                    and confirmation.get("page_state") == "not_published_candidate"
                    and confirmation_hash_ok
                    and confirmation_payload is not None
                    and confirmation_reparsed.get("page_state") == "not_published_candidate"
                    and (
                        confirmation_reparsed.get("identity_verified")
                        or confirmation_reparsed.get("endpoint_target_verified")
                    )
                    and (source.get("identity_verified") or source.get("endpoint_target_verified"))
                    and (confirmation.get("identity_verified") or confirmation.get("endpoint_target_verified"))
                )
            semantic_ok = bool(sources) and confirmations_ok and len(reparsed_sources) == 2 and all(
                source.get("page_state") == "not_published_candidate"
                and (source.get("identity_verified") or source.get("endpoint_target_verified"))
                for source in reparsed_sources
            )
            if capture != "not_applicable" or any(source.get("page_state") == "published" for source in sources):
                bad_state_pairs.append(crn)
        else:
            semantic_ok = False
            if capture not in {"failed", "partial"}:
                bad_state_pairs.append(crn)
        if semantic_ok:
            publication_semantics_ok += 1
        else:
            identity_failures.append(crn)

        source_urls = raw_discovered_urls | {
            str(link.get("url"))
            for source in sources
            for link in (source.get("attachments") or [])
            if link.get("url")
        }
        merged_urls = {str(link.get("url")) for link in (row.get("attachments") or []) if link.get("url")}
        discovered_links += len(source_urls)
        merged_links += len(source_urls & merged_urls)
        if source_urls - merged_urls:
            blockers.append(f"CRN {crn} silently omitted attachment links: {sorted(source_urls - merged_urls)[:3]}")

        record_parses: dict[str, dict[str, Any]] = {}
        for item in row.get("attachments") or []:
            attachments_total += 1
            local_path = _inside(data_root, str(item.get("local_path") or ""))
            downloaded = item.get("download_state") == "downloaded" and local_path is not None and local_path.is_file()
            if downloaded:
                attachment_downloads_ok += 1
            integrity = False
            if downloaded:
                payload = local_path.read_bytes()
                detected = detect_attachment_format(
                    payload,
                    str(item.get("original_filename") or item.get("filename") or ""),
                    str(item.get("declared_mime") or ""),
                )
                integrity = bool(
                    detected.get("valid")
                    and len(payload) == int(item.get("byte_size") or -1)
                    and hashlib.sha256(payload).hexdigest() == item.get("sha256")
                    and detected.get("kind") == item.get("detected_kind")
                )
            if integrity:
                attachment_integrity_ok += 1
            parsed_valid = False
            parsed_path = _inside(data_root, str(item.get("parsed_path") or ""))
            if parsed_path is not None and parsed_path.is_file():
                try:
                    parsed = json.loads(parsed_path.read_text(encoding="utf-8"))
                    digest = str(item.get("sha256") or "")
                    all_units = parsed.get("units") or []
                    units = [unit for unit in all_units if str(unit.get("text") or "").strip()]
                    parsed_valid = bool(
                        item.get("parse_state") == "parsed"
                        and parsed.get("parse_state") == "parsed"
                        and parsed.get("attachment_sha256") == digest
                        and bool(units)
                        and len(all_units) == int(item.get("parsed_unit_count") or -1)
                    )
                    if parsed_valid:
                        record_parses[digest] = parsed
                except Exception:
                    parsed_valid = False
            if parsed_valid:
                attachment_parse_ok += 1
            attachment_details.append(
                {
                    "crn": crn,
                    "url": item.get("url"),
                    "downloaded": downloaded,
                    "integrity": integrity,
                    "parse_state": item.get("parse_state"),
                    "parsed_valid": parsed_valid,
                }
            )
        expected_chunks.extend(make_rag_chunks(row, record_parses))

    if invalid_states:
        blockers.append(f"invalid publication/capture enum values: {invalid_states[:10]}")
    if metadata_mismatches:
        blockers.append(f"schedule/ledger identity mismatches: {metadata_mismatches[:10]} (total {len(metadata_mismatches)})")
    if bad_source_rows:
        blockers.append(f"records without exactly legacy+new_app evidence: {bad_source_rows[:10]} (total {len(bad_source_rows)})")
    if raw_evidence_failures:
        blockers.append(f"missing or SHA-mismatched raw HTML evidence: {raw_evidence_failures[:10]} (total {len(raw_evidence_failures)})")
    if bad_state_pairs:
        blockers.append(f"invalid publication/capture state pairs: {bad_state_pairs[:10]} (total {len(bad_state_pairs)})")
    if identity_failures:
        blockers.append(f"publication evidence failed independent state/identity checks: {identity_failures[:10]} (total {len(identity_failures)})")
    indeterminate = [str(row.get("crn")) for row in ledger if row.get("publication_state") == "indeterminate"]
    failed = [str(row.get("crn")) for row in ledger if row.get("capture_state") == "failed"]
    partial = [str(row.get("crn")) for row in ledger if row.get("capture_state") == "partial"]
    if indeterminate:
        blockers.append(f"indeterminate CRNs remain: {indeterminate[:10]} (total {len(indeterminate)})")
    if failed:
        blockers.append(f"failed captures remain: {failed[:10]} (total {len(failed)})")
    if partial:
        blockers.append(f"partial captures remain: {partial[:10]} (total {len(partial)})")
    bad_attachments = [item for item in attachment_details if not item["downloaded"] or not item["integrity"]]
    if bad_attachments:
        blockers.append(f"attachment download/integrity failures: {bad_attachments[:5]}")
    bad_parses = [item for item in attachment_details if item["parse_state"] != "parsed" or not item["parsed_valid"]]
    if bad_parses:
        blockers.append(f"attachment parse failures/OCR queue: {bad_parses[:5]}")

    chunk_ids = [str(row.get("chunk_id") or "") for row in chunks]
    duplicate_chunks = sorted(key for key, count in Counter(chunk_ids).items() if not key or count > 1)
    if duplicate_chunks:
        blockers.append(f"duplicate/empty chunk IDs: {duplicate_chunks[:10]}")
    actual_chunks_by_id = {str(row.get("chunk_id") or ""): row for row in chunks}
    expected_chunks_by_id = {str(row.get("chunk_id") or ""): row for row in expected_chunks}
    chunk_lineage_ok = bool(
        set(actual_chunks_by_id) == set(expected_chunks_by_id)
        and all(actual_chunks_by_id[key] == expected_chunks_by_id[key] for key in expected_chunks_by_id)
    )
    if not chunk_lineage_ok:
        missing_chunk_ids = sorted(set(expected_chunks_by_id) - set(actual_chunks_by_id))
        extra_chunk_ids = sorted(set(actual_chunks_by_id) - set(expected_chunks_by_id))
        blockers.append(
            f"RAG chunks do not exactly regenerate from page/attachment artifacts; "
            f"missing={missing_chunk_ids[:5]}, extra={extra_chunk_ids[:5]}"
        )
    chunk_parent_ok = sum(
        str(row.get("crn")) in ledger_keys
        and str(row.get("term")) == term
        and row.get("data_role") == "course_syllabus"
        and bool(row.get("source_url"))
        for row in chunks
    )
    chunk_text_ok = sum(len(str(row.get("text") or "").strip()) >= 80 for row in chunks)
    chunks_by_crn = Counter(str(row.get("crn")) for row in chunks)
    unpublished_with_chunks = sorted(
        str(row.get("crn"))
        for row in ledger
        if row.get("publication_state") != "published" and chunks_by_crn[str(row.get("crn"))] > 0
    )
    if unpublished_with_chunks:
        blockers.append(f"non-published CRNs unexpectedly have RAG chunks: {unpublished_with_chunks[:10]}")
    published_without_chunks = sorted(crn for crn in published_crns if chunks_by_crn[crn] == 0)
    if published_without_chunks:
        blockers.append(f"published syllabi without RAG chunks: {published_without_chunks[:10]}")
    if term == "202502" and not published_crns:
        blockers.append("202502 has zero published syllabi; an all-empty corpus cannot pass")
    if chunk_text_ok != len(chunks):
        blockers.append(f"empty/too-short RAG chunks: {len(chunks) - chunk_text_ok}")
    retrieval = _synthetic_retrieval(chunks)
    if retrieval["query_count"] == 0 and published_crns:
        blockers.append("published corpus produced no retrieval benchmark queries")
    if term == "202502" and int(retrieval["query_count"]) < 150:
        blockers.append(f"202502 retrieval benchmark has fewer than 150 evidence queries: {retrieval['query_count']}")
    if retrieval["query_count"] and float(retrieval["recall_at_5"]) < 0.90:
        blockers.append(f"synthetic source Recall@5 below 0.90: {retrieval['recall_at_5']:.4f}")
    if retrieval["query_count"] and float(retrieval["mrr_at_10"]) < 0.80:
        blockers.append(f"synthetic source MRR@10 below 0.80: {retrieval['mrr_at_10']:.4f}")
    if retrieval["query_count"] and float(retrieval["ndcg_at_10"]) < 0.85:
        blockers.append(f"synthetic source nDCG@10 below 0.85: {retrieval['ndcg_at_10']:.4f}")

    manifest_ok = bool(
        int(manifest.get("schedule_crn_count", -1)) == len(schedule)
        and int(manifest.get("ledger_record_count", -1)) == len(ledger)
        and int(manifest.get("rag_chunk_count", -1)) == len(chunks)
        and manifest.get("ledger_sha256") == _sha256_file(ledger_path)
        and manifest.get("chunks_sha256") == _sha256_file(chunks_path)
    )
    if not manifest_ok:
        blockers.append("manifest counts or SHA-256 values do not match corpus files")

    ledger_n = len(ledger)
    inventory_components = {
        "crn_set_f1": _f1(schedule_keys, ledger_keys),
        "identity_match_rate": _ratio(identity_matches, ledger_n),
        "manifest_reconciliation": 1.0 if manifest_ok else 0.0,
        "source_provenance_rate": _ratio(
            sum(bool(row.get("source_authority") and row.get("scraped_at")) for row in ledger), ledger_n
        ),
    }
    inventory_score = (
        10 * inventory_components["crn_set_f1"]
        + 6 * inventory_components["identity_match_rate"]
        + 5 * inventory_components["manifest_reconciliation"]
        + 4 * inventory_components["source_provenance_rate"]
    )
    publication_score = (
        5 * _ratio(sum(row.get("publication_state") in VALID_PUBLICATION_STATES and row.get("capture_state") in VALID_CAPTURE_STATES for row in ledger), ledger_n)
        + 10 * _ratio(publication_semantics_ok, ledger_n)
        + 5 * _ratio(evidence_ok, ledger_n)
    )
    metadata_score = (
        6 * _ratio(metadata_keys_ok, ledger_n)
        + 6 * _ratio(metadata_match, ledger_n)
        + 3 * _ratio(sum(str(row.get("syllabus_id")) == f"syllabus:{term}:{row.get('crn')}" for row in ledger), ledger_n)
    )
    attachment_score = (
        6 * _ratio(merged_links, discovered_links)
        + 6 * _ratio(attachment_integrity_ok, attachments_total)
        + 3 * _ratio(attachment_parse_ok, attachments_total)
    )
    parse_score = (
        5 * _ratio(attachment_parse_ok, attachments_total)
        + 5 * _ratio(chunk_parent_ok, len(chunks))
        + 4 * _ratio(chunk_text_ok, len(chunks))
        + 2 * float(retrieval.get("recall_at_5") or (1.0 if not published_crns else 0.0))
        + 2 * min(1.0, float(retrieval.get("mrr_at_10") or (1.0 if not published_crns else 0.0)) / 0.80)
        + 2 * min(1.0, float(retrieval.get("ndcg_at_10") or (1.0 if not published_crns else 0.0)) / 0.85)
    )
    stable_ids = bool(chunk_ids) and len(chunk_ids) == len(set(chunk_ids)) and all(chunk_ids)
    sorted_ledger = [str(row.get("crn")) for row in ledger] == sorted(str(row.get("crn")) for row in ledger)
    reproducibility_score = (
        2 * (1.0 if stable_ids else 0.0)
        + 1 * (1.0 if sorted_ledger else 0.0)
        + 1 * (1.0 if manifest.get("crawler_version") else 0.0)
        + 1 * (1.0 if manifest_ok else 0.0)
    )
    score = round(inventory_score + publication_score + metadata_score + attachment_score + parse_score + reproducibility_score, 2)
    return {
        "term": term,
        "score": score,
        "passed": score >= 90.0 and not blockers,
        "blockers": blockers,
        "counts": {
            "schedule_crns": len(schedule),
            "ledger_records": len(ledger),
            "published": sum(row.get("publication_state") == "published" for row in ledger),
            "not_published_confirmed": sum(row.get("publication_state") == "not_published_confirmed" for row in ledger),
            "indeterminate": len(indeterminate),
            "complete": sum(row.get("capture_state") == "complete" for row in ledger),
            "partial": len(partial),
            "failed": len(failed),
            "attachment_links": attachments_total,
            "attachment_downloads_ok": attachment_downloads_ok,
            "attachment_integrity_ok": attachment_integrity_ok,
            "attachment_parses_ok": attachment_parse_ok,
            "rag_chunks": len(chunks),
        },
        "category_scores": {
            "inventory_25": round(inventory_score, 2),
            "publication_20": round(publication_score, 2),
            "metadata_15": round(metadata_score, 2),
            "attachments_15": round(attachment_score, 2),
            "parse_and_rag_20": round(parse_score, 2),
            "reproducibility_5": round(reproducibility_score, 2),
        },
        "retrieval": retrieval,
    }


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terms", nargs="+", default=list(DEFAULT_TERMS))
    parser.add_argument("--data-dir", default=str(project_root / "data" / "syllabi"))
    parser.add_argument("--schedule-dir", default=str(project_root / "data" / "schedule"))
    parser.add_argument("--report", default=str(project_root / "outputs" / "syllabus_benchmark" / "report.json"))
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_dir).expanduser().resolve()
    schedule_root = Path(args.schedule_dir).expanduser().resolve()
    requested_terms = [str(term).strip() for term in args.terms]
    required_terms_ok = len(requested_terms) == 2 and set(requested_terms) == set(DEFAULT_TERMS)
    results = [validate_term(term, data_root, schedule_root) for term in requested_terms]
    overall_score = round(min(result["score"] for result in results), 2) if results else 0.0
    blockers = [f"{result['term']}: {blocker}" for result in results for blocker in result.get("blockers", [])]
    if not required_terms_ok:
        blockers.insert(0, f"benchmark must validate exactly {list(DEFAULT_TERMS)}, got {requested_terms}")
    report = {
        "benchmark_version": 2,
        "threshold": 90.0,
        "terms": requested_terms,
        "overall_score": overall_score,
        "passed": required_terms_ok and overall_score >= 90.0 and all(result.get("passed") for result in results) and not blockers,
        "hard_gate_blockers": blockers,
        "term_results": results,
        "notes": {
            "retrieval": "Deterministic evidence-retrieval benchmark with no gold CRN in queries; byte-equivalent syllabus evidence across sections is treated as relevant.",
            "not_published": "Counts as complete only after two HTTP-200 fetches on both public syllabus surfaces, verified raw hashes, and page-identity or exact endpoint-target evidence.",
        },
    }
    report_path = Path(args.report).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.strict and not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
