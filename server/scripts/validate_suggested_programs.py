from __future__ import annotations

"""Strict offline validator for ``data/suggested_programs``.

This validator deliberately does not import pdfplumber.  It can therefore run in CI or
in the production image after the checked-in JSONL/PDF artifacts have been copied.
"""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from ingest_suggested_programs import SOURCE_SPECS, validate_rows


SERVER_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SERVER_ROOT.parent
ALLOWED_DOCUMENT_TYPES = {
    "suggested_program_profile",
    "suggested_program_semester",
    "suggested_program_course",
    "suggested_program_track",
}
ARTIFACT_MARKERS = (
    "\x00",
    "Introducon",
    "Stascal",
    "Visualizaon",
    "Graduaon",
    "ISDISPL",
    "YeaSre",
    "YeSaerm",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: row must be a JSON object")
        rows.append(row)
    return rows


def _course_signature(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("course_code"),
        row.get("course_title"),
        row.get("course_type"),
        row.get("course_group"),
        row.get("su_credits"),
        row.get("ects"),
        row.get("prerequisites"),
        bool(row.get("is_elective_placeholder")),
        bool(row.get("is_one_time_flexible_placement")),
    )


def validate_dataset(data_dir: Path) -> list[str]:
    errors: list[str] = []
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.is_file():
        return [f"missing manifest: {manifest_path}"]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return [f"invalid manifest {manifest_path}: {exc}"]

    sources = manifest.get("sources")
    if not isinstance(sources, list):
        return ["manifest.sources must be a list"]
    if manifest.get("source_count") != len(SOURCE_SPECS) or len(sources) != len(SOURCE_SPECS):
        errors.append(
            f"manifest must contain {len(SOURCE_SPECS)} sources, got {len(sources)}"
        )
    if manifest.get("validation", {}).get("status") != "passed":
        errors.append("manifest validation.status must be passed")

    expected_outputs = {
        f"suggested_programs/{spec.program}/{spec.output_stem}.jsonl": spec
        for spec in SOURCE_SPECS
    }
    actual_outputs = {
        "suggested_programs/" + path.relative_to(data_dir).as_posix()
        for path in data_dir.rglob("*.jsonl")
    }
    if actual_outputs != set(expected_outputs):
        errors.append(
            "JSONL set mismatch: missing="
            f"{sorted(set(expected_outputs) - actual_outputs)}, "
            f"unexpected={sorted(actual_outputs - set(expected_outputs))}"
        )

    manifest_by_output = {
        item.get("output_document"): item for item in sources if isinstance(item, dict)
    }
    all_ids: set[str] = set()
    for output_document, spec in sorted(expected_outputs.items()):
        jsonl_path = data_dir.parent / output_document
        if not jsonl_path.is_file():
            continue
        try:
            rows = _load_jsonl(jsonl_path)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        errors.extend(validate_rows(rows, source_document=output_document))
        entry = manifest_by_output.get(output_document)
        if not entry:
            errors.append(f"{output_document}: missing manifest entry")
            continue
        if entry.get("row_count") != len(rows):
            errors.append(
                f"{output_document}: manifest row_count={entry.get('row_count')} actual={len(rows)}"
            )

        source_document = f"suggested_programs/sources/{spec.filename}"
        source_path = data_dir.parent / source_document
        if not source_path.is_file():
            errors.append(f"{output_document}: missing source PDF {source_document}")
        else:
            actual_hash = _sha256(source_path)
            if entry.get("source_sha256") != actual_hash:
                errors.append(f"{output_document}: manifest source_sha256 mismatch")

        profiles = [row for row in rows if row.get("document_type") == "suggested_program_profile"]
        summaries = [row for row in rows if row.get("document_type") == "suggested_program_semester"]
        courses = [row for row in rows if row.get("document_type") == "suggested_program_course"]
        tracks = [row for row in rows if row.get("document_type") == "suggested_program_track"]
        if len(profiles) != 1:
            errors.append(f"{output_document}: expected one profile, got {len(profiles)}")
        semesters = {
            row.get("semester")
            for row in summaries
            if row.get("term_kind") == "semester"
        }
        if semesters != set(range(1, 9)):
            errors.append(
                f"{output_document}: regular semester summaries must be exactly 1..8, "
                f"got {sorted(x for x in semesters if x is not None)}"
            )
        for summary in summaries:
            if summary.get("term_kind") != "semester" and summary.get("semester") is not None:
                errors.append(f"{output_document}: non-regular term must have semester=null")
        if any(row.get("study_year") is not None or row.get("semester") is not None for row in tracks):
            errors.append(f"{output_document}: track suggestions must not invent year/semester")

        by_period: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        for row in courses:
            period = (row.get("study_year"), row.get("semester"), row.get("term_kind"))
            by_period.setdefault(period, []).append(row)
        for summary in summaries:
            period = (
                summary.get("study_year"),
                summary.get("semester"),
                summary.get("term_kind"),
            )
            summary_counter = Counter(_course_signature(item) for item in summary.get("courses", []))
            course_counter = Counter(_course_signature(item) for item in by_period.get(period, []))
            if summary_counter != course_counter:
                errors.append(f"{output_document}: summary/course mismatch for period {period}")

        for line_number, row in enumerate(rows, start=1):
            chunk_id = row.get("chunk_id")
            if chunk_id in all_ids:
                errors.append(f"{output_document}:{line_number}: globally duplicate chunk_id")
            all_ids.add(str(chunk_id))
            if row.get("document_type") not in ALLOWED_DOCUMENT_TYPES:
                errors.append(
                    f"{output_document}:{line_number}: unsupported document_type "
                    f"{row.get('document_type')!r}"
                )
            if row.get("authority_level") != "official_advisory":
                errors.append(f"{output_document}:{line_number}: authority_level drift")
            if row.get("program") != spec.program:
                errors.append(f"{output_document}:{line_number}: program drift")
            if row.get("source_document") != source_document:
                errors.append(f"{output_document}:{line_number}: source_document drift")
            if row.get("source_sha256") != entry.get("source_sha256"):
                errors.append(f"{output_document}:{line_number}: source_sha256 drift")
            page = row.get("source_page")
            if not isinstance(page, int) or not (1 <= page <= entry.get("source_pages", 0)):
                errors.append(f"{output_document}:{line_number}: invalid source_page {page!r}")
            searchable = " ".join(
                str(row.get(key) or "") for key in ("course_code", "course_title", "text")
            )
            marker = next((item for item in ARTIFACT_MARKERS if item in searchable), None)
            if marker:
                errors.append(
                    f"{output_document}:{line_number}: extraction artifact marker {marker!r}"
                )

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "suggested_programs",
    )
    args = parser.parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    errors = validate_dataset(data_dir)
    if errors:
        print(f"FAILED: {len(errors)} validation error(s)")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    jsonl_files = list(data_dir.rglob("*.jsonl"))
    rows = sum(len(_load_jsonl(path)) for path in jsonl_files)
    print(
        f"PASS: {len(jsonl_files)} plan documents, {rows} rows, "
        "unique IDs, exact semester 1..8 coverage, source hashes verified."
    )


if __name__ == "__main__":
    main()
