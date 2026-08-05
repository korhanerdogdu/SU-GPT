# SUIS course schedule snapshots

This directory contains source data only. Nothing here is automatically inserted into Chroma,
MongoDB, or another RAG index.

## Files

- `<term>.jsonl`: one JSON object per CRN/section.
- `<term>.manifest.json`: provenance, counts, fetch time, and the JSONL SHA-256 checksum.
- `latest.json`: a small pointer to the most recently fetched snapshot.

The current repository loader already recognizes `data/schedule/*.jsonl`. It expects the main
fields `term`, `course_id`, `title`, `crn`, `section`, `component`, `meetings`, and `source_url`.
The generated records also contain a ready-to-embed `text` field and stable `chunk_id`.

## Regenerating a snapshot

From the repository root, using the server virtual environment on Windows:

```powershell
.\server\.venv\Scripts\python.exe server\scripts\scrape_course_schedule.py --term 202601
```

Inside the dev container/Linux environment:

```bash
python server/scripts/scrape_course_schedule.py --term 202601
```

`--term` is deliberately required. SU can publish the next registration term before the calendar
season changes, so a date-only guess can select an older term. Use the six-digit term shown by
SUIS; the snapshot currently requested for this repository is `202601`.

## Record shape

Each JSONL line is a self-contained section record:

- Identity: `term`, `course_id`, `source_course_code`, `crn`, `section`, `component`.
- Search text: `text` (Turkish/English labels, exact meeting facts, and source URL).
- Exact meetings: `meetings[]` with source time plus normalized 24-hour `start_time` and
  `end_time`, day codes/names, location, date range, schedule type, and instructors.
- Flat filter helpers: `subject`, `course_number`, `meeting_days`, `meeting_times`, `locations`,
  and `instructors`.
- Provenance: `source_authority`, `source_endpoint`, `source_url`, `scraped_at`, and
  `authority_level`.
- Stable key: `chunk_id = course_schedule:<term>:<crn>`.

Empty day/time cells from SUIS are represented as `status: "TBA"`. The raw source value is kept
in `time_raw`, while the retrieval-friendly `time` value becomes `TBA`.
