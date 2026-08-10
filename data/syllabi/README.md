# Sabanci University syllabus corpus

This directory is the reproducible, official-source syllabus corpus for the
`202601` and `202502` SUIS terms.  The crawler accounts for every schedule CRN;
an absent syllabus is a stored state, never a silently missing row.

## Layout

- `<term>.jsonl` - one crawl/accounting record per `(term, CRN)`.
- `<term>.chunks.jsonl` - pre-chunked, flat-metadata RAG records.
- `<term>.manifest.json` - counts, crawler version, endpoints, and SHA-256s.
- `raw/<term>/<crn>/` - source HTML from the legacy and new public apps.
- `attachments/<sha-prefix>/<sha256>.<ext>` - deduplicated original files.
- `parsed/<sha-prefix>/<sha256>.json` - page/section-aware extracted content.
- `checkpoints/<term>.jsonl` - resumable crawl journal; the final ledger is
  deterministically sorted and is the authoritative snapshot.

Original attachments are content-addressed.  If the same academic calendar or
syllabus file is linked by multiple sections, its bytes are stored once while
every CRN keeps its own relationship and provenance metadata.

## State semantics

`publication_state` and `capture_state` are deliberately separate:

- `published`: at least one official public syllabus surface contains syllabus
  text or attachments.
- `not_published_confirmed`: both public surfaces returned identity-verified
  empty pages twice.
- `indeterminate`: a timeout, HTTP error, login/error page, identity mismatch,
  or parser error prevented a safe conclusion.
- `complete`: all published material was captured and parsed.
- `partial` / `failed`: explicit retry or manual/OCR work remains.
- `not_applicable`: the syllabus was safely confirmed as not published.

HTTP failures, generic 404 pages, and HTML disguised as PDF/DOCX are never
classified as "not published".

## Build and validate

From the repository root:

```powershell
.\server\.venv\Scripts\python.exe server\scripts\scrape_syllabi.py --terms 202601 202502 --resume
.\server\.venv\Scripts\python.exe server\scripts\validate_syllabus_corpus.py --terms 202601 202502 --strict
.\server\.venv\Scripts\python.exe server\scripts\ingest_syllabi.py --terms 202601 202502 --reset
```

The strict validator writes `outputs/syllabus_benchmark/report.json`, requires
an overall score of at least 90/100, and also enforces hard gates for complete
CRN accounting, attachment integrity, parse success, stable IDs, manifest
reconciliation, and deterministic exact-source retrieval.

The default `hybrid_meta` disk corpus reads `*.chunks.jsonl` directly.  The
separate ingestion command updates the shared Chroma collection for fallback
retrieval paths without relying on ignored `sources/` or upload state.
