from __future__ import annotations

"""Build and run the recommended-program retrieval benchmark.

This benchmark deliberately measures two things together:

* evidence coverage -- does the corpus contain authoritative evidence for the question?
* retrieval -- is at least one accepted evidence chunk in the first ``k`` results?

The headline metric is therefore **Authoritative Evidence Hit Rate@K (AEHR@K)**: the
percentage of all benchmark questions for which a system returns at least one accepted,
fact-bearing chunk in its first K results.  A missing fact is a miss rather than being
dropped from the denominator.  ``conditional_hit_rate@K`` is also reported over only the
questions for which that corpus has accepted evidence, so a coverage gain cannot disguise
a ranking regression.

The baseline corpus contains degree requirements, minors, and the compact course catalog.
The enhanced corpus is the exact same corpus plus ``data/suggested_programs/**/*.jsonl``.
No LLM or network call is involved, making runs deterministic and inexpensive.

Usage (from the repository root)::

    python server/evaluation/benchmark_suggested_programs.py --build-dataset --run
    python server/evaluation/benchmark_suggested_programs.py --run --assert-target 0.90
"""

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


SERVER_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SERVER_ROOT.parent
if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))

from retrieval_lab.corpus import Chunk, Corpus  # noqa: E402
from retrieval_lab.sparse import BM25FIndex  # noqa: E402
from retrieval_lab.text import tokenize_v2  # noqa: E402


DEFAULT_BENCHMARK = PROJECT_ROOT / "data" / "benchmark" / "suggested_programs_v1.jsonl"
DEFAULT_SUGGESTED_DIR = PROJECT_ROOT / "data" / "suggested_programs"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "retrieval_lab" / "suggested_programs"
BENCHMARK_SIZE = 60
DEFAULT_TOP_K = 5

_COURSE_RE = re.compile(r"\b([A-Z]{2,6})\s*([0-9]{3,5}[A-Z]?)\b")
_CATEGORY_ALIASES = {
    "required": "required",
    "required c": "required",
    "mandatory": "required",
    "zorunlu": "required",
    "core": "core",
    "core elective": "core",
    "area": "area",
    "area elective": "area",
    "free": "free",
    "free elective": "free",
    "university": "university",
    "university c": "university",
    "faculty": "faculty",
    "faculty c": "faculty",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def _clean(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _course_code(value: Any) -> str:
    match = _COURSE_RE.search(_clean(value).upper())
    return f"{match.group(1)} {match.group(2)}" if match else _clean(value).upper()


def _is_concrete_course_code(value: Any) -> bool:
    raw = _clean(value).upper()
    return bool(re.fullmatch(r"[A-Z]{2,6}\s*[0-9]{3,5}[A-Z]?", raw))


def _category(value: Any) -> str:
    cleaned = re.sub(r"[^a-z ]+", " ", _clean(value).lower().replace("_", " "))
    cleaned = " ".join(cleaned.split())
    for alias, canonical in _CATEGORY_ALIASES.items():
        if cleaned == alias or cleaned.startswith(alias + " "):
            return canonical
    return cleaned


def _track_label(track: str) -> str:
    return track.replace("_", " ")


def _fingerprint(chunks: Sequence[Chunk]) -> str:
    digest = hashlib.sha256()
    for chunk in sorted(chunks, key=lambda item: item.chunk_id):
        digest.update(chunk.chunk_id.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(chunk.text.encode("utf-8"))
        digest.update(b"\x01")
    return digest.hexdigest()[:16]


def _display_path(path: Path) -> str:
    """Prefer a repo-relative artifact path, but accept pytest/CI temp directories."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _row_to_chunk(row: dict[str, Any], source: str) -> Chunk | None:
    text = _clean(row.get("text"))
    chunk_id = _clean(row.get("chunk_id"))
    if not text or not chunk_id:
        return None
    meta = {key: value for key, value in row.items() if key != "text"}
    meta.setdefault("source", meta.get("source_document") or source)
    meta.setdefault("course_id", meta.get("course_code") or "")
    meta.setdefault("course_title", meta.get("title") or "")
    meta.setdefault("requirement_category", meta.get("course_type") or meta.get("type") or "")
    meta.setdefault("curriculum_term", meta.get("term_code") or "")
    return Chunk(chunk_id=chunk_id, text=text, meta=meta)


def _load_chunk_tree(root: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(root.rglob("*.jsonl")) if root.is_dir() else []:
        for row in _read_jsonl(path):
            chunk = _row_to_chunk(row, path.relative_to(PROJECT_ROOT).as_posix())
            if chunk:
                chunks.append(chunk)
    return chunks


def _load_course_catalog(path: Path) -> list[Chunk]:
    """Turn the compact catalog into auditable retrieval chunks for the baseline."""
    if not path.exists():
        return []
    chunks: list[Chunk] = []
    for row in _read_jsonl(path):
        code = _course_code(row.get("course_id"))
        if not code:
            continue
        title = _clean(row.get("title"))
        su = row.get("su_credits")
        ects = row.get("ects")
        text = (
            f"Official course catalog record for {code} - {title}. "
            f"SU credits: {su}. ECTS: {ects}. Faculty: {_clean(row.get('faculty'))}."
        )
        digest = hashlib.sha256(f"{code}|{title}|{su}|{ects}".encode("utf-8")).hexdigest()[:16]
        chunks.append(
            Chunk(
                chunk_id=f"course_catalog:{code}:{digest}",
                text=text,
                meta={
                    **row,
                    "chunk_id": f"course_catalog:{code}:{digest}",
                    "course_id": code,
                    "course_title": title,
                    "document_type": "course_catalog",
                    "source_document": "course_catalog/current.jsonl",
                    "source": "course_catalog/current.jsonl",
                },
            )
        )
    return chunks


def load_comparison_corpora(
    suggested_dir: Path = DEFAULT_SUGGESTED_DIR,
) -> tuple[Corpus, Corpus]:
    baseline = _load_chunk_tree(PROJECT_ROOT / "data" / "degree_requirements")
    baseline.extend(_load_chunk_tree(PROJECT_ROOT / "data" / "minors"))
    baseline.extend(_load_course_catalog(PROJECT_ROOT / "data" / "course_catalog" / "current.jsonl"))

    # Defensive filtering means a future shared loader change cannot leak suggested records
    # into the baseline.
    baseline = [c for c in baseline if c.meta.get("data_role") != "suggested_program"]
    suggested = [c for c in _load_chunk_tree(suggested_dir) if c.meta.get("data_role") == "suggested_program"]

    def unique(chunks: Iterable[Chunk]) -> list[Chunk]:
        out: list[Chunk] = []
        seen: set[str] = set()
        for chunk in chunks:
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            out.append(chunk)
        return out

    baseline = unique(baseline)
    return Corpus(baseline), Corpus(unique([*baseline, *suggested]))


def _suggested_rows(suggested_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(suggested_dir.rglob("*.jsonl")) if suggested_dir.is_dir() else []:
        for row in _read_jsonl(path):
            if row.get("data_role") == "suggested_program":
                row = dict(row)
                row["_path"] = path.relative_to(PROJECT_ROOT).as_posix()
                rows.append(row)
    return rows


def validate_suggested_source_for_benchmark(
    suggested_dir: Path = DEFAULT_SUGGESTED_DIR,
) -> dict[str, Any]:
    """Hard gate: every published program/track must have one summary for semesters 1..8."""
    rows = _suggested_rows(suggested_dir)
    profiles = {
        (_clean(row.get("program")), _clean(row.get("track")) or "standard")
        for row in rows
        if row.get("document_type") == "suggested_program_profile"
    }
    if not profiles:
        raise RuntimeError(f"No suggested-program profiles found under {suggested_dir}")
    summaries: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in rows:
        if row.get("document_type") != "suggested_program_semester":
            continue
        if row.get("term_kind") != "semester":
            continue
        key = (_clean(row.get("program")), _clean(row.get("track")) or "standard")
        if isinstance(row.get("semester"), int):
            summaries[key].append(int(row["semester"]))
    expected = list(range(1, 9))
    failures: list[str] = []
    for key in sorted(profiles):
        actual = sorted(summaries.get(key, []))
        if actual != expected:
            failures.append(f"{key[0]}/{key[1]} semesters={actual}, expected={expected}")
    unexpected = sorted(set(summaries) - profiles)
    if unexpected:
        failures.append(f"semester summaries without profiles: {unexpected}")
    if failures:
        raise RuntimeError("Suggested-program benchmark quality gate failed: " + "; ".join(failures))

    manifest_path = suggested_dir / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"Missing suggested-program manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validation = manifest.get("validation") or {}
    required_validation = {
        "status": "passed",
        "strict_schema": True,
        "unique_chunk_ids": True,
        "regular_semester_coverage": "exactly_1_through_8_per_plan",
    }
    for key, expected_value in required_validation.items():
        if validation.get(key) != expected_value:
            raise RuntimeError(
                f"Manifest validation is not final/passed: {key}={validation.get(key)!r}, "
                f"expected {expected_value!r}"
            )
    if int(manifest.get("source_count") or 0) != len(profiles):
        raise RuntimeError(
            f"Manifest/source mismatch: source_count={manifest.get('source_count')}, profiles={len(profiles)}"
        )
    return {"profiles": len(profiles), "semester_summaries": sum(map(len, summaries.values()))}


def _gold_for_course(
    row: dict[str, Any], all_rows: Sequence[dict[str, Any]]
) -> list[str]:
    """Course row plus its semester summary are both valid placement evidence."""
    gold = {_clean(row.get("chunk_id"))}
    for other in all_rows:
        if other.get("document_type") not in {"suggested_program_semester", "suggested_semester_plan"}:
            continue
        if all(
            other.get(field) == row.get(field)
            for field in ("program", "track", "semester")
        ):
            gold.add(_clean(other.get("chunk_id")))
    return sorted(item for item in gold if item)


def _baseline_evidence_indexes(
    baseline: Corpus,
) -> tuple[dict[tuple[str, Any], list[str]], dict[tuple[str, str, str], list[str]]]:
    """Index accepted equivalent facts once; dataset construction must stay linear."""
    credits: dict[tuple[str, Any], list[str]] = defaultdict(list)
    requirements: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for chunk in baseline:
        code = _course_code(chunk.course_id)
        if not code:
            continue
        if chunk.meta.get("su_credits") is not None:
            credits[(code, chunk.meta.get("su_credits"))].append(chunk.chunk_id)
        if chunk.program and chunk.requirement_category:
            requirements[(
                chunk.program, code, _category(chunk.requirement_category)
            )].append(chunk.chunk_id)
    return credits, requirements


def _baseline_course_gold(
    indexes: tuple[dict[tuple[str, Any], list[str]], dict[tuple[str, str, str], list[str]]],
    *,
    program: str,
    course_code: str,
    fact: str,
    fact_value: Any,
) -> list[str]:
    credits, requirements = indexes
    if fact == "credits":
        accepted = credits.get((course_code, fact_value), [])
    elif fact == "requirement":
        accepted = requirements.get((program, course_code, _category(fact_value)), [])
    else:
        accepted = []
    return sorted(set(accepted))


def _item(
    *,
    ident: str,
    dimension: str,
    question: str,
    answer: str,
    row: dict[str, Any],
    enhanced_gold: Sequence[str],
    baseline_gold: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "id": ident,
        "language": "en",
        "dimension": dimension,
        "question": question,
        "reference_answer": answer,
        "program": row.get("program"),
        "track": row.get("track"),
        "study_year": row.get("study_year"),
        "semester": row.get("semester"),
        "course_code": row.get("course_code") or row.get("course_id"),
        "source_document": row.get("source_document") or row.get("_path"),
        "source_page": row.get("source_page"),
        "expected_baseline_chunk_ids": sorted(set(baseline_gold)),
        "expected_enhanced_chunk_ids": sorted(set(enhanced_gold) | set(baseline_gold)),
        "evidence_fact": {
            "program": row.get("program"),
            "track": row.get("track"),
            "semester": row.get("semester"),
            "course_code": row.get("course_code") or row.get("course_id"),
            "course_type": row.get("course_type") or row.get("type"),
            "su_credits": row.get("su_credits"),
            "source_file": row.get("source_file"),
            "source_page": row.get("source_page"),
        },
    }


def _balanced_take(rows: Sequence[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    """Round-robin programs with a fixed hash order so neither PDFs nor first-year rows dominate."""
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sorted(
        rows,
        key=lambda item: (
            str(item.get("program")),
            hashlib.sha256(str(item["question"]).encode("utf-8")).hexdigest(),
        ),
    ):
        buckets[str(row.get("program") or "?")].append(row)
    selected: list[dict[str, Any]] = []
    while len(selected) < count and any(buckets.values()):
        for program in sorted(buckets):
            if buckets[program] and len(selected) < count:
                selected.append(buckets[program].pop(0))
    return selected


def build_benchmark(
    *,
    suggested_dir: Path = DEFAULT_SUGGESTED_DIR,
    benchmark_path: Path = DEFAULT_BENCHMARK,
) -> list[dict[str, Any]]:
    validate_suggested_source_for_benchmark(suggested_dir)
    baseline, _enhanced = load_comparison_corpora(suggested_dir)
    rows = _suggested_rows(suggested_dir)
    courses = [
        row
        for row in rows
        if row.get("document_type") in {"suggested_program_course", "suggested_course"}
        and row.get("course_code")
        and _is_concrete_course_code(row.get("course_code"))
        and isinstance(row.get("semester"), int)
        and 1 <= int(row["semester"]) <= 8
        and not row.get("is_elective_placeholder")
    ]
    if not courses:
        raise RuntimeError(f"No suggested_course records found under {suggested_dir}")
    baseline_indexes = _baseline_evidence_indexes(baseline)

    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    serial = 0

    def add(dimension: str, **kwargs: Any) -> None:
        nonlocal serial
        serial += 1
        candidates[dimension].append(_item(ident=f"sp{serial:04d}", dimension=dimension, **kwargs))

    for row in sorted(
        courses,
        key=lambda item: (
            str(item.get("program")), str(item.get("track")),
            int(item.get("semester") or 0), _course_code(item.get("course_code")),
        ),
    ):
        program = _clean(row.get("program"))
        track = _clean(row.get("track")) or "standard"
        semester = int(row.get("semester"))
        code = _course_code(row.get("course_code"))
        title = _clean(row.get("course_title") or row.get("title"))
        label = _track_label(track)
        gold = _gold_for_course(row, rows)

        add(
            "semester",
            question=f"In which semester does the {program} {label} recommended plan place {code} ({title})?",
            answer=f"Semester {semester}.", row=row, enhanced_gold=gold,
        )

        course_type = _clean(row.get("course_type") or row.get("type"))
        if course_type:
            baseline_gold = _baseline_course_gold(
                baseline_indexes, program=program, course_code=code,
                fact="requirement", fact_value=course_type,
            )
            add(
                "requirement_type",
                question=(
                    f"What course type is {code} in semester {semester} of the "
                    f"{program} {label} recommended plan?"
                ),
                answer=course_type, row=row, enhanced_gold=gold, baseline_gold=baseline_gold,
            )

        if row.get("su_credits") is not None:
            baseline_gold = _baseline_course_gold(
                baseline_indexes, program=program, course_code=code,
                fact="credits", fact_value=row.get("su_credits"),
            )
            add(
                "credits",
                question=(
                    f"How many SU credits is {code} in semester {semester} of the "
                    f"{program} {label} recommended plan?"
                ),
                answer=f"{row.get('su_credits')} SU credits.", row=row,
                enhanced_gold=gold, baseline_gold=baseline_gold,
            )

        add(
            "source",
            question=(
                f"Which official source supports placing {code} in semester {semester} "
                f"of the {program} {label} recommended plan?"
            ),
            answer=(
                f"{_clean(row.get('source_file'))}, page {row.get('source_page')}."
            ),
            row=row, enhanced_gold=gold,
        )

    # Track questions are only retained where (program, course, semester) identifies one
    # track. This avoids pretending that a shared fast-track placement has a unique answer.
    track_groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in courses:
        track_groups[(
            _clean(row.get("program")), _course_code(row.get("course_code")),
            int(row.get("semester") or 0),
        )].append(row)
    for (program, code, semester), group in sorted(track_groups.items()):
        tracks = {_clean(row.get("track")) or "standard" for row in group}
        if len(tracks) != 1:
            continue
        row = group[0]
        track = next(iter(tracks))
        add(
            "track",
            question=(
                f"Which {program} recommended-program track places {code} in semester {semester}?"
            ),
            answer=_track_label(track), row=row,
            enhanced_gold=sorted({cid for item in group for cid in _gold_for_course(item, rows)}),
        )

    # Program questions use placements unique across programs. Shared first-year university
    # courses are excluded rather than assigning a false single-program gold label.
    program_groups: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in courses:
        program_groups[(
            _course_code(row.get("course_code")), int(row.get("semester") or 0),
            _clean(row.get("track")) or "standard",
        )].append(row)
    for (code, semester, track), group in sorted(program_groups.items()):
        programs = {_clean(row.get("program")) for row in group}
        if len(programs) != 1:
            continue
        row = group[0]
        program = next(iter(programs))
        add(
            "program",
            question=(
                f"Which program's {_track_label(track)} plan places {code} in semester {semester}?"
            ),
            answer=program, row=row,
            enhanced_gold=sorted({cid for item in group for cid in _gold_for_course(item, rows)}),
        )

    # Eight catalog-entry requirement questions are a regression guard for the newly added
    # PSIR and Management (repository program key MAN, official degree code BAMAN) files
    # (202201..202501). They should be equally answerable in both corpora.
    degree_items: list[dict[str, Any]] = []
    for program in ("PSIR", "MAN"):
        for term in ("202201", "202301", "202401", "202501"):
            path = PROJECT_ROOT / "data" / "degree_requirements" / program / f"{term}.jsonl"
            if not path.exists():
                continue
            profiles = [row for row in _read_jsonl(path) if row.get("document_type") == "degree_requirement_profile"]
            if not profiles:
                continue
            row = dict(profiles[0])
            total = row.get("total_min_su_credits")
            degree_items.append(
                _item(
                    ident=f"degree-{program.lower()}-{term}",
                    dimension="degree_requirement",
                    question=f"What is the total minimum SU-credit requirement for {program} curriculum {term}?",
                    answer=f"{total} SU credits.", row=row,
                    enhanced_gold=[row["chunk_id"]], baseline_gold=[row["chunk_id"]],
                )
            )

    quotas = {
        "program": 10,
        "semester": 10,
        "track": 8,
        "requirement_type": 8,
        "credits": 8,
        "source": 8,
        "degree_requirement": 8,
    }
    candidates["degree_requirement"] = degree_items
    selected: list[dict[str, Any]] = []
    shortages: list[str] = []
    for dimension, wanted in quotas.items():
        available = candidates.get(dimension, [])
        taken = _balanced_take(available, wanted)
        selected.extend(taken)
        if len(taken) < wanted:
            shortages.append(f"{dimension}: wanted {wanted}, found {len(taken)}")
    if shortages:
        raise RuntimeError("Insufficient benchmark coverage: " + "; ".join(shortages))
    if len(selected) != BENCHMARK_SIZE:
        raise AssertionError(f"Expected {BENCHMARK_SIZE} items, selected {len(selected)}")

    # Stable public ids are assigned after dimension/program balancing; source facts remain
    # in each row so every gold label can be checked directly against its JSONL source.
    selected = sorted(selected, key=lambda row: (row["dimension"], row["program"] or "", row["id"]))
    for index, row in enumerate(selected, 1):
        row["id"] = f"suggested-v1-{index:03d}"
    _write_jsonl(benchmark_path, selected)
    return selected


@dataclass(frozen=True)
class SystemResult:
    name: str
    corpus_size: int
    corpus_fingerprint: str
    evidence_coverage_rate: float
    authoritative_evidence_hit_rate: float
    conditional_hit_rate: float
    hits: int
    covered: int
    total: int
    by_dimension: dict[str, dict[str, Any]]
    per_query: list[dict[str, Any]]


def _evaluate_system(
    *, name: str, corpus: Corpus, items: Sequence[dict[str, Any]], gold_key: str, top_k: int,
) -> SystemResult:
    index = BM25FIndex(corpus, tokenizer=tokenize_v2)
    corpus_ids = set(corpus.by_id)
    dimension_totals: Counter[str] = Counter()
    dimension_covered: Counter[str] = Counter()
    dimension_hits: Counter[str] = Counter()
    per_query: list[dict[str, Any]] = []
    hits = covered = 0
    for item in items:
        dimension = str(item["dimension"])
        gold = set(item.get(gold_key) or []) & corpus_ids
        ranking = index.search(tokenize_v2(item["question"]), top=top_k)
        retrieved = [hit.chunk_id for hit in ranking]
        is_covered = bool(gold)
        is_hit = bool(gold & set(retrieved))
        covered += int(is_covered)
        hits += int(is_hit)
        dimension_totals[dimension] += 1
        dimension_covered[dimension] += int(is_covered)
        dimension_hits[dimension] += int(is_hit)
        per_query.append({
            "id": item["id"], "dimension": dimension, "question": item["question"],
            "covered": is_covered, "hit": is_hit, "gold": sorted(gold),
            "retrieved": retrieved,
        })
    total = len(items)
    by_dimension = {
        dimension: {
            "n": dimension_totals[dimension],
            "covered": dimension_covered[dimension],
            "hits": dimension_hits[dimension],
            "evidence_coverage_rate": dimension_covered[dimension] / dimension_totals[dimension],
            f"authoritative_evidence_hit_rate@{top_k}": dimension_hits[dimension] / dimension_totals[dimension],
            f"conditional_hit_rate@{top_k}": (
                dimension_hits[dimension] / dimension_covered[dimension]
                if dimension_covered[dimension] else 0.0
            ),
        }
        for dimension in sorted(dimension_totals)
    }
    return SystemResult(
        name=name,
        corpus_size=len(corpus),
        corpus_fingerprint=_fingerprint(corpus.chunks),
        evidence_coverage_rate=covered / total if total else 0.0,
        authoritative_evidence_hit_rate=hits / total if total else 0.0,
        conditional_hit_rate=hits / covered if covered else 0.0,
        hits=hits,
        covered=covered,
        total=total,
        by_dimension=by_dimension,
        per_query=per_query,
    )


def run_benchmark(
    *, benchmark_path: Path = DEFAULT_BENCHMARK,
    suggested_dir: Path = DEFAULT_SUGGESTED_DIR,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    top_k: int = DEFAULT_TOP_K,
    target: float = 0.90,
) -> dict[str, Any]:
    validate_suggested_source_for_benchmark(suggested_dir)
    items = _read_jsonl(benchmark_path)
    if len(items) < BENCHMARK_SIZE:
        raise RuntimeError(f"Benchmark must contain at least {BENCHMARK_SIZE} items; found {len(items)}")
    baseline, enhanced = load_comparison_corpora(suggested_dir)
    base_result = _evaluate_system(
        name="baseline_without_suggested_programs", corpus=baseline, items=items,
        gold_key="expected_baseline_chunk_ids", top_k=top_k,
    )
    enhanced_result = _evaluate_system(
        name="enhanced_with_suggested_programs", corpus=enhanced, items=items,
        gold_key="expected_enhanced_chunk_ids", top_k=top_k,
    )
    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%SZ")
    dataset_sha = hashlib.sha256(benchmark_path.read_bytes()).hexdigest()

    def summary(result: SystemResult) -> dict[str, Any]:
        return {
            "corpus_size": result.corpus_size,
            "corpus_fingerprint": result.corpus_fingerprint,
            "evidence_coverage_rate": result.evidence_coverage_rate,
            f"authoritative_evidence_hit_rate@{top_k}": result.authoritative_evidence_hit_rate,
            f"conditional_hit_rate@{top_k}": result.conditional_hit_rate,
            "hits": result.hits, "covered": result.covered, "total": result.total,
            "by_dimension": result.by_dimension,
        }

    report = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": now.isoformat(),
        "benchmark": {
            "path": _display_path(benchmark_path),
            "sha256": dataset_sha,
            "n_queries": len(items),
            "dimensions": dict(sorted(Counter(item["dimension"] for item in items).items())),
        },
        "metric": {
            "name": f"Authoritative Evidence Hit Rate@{top_k}",
            "short_name": f"AEHR@{top_k}",
            "definition": (
                f"Fraction of all benchmark questions with at least one accepted authoritative "
                f"evidence chunk in the first {top_k} results. Missing corpus evidence is a miss."
            ),
            "secondary_metrics": {
                "evidence_coverage_rate": "Fraction of questions with accepted evidence in the corpus.",
                f"conditional_hit_rate@{top_k}": (
                    "Hit rate restricted to questions whose accepted evidence exists in that corpus."
                ),
            },
        },
        "systems": {
            base_result.name: summary(base_result),
            enhanced_result.name: summary(enhanced_result),
        },
        "comparison": {
            f"aehr@{top_k}_absolute_delta": (
                enhanced_result.authoritative_evidence_hit_rate
                - base_result.authoritative_evidence_hit_rate
            ),
            "target": target,
            "target_passed": enhanced_result.authoritative_evidence_hit_rate >= target,
        },
    }

    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_jsonl(run_dir / "baseline_per_query.jsonl", base_result.per_query)
    _write_jsonl(run_dir / "enhanced_per_query.jsonl", enhanced_result.per_query)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "latest_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--suggested-dir", type=Path, default=DEFAULT_SUGGESTED_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--build-dataset", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument(
        "--assert-target", type=float, default=None,
        help="Exit non-zero if enhanced AEHR@K is below this threshold.",
    )
    args = parser.parse_args()
    if not args.build_dataset and not args.run:
        parser.error("Choose --build-dataset, --run, or both.")
    if args.build_dataset:
        items = build_benchmark(suggested_dir=args.suggested_dir, benchmark_path=args.benchmark)
        print(f"Wrote {len(items)} benchmark questions to {args.benchmark}")
    if args.run:
        target = args.assert_target if args.assert_target is not None else 0.90
        report = run_benchmark(
            benchmark_path=args.benchmark, suggested_dir=args.suggested_dir,
            output_root=args.output_root, top_k=args.top_k, target=target,
        )
        baseline = report["systems"]["baseline_without_suggested_programs"]
        enhanced = report["systems"]["enhanced_with_suggested_programs"]
        metric = f"authoritative_evidence_hit_rate@{args.top_k}"
        print(f"Baseline {metric}: {baseline[metric]:.2%}")
        print(f"Enhanced {metric}: {enhanced[metric]:.2%}")
        print(f"Absolute delta: {report['comparison'][f'aehr@{args.top_k}_absolute_delta']:+.2%}")
        print(f"Target {target:.0%}: {'PASS' if report['comparison']['target_passed'] else 'FAIL'}")
        print(f"Summary: {args.output_root / 'latest_summary.json'}")
        if args.assert_target is not None and not report["comparison"]["target_passed"]:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
