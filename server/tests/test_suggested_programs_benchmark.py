from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pytest


SERVER_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SERVER_ROOT.parent
if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))

from evaluation import benchmark_suggested_programs as BENCH


def _require_data() -> None:
    if not BENCH.DEFAULT_SUGGESTED_DIR.exists():
        pytest.skip("suggested-program corpus has not been generated yet")


def _read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_comparison_corpora_are_strictly_isolated() -> None:
    _require_data()
    baseline, enhanced = BENCH.load_comparison_corpora()
    assert baseline.chunks
    assert all(chunk.meta.get("data_role") != "suggested_program" for chunk in baseline)
    suggested = [chunk for chunk in enhanced if chunk.meta.get("data_role") == "suggested_program"]
    assert suggested
    assert len(enhanced) == len(baseline) + len(suggested)


def test_benchmark_is_deterministic_and_has_auditable_coverage(tmp_path: Path) -> None:
    _require_data()
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    BENCH.build_benchmark(benchmark_path=first)
    BENCH.build_benchmark(benchmark_path=second)
    assert first.read_bytes() == second.read_bytes()

    rows = _read_rows(first)
    assert len(rows) == 60
    assert len({row["id"] for row in rows}) == 60
    assert Counter(row["dimension"] for row in rows) == {
        "program": 10,
        "semester": 10,
        "track": 8,
        "requirement_type": 8,
        "credits": 8,
        "source": 8,
        "degree_requirement": 8,
    }
    assert {row["program"] for row in rows if row["dimension"] == "degree_requirement"} == {
        "PSIR", "MAN"
    }
    assert all(row["question"] and row["reference_answer"] for row in rows)
    assert all(row["expected_enhanced_chunk_ids"] for row in rows)
    assert all(row["source_document"] for row in rows)


def test_every_suggested_gold_label_matches_its_recorded_fact(tmp_path: Path) -> None:
    _require_data()
    path = tmp_path / "benchmark.jsonl"
    BENCH.build_benchmark(benchmark_path=path)
    rows = _read_rows(path)
    _baseline, enhanced = BENCH.load_comparison_corpora()

    for item in rows:
        gold_chunks = [
            enhanced.by_id[chunk_id]
            for chunk_id in item["expected_enhanced_chunk_ids"]
            if chunk_id in enhanced.by_id
        ]
        assert gold_chunks, item["id"]
        if item["dimension"] == "degree_requirement":
            assert any(chunk.program == item["program"] for chunk in gold_chunks)
            continue
        fact = item["evidence_fact"]
        assert any(
            chunk.meta.get("data_role") == "suggested_program"
            and chunk.program == fact["program"]
            and chunk.meta.get("track") == fact["track"]
            and chunk.meta.get("semester") == fact["semester"]
            for chunk in gold_chunks
        ), item["id"]


@pytest.mark.parametrize("program,degree_code", [("PSIR", "BAPSIR"), ("MAN", "BAMAN")])
def test_new_degree_requirement_years_are_present(program: str, degree_code: str) -> None:
    for term in ("202201", "202301", "202401", "202501"):
        path = PROJECT_ROOT / "data" / "degree_requirements" / program / f"{term}.jsonl"
        if not path.exists():
            pytest.fail(f"missing requested degree-requirement file: {path}")
        profiles = [
            row for row in _read_rows(path)
            if row.get("document_type") == "degree_requirement_profile"
        ]
        assert len(profiles) == 1
        assert profiles[0]["program"] == program
        assert profiles[0]["degree_code"] == degree_code
        assert profiles[0]["curriculum_term"] == term
        assert profiles[0].get("total_min_su_credits")


def test_enhanced_corpus_reaches_the_declared_90_percent_target(tmp_path: Path) -> None:
    """End-to-end acceptance test; no item or failure is removed from the denominator."""
    _require_data()
    benchmark = tmp_path / "benchmark.jsonl"
    BENCH.build_benchmark(benchmark_path=benchmark)
    report = BENCH.run_benchmark(
        benchmark_path=benchmark,
        output_root=tmp_path / "outputs",
        target=0.90,
    )
    baseline = report["systems"]["baseline_without_suggested_programs"]
    enhanced = report["systems"]["enhanced_with_suggested_programs"]
    metric = "authoritative_evidence_hit_rate@5"
    assert enhanced[metric] >= 0.90
    assert enhanced[metric] > baseline[metric]
    assert enhanced["total"] == baseline["total"] == 60
    assert enhanced["evidence_coverage_rate"] == 1.0
