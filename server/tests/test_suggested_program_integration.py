from __future__ import annotations

import json
from pathlib import Path

from modules.bm25_retriever import _is_narrowing
from modules.catalog_retriever import _document_type_filter
from modules.intents import COURSE_RECOMMENDATION
from modules.retrieval_policy import build_metadata_filter
from retrieval_lab.corpus import load_corpus
from scripts.ingest_degree_requirements import _iter_rows


def _write_suggested_row(root: Path) -> dict:
    row = {
        "data_role": "suggested_program",
        "authority_level": "official_advisory",
        "binding_status": "non_binding_recommended_plan",
        "program": "CS",
        "program_name": "Computer Science and Engineering",
        "plan_type": "standard_track",
        "semester": 3,
        "document_type": "suggested_program_course",
        "course_code": "CS 201",
        "course_title": "Programming Fundamentals",
        "source_document": "suggested_programs/sources/cs.pdf",
        "source_page": 1,
        "text": "The official non-binding CS standard track suggests CS 201 in semester 3.",
        "chunk_id": "suggested_program:CS:standard_track:semester_3:CS201:test",
    }
    path = root / "suggested_programs" / "CS" / "standard_track.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return row


def test_suggested_program_rows_enter_benchmark_and_production_ingesters(tmp_path: Path) -> None:
    expected = _write_suggested_row(tmp_path)

    corpus = load_corpus(tmp_path)
    assert len(corpus.chunks) == 1
    assert corpus.chunks[0].chunk_id == expected["chunk_id"]
    assert corpus.chunks[0].meta["documentType"] == "course"
    assert corpus.chunks[0].course_id == "CS 201"

    ingested = list(_iter_rows(tmp_path))
    assert len(ingested) == 1
    chunk_id, document = ingested[0]
    assert chunk_id == expected["chunk_id"]
    assert document.metadata["data_role"] == "suggested_program"
    assert document.metadata["documentType"] == "course"
    assert document.metadata["course_id"] == "CS 201"


def test_recommendation_filter_keeps_exact_requirements_and_program_plan() -> None:
    metadata_filter = build_metadata_filter(
        COURSE_RECOMMENDATION,
        {"major": "CS", "curriculum_term": "202501"},
    )

    assert metadata_filter == {
        "$or": [
            {"$and": [
                {"data_role": "curriculum_requirement"},
                {"program": "CS"},
                {"curriculum_term": "202501"},
            ]},
            {"$and": [
                {"data_role": "suggested_program"},
                {"program": "CS"},
            ]},
        ]
    }
    assert _is_narrowing(metadata_filter)


def test_recommendation_language_selects_suggested_document_types() -> None:
    selected = _document_type_filter("CS için bu dönem hangi dersleri almalıyım?")
    assert "suggested_program_semester" in selected
    assert "suggested_program_course" in selected
    assert "schedule_section" not in selected
