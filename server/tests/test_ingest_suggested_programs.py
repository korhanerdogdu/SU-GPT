from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SERVER_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SERVER_ROOT.parent
SCRIPTS_ROOT = SERVER_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import ingest_suggested_programs as INGEST
import validate_suggested_programs as VALIDATE


DATA_DIR = PROJECT_ROOT / "data" / "suggested_programs"
SOURCE_DIR = DATA_DIR / "sources"


def _rows(program: str, track: str = "standard") -> list[dict]:
    path = DATA_DIR / program / f"{track}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class SuggestedProgramIngestionTests(unittest.TestCase):
    def test_checked_in_dataset_passes_strict_validator(self) -> None:
        self.assertEqual(VALIDATE.validate_dataset(DATA_DIR), [])

    def test_every_plan_has_exact_regular_semester_coverage(self) -> None:
        for spec in INGEST.SOURCE_SPECS:
            rows = _rows(spec.program, spec.output_stem)
            semesters = {
                row["semester"]
                for row in rows
                if row["document_type"] == "suggested_program_semester"
                and row["term_kind"] == "semester"
            }
            self.assertEqual(semesters, set(range(1, 9)), spec.filename)
            for row in rows:
                if (
                    row["document_type"] == "suggested_program_semester"
                    and row["term_kind"] != "semester"
                ):
                    self.assertIsNone(row["semester"], spec.filename)

    def test_cs_tracks_remain_separate_and_retain_distinct_placement(self) -> None:
        placements: dict[str, tuple[int, int | None, str]] = {}
        for track in ("standard", "fast_track_1", "fast_track_2", "fast_track_3"):
            course = next(
                row
                for row in _rows("CS", track)
                if row["document_type"] == "suggested_program_course"
                and row["course_code"] == "CS 201"
            )
            placements[track] = (
                course["study_year"],
                course["semester"],
                course["term_kind"],
            )
        self.assertEqual(placements["standard"], (2, 3, "semester"))
        self.assertEqual(placements["fast_track_1"], (1, 2, "semester"))
        self.assertEqual(placements["fast_track_2"], (1, None, "summer"))
        self.assertEqual(placements["fast_track_3"], (1, 2, "semester"))

    def test_visual_text_layer_repairs_are_present(self) -> None:
        ee = _rows("EE")
        self.assertTrue(
            any(
                row["document_type"] == "suggested_program_course"
                and row["course_code"] == "EE 395"
                and row["term_kind"] == "summer"
                and row["semester"] is None
                for row in ee
            )
        )
        self.assertTrue(
            any(
                row["document_type"] == "suggested_program_course"
                and row["course_title"] == "EE Track or Area Elective"
                for row in ee
            )
        )
        psy = _rows("PSY")
        self.assertTrue(
            any(
                row["document_type"] == "suggested_program_course"
                and row["course_code"] == "PSY 201"
                and row["course_title"] == "Mind and Behavior"
                for row in psy
            )
        )
        dsa_titles = {
            row["course_code"]: row["course_title"]
            for row in _rows("DSA")
            if row["document_type"] == "suggested_program_course" and row["course_code"]
        }
        self.assertEqual(dsa_titles["DSA 301"], "Data Visualization")
        self.assertEqual(dsa_titles["OPIM 390"], "Introduction to Business Analytics")

    def test_conversion_is_byte_deterministic(self) -> None:
        try:
            import pdfplumber  # noqa: F401
        except ImportError:
            self.skipTest("offline PDF regeneration requires optional pdfplumber")
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first = Path(first_dir)
            second = Path(second_dir)
            INGEST.convert(SOURCE_DIR, first)
            INGEST.convert(SOURCE_DIR, second)
            first_files = sorted(path.relative_to(first) for path in first.rglob("*.*"))
            second_files = sorted(path.relative_to(second) for path in second.rglob("*.*"))
            self.assertEqual(first_files, second_files)
            for relative in first_files:
                self.assertEqual((first / relative).read_bytes(), (second / relative).read_bytes())


if __name__ == "__main__":
    unittest.main()
