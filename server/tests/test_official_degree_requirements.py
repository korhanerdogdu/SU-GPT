from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TERMS = ("202201", "202301", "202401", "202501")


def _rows(program: str, term: str) -> list[dict]:
    path = ROOT / "data" / "degree_requirements" / program / f"{term}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class OfficialDegreeRequirementTests(unittest.TestCase):
    def test_official_major_files_have_complete_deterministic_metadata(self):
        for program, degree_code in (("MAN", "BAMAN"), ("PSIR", "BAPSIR")):
            for term in TERMS:
                with self.subTest(program=program, term=term):
                    rows = _rows(program, term)
                    self.assertTrue(rows)
                    self.assertEqual(len({row["chunk_id"] for row in rows}), len(rows))
                    self.assertEqual(
                        sum(row["document_type"] == "degree_requirement_profile" for row in rows), 1
                    )
                    for row in rows:
                        self.assertEqual(row["program"], program)
                        self.assertEqual(row["degree_code"], degree_code)
                        self.assertEqual(row["curriculum_term"], term)
                        self.assertEqual(row["source_document"], f"degree_requirements/{program}/{term}.jsonl")
                        self.assertIn("suis.sabanciuniv.edu/prod/SU_DEGREE", row["source_url"])
                        self.assertIn(f"P_PROGRAM={degree_code}", row["source_url"])
                        self.assertFalse(any(marker in row["text"] for marker in ("Ã", "Â", "�")))
                        if row["document_type"] == "degree_requirement_pool_course":
                            self.assertIsInstance(row["su_credits"], (int, float))
                            self.assertIsInstance(row["ects"], (int, float))

    def test_pool_summary_course_ids_match_their_course_records(self):
        for program in ("MAN", "PSIR"):
            for term in TERMS:
                with self.subTest(program=program, term=term):
                    rows = _rows(program, term)
                    by_pool: dict[str, set[str]] = {}
                    for row in rows:
                        if row["document_type"] == "degree_requirement_pool_course":
                            by_pool.setdefault(row["pool_key"], set()).add(row["course_id"])
                    for row in rows:
                        if row["document_type"] == "degree_requirement_category_pool":
                            self.assertEqual(set(row["course_ids"]), by_pool[row["pool_key"]])

    def test_management_official_minima_and_required_courses(self):
        expected_required = {"ECON 202", "ECON 204", "ENG 300", "MGMT 201", "MGMT 203", "MGMT 300"}
        for term in TERMS:
            with self.subTest(term=term):
                rows = _rows("MAN", term)
                profile = next(row for row in rows if row["document_type"] == "degree_requirement_profile")
                minima = {item["category"]: item for item in profile["categories"]}
                self.assertEqual(profile["total_min_su_credits"], 127)
                self.assertEqual(profile["total_min_ects"], 240)
                self.assertEqual(minima["required_courses"]["min_su_credits"], 15)
                self.assertEqual(minima["required_courses"]["min_courses"], 6)
                self.assertEqual(minima["core_electives"]["min_su_credits"], 18)
                self.assertEqual(minima["core_electives"]["min_courses"], 6)
                required = {
                    row["course_id"]
                    for row in rows
                    if row["document_type"] == "degree_requirement_pool_course"
                    and row["requirement_category"] == "required_courses"
                }
                self.assertEqual(required, expected_required)

    def test_psir_keeps_both_official_core_minima_separate(self):
        for term in TERMS:
            with self.subTest(term=term):
                rows = _rows("PSIR", term)
                profile = next(row for row in rows if row["document_type"] == "degree_requirement_profile")
                minima = {item["category"]: item for item in profile["categories"]}
                self.assertEqual(profile["total_min_su_credits"], 125)
                self.assertEqual(profile["total_min_ects"], 240)
                self.assertEqual(minima["core_electives_political_science"]["min_su_credits"], 12)
                self.assertEqual(minima["core_electives_international_relations"]["min_su_credits"], 12)
                self.assertNotIn("core_electives", minima)
                categories = {
                    row["requirement_category"]
                    for row in rows
                    if row["document_type"] == "degree_requirement_pool_course"
                }
                self.assertIn("core_electives_political_science", categories)
                self.assertIn("core_electives_international_relations", categories)
                rules = {
                    row.get("rule_id")
                    for row in rows
                    if row["document_type"] == "degree_requirement_rule"
                }
                self.assertIn("official_required_summary_discrepancy", rules)


if __name__ == "__main__":
    unittest.main()
