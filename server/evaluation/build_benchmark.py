from __future__ import annotations

"""
Generate data/benchmark/questions.jsonl from the shipped corpus (CLAUDE.md Section 5.1).

WHY GENERATED, NOT HAND-WRITTEN
CLAUDE.md 17 forbids fabricating benchmark answers. Hand-typing "CS needs 125 SU credits" invites
exactly that. So every answerable question here is derived from a real row in
data/degree_requirements/** or data/minors/**: the ground-truth `expected_chunk_ids` are that
row's real chunk_id, and `reference_answer` is built from that row's own fields. If the corpus is
re-scraped and a number changes, re-running this script keeps the benchmark truthful.

The question *wording* is templated (a known limitation - templated phrasing is easier for a
lexical retriever than free student prose, so BM25 scores here are an optimistic bound). The
unanswerable and misleading items are hand-written on purpose, and target gaps CLAUDE.md section H
already documents as real: schedules, instructors and enrolment data are NOT in the corpus.

Usage:  python server/evaluation/build_benchmark.py [--out data/benchmark/questions.jsonl]
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

SERVER_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SERVER_ROOT.parent
DATA_DIR = PROJECT_ROOT / "data"

# The reference student for the benchmark. CS/202401 is the curriculum used in the demo.
PROGRAM = "CS"
TERM = "202401"
TERM_LABEL = "Fall 2024-2025"


def _load(path: Path) -> list[dict]:
    if not path.exists():
        sys.exit(f"missing corpus file: {path}\nRun the offline pipeline first (see CLAUDE.md B).")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _pretty_category(raw: str) -> str:
    return (raw or "").replace("_", " ").strip()


def _rows_by_type(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        out[row.get("document_type", "")].append(row)
    return out


def build() -> list[dict]:
    rows = _load(DATA_DIR / "degree_requirements" / PROGRAM / f"{TERM}.jsonl")
    by_type = _rows_by_type(rows)
    source_doc = f"degree_requirements/{PROGRAM}/{TERM}.jsonl"

    profile = by_type["degree_requirement_profile"][0]
    pool_courses = by_type["degree_requirement_pool_course"]
    rules = by_type["degree_requirement_rule"]

    # Which categories does each course appear in? Needed so a "which category" question is only
    # asked about courses with an unambiguous answer.
    categories_of: dict[str, list[dict]] = defaultdict(list)
    for row in pool_courses:
        categories_of[row["course_id"]].append(row)

    items: list[dict] = []

    def add(**kwargs) -> None:
        item = {
            "id": f"q{len(items) + 1:03d}",
            "language": "en",
            "answerable": True,
            "program": PROGRAM,
            "curriculum_term": TERM,
            "intent": "ders_ayrintisi",
            "expected_sources": [source_doc],
            "expected_chunk_ids": [],
            "reference_answer": "",
            "notes": "",
        }
        item.update(kwargs)
        items.append(item)

    # ---- A. course -> requirement category (unambiguous courses only) ------------------
    unambiguous = sorted(
        (cid for cid, rs in categories_of.items() if len({r["requirement_category"] for r in rs}) == 1),
        key=str,
    )
    # deterministic spread across different categories
    picked: dict[str, str] = {}
    for course_id in unambiguous:
        category = categories_of[course_id][0]["requirement_category"]
        picked.setdefault(category, course_id)
    for category, course_id in sorted(picked.items())[:4]:
        rows_for_course = categories_of[course_id]
        add(
            question=f"In the {PROGRAM} {TERM} curriculum, which requirement category does {course_id} belong to?",
            question_type="factual_lookup",
            reference_answer=f"{course_id} belongs to the {_pretty_category(category)} pool of the {PROGRAM} {TERM} curriculum.",
            expected_chunk_ids=[r["chunk_id"] for r in rows_for_course],
            notes=f"derived from degree_requirement_pool_course rows; category={category}",
        )

    # ---- B. course credits -------------------------------------------------------------
    credit_course = next(
        (c for c in unambiguous if categories_of[c][0].get("su_credits") and categories_of[c][0].get("ects")),
        None,
    )
    if credit_course:
        row = categories_of[credit_course][0]
        add(
            question=f"How many SU credits and ECTS does {credit_course} carry in the {PROGRAM} {TERM} curriculum?",
            question_type="factual_lookup",
            reference_answer=f"{credit_course} carries {row['su_credits']} SU credits and {row['ects']} ECTS.",
            expected_chunk_ids=[row["chunk_id"]],
            notes="derived from su_credits/ects fields",
        )

    # ---- C. program totals (authoritative graduation intent) ---------------------------
    add(
        question=f"What is the minimum total number of SU credits required to graduate from the {PROGRAM} {TERM} curriculum?",
        question_type="course_requirement",
        intent="mezuniyet_durumu",
        reference_answer=f"{profile['total_min_su_credits']} SU credits.",
        expected_chunk_ids=[profile["chunk_id"]],
        notes="derived from degree_requirement_profile.total_min_su_credits",
    )
    add(
        question=f"What is the minimum total ECTS required for the {PROGRAM} {TERM} curriculum?",
        question_type="course_requirement",
        intent="mezuniyet_durumu",
        reference_answer=f"{profile['total_min_ects']} ECTS.",
        expected_chunk_ids=[profile["chunk_id"]],
        notes="derived from degree_requirement_profile.total_min_ects",
    )

    # ---- D. category minimums (these live on the category_pool rows, not the rule rows) --
    for pool in sorted(by_type["degree_requirement_category_pool"], key=lambda r: r.get("requirement_category", "")):
        if not pool.get("min_su_credits"):
            continue
        category_key = pool.get("requirement_category", "")
        category = _pretty_category(category_key)
        # A rule row restating the same minimum is equally valid evidence -> also gold.
        gold = [pool["chunk_id"]] + [
            r["chunk_id"] for r in rules
            if r.get("requirement_category") == category_key
            and r.get("min_su_credits") == pool["min_su_credits"]
        ]
        add(
            question=f"How many SU credits of {category} are required in the {PROGRAM} {TERM} curriculum?",
            question_type="course_requirement",
            intent="mezuniyet_durumu",
            reference_answer=f"{pool['min_su_credits']} SU credits of {category}.",
            expected_chunk_ids=gold,
            notes=f"derived from degree_requirement_category_pool.min_su_credits (category={category_key})",
        )

    # ---- E. an explicit exclusion rule (tests precision, not just recall) --------------
    free_rule = next((r for r in rules if r.get("requirement_category") == "free_electives"), None)
    if free_rule and "School of Languages" in free_rule.get("text", ""):
        add(
            question=f"Do School of Languages (SL) courses count as free electives in the {PROGRAM} {TERM} curriculum?",
            question_type="course_requirement",
            intent="mezuniyet_durumu",
            reference_answer="No. Language courses offered by the School of Languages (SL) do not count as free electives.",
            expected_chunk_ids=[free_rule["chunk_id"]],
            notes="explicit exclusion stated in the free_electives rule text",
        )

    # ---- F. Turkish phrasing over the same ground truth --------------------------------
    add(
        question=f"{PROGRAM} {TERM} müfredatında mezun olmak için gereken minimum toplam SU kredisi kaçtır?",
        question_type="turkish",
        language="tr",
        intent="mezuniyet_durumu",
        reference_answer=f"{profile['total_min_su_credits']} SU kredisi.",
        expected_chunk_ids=[profile["chunk_id"]],
        notes="Turkish rendering of the English total-credits question (same gold chunk)",
    )
    if credit_course:
        row = categories_of[credit_course][0]
        add(
            question=f"{credit_course} dersi {PROGRAM} {TERM} müfredatında kaç SU kredisidir?",
            question_type="turkish",
            language="tr",
            reference_answer=f"{credit_course} dersi {row['su_credits']} SU kredisidir.",
            expected_chunk_ids=[row["chunk_id"]],
            notes="Turkish rendering of the credits question",
        )

    # ---- G. minor requirements ----------------------------------------------------------
    minor_dir = DATA_DIR / "minors"
    minor_file = next(iter(sorted(minor_dir.glob(f"MATH-MINOR/{TERM}.jsonl"))), None)
    if minor_file:
        minor_rows = _load(minor_file)
        minor_profile = next((r for r in minor_rows if r["document_type"] == "minor_requirement_profile"), None)
        if minor_profile:
            add(
                question="What are the requirements of the Mathematics minor programme?",
                question_type="course_requirement",
                intent="minor",
                program=None,
                expected_sources=[f"minors/MATH-MINOR/{TERM}.jsonl"],
                reference_answer=minor_profile["text"][:400],
                expected_chunk_ids=[minor_profile["chunk_id"]],
                notes="minor_requirement_profile row; also checks data_role isolation from majors",
            )

    # ---- H. cross-program scope check ---------------------------------------------------
    ie_file = DATA_DIR / "degree_requirements" / "IE" / f"{TERM}.jsonl"
    if ie_file.exists():
        ie_profile = next(r for r in _load(ie_file) if r["document_type"] == "degree_requirement_profile")
        add(
            question=f"What is the minimum total number of SU credits required for the IE {TERM} curriculum?",
            question_type="factual_lookup",
            program="IE",
            intent="mezuniyet_durumu",
            expected_sources=[f"degree_requirements/IE/{TERM}.jsonl"],
            reference_answer=f"{ie_profile['total_min_su_credits']} SU credits.",
            expected_chunk_ids=[ie_profile["chunk_id"]],
            notes="different program; a correct system must not answer with CS numbers",
        )

    # ---- I. unanswerable: real, documented corpus gaps (CLAUDE.md section H) -------------
    for question, note in [
        (f"What time does {PROGRAM} 455 meet on Mondays, and in which classroom?",
         "schedules are not ingested - data/schedules/** does not exist"),
        (f"Who is the instructor teaching {PROGRAM} 412 next semester?",
         "no instructor/schedule data in the corpus"),
        (f"How many students were enrolled in {PROGRAM} 455 last year?",
         "enrolment statistics were never collected"),
        ("What was the midterm average in CS 300 last spring?",
         "no grade or exam-statistics data in the corpus"),
    ]:
        add(
            question=question,
            question_type="unanswerable",
            answerable=False,
            expected_sources=[],
            expected_chunk_ids=[],
            reference_answer="The available curriculum data does not contain this information.",
            notes=note,
        )

    # ---- J. misleading premise ----------------------------------------------------------
    add(
        question=f"Is CS 999 a required course in the {PROGRAM} {TERM} curriculum?",
        question_type="misleading",
        answerable=False,
        expected_sources=[],
        expected_chunk_ids=[],
        reference_answer="CS 999 does not exist in the curriculum; the premise is false.",
        notes="false premise - a non-existent course code",
    )
    add(
        question=f"Since the {PROGRAM} {TERM} curriculum requires 200 SU credits, how many do I have left after 100?",
        question_type="misleading",
        answerable=False,
        expected_sources=[],
        expected_chunk_ids=[],
        reference_answer=(
            f"The premise is false: the {PROGRAM} {TERM} curriculum requires "
            f"{profile['total_min_su_credits']} SU credits, not 200."
        ),
        notes="false premise embedded in the question - tests whether the system corrects it",
    )

    return items


def verify(items: list[dict]) -> None:
    """Fail loudly if any gold chunk_id is not actually in the corpus."""
    known: set[str] = set()
    for path in sorted(DATA_DIR.glob("degree_requirements/*/*.jsonl")) + sorted(DATA_DIR.glob("minors/*/*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                known.add(json.loads(line)["chunk_id"])
    bad = [(i["id"], cid) for i in items for cid in i["expected_chunk_ids"] if cid not in known]
    if bad:
        sys.exit(f"ground truth references unknown chunk_ids: {bad}")
    print(f"verified: every gold chunk_id exists in the corpus ({len(known)} chunks known)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the adviSU retrieval benchmark from the corpus.")
    parser.add_argument("--out", default=str(PROJECT_ROOT / "data" / "benchmark" / "questions.jsonl"))
    args = parser.parse_args()

    items = build()
    verify(items)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    answerable = sum(1 for i in items if i["answerable"])
    print(f"wrote {len(items)} questions -> {out_path}")
    print(f"  answerable: {answerable} | unanswerable/misleading: {len(items) - answerable}")
    kinds: dict[str, int] = defaultdict(int)
    for item in items:
        kinds[item["question_type"]] += 1
    print("  by type:", dict(sorted(kinds.items())))


if __name__ == "__main__":
    main()
