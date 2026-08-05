from __future__ import annotations

"""
Automated invariant tests for the adviSU advising backend (roadmap section 23).

Runs standalone (no pytest needed):  python server/tests/test_advising.py
Add --integration to also exercise the Chroma vectorstore (loads embeddings, slower).
Functions are named test_* so `pytest` works too if installed.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("MONGO_URI", "mongodb://localhost:27017")  # lazy; never dialed in unit tests

from modules import (
    content_safety,
    curriculum_registry,
    degree_audit,
    course_planner,
    intents,
    major_advisor,
    schedule_planner,
)
from modules.llm import detect_language
from modules.retrieval_policy import build_metadata_filter, check_profile
from modules.bm25_retriever import _tokenize, _is_narrowing
from modules.conversation_memory import _automatic_title, extract_course_code, resolve_reference
from modules.course_commands import parse_course_history_command
from modules.profile_commands import parse_academic_profile_command, resolve_profile_update
from modules.export_utils import course_rows
from modules.mongodb import _norm_status, ELIGIBLE_FOR_CREDIT
from modules.response_formatter import audit_answer, audit_summary, ensure_summary_section, sanitize_student_answer


# ---- retrieval policy / profile scoping ---------------------------------------------
def test_profile_filter_scopes_program_and_term():
    f = build_metadata_filter("graduation_status", {"major": "CS", "curriculum_term": "202401"})
    clauses = f["$and"]
    assert {"data_role": "curriculum_requirement"} in clauses
    assert {"program": "CS"} in clauses
    assert {"curriculum_term": "202401"} in clauses


def test_minor_filter_uses_minor_role():
    assert build_metadata_filter("minor", {}) == {"data_role": "minor_requirement"}


def test_missing_profile_fields_blocks_audit():
    g = check_profile("graduation_status", {})
    assert not g.ok and g.missing_fields


def test_missing_curriculum_is_unavailable_not_fallback():
    g = check_profile("graduation_status", {"major": "IE", "curriculum_term": "202101"})
    assert not g.ok and g.data_unavailable  # IE 202101 not in corpus -> safe limitation


def test_valid_cs_curriculum_passes_gate():
    assert check_profile("graduation_status", {"major": "CS", "curriculum_term": "202401"}).ok


# ---- registry -----------------------------------------------------------------------
def test_registry_has_nine_majors():
    progs = curriculum_registry.list_major_programs()
    for p in ("CS", "IE", "EE", "ME", "BIO", "DSA", "ECON", "PSY", "MAT"):
        assert p in progs
    assert curriculum_registry.has_major_curriculum("CS", "202401")
    assert not curriculum_registry.has_major_curriculum("IE", "202101")


# ---- deterministic audit ------------------------------------------------------------
def test_audit_unavailable_when_file_missing():
    assert degree_audit.audit("IE", "202101", [])["status"] == "unavailable"


def test_audit_math_choice_not_flagged_missing():
    # completing MATH 201 satisfies the MATH201-or-212 choice; MATH 212 must NOT be "missing"
    a = degree_audit.audit("CS", "202401", ["MATH 201"])
    assert "MATH 212" not in a["missing_required_courses"]


def test_audit_elective_overflow_and_scoping():
    a = degree_audit.audit("CS", "202401", ["CS 306", "CS 307", "CS 412", "CS 455"])
    cats = {c["category"]: c for c in a["categories"]}
    # CS 455 is an area elective; the three CS 3xx are core -> core filled, CS455 -> area
    assert cats["area_electives"]["completed_su_credits"] >= 3
    assert cats["core_electives"]["completed_su_credits"] >= 9
    assert a["reliability"] == "authoritative"


def test_audit_missing_required_detected():
    a = degree_audit.audit("CS", "202401", [])
    for code in ("CS 395", "ENS 491", "ENS 492"):
        assert code in a["missing_required_courses"]


# ---- BM25 tokenization --------------------------------------------------------------
def test_bm25_merges_course_code_token():
    toks = _tokenize("is CS 455 an area elective")
    assert "cs455" in toks and "cs" in toks and "455" in toks


def test_bm25_only_runs_on_narrowing_filter():
    assert _is_narrowing({"$and": [{"program": "CS"}]})
    assert not _is_narrowing({"data_role": "curriculum_requirement"})


# ---- conversation memory ------------------------------------------------------------
def test_reference_resolution_injects_last_course():
    q = resolve_reference("can I take it next semester?", {"last_course_id": "CS 455"})
    assert "CS 455" in q
    assert extract_course_code("does CS412 count") == "CS 412"


def test_followup_resolution_keeps_previous_intent():
    q = resolve_reference("yani ne almam lazım?", {"last_intent": "graduation_status"})
    assert "graduation_status" in q


def test_automatic_title_uses_multiple_turns():
    title = _automatic_title(
        [
            {"user": "Merhaba, mezuniyet durumumu hesaplar mısın?", "assistant": "..."},
            {"user": "Yani ne almam lazım?", "assistant": "..."},
        ]
    )
    assert "mezuniyet" in title.lower()
    assert "dönem planı" in title.lower()
    assert len(title) <= 70


def test_course_history_command_parser():
    command = parse_course_history_command("CS 201 ve MATH101'i aldım")
    assert command is not None
    assert command.status == "completed"
    assert command.course_codes == ("CS 201", "MATH 101")
    assert parse_course_history_command("CS 201 hakkında bilgi ver") is None


def test_course_history_shorthand_repeats_subject():
    command = parse_course_history_command("CS 445 412 404 aldım tüm university coursesı aldım")
    assert command is not None
    assert command.course_codes == ("CS 445", "CS 412", "CS 404")


def test_profile_command_parses_major_and_curriculum_term():
    command = parse_academic_profile_command("Bölümüm CS, müfredat dönemim 202401")
    assert command is not None and command.major == "CS" and command.curriculum_term == "202401"
    update, error = resolve_profile_update(command, {})
    assert error is None
    assert update["degree_code"] == "BSCS"
    natural = parse_academic_profile_command("Majorum cs cirriculum termim Fall 2022")
    assert natural is not None
    assert natural.major == "CS" and natural.curriculum_term == "202201"


def test_course_export_contains_totals():
    rows = course_rows([
        {"code": "CS 201", "title": "X", "status": "completed", "su_credits": 3, "ects": 6},
        {"code": "CS 202", "title": "Y", "status": "failed", "su_credits": 3, "ects": 6},
    ])
    assert rows[-1]["code"] == "TOTAL"
    assert rows[-1]["total_su_credits"] == 3
    assert rows[0]["total_su_credits"] == 3


def test_summary_is_always_available_for_long_answers():
    answer, summary = ensure_summary_section(
        "İlk önemli sonuç budur. İkinci önemli sonuç budur.", language="tr"
    )
    assert summary
    assert "Kısa Özet" in answer


def test_audit_summary_contains_total_remaining_and_status():
    audit = degree_audit.audit("CS", "202401", ["CS 201"])
    summary = audit_summary(audit, language="tr")
    assert str(audit["total_min_su_credits"]) in summary
    assert str(audit["remaining_su_credits"]) in summary
    assert "henüz" in summary
    answer, rendered_summary = audit_answer(audit, language="tr")
    assert rendered_summary == summary and "MongoDB" not in answer


def test_student_answer_hides_implementation_narration():
    raw = (
        "Mezuniyet durumunuzu hesaplamak için MongoDB ders geçmişinizi kullanacağım.\n\n"
        "Sonuç: 14/125 SU.\n\nKaynaklar:\n[Source: Deterministic degree audit engine]"
    )
    cleaned = sanitize_student_answer(raw)
    assert "MongoDB" not in cleaned and "Source:" not in cleaned
    assert "14/125" in cleaned


# ---- course-history status ----------------------------------------------------------
def test_status_eligibility():
    assert _norm_status("COMPLETED") == "completed"
    import pytest
    with pytest.raises(ValueError):
        _norm_status("garbage")
    assert "failed" not in ELIGIBLE_FOR_CREDIT and "withdrawn" not in ELIGIBLE_FOR_CREDIT
    assert {"completed", "transfer", "exempted"} == ELIGIBLE_FOR_CREDIT


# ---- retrieval modes (CLAUDE.md Section 3) -------------------------------------------
def test_mode_normalization():
    from modules import retrieval_modes as rm
    assert rm.normalize_mode(None) == "hybrid_rerank"        # default when unset
    assert rm.normalize_mode("") == "hybrid_rerank"
    assert rm.normalize_mode("garbage") == "hybrid_rerank"   # unknown never crashes /ask
    assert rm.normalize_mode("DENSE") == "dense"
    assert rm.normalize_mode("hybrid-rerank") == "hybrid_rerank"
    assert rm.normalize_mode("llm") == "llm_only"
    # `hybrid_meta` joined the set in the 2026-07-28 retrieval benchmark and is now the
    # production default; the five original modes must all still be selectable so the
    # evaluation track can keep ablating components and BM25 stays available as a fallback.
    assert set(rm.RETRIEVAL_MODES) == {
        "llm_only", "bm25", "dense", "hybrid", "hybrid_rerank", "hybrid_meta",
    }
    assert rm.normalize_mode("hybrid_meta") == "hybrid_meta"


def test_llm_only_retrieves_nothing():
    """The baseline must touch no corpus at all - that is what makes it a baseline."""
    from modules import retrieval_modes as rm
    out = rm.retrieve(
        "llm_only", vectorstore=None, query="mezuniyetime ne kaldi",
        top_k=6, candidate_k=20, metadata_filter={"program": "CS"},
        hybrid_search=lambda: (_ for _ in ()).throw(AssertionError("llm_only must not retrieve")),
    )
    assert out.documents == [] and out.candidate_count == 0 and not out.used_retrieval


def test_legacy_evaluation_adapter_tracks_current_retrieval_contract():
    from modules import retrieval_modes as rm
    report = rm.run_retrieval_mode("llm_only", "question", None, top_k=6)
    assert report["mode"] == "llm_only"
    assert report["results"] == [] and report["context_documents"] == []
    assert report["timings_ms"] == {"retrieval_ms": 0.0, "rerank_ms": 0.0}


def test_hybrid_mode_skips_rerank_but_keeps_scope():
    """Ablating the reranker must change ranking only - never widen the corpus."""
    from langchain_core.documents import Document
    from modules import retrieval_modes as rm
    corpus = [Document(page_content=f"CS doc {i}", metadata={"program": "CS"}) for i in range(9)]
    out = rm.retrieve(
        "hybrid", vectorstore=None, query="area electives", top_k=4, candidate_k=20,
        metadata_filter={"program": "CS"}, hybrid_search=lambda: corpus,
    )
    assert not out.reranked, "hybrid must not rerank"
    assert len(out.documents) == 4, "top_k not honored"
    assert out.candidate_count == 9
    assert {d.metadata["program"] for d in out.documents} == {"CS"}


def test_bm25_gate_relaxes_only_for_standalone_mode():
    """Default hybrid behaviour stays narrowing-only; the bm25 baseline opts out explicitly."""
    import inspect
    from modules.bm25_retriever import annotate_bm25
    assert inspect.signature(annotate_bm25).parameters["require_narrowing"].default is True


# ---- optional integration (vectorstore) ---------------------------------------------
def integration_tests():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "ingest", str(Path(__file__).resolve().parents[1] / "scripts" / "ingest_degree_requirements.py"))
    ing = importlib.util.module_from_spec(spec); spec.loader.exec_module(ing)
    from modules.catalog_retriever import retrieve_documents
    vs = ing.get_vectorstore()
    assert vs._collection.count() > 20000, "corpus not ingested"

    pf = build_metadata_filter("graduation_status", {"major": "CS", "curriculum_term": "202401"})
    docs = retrieve_documents(vs, "area electives list", k=12, metadata_filter=pf)
    assert docs and {d.metadata.get("program") for d in docs} <= {"CS"}, "cross-program leak!"

    hit = retrieve_documents(vs, "is CS 455 an area elective", k=1, metadata_filter=pf)
    assert hit and hit[0].metadata.get("course_id") == "CS 455", "hybrid exact-code retrieval failed"

    mi = build_metadata_filter("minor", {})
    md = retrieve_documents(vs, "mathematics minor required courses", k=3, metadata_filter=mi)
    assert all(d.metadata.get("data_role") == "minor_requirement" for d in md)
    print("  integration: scope-isolation, hybrid exact-code, minor filter all OK")


# ---- deterministic academic-stage planner ---------------------------------------------------

_FRESHMAN_DONE = [
    "IF 100", "MATH 101", "NS 101", "HIST 191", "SPS 101", "TLL 101", "CIP 101N",
    "MATH 102", "NS 102", "AL 102", "HIST 192", "SPS 102", "TLL 102",
]


def test_planner_canonical_names_match_catalog():
    # Failure D: names come from the canonical catalog, never invented.
    assert course_planner.official_name("CS 308") == "Software Engineering"
    assert course_planner.official_name("CS 300") == "Data Structures"
    assert course_planner.official_name("CS 301") == "Algorithms"
    assert course_planner.official_name("SPS 303") == "Law and Ethics"
    assert course_planner.official_name("EE 417") == "Computer Vision"
    assert course_planner.official_name("CS 412") == "Machine Learning"


def test_planner_course_level():
    assert course_planner.course_level("CS 445") == 400
    assert course_planner.course_level("CS 201") == 200
    assert course_planner.course_level("IF 100") == 100


def test_planner_sophomore_gets_no_4xx_in_current_pool():
    # Failure C: sophomore NLP student -> 4XX interest courses are FUTURE TARGETS, not current.
    nlp = ["CS445", "CS455", "CS412", "EE417"]
    ctx = course_planner.build_context("CS", _FRESHMAN_DONE + ["CS 201"], nlp, term="202401")
    pool_start = ctx.index("CANDIDATE POOL")
    future_start = ctx.index("FUTURE TARGETS (")
    pool, future = ctx[pool_start:future_start], ctx[future_start:]
    for code in ("CS 445", "CS 455", "CS 412", "EE 417"):
        assert code not in pool, f"{code} must not be in the current candidate pool for a sophomore"
        assert code in future, f"{code} should appear under future targets"
    # 200-level required foundations ARE offered now
    assert "CS 204" in pool and "MATH 203" in pool


def test_planner_prioritizes_missing_university_courses():
    a = course_planner.analyze("CS", ["IF 100", "MATH 101", "NS 101"])
    assert a.stage == "freshman_foundation"
    missing = course_planner.missing_university_courses("CS", ["IF 100", "MATH 101", "NS 101"])
    codes = " ".join(missing)
    assert "MATH 102" in codes and "AL 102" in codes  # still owed
    assert "MATH 101" not in codes  # already done


def test_university_debt_still_obeys_prerequisites():
    plan = course_planner.build_plan("CS", "202401", [])
    codes = {item.code for item in plan.recommended}
    assert "MATH101" in codes and "NS101" in codes
    assert "MATH102" not in codes and "NS102" not in codes
    assert {code for code, _ in plan.blocked_required} >= {"MATH102", "NS102"}


def test_single_transferred_4xx_does_not_promote_sophomore():
    completed = _FRESHMAN_DONE + ["CS 201", "CS 412"]
    plan = course_planner.build_plan("CS", "202401", completed, ["CS 445"])
    assert plan.stage == "sophomore_foundation"
    assert all(course_planner.course_level(item.code) < 400 for item in plan.recommended)


def test_single_transferred_3xx_does_not_promote_sophomore():
    completed = _FRESHMAN_DONE + ["CS 201", "ENS 491"]
    plan = course_planner.build_plan("CS", "202401", completed, ["CS 445"])
    assert plan.stage == "sophomore_foundation"
    assert all(course_planner.course_level(item.code) < 400 for item in plan.recommended)


def test_authoritative_academic_year_enforces_sophomore_cap():
    completed = _FRESHMAN_DONE + ["CS 201", "CS 300", "CS 301", "CS 412", "CS 455"]
    plan = course_planner.build_plan(
        "CS", "202401", completed, ["CS 445"], academic_year=2
    )
    assert plan.stage == "sophomore_foundation"
    assert all(course_planner.course_level(item.code) < 400 for item in plan.recommended)


def test_generated_plan_validator_rejects_critical_violations():
    errors = course_planner.validate_proposed_plan(
        [
            {"code": "CS 201", "title": "Invented Name"},
            {"code": "CS 445", "title": course_planner.official_name("CS 445")},
            {"code": "CS 300", "title": course_planner.official_name("CS 300")},
            {"code": "CS 201", "title": course_planner.official_name("CS 201")},
        ],
        completed_codes=_FRESHMAN_DONE + ["CS 201"],
        stage="sophomore_foundation",
        maximum_su_credits=6,
    )
    assert "CS201:already_completed" in errors
    assert "CS201:noncanonical_title" in errors
    assert "CS445:level_exceeds_stage" in errors
    assert "CS300:unmet_prerequisite" in errors
    assert "CS201:duplicate" in errors
    assert "credit_limit_exceeded" in errors


def test_unknown_prerequisite_data_is_never_interpreted_as_no_prerequisite():
    errors = course_planner.validate_proposed_plan(
        [{"code": "CS 445", "title": course_planner.official_name("CS 445")}],
        completed_codes=_FRESHMAN_DONE + ["CS 201", "CS 204", "CS 300"],
        stage="senior_completion",
    )
    assert "CS445:prerequisite_data_unavailable" in errors


def test_alternative_prerequisite_path_is_supported():
    assert course_planner._prereqs_met("CS 306", {"DSA201"})
    assert course_planner._prereqs_met("CS 306", {"CS204"})
    assert not course_planner._prereqs_met("CS 306", {"IF100"})


def test_schedule_never_drops_a_required_secondary_component():
    lecture = schedule_planner.Section(
        course_id="CS 204", title="Advanced Programming", crn="1", section="A",
        component="Primary", instructors="", locations="",
        slots=[schedule_planner.Slot("M", 600, 660)],
    )
    conflicting_lab = schedule_planner.Section(
        course_id="CS 204", title="Advanced Programming", crn="2", section="A1",
        component="Laboratory", instructors="", locations="",
        slots=[schedule_planner.Slot("M", 630, 690)],
    )
    assert schedule_planner._candidate_blocks([lecture, conflicting_lab]) == []


def test_planner_prereq_blocks_required_3xx_for_sophomore():
    # A CS sophomore who only finished CS 201 must see CS 300 (needs CS 204) BLOCKED, and CS 204
    # (needs CS 201, satisfied) ELIGIBLE. Foundations are the program's REAL required courses.
    a = course_planner.analyze("CS", _FRESHMAN_DONE + ["CS 201"], term="202401")
    blocked = [code for code, _ in a.blocked_foundations]
    eligible = [pc.code for pc in a.eligible_foundations]
    assert "CS300" in blocked and "CS204" in eligible
    assert "CS300" not in eligible


def test_plan_never_recommends_non_required_elective_phys113():
    # Failure: PHYS 113 is an ELECTIVE for CS, not a requirement. The planner must never present
    # it (or any non-required course) as a next-semester foundation.
    plan = course_planner.build_plan("CS", "202401", _FRESHMAN_DONE + ["CS 201"], [])
    codes = {i.code for i in plan.recommended}
    assert "PHYS113" not in codes
    assert not any(course_planner.course_level(c) >= 400 for c in codes)  # no 4XX for a sophomore


def test_plan_dedupes_math_choice_and_names_linear_algebra():
    # MATH 201 (Linear Algebra) OR MATH 212 is a choice pool: recommend exactly one, and never
    # both; and the linear-algebra course must be named from the catalog, not invented.
    plan = course_planner.build_plan("CS", "202401", _FRESHMAN_DONE + ["CS 201"], [])
    codes = [i.code for i in plan.recommended]
    assert not ("MATH201" in codes and "MATH212" in codes)
    assert "MATH212" not in plan.deferred_required  # the alternative is not "also owed"
    assert course_planner.official_name("MATH 201") == "Linear Algebra"
    # If the student already did MATH 212, no linear-algebra course is recommended again.
    done = course_planner.build_plan("CS", "202401", _FRESHMAN_DONE + ["CS 201", "MATH 212"], [])
    assert all(c not in ("MATH201", "MATH212") for c in [i.code for i in done.recommended])


def test_plan_is_difficulty_balanced():
    # No more than two courses may share a subject prefix, so a term is never three MATH courses.
    plan = course_planner.build_plan("CS", "202401", _FRESHMAN_DONE + ["CS 201"], [])
    from collections import Counter
    subjects = Counter(course_planner.re.match(r"[A-Z]+", c.code).group(0) for c in plan.recommended)
    assert all(count <= 2 for count in subjects.values()), subjects


def test_plan_excludes_unreachable_required_and_offers_light_fillers():
    # The reported "sonraki dönem" bug: unreachable 3XX (CS 300/301/303) must NOT be recommended;
    # lighter University-category courses balance the term instead.
    plan = course_planner.build_plan("CS", "202401", _FRESHMAN_DONE + ["CS 201"], [])
    codes = {i.code for i in plan.recommended}
    for blocked in ("CS300", "CS301", "CS303", "CS395"):
        assert blocked not in codes
    assert "CS204" in codes                      # the reachable required course IS offered
    assert codes & {"PROJ201", "SPS303"}         # a light University filler rounds out the term


def test_normal_plan_reaches_fifteen_su_credit_floor():
    plan = course_planner.build_plan("CS", "202401", _FRESHMAN_DONE + ["CS 201"], [])
    assert sum(item.su for item in plan.recommended) >= 15
    assert plan.credit_shortfall == 0


def test_heavy_plan_reaches_eighteen_su_credit_floor():
    plan = course_planner.build_plan(
        "CS", "202401", _FRESHMAN_DONE + ["CS 201"], [],
        target=6, minimum_su_credits=18,
    )
    assert sum(item.su for item in plan.recommended) >= 18
    assert len(plan.recommended) >= 6


def test_explicit_four_course_limit_may_stay_below_credit_floor():
    plan = course_planner.build_plan(
        "CS", "202401", _FRESHMAN_DONE + ["CS 201"], [],
        target=4, minimum_su_credits=0, exact_course_count=True,
    )
    assert len(plan.recommended) == 4
    assert sum(item.su for item in plan.recommended) < 15


def test_heavy_weekly_timetable_surfaces_unavoidable_credit_shortfall():
    candidates = course_planner.build_plan(
        "CS", "202401", _FRESHMAN_DONE + ["CS 201"], [],
        target=8, minimum_su_credits=18,
    )
    timetable = schedule_planner.build_timetable_for_load(
        [item.code for item in candidates.recommended],
        target_courses=6,
        minimum_su_credits=18,
        term="202601",
    )
    placed_codes = [course_planner.normalize_code(lecture.course_id) for lecture, _ in timetable.placed]
    placed_su = sum(int((course_planner.resolve(code) or {}).get("su_credits") or 0) for code in placed_codes)
    assert len(timetable.placed) >= 6
    # The fail-closed prerequisite policy leaves no verified conflict-free 18-SU subset in this
    # frozen schedule. The service must report that fact instead of adding an unverified elective.
    assert placed_su == timetable.placed_su_credits
    assert timetable.credit_shortfall == 18 - placed_su
    assert timetable.credit_shortfall > 0
    assert timetable.unplaced or timetable.not_offered


# ---- major-selection mini-test --------------------------------------------------------------
def test_major_quiz_question_detection():
    assert major_advisor.is_major_question("Hangi bölümü seçmeliyim?")
    assert major_advisor.is_major_question("which major should I pick")
    assert not major_advisor.is_major_question("mezuniyet durumum ne")


def test_major_quiz_answer_detection():
    assert major_advisor.looks_like_answers("1a 2c 3a 4b 5a")
    assert major_advisor.looks_like_answers("a c b a b")
    assert not major_advisor.looks_like_answers("mezuniyet durumumu hesapla")


def test_major_quiz_scores_definitive_best():
    # All software/AI answers -> CS; the recommendation is definitive (one best-fit + a runner-up).
    rec = major_advisor.evaluate("1a 2a 3a 4a 5a", current_major="CS", language="tr")
    assert rec.best == "CS" and rec.second is not None
    assert "CS" in rec.summary
    # Mechanical/robotics answers point to ME, not CS.
    rec2 = major_advisor.evaluate("1c 2a 3d 4c 5c", current_major="CS", language="tr")
    assert rec2.best in {"ME", "EE"}


def test_major_quiz_keyword_fallback():
    rec = major_advisor.evaluate("psikolojiye ve insan davranışına çok ilgim var", language="tr")
    assert rec.best == "PSY" and rec.used_keywords


# ---- weekly timetable (option 5: real, conflict-free schedule) ------------------------------
_SOPHOMORE_PLAN = ["CS 204", "MATH 201", "MATH 203", "PROJ 201", "SPS 303"]


def test_timetable_is_conflict_free():
    tt = schedule_planner.build_timetable(_SOPHOMORE_PLAN)
    secs = tt.all_sections
    assert secs, "no sections chosen"
    for i in range(len(secs)):
        for j in range(i + 1, len(secs)):
            assert not secs[i].conflicts_with(secs[j]), (
                f"clash: {secs[i].course_id} {secs[i].crn} vs {secs[j].course_id} {secs[j].crn}"
            )
    assert len(tt.placed) == len(_SOPHOMORE_PLAN)   # all offered courses got a slot
    assert tt.crns and len(tt.crns) == len(set(tt.crns))  # CRNs present and unique
    # TBA meetings (PROJ 201) are kept, never dropped.
    assert any(s.course_id.replace(" ", "") == "PROJ201" for s in secs)


def test_timetable_reports_not_offered_course():
    tt = schedule_planner.build_timetable(["CS 204", "ZZZ 999"])
    assert "ZZZ999" in tt.not_offered
    assert all(s.course_id.replace(" ", "") != "ZZZ999" for s in tt.all_sections)


def test_timetable_structured_content_carries_crns_and_crn_column():
    tt = schedule_planner.build_timetable(_SOPHOMORE_PLAN)
    sc = schedule_planner.timetable_structured_content(tt, language="tr")
    assert sc["kind"] == "course_schedule"
    assert sc["crns"] == tt.crns and sc["crns"]
    table = sc["tables"][0]
    assert table["exportable"] is True
    assert any(col["key"] == "crn" for col in table["columns"])  # Excel export includes CRNs


def test_timetable_payload_is_calendar_ready_and_matches_example_crns():
    tt = schedule_planner.build_timetable(_SOPHOMORE_PLAN, term="202601")
    payload = schedule_planner.timetable_payload(
        tt,
        language="tr",
        generated_at="2026-07-28T20:00:00Z",
    )
    assert payload["schema_version"] == 1
    assert payload["term"] == "202601"
    assert payload["term_label"] == "Fall 2026-2027 (Güz)"
    assert payload["origin"] == "chatbot"
    assert payload["generated_at"] == "2026-07-28T20:00:00Z"
    assert payload["crns"] == [
        "10218", "10223", "10842", "10855", "10875", "10880", "10775", "11030", "11045",
    ]
    assert [course["course_id"] for course in payload["courses"]] == _SOPHOMORE_PLAN

    sections = [section for course in payload["courses"] for section in course["sections"]]
    cs_lecture = next(section for section in sections if section["crn"] == "10218")
    assert [(m["day_codes"], m["start_time"], m["end_time"]) for m in cs_lecture["meetings"]] == [
        (["M"], "16:40", "18:30"),
        (["W"], "12:40", "13:30"),
    ]
    assert cs_lecture["meetings"][0]["location"] == "Fac. of Engin. and Nat. Sci. G077"
    proj = next(section for section in sections if section["crn"] == "10775")
    assert proj["tba"] is True
    assert not any(m["start_time"] for m in proj["meetings"])
    assert payload["conflicts"] == []


def test_schedule_catalog_search_is_bounded_and_exposes_official_meetings():
    result = schedule_planner.search_sections_payload(term="202601", search="CS204", limit=2)
    assert result["term"] == "202601" and result["limit"] == 2
    assert result["total"] > 2 and result["has_more"] is True
    assert len(result["sections"]) == 2
    assert all(section["course_id"] == "CS 204" for section in result["sections"])
    assert all("meetings" in section for section in result["sections"])

    exact = schedule_planner.course_sections_payload("cs204", term="202601")
    by_crn = {section["crn"]: section for section in exact["sections"]}
    assert {"10218", "10223"} <= set(by_crn)
    assert by_crn["10218"]["component"] == "Primary"
    assert by_crn["10223"]["component"] == "Laboratory"
    assert by_crn["10223"]["meetings"][0]["day_codes"] == ["F"]


def test_manual_schedule_canonicalization_allows_blank_crn_and_detects_conflict():
    payload = {
        "schema_version": 1,
        "term": "202601",
        "term_label": "Fall 2026-2027 (Güz)",
        "source": "manual",
        "courses": [
            {
                "course_id": "CUSTOM 101",
                "title": "Özel çalışma",
                "sections": [{
                    "crn": "",
                    "section": "M1",
                    "component": "Manual",
                    "meetings": [{"day_codes": ["M"], "start_time": "10:00", "end_time": "11:00"}],
                }],
            },
            {
                "course_id": "CS 204",
                "title": "Advanced Programming",
                "sections": [{
                    "crn": "10218",
                    "section": "0",
                    "component": "Primary",
                    "meetings": [{"day_codes": ["M"], "start_time": "10:30", "end_time": "11:30"}],
                }],
            },
        ],
    }
    clean = schedule_planner.validate_schedule_payload(payload)
    assert clean["origin"] == "manual"
    assert clean["source"]["authority"] == "official"
    assert clean["crns"] == ["10218"]
    assert clean["courses"][0]["sections"][0]["tba"] is False
    assert clean["conflicts"] == [{
        "first_course_id": "CUSTOM 101", "first_crn": "",
        "second_course_id": "CS 204", "second_crn": "10218",
        "day_code": "M", "start_time": "10:30", "end_time": "11:00",
    }]


def test_timetable_real_times_are_24h_and_ordered():
    tt = schedule_planner.build_timetable(["MATH 203"])
    lecture = tt.placed[0][0]
    assert lecture.slots, "MATH 203 lecture should have real meeting times"
    for slot in lecture.slots:
        assert 0 <= slot.start < slot.end <= 24 * 60


# ---- content safety -----------------------------------------------------------------
def test_ordinary_academic_questions_are_never_blocked():
    safe = [
        "Mezuniyetime kaç kredi kaldı?",
        "Which courses count as area electives for CS?",
        "CS 307 dersini kim veriyor?",
        "Bu dönem hangi dersleri almalıyım?",
        "Bana NLP alanında ders öner",
    ]
    for question in safe:
        assert not content_safety.classify(question).blocked, question


def test_cs_security_vocabulary_is_not_treated_as_violence():
    # A CS school teaches CS 432 Computer and Network Security; these are ordinary questions.
    for question in [
        "How do I kill the process that holds port 8000?",
        "What is an attack vector in CS 432?",
        "Explain denial-of-service and buffer overflow exploits",
        "Bu ödevde process'i öldürmem gerekiyor, nasıl yaparım?",
    ]:
        assert not content_safety.classify(question).blocked, question


def test_each_unsafe_category_is_detected():
    cases = {
        "I want to kill myself": content_safety.SELF_HARM,
        "kendimi öldürmek istiyorum": content_safety.SELF_HARM,
        "seni öldüreceğim": content_safety.VIOLENCE,
        "I will kill you": content_safety.VIOLENCE,
        "fuck this stupid system": content_safety.PROFANITY,
        "amk ne biçim sistem": content_safety.PROFANITY,
    }
    for text, expected in cases.items():
        verdict = content_safety.classify(text)
        assert verdict.category == expected, f"{text!r} -> {verdict.category!r}"


def test_self_harm_outranks_violence_and_offers_help():
    verdict = content_safety.classify("I want to kill myself")
    assert verdict.category == content_safety.SELF_HARM
    for language in ("tr", "en"):
        body = content_safety.response_for(verdict.category, language)
        assert "112" in body, "self-harm response must point at real help"


def test_unsafe_categories_get_different_responses():
    seen = {
        content_safety.response_for(category, "tr")
        for category in (
            content_safety.SELF_HARM,
            content_safety.HATE,
            content_safety.VIOLENCE,
            content_safety.SEXUAL_HARASSMENT,
            content_safety.PROFANITY,
        )
    }
    assert len(seen) == 5, "every category must have its own response"


# ---- language detection --------------------------------------------------------------
def test_language_follows_the_question_not_the_alphabet():
    assert detect_language("Mezuniyetime kaç kredi kaldı?") == "tr"
    assert detect_language("How many credits do I need to graduate?") == "en"
    # Regression: a Turkish proper noun inside an English question used to force a Turkish answer.
    assert detect_language("Who teaches CS 412, is it Yücel Saygın?") == "en"
    assert detect_language("CS 300 dersini kim veriyor?") == "tr"


# ---- canonical intent vocabulary -----------------------------------------------------
def test_trained_labels_translate_to_english_intents():
    assert intents.to_canonical("mezuniyet_durumu") == intents.GRADUATION_STATUS
    assert intents.to_canonical("ders_onerisi") == intents.COURSE_RECOMMENDATION
    # Already-English input passes through unchanged, so the map is idempotent.
    assert intents.to_canonical(intents.GRADUATION_STATUS) == intents.GRADUATION_STATUS
    # The evaluation scripts must be able to get the recorded label back.
    assert intents.to_legacy(intents.GRADUATION_STATUS) == "mezuniyet_durumu"


def test_retrieval_policy_is_keyed_by_english_intents():
    assert build_metadata_filter(intents.GRADUATION_STATUS, {"major": "CS"}) is not None
    assert build_metadata_filter("mezuniyet_durumu", {"major": "CS"}) is None, (
        "legacy labels must not silently keep working, or the rename is only cosmetic"
    )


def _run() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS {t.__name__}")
        except Exception as exc:
            failed += 1
            print(f"  FAIL {t.__name__}: {exc}")
    if "--integration" in sys.argv or os.getenv("RUN_INTEGRATION") == "1":
        try:
            integration_tests()
        except Exception as exc:
            failed += 1
            print(f"  FAIL integration: {exc}")
    print(f"\n{len(tests)} unit tests, {failed} failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run())
