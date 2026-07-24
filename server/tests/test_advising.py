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

from modules import curriculum_registry, degree_audit
from modules.retrieval_policy import build_metadata_filter, check_profile
from modules.bm25_retriever import _tokenize, _is_narrowing
from modules.conversation_memory import _automatic_title, extract_course_code, resolve_reference
from modules.course_commands import parse_course_history_command
from modules.mongodb import _norm_status, ELIGIBLE_FOR_CREDIT
from modules.response_formatter import ensure_summary_section


# ---- retrieval policy / profile scoping ---------------------------------------------
def test_profile_filter_scopes_program_and_term():
    f = build_metadata_filter("mezuniyet_durumu", {"major": "CS", "curriculum_term": "202401"})
    clauses = f["$and"]
    assert {"data_role": "curriculum_requirement"} in clauses
    assert {"program": "CS"} in clauses
    assert {"curriculum_term": "202401"} in clauses


def test_minor_filter_uses_minor_role():
    assert build_metadata_filter("minor", {}) == {"data_role": "minor_requirement"}


def test_missing_profile_fields_blocks_audit():
    g = check_profile("mezuniyet_durumu", {})
    assert not g.ok and g.missing_fields


def test_missing_curriculum_is_unavailable_not_fallback():
    g = check_profile("mezuniyet_durumu", {"major": "IE", "curriculum_term": "202101"})
    assert not g.ok and g.data_unavailable  # IE 202101 not in corpus -> safe limitation


def test_valid_cs_curriculum_passes_gate():
    assert check_profile("mezuniyet_durumu", {"major": "CS", "curriculum_term": "202401"}).ok


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
    q = resolve_reference("yani ne almam lazım?", {"last_intent": "mezuniyet_durumu"})
    assert "mezuniyet_durumu" in q


def test_automatic_title_uses_multiple_turns():
    title = _automatic_title(
        [
            {"user": "Merhaba, mezuniyet durumumu hesaplar mısın?", "assistant": "..."},
            {"user": "Yani ne almam lazım?", "assistant": "..."},
        ]
    )
    assert "mezuniyet" in title.lower()
    assert len(title) <= 70


def test_course_history_command_parser():
    command = parse_course_history_command("CS 201 ve MATH101'i aldım")
    assert command is not None
    assert command.status == "completed"
    assert command.course_codes == ("CS 201", "MATH 101")
    assert parse_course_history_command("CS 201 hakkında bilgi ver") is None


def test_summary_is_always_available_for_long_answers():
    answer, summary = ensure_summary_section(
        "İlk önemli sonuç budur. İkinci önemli sonuç budur.", language="tr"
    )
    assert summary
    assert "Kısa Özet" in answer


# ---- course-history status ----------------------------------------------------------
def test_status_eligibility():
    assert _norm_status("COMPLETED") == "completed"
    assert _norm_status("garbage") == "completed"  # unknown -> default completed
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
    assert set(rm.RETRIEVAL_MODES) == {"llm_only", "bm25", "dense", "hybrid", "hybrid_rerank"}


def test_llm_only_retrieves_nothing():
    """The baseline must touch no corpus at all - that is what makes it a baseline."""
    from modules import retrieval_modes as rm
    out = rm.retrieve(
        "llm_only", vectorstore=None, query="mezuniyetime ne kaldi",
        top_k=6, candidate_k=20, metadata_filter={"program": "CS"},
        hybrid_search=lambda: (_ for _ in ()).throw(AssertionError("llm_only must not retrieve")),
    )
    assert out.documents == [] and out.candidate_count == 0 and not out.used_retrieval


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

    pf = build_metadata_filter("mezuniyet_durumu", {"major": "CS", "curriculum_term": "202401"})
    docs = retrieve_documents(vs, "area electives list", k=12, metadata_filter=pf)
    assert docs and {d.metadata.get("program") for d in docs} <= {"CS"}, "cross-program leak!"

    hit = retrieve_documents(vs, "is CS 455 an area elective", k=1, metadata_filter=pf)
    assert hit and hit[0].metadata.get("course_id") == "CS 455", "hybrid exact-code retrieval failed"

    mi = build_metadata_filter("minor", {})
    md = retrieve_documents(vs, "mathematics minor required courses", k=3, metadata_filter=mi)
    assert all(d.metadata.get("data_role") == "minor_requirement" for d in md)
    print("  integration: scope-isolation, hybrid exact-code, minor filter all OK")


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
