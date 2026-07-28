from __future__ import annotations

"""
Unit + regression tests for the retrieval lab and its production adapter.

These cover the invariants that silently destroy retrieval quality when they break:
course codes surviving tokenization, Turkish characters not being treated as separators,
rank fusion being rank-based rather than raw-score addition, metric definitions, metadata
scoping, and a small deterministic set of representative adviSU queries that must keep
finding their gold evidence.

Run:  cd server && ../.venv/bin/python -m pytest tests/test_retrieval_lab.py -q
"""

import json
import sys
from pathlib import Path

import pytest

SERVER_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SERVER_ROOT.parent
if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))

from retrieval_lab import metrics as M
from retrieval_lab.corpus import load_corpus
from retrieval_lab.fusion import (
    apply_metadata_boost,
    build_program_lexicon,
    extract_query_metadata,
    infer_record_types,
    minmax_fusion,
    reciprocal_rank_fusion,
)
from retrieval_lab.sparse import BM25FIndex
from retrieval_lab.text import (
    extract_course_codes,
    extract_term_codes,
    fold,
    tokenize_original,
    tokenize_v2,
)


# --- fixtures ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def corpus():
    return load_corpus()


@pytest.fixture(scope="module")
def bm25f(corpus):
    return BM25FIndex(corpus, tokenizer=tokenize_v2)


@pytest.fixture(scope="module")
def lexicon(corpus):
    return build_program_lexicon(corpus)


# --- tokenization -------------------------------------------------------------------------

class TestTokenization:
    def test_course_code_survives_split_form(self):
        assert "cs455" in tokenize_v2("CS 455 prerequisites")

    def test_course_code_survives_merged_form(self):
        toks = tokenize_v2("is CS455 required")
        assert "cs455" in toks and "cs" in toks and "455" in toks

    def test_merged_and_split_forms_share_a_token(self):
        """'CS 455' and 'CS455' must produce an overlapping token or they never match."""
        assert set(tokenize_v2("CS 455")) & set(tokenize_v2("CS455"))

    def test_turkish_characters_are_not_separators(self):
        """The shipped tokenizer shatters Turkish words; v2 must not."""
        assert tokenize_v2("müfredatında") == ["mufredatinda"]

    def test_original_tokenizer_shatters_turkish(self):
        """Regression guard on the BASELINE's known defect - if this changes, the baseline moved."""
        assert tokenize_original("müfredatında") == ["m", "fredat", "nda"]

    def test_turkish_dotted_capital_i(self):
        assert "istanbul" in tokenize_v2("İstanbul")

    def test_fold_is_idempotent(self):
        assert fold(fold("Öğrenci ÇĞIÖŞÜ")) == fold("Öğrenci ÇĞIÖŞÜ")

    def test_numbers_preserved(self):
        assert "202401" in tokenize_v2("the 202401 curriculum")


class TestEntityExtraction:
    def test_extract_course_code_variants(self):
        assert extract_course_codes("Does CS414 count?") == ["CS 414"]
        assert extract_course_codes("Does CS 414 count?") == ["CS 414"]

    def test_extract_multiple_course_codes(self):
        assert extract_course_codes("CS 201 before CS 204?") == ["CS 201", "CS 204"]

    def test_extract_explicit_term(self):
        assert extract_term_codes("the 202401 curriculum") == ["202401"]

    def test_bare_year_widens_to_fall_term(self):
        assert extract_term_codes("the 2024 curriculum") == ["202401"]

    def test_no_false_term_from_course_number(self):
        assert extract_term_codes("CS 455") == []

    def test_program_extracted_from_full_name(self, lexicon):
        meta = extract_query_metadata(
            "I'm a Computer Science and Engineering Undergraduate Program student", lexicon
        )
        assert "CS" in meta.programs

    def test_minor_detected_in_both_languages(self, lexicon):
        assert extract_query_metadata("the Mathematics minor", lexicon).is_minor
        assert extract_query_metadata("matematik yandal", lexicon).is_minor


class TestRecordTypeInference:
    def test_aggregate_question_wants_category_rows(self):
        types = infer_record_types("How many SU credits of free electives are required?", False)
        assert "degree_requirement_category_pool" in types

    def test_named_course_overrides_aggregate_phrasing(self):
        """'how many credits is CS 414' is a course-level question despite 'how many credits'."""
        types = infer_record_types("How many credits is CS 414?", True)
        assert "degree_requirement_pool_course" in types
        assert "degree_requirement_category_pool" not in types

    def test_no_signal_returns_empty(self):
        assert infer_record_types("hello there", False) == []


# --- fusion ------------------------------------------------------------------------------

class TestFusion:
    def test_rrf_is_scale_free(self):
        """Raw scores on wildly different scales must not change an RRF result."""
        a = [("x", 1000.0), ("y", 999.0)]
        b = [("y", 0.02), ("x", 0.01)]
        scaled_a = [("x", 1_000_000.0), ("y", 999_000.0)]
        assert reciprocal_rank_fusion([a, b]) == reciprocal_rank_fusion([scaled_a, b])

    def test_rrf_rewards_agreement(self):
        a = [("agreed", 1.0), ("only_a", 0.9)]
        b = [("agreed", 1.0), ("only_b", 0.9)]
        assert reciprocal_rank_fusion([a, b])[0][0] == "agreed"

    def test_minmax_normalizes_before_summing(self):
        a = [("x", 100.0), ("y", 0.0)]
        b = [("y", 1.0), ("x", 0.0)]
        fused = dict(minmax_fusion([a, b]))
        assert fused["x"] == pytest.approx(fused["y"])

    def test_rrf_handles_empty_ranking(self):
        assert reciprocal_rank_fusion([[], [("a", 1.0)]])[0][0] == "a"

    def test_metadata_boost_never_drops_candidates(self, corpus, lexicon):
        ranking = [(c.chunk_id, 1.0 / (i + 1)) for i, c in enumerate(corpus.chunks[:50])]
        meta = extract_query_metadata("CS 414 in the 202401 CS curriculum", lexicon)
        boosted = apply_metadata_boost(ranking, corpus, meta, top=50)
        assert set(cid for cid, _ in boosted) == set(cid for cid, _ in ranking)


# --- metrics -------------------------------------------------------------------------------

class TestMetrics:
    def test_recall_any_gold(self):
        assert M.recall_at_k(["a", "b", "c"], {"c"}, 3) == 1.0
        assert M.recall_at_k(["a", "b", "c"], {"c"}, 2) == 0.0

    def test_evidence_set_recall_is_a_fraction(self):
        assert M.evidence_set_recall_at_k(["a", "b"], {"a", "b", "c"}, 10) == pytest.approx(2 / 3)

    def test_all_gold_requires_everything(self):
        assert M.all_gold_at_k(["a", "b"], {"a", "b"}, 10) == 1.0
        assert M.all_gold_at_k(["a"], {"a", "b"}, 10) == 0.0

    def test_reciprocal_rank(self):
        assert M.reciprocal_rank(["x", "y", "gold"], {"gold"}) == pytest.approx(1 / 3)

    def test_ndcg_perfect_ranking_is_one(self):
        assert M.ndcg_at_k(["a", "b", "c"], {"a", "b"}, 10) == pytest.approx(1.0)

    def test_ndcg_penalizes_lower_rank(self):
        assert M.ndcg_at_k(["x", "a"], {"a"}, 10) < M.ndcg_at_k(["a", "x"], {"a"}, 10)

    def test_hit_at_1(self):
        assert M.hit_at_1(["a"], {"a"}) == 1.0
        assert M.hit_at_1(["b", "a"], {"a"}) == 0.0

    def test_empty_gold_scores_zero_not_crash(self):
        assert M.evidence_set_recall_at_k(["a"], set(), 10) == 0.0
        assert M.ndcg_at_k(["a"], set(), 10) == 0.0

    def test_paired_bootstrap_detects_clear_improvement(self):
        base = {f"q{i}": 0.0 for i in range(100)}
        cand = {f"q{i}": 1.0 for i in range(100)}
        cmp = M.paired_bootstrap(base, cand, metric="recall@10",
                                 baseline_name="b", candidate_name="c", n_boot=500)
        assert cmp.delta == pytest.approx(1.0)
        assert cmp.significant and cmp.improved == 100 and cmp.harmed == 0

    def test_paired_bootstrap_finds_no_effect_when_identical(self):
        same = {f"q{i}": float(i % 2) for i in range(100)}
        cmp = M.paired_bootstrap(same, dict(same), metric="recall@10",
                                 baseline_name="b", candidate_name="c", n_boot=500)
        assert cmp.delta == 0.0 and not cmp.significant

    def test_paired_bootstrap_requires_overlap(self):
        with pytest.raises(ValueError):
            M.paired_bootstrap({"a": 1.0}, {"b": 1.0}, metric="m",
                               baseline_name="b", candidate_name="c", n_boot=10)


# --- corpus / index ---------------------------------------------------------------------------

class TestCorpus:
    def test_corpus_loads(self, corpus):
        assert len(corpus) > 10_000

    def test_chunk_ids_unique(self, corpus):
        assert len(corpus.by_id) == len(corpus.chunks)

    def test_fingerprint_is_stable(self, corpus):
        assert corpus.fingerprint == load_corpus().fingerprint

    def test_empty_query_does_not_crash(self, bm25f):
        assert bm25f.search(tokenize_v2(""), top=10) == []

    def test_nonsense_query_returns_no_or_few_hits(self, bm25f):
        assert len(bm25f.search(tokenize_v2("zzzqqq xylophone"), top=10)) == 0


# --- production adapter -------------------------------------------------------------------------

class TestLabRetrieverAdapter:
    def test_where_filter_shapes(self):
        from modules.lab_retriever import _matches

        meta = {"program": "CS", "curriculum_term": "202401", "data_role": "curriculum_requirement"}
        assert _matches(meta, {"program": "CS"})
        assert not _matches(meta, {"program": "IE"})
        assert _matches(meta, {"$and": [{"program": "CS"}, {"curriculum_term": "202401"}]})
        assert not _matches(meta, {"$and": [{"program": "CS"}, {"curriculum_term": "202301"}]})
        assert _matches(meta, {"data_role": {"$in": ["curriculum_requirement", "minor_requirement"]}})
        assert not _matches(meta, {"data_role": {"$in": ["minor_requirement"]}})
        assert _matches(meta, None)

    def test_search_returns_documents_with_citation_metadata(self):
        from modules.lab_retriever import lab_search

        docs = lab_search("How many SU credits does CS 414 carry in CS 202401?", top_k=5)
        assert docs, "expected results for a well-formed curriculum question"
        for d in docs:
            assert d.page_content
            assert d.metadata.get("chunk_id")
            assert d.metadata.get("source")

    def test_metadata_filter_scopes_results(self):
        from modules.lab_retriever import lab_search

        docs = lab_search("free electives requirement", top_k=10,
                          metadata_filter={"program": "IE"})
        assert docs and all(d.metadata.get("program") == "IE" for d in docs)

    def test_empty_query_returns_empty(self):
        from modules.lab_retriever import lab_search

        assert lab_search("", top_k=5) == []


# --- deterministic retrieval regression suite ------------------------------------------------------

REGRESSION_SPLIT = PROJECT_ROOT / "data" / "benchmark" / "retrieval_test.jsonl"


def _regression_items(n: int = 25) -> list[dict]:
    if not REGRESSION_SPLIT.exists():
        pytest.skip("benchmark not built; run build_retrieval_benchmark.py")
    rows = [json.loads(l) for l in REGRESSION_SPLIT.read_text(encoding="utf-8").splitlines() if l.strip()]
    scorable = [r for r in rows if r.get("expected_chunk_ids")]
    # deterministic: first N by id, no sampling
    return sorted(scorable, key=lambda r: r["id"])[:n]


class TestRetrievalRegression:
    """Representative adviSU queries whose gold evidence must stay within reach."""

    def test_gold_found_within_top_10_for_most_queries(self):
        from modules.lab_retriever import lab_search

        items = _regression_items()
        hits = 0
        for item in items:
            docs = lab_search(item["question"], top_k=10)
            got = {d.metadata.get("chunk_id") for d in docs}
            if got & set(item["expected_chunk_ids"]):
                hits += 1
        rate = hits / len(items)
        assert rate >= 0.80, f"Recall@10 on the regression set dropped to {rate:.2%}"

    def test_catalog_year_evidence_is_not_outranked(self):
        """A query naming a catalog term must not rank another term's row first."""
        from modules.lab_retriever import lab_search

        items = [i for i in _regression_items(60) if i.get("curriculum_term") and i.get("program")]
        checked = failures = 0
        for item in items[:20]:
            docs = lab_search(item["question"], top_k=1)
            if not docs:
                continue
            checked += 1
            top = docs[0].metadata
            if top.get("curriculum_term") and top.get("curriculum_term") != item["curriculum_term"]:
                failures += 1
        assert checked, "no scoped items to check"
        assert failures / checked <= 0.20, f"{failures}/{checked} queries ranked the wrong catalog year first"

    def test_course_code_preserved_end_to_end(self):
        from modules.lab_retriever import lab_search

        docs = lab_search("Which requirement category does CS 414 belong to in CS 202401?", top_k=5)
        assert any(d.metadata.get("course_id") == "CS 414" for d in docs)
