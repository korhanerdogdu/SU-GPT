from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.claim_coverage import claim_coverage_supported


def test_grounded_answer_over_matching_evidence_is_supported():
    evidence = ["CS 306 Database Systems is a 3 SU core elective course with prerequisite CS 300."]
    answer = "CS 306 Database Systems is a 3 SU core elective, and its prerequisite is CS 300."
    assert claim_coverage_supported(answer, evidence)


def test_fabricated_claim_riding_on_real_citation_is_not_supported():
    # The citation label ("Official catalog") is real and authorized; the *content* of the
    # sentence about it is not backed by anything the evidence text actually says.
    evidence = ["CS 201 is an official catalog course."]
    answer = "CS 201 guarantees every student a one-million-dollar salary."
    assert not claim_coverage_supported(answer, evidence)


def test_turkish_grounded_answer_is_supported():
    evidence = ["CS 306 Veritabanı Sistemleri dersi 3 SU kredisindedir ve CS 300 ön koşuludur."]
    answer = "CS 306 Veritabanı Sistemleri dersi 3 SU kredisindedir, ön koşulu CS 300'dür."
    assert claim_coverage_supported(answer, evidence)


def test_no_evidence_text_is_never_supported():
    assert not claim_coverage_supported("CS 306 is a core elective.", [])
    assert not claim_coverage_supported("CS 306 is a core elective.", [""])


def test_terse_answer_with_no_checkable_claim_sentences_is_supported():
    # A bare course code plus citation has nothing this lexical check can confirm or deny --
    # citations_authorized already covers whether the label itself is real.
    evidence = ["CS 306 Database Systems, 3 SU."]
    assert claim_coverage_supported("CS 306.", evidence)


def test_honest_hedge_about_missing_admit_term_is_not_penalized():
    # Regression: a model correctly saying "I can't determine this without your admit term" was
    # being scored as an unsupported claim because that sentence naturally has low lexical
    # overlap with evidence about a *different* fact -- punishing exactly the honesty this
    # pipeline wants (CLAUDE.md: missing critical data -> explicit abstention, never a guess).
    evidence = [
        "For Computer Science and Engineering Undergraduate Program (BSCS), CS 303 - Logic and "
        "Digital System Design is a required course. A minimum of 29 SU credits of required "
        "courses is required for Fall 2022-2023 (202201)."
    ]
    answer = (
        "According to the degree requirements for the Computer Science and Engineering "
        "Undergraduate Program (BSCS), CS 303 is a required course. "
        "For Fall 2022-2023 (202201), a minimum of 29 SU credits of required courses is required. "
        "Since your admit term is not specified, we cannot determine the exact requirement for "
        "your case. Please provide your admit term for a more accurate answer."
    )
    assert claim_coverage_supported(answer, evidence)


def test_turkish_no_information_hedge_is_not_penalized():
    evidence = [
        "For Computer Science and Engineering Undergraduate Program (BSCS), CS 210 belongs to "
        "the faculty courses fens pool."
    ]
    answer = (
        "CS 210 dersinin ön koşulu hakkında bilgi bulunmamaktadır. "
        "Sadece CS 210 dersinin Computer Science and Engineering Undergraduate Program (BSCS) "
        "için faculty courses fens poolunda yer aldığı belirtilmiştir."
    )
    assert claim_coverage_supported(answer, evidence)


def test_KNOWN_LIMITATION_a_wrong_number_surrounded_by_correct_vocabulary_is_not_caught():
    # Documents a real gap, not a bug to fix here: this is lexical-overlap grounding, not
    # semantic entailment (see the module docstring). A single wrong digit inside an otherwise
    # evidence-matching sentence only costs that one token out of ~8-9 significant tokens in the
    # sentence, comfortably clearing _MIN_OVERLAP. Catching this needs an actual entailment
    # check (e.g. comparing extracted numeric claims against the evidence's own numbers
    # per-field, or an NLI model) -- out of scope for a deterministic lexical checker.
    evidence = [
        "For Computer Science and Engineering Undergraduate Program (BSCS), CS 303 is a "
        "required course. A minimum of 29 SU credits of required courses is required."
    ]
    wrong_number_answer = (
        "CS 303 is a required course for the Computer Science and Engineering Undergraduate "
        "Program (BSCS). A minimum of 92 SU credits of required courses is required."
    )
    # Asserting True here documents the limitation (this SHOULD be False in an ideal system,
    # i.e. 92 SU is fabricated) -- if this ever starts failing, claim_coverage_supported has
    # gained real numeric-entailment ability and this test (and its comment) should be updated,
    # not treated as a regression.
    assert claim_coverage_supported(wrong_number_answer, evidence)


def test_one_fabricated_sentence_among_grounded_ones_fails_the_whole_answer():
    evidence = ["CS 306 Database Systems is a 3 SU core elective course."]
    answer = (
        "CS 306 Database Systems is a 3 SU core elective course. "
        "It also comes with a free trip to the moon for every enrolled student."
    )
    assert not claim_coverage_supported(answer, evidence)
