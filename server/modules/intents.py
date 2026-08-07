"""Canonical intent vocabulary for adviSU — English, and the single translation boundary.

Every intent identifier used by application code is English. The strings below are the ones
`main.py`, `rag_router`, `retrieval_policy`, `llm` and the tests compare against.

WHY A TRANSLATION MAP EXISTS
The shipped intent classifier is a committed artifact: `server/models/intent_bert_classifier.joblib`
was trained with Turkish labels baked into its `classes_`, and it was selected over TF-IDF by a
recorded benchmark (`data/benchmark/intent_model_selection.json`, macro-F1 0.737 vs 0.348). The
retrieval and answer benchmarks under `data/benchmark/*.jsonl` also carry those labels on ~1,200
committed rows. Renaming the labels inside those artifacts would either silently invalidate every
measured number or force a retrain that this change has no reason to run.

So the rename happens at the boundary instead: the classifier keeps speaking its historical
vocabulary, `intent_detector` translates once on the way out, and everything downstream is English.
`to_legacy` exists for the evaluation scripts, which must keep reading and writing the recorded
labels to stay reproducible.
"""

from __future__ import annotations

# --- Intents the classifier can produce -------------------------------------------------
GRADUATION_STATUS = "graduation_status"
COURSE_RECOMMENDATION = "course_recommendation"
STUDY_PLAN = "study_plan"
MAJOR_SELECTION = "major_selection"
SPECIALIZATION = "specialization"
COURSE_DETAIL = "course_detail"
OTHER = "other"

# --- Intents added by the deterministic regex guards in main.py / rag_router -------------
MINOR = "minor"
REVIEW = "review"
EXAM = "exam"
GRADUATION_PLAN = "graduation_plan"
WEEKLY_SCHEDULE = "weekly_schedule"
UNIVERSITY_COURSES = "university_courses"
GPA_PROJECTION = "gpa_projection"

# --- Intents that describe a handled request rather than a classified question ----------
COURSE_HISTORY_UPDATE = "course_history_update"
ACADEMIC_PROFILE_UPDATE = "academic_profile_update"
LLM_ONLY = "llm_only"
SAFETY_BLOCKED = "safety_blocked"
RATE_LIMITED = "rate_limited"
ERROR = "error"

#: Labels the trained models emit -> the English names the application uses.
LEGACY_TO_CANONICAL: dict[str, str] = {
    "mezuniyet_durumu": GRADUATION_STATUS,
    "ders_onerisi": COURSE_RECOMMENDATION,
    "calisma_plani": STUDY_PLAN,
    "major_secimi": MAJOR_SELECTION,
    "alanda_ozellesme": SPECIALIZATION,
    "ders_ayrintisi": COURSE_DETAIL,
    "diger": OTHER,
    "ders_programi": WEEKLY_SCHEDULE,
    "universite_dersleri": UNIVERSITY_COURSES,
}

CANONICAL_TO_LEGACY: dict[str, str] = {
    canonical: legacy for legacy, canonical in LEGACY_TO_CANONICAL.items()
}

#: Intents that are answered from the curriculum corpus (as opposed to review/exam chunks).
CURRICULUM_INTENTS: frozenset[str] = frozenset(
    {
        GRADUATION_STATUS,
        COURSE_RECOMMENDATION,
        COURSE_DETAIL,
        STUDY_PLAN,
        MAJOR_SELECTION,
        SPECIALIZATION,
    }
)


def to_canonical(intent: str | None) -> str:
    """Translate a trained-model label to its English name. English input passes through."""
    if not intent:
        return OTHER
    value = str(intent).strip()
    return LEGACY_TO_CANONICAL.get(value, value)


def to_legacy(intent: str | None) -> str:
    """Translate back, for evaluation code that must keep writing the recorded labels."""
    if not intent:
        return "diger"
    value = str(intent).strip()
    return CANONICAL_TO_LEGACY.get(value, value)
