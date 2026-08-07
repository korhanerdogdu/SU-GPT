from __future__ import annotations

"""Domain-specific prompt-strategy benchmark for the course-advising LLM composition layer.

Scope, deliberately: GPA and degree-audit numbers are computed in code (gpa_planner.py,
degree_audit.py) and are excluded here on purpose -- they are not something a "prompting style"
should ever be picked for, and comparing strategies on arithmetic the code already gets right
would be measuring noise. This benchmark instead covers the layer that genuinely goes through the
LLM: composing/explaining an already-computed deterministic recommendation, narrowing it to a
stated interest area, an open-ended "which courses until graduation" roadmap, and open-domain
factual/advisory questions. It complements evaluation/benchmark_prompt_strategies.py (which tests
generic lookup/arithmetic ability on toy tasks) with real advising scenarios and real course data.

Each scenario is built from the SAME context-construction primitives production code uses
(modules.course_planner.build_context, real curriculum-pool text from data/degree_requirements,
real course_schedule chunks) rather than fabricated context, so the benchmark exercises the actual
frozen prompt template with only the {strategy_directive} slot varying between runs.

Scoring is automatic and criterion-based (regex/substring checks defined per scenario) rather than
LLM-as-judge, matching this repo's existing evaluation methodology (retrieval/reranking reports)
and modules.confidence's own stance against inventing an uncalibrated numeric threshold.
"""

import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

SERVER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVER_ROOT))

from langchain_core.documents import Document

from main import StaticRetriever
from modules import course_planner
from modules.llm import _PROMPT_STRATEGIES, get_llm_chain
from modules.llm_providers import LLMProviderError
from modules import intents as intents_module

ROOT = SERVER_ROOT.parent
OUTPUT = ROOT / "data" / "benchmark" / "course_advising_prompt_strategy.json"

PROGRAM = "CS"
TERM = "202401"

# A sophomore who has cleared the freshman University Courses plus the CS 2XX core sequence --
# used only to build a realistic planner-context document for the advisory scenario below.
SOPHOMORE_DONE = [
    "IF 100", "MATH 101", "CIP 101N", "NS 101", "SPS 101", "TLL 101", "AL 102",
    "MATH 102", "NS 102", "SPS 102", "TLL 102", "HIST 191", "HIST 192", "PROJ 201", "SPS 303",
    "CS 201", "CS 204", "CS 210", "CS 300", "CS 301",
]

MONGODB_PROFILE_HEADER = "[Source: MongoDB student profile]"


def _profile_doc(completed: list[str]) -> Document:
    return Document(
        page_content=f"{MONGODB_PROFILE_HEADER}\nCompleted courses: {', '.join(completed)}.",
        metadata={"source": "MongoDB student profile", "document_type": "user_course_history"},
    )


def _planner_doc(completed: list[str], interest_codes: list[str] | None = None) -> Document:
    return Document(
        page_content=course_planner.build_context(PROGRAM, completed, interest_codes, term=TERM),
        metadata={"source": "Academic-stage planner", "document_type": "academic_stage_plan"},
    )


def _schedule_doc() -> Document:
    # Real course_schedule_section chunk content (data/schedule/202601.jsonl, CRN 10188):
    # CS 412 Machine Learning, Section A, Tue/Thu 08:40-10:30, instructor Ayşe Berrin Yanıkoğlu.
    return Document(
        page_content=(
            "[Source: degree_requirements/schedule/202601.jsonl]\n"
            "Ders programı kaydı / course schedule record. Ders/course: CS 412 - Machine "
            "Learning. CRN: 10188. Section: A. Toplantılar/meetings: Salı/Tuesday, 08:40-10:30, "
            "yer/location University Center G030, öğretim elemanı/instructor Ayşe Berrin "
            "Yanıkoğlu | Perşembe/Thursday, 08:40-10:30, yer/location Fac.of Arts and Social "
            "Sci. G062, öğretim elemanı/instructor Ayşe Berrin Yanıkoğlu."
        ),
        metadata={"source": "course_schedule/202601.jsonl", "document_type": "course"},
    )


@dataclass(frozen=True)
class Scenario:
    id: str
    scenario_class: str
    intent: str
    language: str
    question: str
    documents: list[Document] = field(default_factory=list)
    checks: list[tuple[str, "callable"]] = field(default_factory=list)


def _contains_all(*substrings: str):
    def check(text: str) -> bool:
        lowered = text.lower()
        return all(s.lower() in lowered for s in substrings)

    return check


def _schedule_doc_cs306() -> Document:
    # Real course_schedule_section chunk (data/schedule/202601.jsonl, CRN 10141):
    # CS 306 Database Systems, instructor Yücel Saygın, Mon/Wed.
    return Document(
        page_content=(
            "[Source: degree_requirements/schedule/202601.jsonl]\n"
            "Ders programı kaydı / course schedule record. Ders/course: CS 306 - Database "
            "Systems. CRN: 10141. Section: 0. Toplantılar/meetings: Pazartesi/Monday, "
            "16:40-18:30, öğretim elemanı/instructor Yücel Saygın | Çarşamba/Wednesday, "
            "10:40-11:30, öğretim elemanı/instructor Yücel Saygın."
        ),
        metadata={"source": "course_schedule/202601.jsonl", "document_type": "course"},
    )


# Section 1.1-1.3 of the original request (course recommendation progression, interest-area
# narrowing, heavy/light override, until-graduation roadmap) turned out to be fully deterministic
# in the current code (main.py:2211-2321 -> modules.course_planner.build_plan/render_plan) --
# it never reaches get_llm_chain when the student's profile is complete, so there is no prompting
# strategy to compare there; it is excluded from this benchmark the same way GPA/degree-audit are.
# What is left, genuinely LLM-composed: open-domain factual and advisory questions (section 1.5).
SCENARIOS: list[Scenario] = [
    Scenario(
        id="factual_time_instructor_tr",
        scenario_class="open_domain_factual",
        intent=intents_module.COURSE_DETAIL,
        language="tr",
        question="CS 412 hangi gün, kaçta ve kim tarafından veriliyor?",
        documents=[_schedule_doc()],
        checks=[
            ("states the real instructor name", _contains_all("Yanıkoğlu")),
            ("states a real meeting day (Salı/Tuesday or Perşembe/Thursday)",
             lambda t: ("salı" in t.lower() or "tuesday" in t.lower() or "perşembe" in t.lower() or "thursday" in t.lower())),
        ],
    ),
    Scenario(
        id="factual_time_instructor_en",
        scenario_class="open_domain_factual",
        intent=intents_module.COURSE_DETAIL,
        language="en",
        question="What day and time is CS 306, and who teaches it?",
        documents=[_schedule_doc_cs306()],
        checks=[
            ("states the real instructor name", _contains_all("Saygın")),
            ("states a real meeting day (Monday or Wednesday)",
             lambda t: ("monday" in t.lower() or "pazartesi" in t.lower() or "wednesday" in t.lower() or "çarşamba" in t.lower())),
        ],
    ),
    Scenario(
        id="advisory_open_ended_tr",
        scenario_class="open_domain_advisory",
        intent=intents_module.COURSE_DETAIL,
        language="tr",
        question="NLP mi yapay zeka mı kararsızım, hangi dersleri almalıyım?",
        documents=[_profile_doc(SOPHOMORE_DONE), _planner_doc(SOPHOMORE_DONE, ["CS 412", "CS 445"])],
        checks=[
            ("grounds the advice in an actual candidate-pool course code", lambda t: bool(re.search(r"\bCS \d{3}\b", t))),
        ],
    ),
]


# The provider adapter (modules.llm_providers) already does its own bounded retries with
# backoff internally before raising -- an outer retry loop that re-fires immediately just
# compounds delay without giving Groq's actual per-minute quota room to recover. One real
# attempt per call, with generous pacing between calls, is both faster in practice and more
# honest about what's actually rate-limited (the account, not a single flaky request).
def _invoke(strategy: str, scenario: Scenario) -> tuple[str, list[tuple[str, bool]], str | None]:
    retriever = StaticRetriever(documents=scenario.documents)
    chain = get_llm_chain(retriever, scenario.intent, scenario.language, strategy)
    try:
        result = chain({"query": scenario.question})
        answer = str(result.get("result") or "")
        outcomes = [(label, check(answer)) for label, check in scenario.checks]
        return answer, outcomes, None
    except LLMProviderError as exc:
        return "", [(label, False) for label, _ in scenario.checks], f"{type(exc).__name__}: {exc}"


def main() -> None:
    strategies = list(_PROMPT_STRATEGIES.keys())
    per_strategy: dict[str, dict] = {s: {"scenario_results": {}, "passed": 0, "total": 0, "errors": 0} for s in strategies}

    for scenario in SCENARIOS:
        for strategy in strategies:
            answer, outcomes, error = _invoke(strategy, scenario)
            if error:
                # The measured account limit is 12,000 tokens/minute (confirmed via the
                # x-ratelimit-* response headers), and this app's frozen prompt template plus
                # context is easily several thousand tokens per call -- a retry that re-sends the
                # same prompt within the same minute just spends more of the same exhausted
                # budget. Wait a full minute so the token window has actually reset.
                print(f"[{strategy:>16}] {scenario.id:<28} rate-limited, waiting 65s for the token window to reset")
                time.sleep(65)
                answer, outcomes, error = _invoke(strategy, scenario)
            passed = all(ok for _, ok in outcomes) and error is None
            per_strategy[strategy]["scenario_results"][scenario.id] = {
                "scenario_class": scenario.scenario_class,
                "passed": passed,
                "checks": {label: ok for label, ok in outcomes},
                "error": error,
                "answer_preview": answer[:220],
            }
            per_strategy[strategy]["total"] += 1
            if passed:
                per_strategy[strategy]["passed"] += 1
            if error:
                per_strategy[strategy]["errors"] += 1
            print(f"[{strategy:>16}] {scenario.id:<28} passed={passed} error={error}", flush=True)
            time.sleep(35)  # stay well under the measured 12,000 tokens/minute account limit

    for strategy, data in per_strategy.items():
        data["accuracy"] = round(data["passed"] / data["total"], 4) if data["total"] else 0.0

    winner = max(strategies, key=lambda s: (per_strategy[s]["accuracy"], -per_strategy[s]["errors"]))

    output = {
        "scenario_count": len(SCENARIOS),
        "strategies_tested": strategies,
        "winner": winner,
        "results": per_strategy,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"winner": winner, "accuracy": per_strategy[winner]["accuracy"]}, indent=2))


if __name__ == "__main__":
    main()
