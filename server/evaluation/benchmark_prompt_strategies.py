from __future__ import annotations

"""Small reproducible lookup/algorithmic-prompt benchmark.

It uses one batched model call per strategy to limit provider cost. Every answer is an exact
lookup or arithmetic result stated in the prompt, so scoring is automatic and does not require
subjective grading.
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.llm import _PROMPT_STRATEGIES, _build_llm


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data" / "benchmark" / "prompt_strategy_selection.json"

INCIDENT = """
Node 0 is connected to node 19.
Node 1 is connected to nodes 2, 18.
Node 2 is connected to nodes 1, 9.
Node 3 is connected to nodes 4, 8, 16.
Node 4 is connected to nodes 3, 5.
Node 5 is connected to nodes 4, 10, 12, 14.
Node 6 is connected to node 17.
Node 7 is not connected to any other node.
Node 8 is connected to nodes 3, 11.
Node 9 is connected to nodes 2, 16.
Node 10 is connected to node 5.
Node 11 is connected to nodes 8, 16.
Node 12 is connected to nodes 5, 19.
Node 13 is connected to node 16.
Node 14 is connected to nodes 5, 17, 19.
Node 15 is connected to node 18.
Node 16 is connected to nodes 3, 9, 11, 13.
Node 17 is connected to nodes 6, 14.
Node 18 is connected to nodes 1, 15.
Node 19 is connected to nodes 0, 12, 14.
""".strip()

TASKS = [
    {"id": "g1", "context": INCIDENT, "question": "Is there an edge between 1 and 18?", "answer": "yes"},
    {"id": "g2", "context": INCIDENT, "question": "Is there an edge between 7 and 8?", "answer": "no"},
    {"id": "g3", "context": INCIDENT, "question": "How many neighbours does node 16 have?", "answer": "4"},
    {
        "id": "a1",
        "context": "Student completed 96 SU. Curriculum minimum is 125 SU.",
        "question": "How many SU remain?",
        "answer": "29",
    },
    {
        "id": "a2",
        "context": "CS 201 status=completed; CS 300 status=enrolled; CS 307 status=withdrawn. Only completed, transfer and exempted count.",
        "question": "How many listed courses currently count toward graduation credit?",
        "answer": "1",
    },
    {
        "id": "a3",
        "context": "CS 455: title=Deep Learning, SU=3, ECTS=6, instructor=Ada Example.",
        "question": "Return the exact SU credit value for CS 455.",
        "answer": "3",
    },
]


def _parse_json(text: str) -> list[dict]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    match = re.search(r"\[[\s\S]*\]", cleaned)
    if not match:
        return []
    try:
        value = json.loads(match.group(0))
        return value if isinstance(value, list) else []
    except json.JSONDecodeError:
        return []


def main() -> None:
    llm = _build_llm()
    results = {}
    compact_tasks = [
        {"id": row["id"], "context": row["context"], "question": row["question"]}
        for row in TASKS
    ]
    expected = {row["id"]: row["answer"] for row in TASKS}

    for strategy, directive in _PROMPT_STRATEGIES.items():
        prompt = f"""
You are evaluating an exact lookup strategy.
Strategy directive: {directive}

Solve every independent task using only its context. Return ONLY a valid JSON array with
objects shaped {{"id":"...", "answer":"..."}}. Keep each answer to the exact short value
requested. Do not add explanations.

Tasks:
{json.dumps(compact_tasks, ensure_ascii=False)}
"""
        response = llm.invoke(prompt)
        text = str(getattr(response, "content", response))
        parsed = _parse_json(text)
        predicted = {
            str(item.get("id")): str(item.get("answer", "")).strip().lower()
            for item in parsed
            if isinstance(item, dict)
        }
        correct = sum(predicted.get(task_id) == answer for task_id, answer in expected.items())
        results[strategy] = {
            "correct": correct,
            "total": len(TASKS),
            "accuracy": round(correct / len(TASKS), 6),
            "parse_success": len(parsed) == len(TASKS),
        }

    priority = ["structured_lookup", "algorithmic", "lookup", "basic"]
    winner = max(
        priority,
        key=lambda name: (results[name]["accuracy"], -priority.index(name)),
    )
    output = {
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "model_calls": len(_PROMPT_STRATEGIES),
        "task_count": len(TASKS),
        "selection_metric": "exact_match_accuracy",
        "results": results,
        "winner": winner,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
