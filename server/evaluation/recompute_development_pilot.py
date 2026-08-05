"""Recompute exploratory provider-pilot statistics from committed redacted rows.

No prompt, response, reasoning, header, credential, or raw provider error is needed. The output is
written to stdout so the tracked evidence remains immutable unless a reviewer explicitly captures
it elsewhere.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from .provider_stats import analyze_paired_results
except ImportError:  # direct script execution
    from provider_stats import analyze_paired_results


def load_redacted_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or "outcomes" not in row:
                raise ValueError(f"invalid redacted row at {path}:{line_number}")
            rows.append(row)
    if not rows:
        raise ValueError(f"no rows found in {path}")
    return rows


def recompute(run_directory: Path, *, draws: int = 10_000) -> dict[str, Any]:
    files = sorted(run_directory.glob("*.redacted.jsonl"))
    if not files:
        raise ValueError(f"no redacted pilot artifacts found in {run_directory}")
    return {
        "schema_version": 1,
        "source": "committed_redacted_rows_only",
        "run_directory": run_directory.name,
        "configurations": {
            path.stem.removesuffix(".redacted"): {
                "row_count": len(rows := load_redacted_rows(path)),
                "analysis": analyze_paired_results(rows, draws=draws),
            }
            for path in files
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--draws", type=int, default=10_000)
    args = parser.parse_args()
    print(json.dumps(recompute(args.run_directory, draws=args.draws), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
