#!/usr/bin/env python3
from __future__ import annotations

"""
Publication figures for the reranking benchmark (second stage).

Companion to make_retrieval_figures.py, which draws the first-stage figures. The two scripts
share the palette, the method->colour discipline, the outside-right legends and the
"fail loudly, never substitute a placeholder" loading rules, so the two figure sets read as
one report. Nothing in here is typed in by hand: every plotted number comes out of a
reranker_metrics.json, a rankings/*.jsonl or the candidate-ceiling files, and a missing file,
key or column is a hard error rather than a zero.

INPUTS
    <run_dir>/reranker_metrics.json      {"<reranker>@<depth>": {hit@K, recall@K, mrr@10,
                                         ndcg@10, map@10, evidence_set_recall@K, all_gold@K,
                                         n_queries, depth, latency_{mean,p50,p95}_ms,
                                         model_load_seconds, trunc_* (optional)}}
    <run_dir>/rankings/<name>_at_<d>.jsonl   one row per query: query_id, reranker, depth,
                                         rank_of_first_gold, orig_rank_of_first_gold,
                                         num_gold, subgroups[], per-query metrics, top10[]
    --ceiling <...>/candidate_ceiling.csv    candidate_depth, n_queries, candidate_hit,
                                         candidate_allgold, queries_without_gold,
                                         oracle_hit@1, oracle_recall@{1,3,5}, oracle_hit@10,
                                         oracle_mrr@10, oracle_ndcg@10, oracle_allgold@10
    <same dir>/oracle_metrics.json       {"achieved": {...}, "by_depth": [...]}

Several run dirs may be passed at once; they are merged into one keyspace. Two dirs that
publish the same "<reranker>@<depth>" key with different numbers is an error (the report
would silently depend on argument order). A run dir that only contains some of the
rerankers is fine -- entries are discovered, never assumed.

OUTPUTS (under <out_dir>/, default <run_dir>/figures), each as .png (200 dpi) + .svg + .pdf
    fig1_rerank_recall_at_k        Recall@K and Hit@K for the finalists + the oracle ceiling
    fig2_gold_rank_ecdf            CDF of the rank of the first relevant chunk
    fig3_hit1_recall3_improvement  paired-bootstrap improvement over the baseline, 95% CI
    fig4_rank_shift                per-query gold-rank movement for the winner
    fig5_win_tie_loss              queries moved up / unchanged / down, + severe regressions
    fig6_quality_latency           Hit@1 vs P95 latency, Pareto frontier, 1000 ms budget
    fig7_subgroup_delta            Hit@1 delta vs baseline per query subgroup (diverging)
    fig9_candidate_depth           quality and latency vs candidate depth (fig8 is not ours)
    fig10_oracle_gap               remaining headroom: oracle minus achieved

USAGE
    python server/evaluation/make_reranking_figures.py \
        outputs/reranking/dev__cheap outputs/reranking/dev__bge outputs/reranking/dev__fusebge \
        --ceiling outputs/retrieval_lab/dev__pools/reranking/candidate_ceiling.csv \
        [--baseline original_ranking@10] [--out-dir <dir>] [--highlight <reranker@depth>]


--- WHAT "IMPROVEMENT" MEANS HERE, AND WHERE THE INTERVALS COME FROM ---

Every reranker is scored on the SAME frozen candidate pools, so a difference between two
rows is a difference in ordering ability alone. That makes the comparison paired, and the
intervals in fig3 are a paired bootstrap computed in this script from the per-query files:
10 000 resamples of the QUERY set (seed 20260727, one shared resample index matrix so all
rerankers are compared on identical draws), percentile CI at 2.5 / 97.5. No CI is read from
a file and none is assumed -- a metric without per-query data simply does not get an
interval. "Significant" is used in the narrow sense of "the 95% CI excludes zero".

Queries are matched by query_id, so a run scored on a subset of the benchmark is compared
on the intersection and the n is printed in the label rather than being silently averaged
into a different denominator. A run whose n_queries is far below the largest run in the
frame (see --min-queries-frac) is dropped from the aggregate figures and named on the
console, because a 12-query smoke run drawn next to a 238-query run reads as a result.


--- RANK SEMANTICS (the one contract subtlety worth stating) ---

`rank_of_first_gold` is the rank AFTER reranking, inside the candidate window of `depth`;
it is null when no gold chunk is in that window at all. `orig_rank_of_first_gold` is the
rank in the FULL first-stage list, which runs deeper than the reranking window (values of
11, 31, 57 occur), and is null only when the first stage never retrieved a gold chunk.

Those nulls are not rank 0 and are never plotted as such. In fig2 they are excluded from
the curve and counted in the footnote; in figs 4 and 5 they sit in an explicit
"below top-K / not in window" band. For fig4 the original rank is clamped to the reranking
window first: a gold chunk that sat at rank 57 was never inside a depth-10 pool, so calling
its disappearance a regression would blame the reranker for the first stage.


--- DESIGN CHOICES (inherited from make_retrieval_figures.py) ---

Palette: Okabe-Ito in the validated order blue / vermillion / bluish-green / orange /
reddish-purple / sky-blue, whose worst adjacent pair stays above the deuteranopia dE floor.
Colour follows the reranker, not its rank, and one map is built once and reused by every
figure. Baseline and winner take the two highest-contrast slots and additionally carry a
dashed-vs-solid line and a heavier weight, so identity never rests on hue alone.
Non-finalists share one neutral grey; in the figures where every reranker appears (3, 5, 7)
identity is carried by the axis label.

Figures 4 and 5 are the deliberate exception: there colour encodes OUTCOME (improved /
unchanged / regressed), which is the question those figures answer.

Diverging colour (fig7) is RdBu_r with vmin = -vmax pinned symmetrically about zero, so a
regression is a different hue and not merely a paler blue. Sequential ramps would hide it.

Proportion axes are anchored at 0 and never truncated to widen a gap. Where the interesting
spread is genuinely small (fig1 at K=1 and K=3) the fix is a labelled inset plus direct
delta annotations, not a cropped y-axis.
"""

import argparse
import csv
import json
import math
import os
import textwrap
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import NoReturn

import matplotlib

matplotlib.use("Agg")  # figures are written to disk; never needs a display

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# --------------------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------------------

RECALL_K: tuple[int, ...] = (1, 3, 5, 10)
HIT_K: tuple[int, ...] = (1, 3, 5, 10)
ECDF_MAX_RANK = 10          # the per-query files only preserve top10, so the CDF stops there
LATENCY_BUDGET_MS = 1000.0  # production ceiling drawn in fig6
NO_COST_MS = 0.05           # below this a "reranker" did no work; see fig6's split axis
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260727

# Okabe-Ito in the validated order (see module docstring).
PALETTE: tuple[str, ...] = ("#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9")
NEUTRAL = "#7A7A7A"
INK = "#1A1A1A"
INK_MUTED = "#5A5A5A"
GRID = "#DCDCDC"
SURFACE = "#FFFFFF"
ORACLE = "#4A4A4A"          # the ceiling is a reference, not a competing method

# outcome classes (used only where colour means outcome)
C_IMPROVED = "#009E73"
C_TIED = "#C4C4C4"
C_HARMED = "#D55E00"
C_SEVERE = "#8C2D04"

MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")
DASHES = ("-", "--", "-.", ":", (0, (5, 1, 1, 1)), (0, (3, 1, 3, 1, 1, 1)))

# Family detection is substring-based on the reranker name. ORDER MATTERS: a fusion of a
# cross-encoder with the first stage ("fuse_bge_first") is a fusion method first, and a
# fusion that also applies rules ("fuse_bge_first_rules") is still a fusion, so fusion is
# tested before both rule and cross_encoder.
FAMILY_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("baseline", ("original_ranking", "identity", "no_rerank", "first_stage", "passthrough")),
    ("fusion", ("fuse", "fusion", "rrf", "blend", "ensemble", "hybrid", "combo")),
    ("rule", ("rule", "metadata", "heuristic", "lexical_boost", "regex")),
    ("cross_encoder", ("msmarco", "bge", "qwen", "monot5", "minilm", "cross", "_ce", "ce_",
                       "jina", "mxbai", "colbert", "gte", "e5", "electra", "deberta", "llm")),
)
FAMILY_ORDER = ("baseline", "rule", "cross_encoder", "fusion", "other")
# The two families fig1 must contrast explicitly, per the report spec.
POINTWISE_FAMILY = "cross_encoder"
FUSION_FAMILY = "fusion"

CEILING_COLUMNS = ("candidate_depth", "n_queries", "candidate_hit", "candidate_allgold",
                   "queries_without_gold", "oracle_hit@1", "oracle_recall@1",
                   "oracle_recall@3", "oracle_recall@5", "oracle_hit@10", "oracle_mrr@10",
                   "oracle_ndcg@10", "oracle_allgold@10")

REQUIRED_METRIC_FIELDS = ("hit@1", "hit@3", "hit@10", "recall@1", "recall@3", "recall@5",
                          "recall@10", "mrr@10", "ndcg@10", "n_queries", "depth",
                          "latency_p95_ms")
REQUIRED_ROW_FIELDS = ("query_id", "rank_of_first_gold", "orig_rank_of_first_gold",
                       "subgroups", "hit@1", "recall@3")


# --------------------------------------------------------------------------------------
# loading -- every failure is fatal and says exactly what is wrong and how to fix it
# --------------------------------------------------------------------------------------


def die(msg: str) -> NoReturn:
    raise SystemExit(f"make_reranking_figures: ERROR: {msg}")


@dataclass
class Entry:
    """One (reranker, candidate depth) cell of the experiment."""

    key: str
    name: str
    depth: int
    run_dir: Path
    metrics: dict
    rankings_path: Path
    rows: list[dict] = field(default_factory=list)

    @property
    def n_queries(self) -> int:
        return int(self.metrics["n_queries"])

    @property
    def family(self) -> str:
        low = self.name.lower()
        for family, needles in FAMILY_PATTERNS:
            if any(n in low for n in needles):
                return family
        return "other"


def parse_key(key: str, path: Path) -> tuple[str, int]:
    if "@" not in key:
        die(f"{path}: key {key!r} is not '<reranker>@<depth>'. "
            f"Re-run run_reranking_lab.py; its keys always carry the depth.")
    name, _, depth_s = key.rpartition("@")
    if not name:
        die(f"{path}: key {key!r} has an empty reranker name")
    try:
        return name, int(depth_s)
    except ValueError:
        die(f"{path}: key {key!r} has a non-integer depth {depth_s!r}")


def load_run(run_dir: Path) -> dict[str, Entry]:
    if not run_dir.is_dir():
        die(f"run directory not found: {run_dir}")
    mpath = run_dir / "reranker_metrics.json"
    if not mpath.is_file():
        die(f"{run_dir} has no reranker_metrics.json.\n"
            f"  Either the run is still in progress, or it was never scored. Produce it with:\n"
            f"    python server/evaluation/run_reranking_lab.py --run outputs/retrieval_lab/dev__pools "
            f"--split dev --rerankers <names> --depths <depths>")
    try:
        raw = json.loads(mpath.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        die(f"{mpath} is not valid JSON: {exc}")
    if not isinstance(raw, dict) or not raw:
        die(f"{mpath} should be a non-empty mapping of '<reranker>@<depth>' -> metrics")

    out: dict[str, Entry] = {}
    for key, metrics in raw.items():
        if not isinstance(metrics, dict):
            die(f"{mpath}: entry {key!r} is not a mapping of metric -> value")
        name, depth = parse_key(key, mpath)
        missing = [f for f in REQUIRED_METRIC_FIELDS if f not in metrics]
        if missing:
            die(f"{mpath}: entry {key!r} is missing metric field(s) {missing}. "
                f"Found: {sorted(metrics)}")
        if int(metrics["depth"]) != depth:
            die(f"{mpath}: entry {key!r} declares depth={metrics['depth']}, which contradicts "
                f"its own key. Refusing to guess which one is right.")
        for f in REQUIRED_METRIC_FIELDS:
            v = metrics[f]
            if v is None or (isinstance(v, float) and math.isnan(v)):
                die(f"{mpath}: entry {key!r} has a null/NaN value for required field '{f}'")
        rankings = run_dir / "rankings" / f"{name}_at_{depth}.jsonl"
        out[key] = Entry(key=key, name=name, depth=depth, run_dir=run_dir,
                         metrics=metrics, rankings_path=rankings)
    return out


def merge_runs(run_dirs: list[Path]) -> dict[str, Entry]:
    merged: dict[str, Entry] = {}
    for run_dir in run_dirs:
        for key, entry in load_run(run_dir).items():
            prev = merged.get(key)
            if prev is None:
                merged[key] = entry
                continue
            # Same cell scored twice. Identical numbers are harmless (re-runs of the same
            # config); different numbers would make the report depend on argument order.
            differing = [f for f in REQUIRED_METRIC_FIELDS
                         if not _close(prev.metrics[f], entry.metrics[f])]
            if differing:
                die(f"'{key}' appears in both {prev.run_dir} and {entry.run_dir} with "
                    f"different values for {differing}. Pass only one of those run dirs, or "
                    f"re-run the duplicate so the two agree.")
            print(f"  note: '{key}' appears in {prev.run_dir.name} and {entry.run_dir.name} "
                  f"with identical metrics; using {prev.run_dir.name}")
    if not merged:
        die("no reranker entries found in any of the given run directories")
    return merged


def _close(a, b) -> bool:
    try:
        return math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-12)
    except (TypeError, ValueError):
        return a == b


def load_rows(entry: Entry) -> None:
    """Read the per-query rankings file for one entry. Fatal if it is absent or malformed."""
    if entry.rows:
        return
    path = entry.rankings_path
    if not path.is_file():
        die(f"'{entry.key}' is scored in {entry.run_dir / 'reranker_metrics.json'} but its "
            f"per-query file {path} is missing.\n"
            f"  figs 2-5 and 7 are computed from the per-query rows, not from the summary, so "
            f"this cannot be worked around. Re-run run_reranking_lab.py for '{entry.name}' "
            f"at depth {entry.depth}.")
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError as exc:
                die(f"{path}:{lineno} is not valid JSON: {exc}")
            missing = [f for f in REQUIRED_ROW_FIELDS if f not in r]
            if missing:
                die(f"{path}:{lineno} is missing field(s) {missing}")
            for f in ("rank_of_first_gold", "orig_rank_of_first_gold"):
                v = r[f]
                if v is not None and not isinstance(v, int):
                    die(f"{path}:{lineno}: '{f}' should be an integer rank or null, got {v!r}")
            if not isinstance(r["subgroups"], list):
                die(f"{path}:{lineno}: 'subgroups' should be a list of strings")
            rows.append(r)
    if not rows:
        die(f"{path} is empty; no per-query rows to plot")
    if len(rows) != entry.n_queries:
        die(f"{path} has {len(rows)} rows but reranker_metrics.json says n_queries="
            f"{entry.n_queries} for '{entry.key}'. The two artifacts are out of sync; re-run "
            f"run_reranking_lab.py for this reranker.")
    entry.rows = rows


def load_ceiling(path: Path) -> tuple[dict[int, dict], dict]:
    """Returns (by candidate depth -> ceiling row, oracle_metrics.json contents)."""
    if not path.is_file():
        die(f"--ceiling file not found: {path}\n"
            f"  Produce it with:\n"
            f"    python server/evaluation/candidate_ceiling.py --run outputs/retrieval_lab/dev__pools "
            f"--split dev")
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames
        if header is None:
            die(f"{path} is empty (no header row)")
        missing = [c for c in CEILING_COLUMNS if c not in header]
        if missing:
            die(f"{path} is missing required column(s): {missing}. Found: {header}")
        rows = list(reader)
    if not rows:
        die(f"{path} has a header but no depth rows")

    by_depth: dict[int, dict] = {}
    for row in rows:
        try:
            depth = int(row["candidate_depth"])
        except (TypeError, ValueError):
            die(f"{path}: candidate_depth {row.get('candidate_depth')!r} is not an integer")
        parsed = {"candidate_depth": depth}
        for col in CEILING_COLUMNS[1:]:
            raw = row.get(col, "")
            if raw is None or str(raw).strip() == "":
                die(f"{path}: empty value for '{col}' at candidate_depth={depth}")
            try:
                parsed[col] = float(raw)
            except ValueError:
                die(f"{path}: value {raw!r} in column '{col}' is not a number")
        by_depth[depth] = parsed

    ometrics_path = path.parent / "oracle_metrics.json"
    if not ometrics_path.is_file():
        die(f"{ometrics_path} not found next to the --ceiling CSV.\n"
            f"  candidate_ceiling.py writes both files; a CSV without its JSON means the pair "
            f"is from different runs. Re-run candidate_ceiling.py.")
    try:
        ometrics = json.loads(ometrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        die(f"{ometrics_path} is not valid JSON: {exc}")
    if "achieved" not in ometrics:
        die(f"{ometrics_path} has no 'achieved' block")
    return by_depth, ometrics


# --------------------------------------------------------------------------------------
# selection & shared styling
# --------------------------------------------------------------------------------------


def canonical_order(entries: dict[str, Entry], keys: list[str]) -> list[str]:
    """Stable display order: family, then name, then depth.

    Deliberately NOT sorted by score -- the colour map is derived from this order, and a
    score-derived order would repaint every reranker whenever the numbers shift slightly.
    """
    return sorted(keys, key=lambda k: (FAMILY_ORDER.index(entries[k].family),
                                       entries[k].name, entries[k].depth))


def pick_winner(entries: dict[str, Entry], keys: list[str], baseline: str,
                highlight: str | None) -> str:
    if highlight:
        if highlight not in entries:
            die(f"--highlight '{highlight}' is not an entry in the given run dirs "
                f"(have: {', '.join(sorted(entries))})")
        return highlight
    contenders = [k for k in keys if k != baseline]
    if not contenders:
        return baseline
    # Winner = best Hit@1 (the headline reranking metric); ties broken by MRR@10, then
    # nDCG@10, then name, so the choice is reproducible run to run.
    return max(contenders, key=lambda k: (entries[k].metrics["hit@1"],
                                          entries[k].metrics["mrr@10"],
                                          entries[k].metrics["ndcg@10"], k))


def select_finalists(entries: dict[str, Entry], keys: list[str], baseline: str,
                     winner: str, max_n: int) -> list[str]:
    """Baseline + winner + the best pointwise + the best fusion, topped up by Hit@1.

    Chosen from the data, never from a hardcoded name list, so a run set without (say) any
    fusion reranker simply yields fewer picks instead of failing.
    """
    picks: dict[str, None] = dict.fromkeys([baseline, winner])
    for family in (POINTWISE_FAMILY, FUSION_FAMILY, "rule"):
        members = [k for k in keys if entries[k].family == family and k != baseline]
        if members:
            picks.setdefault(max(members, key=lambda k: (entries[k].metrics["hit@1"], k)))
    for k in sorted(keys, key=lambda k: (-entries[k].metrics["hit@1"], k)):
        if len(picks) >= max_n:
            break
        picks.setdefault(k)
    return [k for k in canonical_order(entries, list(picks)) if k in entries][:max_n]


class Style:
    """One entry -> one colour/marker/dash, built once and shared by every figure."""

    def __init__(self, entries: dict[str, Entry], all_keys: list[str], finalists: list[str],
                 baseline: str, winner: str, n_ref: int):
        self.entries = entries
        self.baseline = baseline
        self.winner = winner
        self.n_ref = n_ref
        self.order = canonical_order(entries, all_keys)
        self.color: dict[str, str] = {}
        self.marker: dict[str, str] = {}
        self.dash: dict[str, object] = {}

        reserved = [baseline, winner] if winner != baseline else [baseline]
        rest = [k for k in canonical_order(entries, finalists) if k not in reserved]
        for i, k in enumerate(reserved + rest):
            self.color[k] = PALETTE[i % len(PALETTE)]
        for k in all_keys:
            self.color.setdefault(k, NEUTRAL)
        for i, k in enumerate(self.order):
            self.marker[k] = MARKERS[i % len(MARKERS)]
            self.dash[k] = DASHES[i % len(DASHES)]
        # Baseline always dashed, winner always solid and heavy: the pair the reader must
        # never confuse stays separable in print and under colour-blindness.
        self.dash[baseline] = (0, (6, 2))
        self.dash[winner] = "-"

    def label(self, key: str, *, short: bool = False) -> str:
        entry = self.entries[key]
        base = key if not short else key
        suffix = ""
        if key == self.baseline:
            suffix = "  (baseline, no reranking)" if not short else "  (baseline)"
        elif key == self.winner:
            suffix = "  ★ winner"
        if entry.n_queries != self.n_ref:
            suffix += f"  [n={entry.n_queries}]"
        return base + suffix

    def linewidth(self, key: str) -> float:
        return 2.6 if key == self.winner else 2.0


# --------------------------------------------------------------------------------------
# paired bootstrap -- computed here from per-query rows, never read from a file
# --------------------------------------------------------------------------------------


class Bootstrap:
    """Paired bootstrap over the query set, with one shared resample index per query set.

    Sharing the index matrix across rerankers means every method is compared on identical
    draws, so the intervals in fig3 are mutually consistent rather than each being a
    separate random experiment.
    """

    def __init__(self, draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED):
        self.draws = draws
        self.seed = seed
        self._idx: dict[int, np.ndarray] = {}

    def index(self, n: int) -> np.ndarray:
        if n not in self._idx:
            rng = np.random.default_rng(self.seed)
            self._idx[n] = rng.integers(0, n, size=(self.draws, n))
        return self._idx[n]

    def delta_ci(self, base_vals: np.ndarray, meth_vals: np.ndarray) -> dict:
        n = len(base_vals)
        if n == 0:
            die("paired bootstrap got an empty query set; the two rankings files share no "
                "query_id, so they cannot be compared")
        diff = meth_vals - base_vals
        boot = diff[self.index(n)].mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        p = 2.0 * min((boot <= 0).mean(), (boot >= 0).mean())
        return {"delta": float(diff.mean()), "ci_low": float(lo), "ci_high": float(hi),
                "p_value": float(min(1.0, p)), "n": n,
                "significant": bool(lo > 0 or hi < 0)}


def aligned(base: Entry, meth: Entry, col: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Per-query vectors for the queries both entries scored, in a stable order."""
    b = {r["query_id"]: r for r in base.rows}
    m = {r["query_id"]: r for r in meth.rows}
    qids = [q for q in (r["query_id"] for r in base.rows) if q in m]
    for q in qids:
        for src, e in ((b, base), (m, meth)):
            if col not in src[q]:
                die(f"{e.rankings_path}: query {q} has no '{col}' column")
    return (np.array([float(b[q][col]) for q in qids]),
            np.array([float(m[q][col]) for q in qids]), qids)


# --------------------------------------------------------------------------------------
# figure plumbing (identical conventions to make_retrieval_figures.py)
# --------------------------------------------------------------------------------------


def apply_rc() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.size": 10,
        "axes.titlesize": 12.5,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "legend.frameon": False,
        "figure.dpi": 110,
    })


def tidy(ax, *, grid_axis: str = "y") -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(True, axis=grid_axis, zorder=0)
    ax.set_axisbelow(True)


def save(fig, out_dir: Path, basename: str, written: list[Path]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext, kwargs in ((".png", {"dpi": 200}), (".svg", {}), (".pdf", {})):
        path = out_dir / f"{basename}{ext}"
        fig.savefig(path, **kwargs)
        written.append(path)
    plt.close(fig)


def footnote(fig, text: str) -> None:
    """Caption strip along the bottom.

    Constrained layout knows nothing about a raw fig.text, so the strip is reserved by
    shrinking the layout rect first -- otherwise the caption lands on the x label.
    """
    wrap_at = max(60, int(fig.get_figwidth() * 17))   # ~17 chars per inch at 8 pt
    lines = textwrap.wrap(text, wrap_at)
    strip = (0.16 * len(lines) + 0.10) / fig.get_figheight()
    engine = fig.get_layout_engine()
    if engine is not None:
        engine.set(rect=(0.0, strip, 1.0, 1.0 - strip))
    fig.text(0.006, 0.006, "\n".join(lines), fontsize=8, color=INK_MUTED,
             ha="left", va="bottom")


def side_legend(fig, ax=None, handles=None, labels=None, **kwargs):
    """Legend in a reserved column to the right of the plot, so it can never cover data."""
    if handles is None:
        handles, labels = ax.get_legend_handles_labels()
    if labels is None:
        return fig.legend(handles=handles, loc="outside right upper", **kwargs)
    return fig.legend(handles, labels, loc="outside right upper", **kwargs)


def rank_or_inf(v) -> float:
    return math.inf if v is None else float(v)


# --------------------------------------------------------------------------------------
# figure 1 -- Recall@K / Hit@K with the oracle ceiling
# --------------------------------------------------------------------------------------


def fig1_rerank_recall_at_k(entries, style: Style, finalists, ceiling_row, out_dir,
                            written) -> None:
    rec_ks = [k for k in RECALL_K
              if all(f"recall@{k}" in entries[f].metrics for f in finalists)]
    hit_ks = [k for k in HIT_K if all(f"hit@{k}" in entries[f].metrics for f in finalists)]
    if not rec_ks:
        die("no recall@K metric is present for every finalist; cannot draw fig1")

    fig = plt.figure(figsize=(13.2, 5.9), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0])
    ax_r = fig.add_subplot(gs[0, 0])
    ax_h = fig.add_subplot(gs[0, 1])

    def draw(ax, ks, prefix, ylabel, title):
        tidy(ax, grid_axis="both")
        xs = list(range(len(ks)))
        # Evenly spaced categorical x: K = 1,3,5,10 is far from uniform, and on a numeric
        # axis the interesting low-K region collapses into the left margin.
        order = sorted(finalists, key=lambda k: (k == style.winner,
                                                 entries[k].metrics[f"{prefix}@{ks[-1]}"]))
        for key in order:
            ys = [entries[key].metrics[f"{prefix}@{k}"] for k in ks]
            ax.plot(xs, ys, color=style.color[key], marker=style.marker[key], markersize=7,
                    linestyle=style.dash[key], linewidth=style.linewidth(key),
                    markeredgecolor=SURFACE, markeredgewidth=1.0,
                    label=style.label(key), zorder=3)
        # Oracle ceiling: only at the K values the ceiling file actually carries.
        ok, oy = [], []
        for i, k in enumerate(ks):
            col = f"oracle_{prefix}@{k}"
            if col in ceiling_row:
                ok.append(i)
                oy.append(ceiling_row[col])
        if ok:
            ax.plot(ok, oy, color=ORACLE, linestyle=(0, (1, 1.6)), linewidth=2.0,
                    marker="_", markersize=13, markeredgewidth=2.0, zorder=4,
                    label="oracle ceiling (perfect reranking\nof the same candidates)")
            # At the right edge the label has to hang inside the axes, or it is clipped.
            at_edge = ok[-1] >= len(ks) - 1
            ax.annotate(f"{oy[-1]:.3f}", xy=(ok[-1], oy[-1]),
                        xytext=(-4, 7) if at_edge else (6, 6),
                        textcoords="offset points", fontsize=8.5, color=ORACLE,
                        ha="right" if at_edge else "left", va="bottom", fontweight="bold")
        ax.set_xticks(xs, [str(k) for k in ks])
        ax.set_ylim(0, 1.0)   # proportion axis anchored at 0 -- never cropped to widen gaps
        ax.set_xlim(-0.18, len(ks) - 0.82)
        ax.set_xlabel("K (rank cut-off)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        return xs

    xs_r = draw(ax_r, rec_ks, "recall", "Recall@K (mean fraction of the gold set)",
                "Recall@K after reranking")
    draw(ax_h, hit_ks, "hit", "Hit@K (fraction of queries)", "Hit@K after reranking")

    # The K=1..3 spread is a few points wide on a 0-1 axis. Rather than truncate the axis,
    # magnify it in a labelled inset and write the deltas vs the baseline next to it.
    zoom_ks = [k for k in rec_ks if k <= 3]
    if len(zoom_ks) >= 2:
        ins = ax_r.inset_axes([0.085, 0.085, 0.50, 0.40])
        zi = [rec_ks.index(k) for k in zoom_ks]
        vals = []
        for key in finalists:
            ys = [entries[key].metrics[f"recall@{k}"] for k in zoom_ks]
            vals += ys
            ins.plot(zi, ys, color=style.color[key], marker=style.marker[key], markersize=5,
                     linestyle=style.dash[key], linewidth=style.linewidth(key) * 0.8,
                     markeredgecolor=SURFACE, markeredgewidth=0.8, zorder=3)
        for k in zoom_ks:
            col = f"oracle_recall@{k}"
            if col in ceiling_row:
                vals.append(ceiling_row[col])
        oz = [(zi[i], ceiling_row[f"oracle_recall@{k}"]) for i, k in enumerate(zoom_ks)
              if f"oracle_recall@{k}" in ceiling_row]
        if oz:
            ins.plot([p[0] for p in oz], [p[1] for p in oz], color=ORACLE,
                     linestyle=(0, (1, 1.6)), linewidth=1.6, zorder=4)
        lo, hi = min(vals), max(vals)
        pad = max(0.012, (hi - lo) * 0.28)
        ins.set_ylim(lo - pad, hi + pad)
        ins.set_xlim(zi[0] - 0.22, zi[-1] + 0.22)
        ins.set_xticks(zi, [f"K={k}" for k in zoom_ks])
        ins.tick_params(labelsize=8, length=2)
        ins.set_facecolor("#FBFCFE")
        for side in ("top", "right"):
            ins.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ins.spines[side].set_color(GRID)
        ins.grid(True, axis="y", zorder=0)
        ins.set_axisbelow(True)
        ins.set_title("zoom: y-axis magnified", fontsize=8, color=INK_MUTED, loc="left",
                      pad=4, fontweight="normal")
        ax_r.indicate_inset_zoom(ins, edgecolor=INK_MUTED, alpha=0.45, linewidth=1.0)

        # Direct delta labels at K=1 so the small differences are readable as numbers too.
        # Parked in the empty right-hand block below the curves, clear of the oracle label.
        base_v = entries[style.baseline].metrics[f"recall@{zoom_ks[0]}"]
        stack = sorted(finalists, key=lambda k: -entries[k].metrics[f"recall@{zoom_ks[0]}"])
        top = 0.645
        ax_r.annotate(f"Δ Recall@{zoom_ks[0]} vs baseline", xy=(0.615, top + 0.055),
                      xycoords="axes fraction", fontsize=8.5, color=INK_MUTED,
                      ha="left", va="top")
        for i, key in enumerate(stack):
            d = entries[key].metrics[f"recall@{zoom_ks[0]}"] - base_v
            txt = "baseline" if key == style.baseline else f"{d:+.3f}"
            ax_r.annotate(f"{key}  {txt}", xy=(0.615, top - i * 0.052),
                          xycoords="axes fraction", fontsize=8.5,
                          color=style.color[key], ha="left", va="top",
                          fontweight="bold" if key == style.winner else "normal")

    handles, labels = ax_r.get_legend_handles_labels()
    side_legend(fig, handles=handles, labels=labels)
    depth = int(ceiling_row["candidate_depth"])
    missing_ks = [k for k in rec_ks if f"oracle_recall@{k}" not in ceiling_row]
    note = (f"All rerankers see the same frozen candidate pool (depth {depth}, "
            f"n = {entries[style.baseline].n_queries} dev queries), so differences are ordering "
            f"ability alone. Recall@K = mean fraction of a query's gold set inside the top K; "
            f"Hit@K = fraction of queries with at least one gold chunk there. The oracle line "
            f"is the best any reordering of those same candidates could reach "
            f"(candidate_ceiling.csv).")
    if missing_ks:
        note += (f" The ceiling file carries no oracle_recall@{'/'.join(map(str, missing_ks))} "
                 f"column, so the oracle line stops before those K -- it is not extrapolated.")
    footnote(fig, note)
    save(fig, out_dir, "fig1_rerank_recall_at_k", written)


# --------------------------------------------------------------------------------------
# figure 2 -- ECDF of the rank of the first relevant chunk
# --------------------------------------------------------------------------------------


def fig2_gold_rank_ecdf(entries, style: Style, finalists, out_dir, written) -> None:
    fig, ax = plt.subplots(figsize=(11.6, 5.6), layout="constrained")
    tidy(ax, grid_axis="both")

    ranks = np.arange(1, ECDF_MAX_RANK + 1)
    unreachable: dict[str, int] = {}
    plateau: list[tuple[str, float]] = []
    curves: dict[str, list[float]] = {}
    order = sorted(finalists, key=lambda k: k == style.winner)
    for key in order:
        e = entries[key]
        n = len(e.rows)
        vals = [rank_or_inf(r["rank_of_first_gold"]) for r in e.rows]
        # A null rank is "no relevant chunk anywhere in the candidate window": it is not
        # rank 0 and never enters the curve. It only lowers the plateau, which is the point.
        unreachable[key] = sum(1 for v in vals if math.isinf(v))
        ys = [sum(1 for v in vals if v <= k) / n for k in ranks]
        curves[key] = ys
        plateau.append((key, ys[-1]))
        ax.step(ranks, ys, where="post", color=style.color[key],
                linestyle=style.dash[key], linewidth=style.linewidth(key), zorder=3,
                label=style.label(key))
        ax.plot(ranks, ys, linestyle="none", marker=style.marker[key], markersize=5,
                color=style.color[key], markeredgecolor=SURFACE, markeredgewidth=0.8,
                zorder=4)

    # The plateau is the ceiling every curve shares (it is the candidate pool's recall), so
    # it is drawn once as a reference rather than left for the reader to infer.
    top = max(v for _, v in plateau)
    ax.axhline(top, color=ORACLE, linewidth=1.0, linestyle=(0, (1, 2)), zorder=2)
    ax.annotate(f"candidate-pool ceiling {top:.3f}\n(no reordering can pass this)",
                xy=(ECDF_MAX_RANK, top), xytext=(-4, -8), textcoords="offset points",
                fontsize=8.5, color=ORACLE, ha="right", va="top")

    ax.set_xticks(ranks, [str(r) for r in ranks])
    ax.set_xlim(0.85, ECDF_MAX_RANK + 0.15)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Rank of the first relevant chunk (1 = top of the list)")
    ax.set_ylabel("Fraction of queries with a relevant chunk at or above that rank")
    ax.set_title("How deep the user must read before hitting relevant evidence")

    # Every curve lives in a narrow band near the top, because the first stage already puts
    # a relevant chunk first for ~4 queries in 5. The main axes stay anchored at 0; the band
    # is magnified in an inset placed in the empty lower half.
    lo = min(min(ys) for ys in curves.values())
    hi = max(max(max(ys) for ys in curves.values()), top)
    ins = ax.inset_axes([0.30, 0.14, 0.66, 0.46])
    for key in order:
        ins.step(ranks, curves[key], where="post", color=style.color[key],
                 linestyle=style.dash[key], linewidth=style.linewidth(key) * 0.9, zorder=3)
        ins.plot(ranks, curves[key], linestyle="none", marker=style.marker[key],
                 markersize=4.5, color=style.color[key], markeredgecolor=SURFACE,
                 markeredgewidth=0.8, zorder=4)
    ins.axhline(top, color=ORACLE, linewidth=1.0, linestyle=(0, (1, 2)), zorder=2)
    pad = max(0.006, (hi - lo) * 0.10)
    ins.set_ylim(lo - pad, hi + pad)
    ins.set_xlim(0.85, ECDF_MAX_RANK + 0.15)
    ins.set_xticks(ranks, [str(r) for r in ranks])
    ins.tick_params(labelsize=8, length=2)
    ins.set_facecolor("#FBFCFE")
    for side in ("top", "right"):
        ins.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ins.spines[side].set_color(GRID)
    ins.grid(True, axis="both", zorder=0)
    ins.set_axisbelow(True)
    ins.set_title("zoom: y-axis magnified (main axes stay anchored at 0)",
                  fontsize=8, color=INK_MUTED, loc="left", pad=4, fontweight="normal")
    indicator = ax.indicate_inset_zoom(ins, edgecolor=INK_MUTED, alpha=0.45, linewidth=1.0)
    # The zoom spans the full x range, so the rectangle alone says what is magnified; the
    # connector lines would only rake diagonally across the curves.
    # Matplotlib >= 3.10 returns one InsetIndicator artist; older versions return a tuple.
    connectors = getattr(indicator, "connectors", None)
    if connectors is None:
        connectors = indicator[1]
    for line in connectors:
        line.set_visible(False)

    side_legend(fig, ax)
    detail = "; ".join(f"{k}: {v}" for k, v in unreachable.items())
    footnote(fig, "Empirical CDF over the dev queries; higher and further left is better. "
                  "Queries whose candidate window contains no relevant chunk are NOT drawn "
                  "at rank 0 -- they are unreachable by any reranker and simply never reach "
                  "the curve, which is why every line plateaus below 1.0. Unreachable "
                  f"queries per method -- {detail} (of "
                  f"{len(entries[style.baseline].rows)} queries). The per-query files keep "
                  f"the top {ECDF_MAX_RANK} only, so the curve stops at rank {ECDF_MAX_RANK}.")
    save(fig, out_dir, "fig2_gold_rank_ecdf", written)


# --------------------------------------------------------------------------------------
# figure 3 -- paired-bootstrap improvement over the baseline
# --------------------------------------------------------------------------------------


def compute_improvements(entries, keys, baseline, boot: Bootstrap,
                         metrics=("hit@1", "recall@3")) -> dict[str, dict[str, dict]]:
    base = entries[baseline]
    out: dict[str, dict[str, dict]] = {}
    for key in keys:
        if key == baseline:
            continue
        per_metric = {}
        for col in metrics:
            b, m, _ = aligned(base, entries[key], col)
            per_metric[col] = boot.delta_ci(b, m)
        out[key] = per_metric
    return out


def fig3_hit1_recall3_improvement(entries, style: Style, improvements, baseline, out_dir,
                                  written) -> None:
    cols = ("hit@1", "recall@3")
    titles = ("Hit@1  (a relevant chunk is ranked first)",
              "Recall@3  (gold set covered in the top 3)")
    keys = sorted(improvements, key=lambda k: improvements[k]["hit@1"]["delta"])
    if not keys:
        die("no reranker to compare against the baseline; fig3 needs at least one non-baseline "
            "entry at the primary candidate depth")

    ys = np.arange(len(keys))
    height = max(3.6, 0.46 * len(keys) + 2.9)
    fig = plt.figure(figsize=(13.4, height), layout="constrained")
    axes = fig.subplots(1, 2, sharey=True)

    for ax, col, title in zip(axes, cols, titles):
        tidy(ax, grid_axis="x")
        for i, key in enumerate(keys):
            r = improvements[key][col]
            colour = style.color.get(key, NEUTRAL)
            sig = r["significant"]
            ax.errorbar(r["delta"], i, xerr=[[r["delta"] - r["ci_low"]],
                                             [r["ci_high"] - r["delta"]]],
                        fmt="none", ecolor=colour, elinewidth=2.0, capsize=4, capthick=1.6,
                        alpha=0.95, zorder=3)
            # Filled = the 95% CI excludes zero; hollow = it does not. Texture rather than a
            # second hue keeps the significance channel readable in greyscale.
            ax.plot(r["delta"], i, marker=style.marker.get(key, "o"), markersize=9,
                    color=colour if sig else SURFACE, markeredgecolor=colour,
                    markeredgewidth=1.8, zorder=4)
        ax.axvline(0, color=INK, linewidth=1.8, zorder=5)
        span = max(max(abs(improvements[k][col]["ci_low"]),
                       abs(improvements[k][col]["ci_high"])) for k in keys)
        span = max(span, 0.01)
        ax.set_xlim(-span * 1.42, span * 1.42)
        for i, key in enumerate(keys):
            r = improvements[key][col]
            right = r["delta"] >= 0
            ax.annotate(f"{r['delta']:+.3f}",
                        xy=(r["ci_high"] if right else r["ci_low"], i),
                        xytext=(6 if right else -6, 0), textcoords="offset points",
                        fontsize=8.5, color=INK, ha="left" if right else "right",
                        va="center", zorder=6)
        ax.set_xlabel(f"Δ {col} vs baseline (absolute)")
        ax.set_title(title, fontsize=11)
        ax.annotate("worse", xy=(0, 1.005), xycoords=("data", "axes fraction"),
                    xytext=(-6, 0), textcoords="offset points", fontsize=8,
                    color=INK_MUTED, ha="right", va="bottom")
        ax.annotate("better", xy=(0, 1.005), xycoords=("data", "axes fraction"),
                    xytext=(6, 0), textcoords="offset points", fontsize=8,
                    color=INK_MUTED, ha="left", va="bottom")

    axes[0].set_yticks(ys, [style.label(k) for k in keys])
    for tick, key in zip(axes[0].get_yticklabels(), keys):
        if key == style.winner:
            tick.set_fontweight("bold")
    axes[0].set_ylim(-0.7, len(keys) - 0.3)

    n = improvements[keys[0]]["hit@1"]["n"]
    side_legend(fig, handles=[
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#9E9E9E",
               markeredgecolor="#9E9E9E", markersize=9,
               label="95% CI excludes 0"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=SURFACE,
               markeredgecolor="#9E9E9E", markeredgewidth=1.8, markersize=9,
               label="CI spans 0\n(no reliable effect)"),
        Line2D([0], [0], color="#9E9E9E", lw=2.0, label="95% paired\nbootstrap CI"),
        Line2D([0], [0], color=INK, lw=1.8, label=f"0 = baseline\n{baseline}"),
    ], labels=None)
    footnote(fig, f"Absolute change vs '{baseline}' on the same frozen candidate pools. "
                  f"Intervals are a paired bootstrap computed in this script from the "
                  f"per-query rankings files: {BOOTSTRAP_DRAWS:,} resamples of the query set "
                  f"(seed {BOOTSTRAP_SEED}), percentile CI, n = {n} paired queries, one shared "
                  f"set of draws for every reranker. Dot colour = reranker, shared across all "
                  f"figures. Negative results are shown, not clipped.")
    save(fig, out_dir, "fig3_hit1_recall3_improvement", written)


# --------------------------------------------------------------------------------------
# figure 4 -- per-query gold-rank movement for the winner
# --------------------------------------------------------------------------------------


def fig4_rank_shift(entries, style: Style, winner, out_dir, written) -> None:
    e = entries[winner]
    depth = e.depth
    out_band = depth + 1  # display row for "not in the reranking window"

    def clamp(v) -> float:
        # A gold chunk deeper than the candidate window was never visible to the reranker,
        # so it is displayed in the out-of-window band rather than counted as a regression.
        r = rank_or_inf(v)
        return r if r <= depth else math.inf

    moved, unchanged, unreachable = [], 0, 0
    for r in e.rows:
        before = clamp(r["orig_rank_of_first_gold"])
        after = clamp(r["rank_of_first_gold"])
        if math.isinf(before) and math.isinf(after):
            unreachable += 1
            continue
        if before == after:
            unchanged += 1
            continue
        moved.append((r["query_id"], before, after))
    if not moved:
        fig, ax = plt.subplots(figsize=(9.0, 3.4), layout="constrained")
        ax.axis("off")
        ax.text(0.5, 0.5, f"{winner} changed the rank of the first relevant chunk\n"
                          f"for 0 of {len(e.rows)} queries (nothing to plot)",
                ha="center", va="center", fontsize=12, color=INK)
        ax.set_title("Per-query gold-rank movement")
        save(fig, out_dir, "fig4_rank_shift", written)
        return

    moved.sort(key=lambda t: (t[2] - t[1] if math.isfinite(t[2] - t[1])
                              else (math.inf if math.isinf(t[2]) else -math.inf),
                              t[1]))
    disp = lambda v: out_band if math.isinf(v) else v  # noqa: E731

    fig = plt.figure(figsize=(13.4, 6.0), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[3.3, 1.0])
    ax = fig.add_subplot(gs[0, 0])
    ax_s = fig.add_subplot(gs[0, 1])
    tidy(ax, grid_axis="y")

    n_up = sum(1 for _, b, a in moved if a < b)
    n_down = len(moved) - n_up
    for i, (_qid, before, after) in enumerate(moved):
        b, a = disp(before), disp(after)
        colour = C_IMPROVED if after < before else C_HARMED
        ax.annotate("", xy=(i, a), xytext=(i, b),
                    arrowprops=dict(arrowstyle="-|>", color=colour, linewidth=1.7,
                                    shrinkA=0, shrinkB=0, mutation_scale=9), zorder=3)
        ax.plot(i, b, marker="o", markersize=4.2, color=SURFACE, markeredgecolor=colour,
                markeredgewidth=1.3, zorder=4)

    ax.axhspan(out_band - 0.5, out_band + 0.5, color="#F0F0F0", zorder=1)
    ax.annotate(f"not in the top-{depth} window\n(no relevant chunk to rank)",
                xy=(len(moved) - 0.5, out_band), xytext=(-4, 0), textcoords="offset points",
                fontsize=8.5, color=INK_MUTED, ha="right", va="center")
    ax.invert_yaxis()
    ax.set_yticks(list(range(1, depth + 1)) + [out_band],
                  [str(r) for r in range(1, depth + 1)] + [f">{depth}"])
    ax.set_ylim(out_band + 0.7, 0.4)
    ax.set_xlim(-1.0, len(moved))
    # The x position is a sort slot, not a quantity: numbering it would invite the reader to
    # read a trend into what is only an ordering.
    ax.set_xticks([])
    ax.set_xlabel(f"One column per query whose first-relevant rank moved, ordered by size of "
                  f"the move (n = {len(moved)} of {len(e.rows)})")
    ax.set_ylabel("Rank of the first relevant chunk (1 = best)")
    ax.set_title(f"Where '{winner}' moved the first relevant chunk")
    ax.annotate("arrow tail = before (first stage)\narrow head = after (reranked)",
                xy=(0.012, 0.135), xycoords="axes fraction", fontsize=8.5, color=INK_MUTED,
                ha="left", va="bottom")

    # Summary panel: the queries the main panel deliberately does not draw are still counted.
    tidy(ax_s, grid_axis="x")
    cats = [("moved up", n_up, C_IMPROVED), ("unchanged", unchanged, C_TIED),
            ("moved down", n_down, C_HARMED),
            (f"never in top-{depth}", unreachable, "#B0B0B0")]
    ys = np.arange(len(cats))
    ax_s.barh(ys, [c[1] for c in cats], color=[c[2] for c in cats], height=0.62, zorder=3)
    for i, (_lab, v, _c) in enumerate(cats):
        ax_s.annotate(f"{v}  ({v / len(e.rows):.1%})", xy=(v, i), xytext=(5, 0),
                      textcoords="offset points", fontsize=9, color=INK, va="center")
    ax_s.set_yticks(ys, [c[0] for c in cats])
    ax_s.invert_yaxis()
    ax_s.set_xlim(0, max(c[1] for c in cats) * 1.42)
    ax_s.set_xlabel(f"Queries (n = {len(e.rows)})")
    ax_s.set_title("All queries, accounted for", fontsize=11)

    footnote(fig, "'Before' is the rank of the first relevant chunk in the first-stage list, "
                  f"clamped to the top-{depth} reranking window: a gold chunk that sat deeper "
                  f"than {depth} was never shown to the reranker, so it starts in the "
                  f">{depth} band instead of being scored as a regression the reranker caused. "
                  "Only queries whose rank changed are drawn; the right panel accounts for "
                  "every query, including the unchanged majority.")
    save(fig, out_dir, "fig4_rank_shift", written)


# --------------------------------------------------------------------------------------
# figure 5 -- win / tie / loss vs the baseline, plus severe regressions
# --------------------------------------------------------------------------------------


def win_tie_loss(base: Entry, meth: Entry) -> dict:
    b = {r["query_id"]: rank_or_inf(r["rank_of_first_gold"]) for r in base.rows}
    counts = {"up": 0, "same": 0, "down": 0, "n": 0,
              "top3_to_out": 0, "top1_to_below3": 0}
    for r in meth.rows:
        qid = r["query_id"]
        if qid not in b:
            continue
        before, after = b[qid], rank_or_inf(r["rank_of_first_gold"])
        counts["n"] += 1
        if after < before:
            counts["up"] += 1
        elif after > before:
            counts["down"] += 1
            # Severe regressions: the two that actually change what a user sees.
            if before <= 3 and (math.isinf(after) or after > ECDF_MAX_RANK):
                counts["top3_to_out"] += 1
            if before <= 1 and after > 3:
                counts["top1_to_below3"] += 1
        else:
            counts["same"] += 1
    return counts


def fig5_win_tie_loss(entries, style: Style, keys, baseline, out_dir, written) -> None:
    rows = []
    for key in keys:
        if key == baseline:
            continue
        c = win_tie_loss(entries[baseline], entries[key])
        rows.append((key, c))
    if not rows:
        die("fig5 needs at least one non-baseline entry at the primary candidate depth")
    rows.sort(key=lambda t: (t[1]["up"] - t[1]["down"]))

    ys = np.arange(len(rows))
    height = max(3.8, 0.5 * len(rows) + 3.0)
    fig = plt.figure(figsize=(13.6, height), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[2.9, 1.0])
    ax = fig.add_subplot(gs[0, 0])
    ax_s = fig.add_subplot(gs[0, 1], sharey=ax)
    tidy(ax, grid_axis="x")
    tidy(ax_s, grid_axis="x")

    up = np.array([r[1]["up"] for r in rows], dtype=float)
    same = np.array([r[1]["same"] for r in rows], dtype=float)
    down = np.array([r[1]["down"] for r in rows], dtype=float)
    total = up + same + down

    # Colour encodes OUTCOME here, not method: each bar is one reranker, already named on
    # the y axis, and the reader's question is "how many queries got better vs worse".
    seg = dict(height=0.6, zorder=3, edgecolor=SURFACE, linewidth=2)
    ax.barh(ys, up, color=C_IMPROVED, label="moved up (rank improved)", **seg)
    ax.barh(ys, same, left=up, color=C_TIED, label="unchanged", **seg)
    ax.barh(ys, down, left=up + same, color=C_HARMED, label="moved down (rank worsened)",
            **seg)

    for i in range(len(rows)):
        for value, left, colour in ((up[i], 0.0, "#FFFFFF"), (same[i], up[i], INK),
                                    (down[i], up[i] + same[i], "#FFFFFF")):
            if value <= 0 or value / max(total[i], 1) < 0.045:
                continue  # too narrow for a legible label; the summary column carries it
            ax.text(left + value / 2, i, f"{int(value)}", ha="center", va="center",
                    fontsize=9, color=colour, zorder=5)
        ax.annotate(f"+{int(up[i])} / -{int(down[i])}", xy=(total[i], i), xytext=(6, 0),
                    textcoords="offset points", fontsize=8.5, color=INK_MUTED, va="center")

    ax.set_yticks(ys, [style.label(r[0]) for r in rows])
    for tick, r in zip(ax.get_yticklabels(), rows):
        if r[0] == style.winner:
            tick.set_fontweight("bold")
    ax.set_xlim(0, max(total) * 1.10 if len(total) else 1)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlabel(f"Benchmark queries (n = {int(max(total)) if len(total) else 0})")
    # The baseline's name lives in the x label and the footnote: spelled out in the title it
    # runs into the second panel's title on a two-panel figure.
    ax.set_title("Per-query movement of the first relevant chunk")

    # Severe regressions get their own panel: they are 1-3 queries each and would be an
    # invisible sliver inside the stacked bar, yet they are the ones that break an answer.
    w = 0.34
    sev1 = [r[1]["top3_to_out"] for r in rows]
    sev2 = [r[1]["top1_to_below3"] for r in rows]
    ax_s.barh(ys - w / 2, sev1, height=w, color=C_SEVERE, zorder=3,
              label=f"top-3 → below top-{ECDF_MAX_RANK}")
    ax_s.barh(ys + w / 2, sev2, height=w, color="#E8A33D", zorder=3,
              label="top-1 → below top-3")
    for i in range(len(rows)):
        for v, off in ((sev1[i], -w / 2), (sev2[i], w / 2)):
            ax_s.annotate(str(int(v)), xy=(v, i + off), xytext=(4, 0),
                          textcoords="offset points", fontsize=8.5,
                          color=INK if v else INK_MUTED, va="center")
    ax_s.set_xlim(0, max(max(sev1 + sev2), 1) * 1.45)
    ax_s.set_xlabel("Queries")
    ax_s.set_title("Severe regressions", fontsize=11)
    ax_s.tick_params(labelleft=False)

    handles, labels = ax.get_legend_handles_labels()
    sh, sl = ax_s.get_legend_handles_labels()
    # One legend outside the frame for both panels: an in-axes legend on the narrow right
    # panel lands on the longest bar.
    side_legend(fig, handles=handles + sh, labels=labels + sl)
    note = ("Movement is the rank of the first relevant chunk after reranking vs its "
            f"rank under '{baseline}' on the same candidates. A query with no relevant "
            "chunk in the window counts as unchanged only when it is absent under both. "
            "The right panel is a subset of 'moved down', broken out because a top-3 result "
            "falling out of the top-10 changes what the user actually reads; it is drawn "
            "separately, not hidden inside the bar.")
    depths = {entries[k].depth for k, _ in rows} | {entries[baseline].depth}
    if max(depths) <= ECDF_MAX_RANK:
        d = max(depths)
        note += (f" NOTE: at candidate depth {d} a reranker only permutes those {d} "
                 f"candidates, so a relevant chunk cannot fall below rank "
                 f"{ECDF_MAX_RANK} at all -- the 'top-3 → below top-{ECDF_MAX_RANK}' series "
                 f"is zero by construction here, not by luck.")
    footnote(fig, note)
    save(fig, out_dir, "fig5_win_tie_loss", written)


# --------------------------------------------------------------------------------------
# figure 6 -- quality vs latency
# --------------------------------------------------------------------------------------


def pareto_frontier(points: list[tuple[str, float, float]]) -> list[tuple[str, float, float]]:
    """Entries not dominated on (minimise latency, maximise Hit@1).

    An entry is dominated when another is at least as fast AND at least as accurate, with at
    least one strictly better. Exact ties are kept, so a duplicated configuration is visible
    rather than silently dropped.
    """
    keep = []
    for name, x, y in points:
        dominated = any((ox <= x and oy >= y) and (ox < x or oy > y)
                        for oname, ox, oy in points if oname != name)
        if not dominated:
            keep.append((name, x, y))
    return sorted(keep, key=lambda p: p[1])


def _place_labels(jobs, placed=None, fontsize=8.5) -> list:
    """Greedy label placement over (axes, name, x, y) jobs.

    ONE collision list is threaded through every job, including jobs on a different axes of
    the same figure: the point clusters here are tight enough that two independent passes
    would happily stack two labels on the same pixels. Jobs are placed in the order given,
    so the caller puts the labels that must be readable (frontier, baseline, winner) first
    and lets the crowded remainder move.
    """
    placed = [] if placed is None else placed
    # Ordered by preference: right/left first (least vertical displacement), then further out.
    candidates = ((9, 9), (9, -12), (-9, 9), (-9, -12), (9, 24), (-9, 24), (9, -26),
                  (-9, -26), (9, 38), (-9, 38))
    for ax, name, x, y in jobs:
        fig = ax.figure
        fig.canvas.draw()  # transforms must be current before measuring text extents
        for dx, dy in candidates:
            txt = ax.annotate(name, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
                              fontsize=fontsize, color=INK,
                              ha="left" if dx > 0 else "right", va="center", zorder=6)
            bb = txt.get_window_extent(renderer=fig.canvas.get_renderer())
            box = (bb.x0 - 2, bb.y0 - 1, bb.x1 + 2, bb.y1 + 1)
            if not any(box[0] < p[2] and p[0] < box[2] and box[1] < p[3] and p[1] < box[3]
                       for p in placed):
                placed.append(box)
                break
            txt.remove()
        else:
            # Every candidate collided; keep the label anyway rather than dropping a method.
            txt = ax.annotate(name, xy=(x, y), xytext=(9, 7), textcoords="offset points",
                              fontsize=fontsize - 0.5, color=INK_MUTED, ha="left",
                              va="center", zorder=6)
            bb = txt.get_window_extent(renderer=fig.canvas.get_renderer())
            placed.append((bb.x0, bb.y0, bb.x1, bb.y1))
    return placed


def fig6_quality_latency(entries, style: Style, keys, baseline, winner, out_dir,
                         written) -> None:
    points = [(k, float(entries[k].metrics["latency_p95_ms"]),
               float(entries[k].metrics["hit@1"])) for k in keys]
    if not points:
        die("no entry has a latency_p95_ms value; cannot draw fig6")
    frontier = pareto_frontier(points)
    frontier_names = {p[0] for p in frontier}

    # A no-op reranker's P95 is ~1e-4 ms. Putting it on the same log axis as a 20 s
    # cross-encoder would add five empty decades and squash the region that matters, so the
    # zero-cost column is split off behind an axis break instead of being clipped away.
    zero = [p for p in points if p[1] < NO_COST_MS]
    rest = [p for p in points if p[1] >= NO_COST_MS]
    if not rest:
        die("every entry has a P95 latency below "
            f"{NO_COST_MS} ms; nothing to place on a latency axis")

    fig = plt.figure(figsize=(13.6, 6.2), layout="constrained")
    if zero:
        gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 15.0], wspace=0.035)
        ax0 = fig.add_subplot(gs[0, 0])
        ax = fig.add_subplot(gs[0, 1], sharey=ax0)
    else:
        ax0 = None
        ax = fig.add_subplot(fig.add_gridspec(1, 1)[0, 0])

    def draw_points(target, pts):
        for name, x, y in pts:
            emphasised = name in (baseline, winner)
            target.scatter(x, y, s=200 if emphasised else 115,
                           color=style.color.get(name, NEUTRAL),
                           marker=style.marker.get(name, "o"),
                           edgecolor=INK if emphasised else SURFACE,
                           linewidth=1.8 if emphasised else 1.4, zorder=5)

    xs = [p[1] for p in rest]
    lo_x = min(xs) / 2.2
    hi_x = max(xs) * 2.6
    tidy(ax, grid_axis="both")
    ax.set_xscale("log")
    ax.set_xlim(lo_x, hi_x)
    ax.set_ylim(0, 1.0)
    draw_points(ax, rest)

    # Frontier: the step enters from the left edge at the height of the fastest kept point,
    # so the segment that lives in the zero-cost column is still visible as a level.
    fpts = [p for p in frontier if p[1] >= NO_COST_MS]
    fzero = [p for p in frontier if p[1] < NO_COST_MS]
    fx = ([lo_x] if fzero else []) + [p[1] for p in fpts]
    fy = ([max(p[2] for p in fzero)] if fzero else []) + [p[2] for p in fpts]
    if fx:
        ax.plot(fx, fy, drawstyle="steps-post", color=INK_MUTED, linewidth=1.4,
                linestyle=(0, (5, 3)), zorder=2)

    if LATENCY_BUDGET_MS < hi_x:
        ax.axvline(LATENCY_BUDGET_MS, color=C_SEVERE, linewidth=1.6, linestyle=(0, (4, 2)),
                   zorder=3)
        ax.annotate(f"{LATENCY_BUDGET_MS:.0f} ms production budget\n"
                    f"← usable            too slow →",
                    xy=(LATENCY_BUDGET_MS, 0.035), xytext=(6, 0),
                    textcoords="offset points", fontsize=8.5, color=C_SEVERE,
                    ha="left", va="bottom", fontweight="bold")
    ax.set_xlabel("P95 reranking latency per query (ms, LOG scale)")
    ax.set_title("Reranking quality against its latency cost")

    if ax0 is not None:
        tidy(ax0, grid_axis="y")
        draw_points(ax0, zero)
        if fzero:
            ax0.plot([-1, 1], [max(p[2] for p in fzero)] * 2, color=INK_MUTED,
                     linewidth=1.4, linestyle=(0, (5, 3)), zorder=2)
        ax0.set_xlim(-1, 1)
        ax0.set_xticks([0], ["≈0"])
        ax0.set_ylim(0, 1.0)
        ax0.set_ylabel("Hit@1 (fraction of queries)")
        ax0.annotate("no reranker\n(< %.2g ms)" % NO_COST_MS, xy=(0, 0.035),
                     xytext=(0, 0), textcoords="offset points", fontsize=8.5,
                     color=INK_MUTED, ha="center", va="bottom")
        ax.tick_params(labelleft=False)
        ax.spines["left"].set_visible(False)
        # Axis-break marks on the x spine only -- that is the axis being broken. Drawn at
        # the top edge as well they read as stray marks, since there is no top spine.
        kw = dict(marker=[(-1, -2.2), (1, 2.2)], markersize=8, linestyle="none",
                  color=INK_MUTED, mec=INK_MUTED, mew=1.1, clip_on=False)
        ax0.plot([1], [0], transform=ax0.transAxes, **kw)
        ax.plot([0], [0], transform=ax.transAxes, **kw)
    else:
        ax.set_ylabel("Hit@1 (fraction of queries)")

    handles = [
        Line2D([0], [0], color=INK_MUTED, lw=1.4, linestyle=(0, (5, 3)),
               label="Pareto frontier:\nnothing is both faster\nand more accurate"),
        Line2D([0], [0], color=C_SEVERE, lw=1.6, linestyle=(0, (4, 2)),
               label=f"{LATENCY_BUDGET_MS:.0f} ms latency budget"),
        Line2D([0], [0], marker=style.marker.get(baseline, "o"), color="none",
               markerfacecolor=style.color.get(baseline, NEUTRAL), markeredgecolor=INK,
               markersize=11, label=f"{baseline}\n(baseline, no reranking)"),
    ]
    if winner != baseline:
        handles.append(Line2D([0], [0], marker=style.marker.get(winner, "o"), color="none",
                              markerfacecolor=style.color.get(winner, NEUTRAL),
                              markeredgecolor=INK, markersize=11,
                              label=f"{winner}\n(winner)"))
    on_frontier = ", ".join(n for n, _, _ in frontier)
    side_legend(fig, handles=handles, labels=None)
    footnote(fig, "P95 of the per-query reranking call, measured in the run itself; one-off "
                  "model load time is excluded (it is reported as model_load_seconds in "
                  "reranker_metrics.json and can reach several minutes). The x axis is "
                  "logarithmic -- equal distances are equal RATIOS, not equal milliseconds. "
                  f"On the frontier: {on_frontier}. Marks with an ink ring are the baseline "
                  "and the winner.")
    # Priority order matters: the frontier, the baseline and the winner are the marks the
    # report points at, so they get first pick of the free space.
    def priority(p):
        return (p[0] not in (baseline, winner), p[0] not in frontier_names, -p[2])

    jobs = [(ax0, *p) for p in sorted(zero, key=priority)] if zero else []
    jobs += [(ax, *p) for p in sorted(rest, key=priority)]
    _place_labels(jobs)
    save(fig, out_dir, "fig6_quality_latency", written)


# --------------------------------------------------------------------------------------
# figure 7 -- Hit@1 delta by query subgroup
# --------------------------------------------------------------------------------------


def fig7_subgroup_delta(entries, style: Style, keys, baseline, out_dir, written,
                        metric: str = "hit@1", min_n: int = 1) -> None:
    base = entries[baseline]
    base_by_q = {r["query_id"]: r for r in base.rows}
    groups = Counter()
    for r in base.rows:
        groups.update(set(r["subgroups"]))
    subgroups = [g for g, n in sorted(groups.items()) if n >= min_n]
    if not subgroups:
        die(f"no subgroup in {base.rankings_path} has at least {min_n} queries; fig7 needs "
            f"the 'subgroups' field to be populated")

    methods = [k for k in keys if k != baseline]
    if not methods:
        die("fig7 needs at least one non-baseline entry at the primary candidate depth")

    # Column 0 is every query, so each row can be read against its own overall effect (the
    # same number fig3 plots) instead of against nothing.
    columns: list[tuple[str, str | None]] = [("all queries", None)]
    columns += [(g, g) for g in subgroups]

    grid = np.full((len(methods), len(columns)), np.nan)
    ns: dict[str, int] = {}
    for i, key in enumerate(methods):
        rows = entries[key].rows
        for j, (label, g) in enumerate(columns):
            b_vals, m_vals = [], []
            for r in rows:
                if g is not None and g not in r["subgroups"]:
                    continue
                br = base_by_q.get(r["query_id"])
                if br is None:
                    continue  # scored on a different query set; excluded from this cell
                if metric not in r or metric not in br:
                    die(f"{entries[key].rankings_path}: query {r['query_id']} has no "
                        f"'{metric}' column")
                m_vals.append(float(r[metric]))
                b_vals.append(float(br[metric]))
            ns[label] = max(ns.get(label, 0), len(b_vals))
            if b_vals:
                grid[i, j] = float(np.mean(m_vals) - np.mean(b_vals))

    # Rows ordered by overall effect, best at the top -- the same ordering convention as
    # figs 3 and 5, so a reader moving between them meets the rerankers in the same order.
    row_order = sorted(range(len(methods)),
                       key=lambda i: (-grid[i, 0] if np.isfinite(grid[i, 0]) else 0.0,
                                      methods[i]))
    methods = [methods[i] for i in row_order]
    grid = grid[row_order, :]

    fig_w = max(9.0, 0.92 * len(columns) + 5.4)
    fig_h = max(4.0, 0.44 * len(methods) + 3.4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), layout="constrained")

    finite = grid[np.isfinite(grid)]
    vmax = float(np.max(np.abs(finite))) if finite.size else 0.01
    vmax = max(vmax, 0.01)
    # Diverging, pinned symmetrically about zero: white IS "no change", so a regression is a
    # different hue rather than a paler shade of the improvement colour.
    im = ax.imshow(grid, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(columns)),
                  [f"{lab}\n(n={ns.get(lab, 0)})" for lab, _ in columns],
                  rotation=40, ha="right", rotation_mode="anchor")
    ax.get_xticklabels()[0].set_fontweight("bold")
    ax.set_yticks(range(len(methods)), [style.label(k) for k in methods])
    for tick, key in zip(ax.get_yticklabels(), methods):
        if key == style.winner:
            tick.set_fontweight("bold")
    ax.axvline(0.5, color=SURFACE, linewidth=5, zorder=4)   # separates the overall column

    for i in range(len(methods)):
        for j in range(len(columns)):
            v = grid[i, j]
            if not np.isfinite(v):
                ax.text(j, i, "-", ha="center", va="center", fontsize=8.5, color=INK_MUTED)
                continue
            ax.text(j, i, f"{v:+.2f}".replace("+0.00", "0.00").replace("-0.00", "0.00"),
                    ha="center", va="center", fontsize=8,
                    color="#FFFFFF" if abs(v) > 0.62 * vmax else INK)

    ax.set_xticks(np.arange(-0.5, len(columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(methods), 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)

    cbar = fig.colorbar(im, ax=ax, shrink=0.86, pad=0.015)
    cbar.set_label(f"Δ {metric} vs baseline")
    cbar.outline.set_visible(False)
    ax.set_title(f"Where reranking helps and where it hurts ({metric} vs '{baseline}')")
    ax.set_xlabel("Query subgroup (first column = all queries, the overall effect)")
    footnote(fig, f"Cell = mean {metric} of that reranker minus the baseline's mean on the "
                  f"same subgroup's queries; rows are ordered by the overall effect, best "
                  f"first. The colour scale is diverging and centred exactly at 0 with "
                  f"symmetric limits (±{vmax:.3f}), so red cells are real regressions and "
                  f"not a lighter shade of blue -- but one badly-behaved reranker sets that "
                  f"range, which is why the well-behaved rows look pale; read their printed "
                  f"values. Subgroups overlap (a query carries several tags) and small-n "
                  f"subgroups are noisy -- n is in the tick label. '-' = the reranker "
                  f"scored no query in that subgroup.")
    save(fig, out_dir, "fig7_subgroup_delta", written)


# --------------------------------------------------------------------------------------
# figure 9 -- candidate depth: quality and latency (fig8 belongs to another section)
# --------------------------------------------------------------------------------------


def fig9_candidate_depth(entries, style: Style, ceiling, all_keys, baseline, out_dir,
                         written) -> None:
    by_name: dict[str, dict[int, str]] = {}
    for key in all_keys:
        by_name.setdefault(entries[key].name, {})[entries[key].depth] = key
    multi = {n: d for n, d in by_name.items() if len(d) >= 2}

    depths_all = sorted({d for d in ceiling} | {entries[k].depth for k in all_keys})
    xpos = {d: i for i, d in enumerate(depths_all)}

    fig = plt.figure(figsize=(12.6, 7.4), layout="constrained")
    gs = fig.add_gridspec(2, 1, height_ratios=[1.25, 1.0], hspace=0.06)
    ax_q = fig.add_subplot(gs[0, 0])
    ax_l = fig.add_subplot(gs[1, 0], sharex=ax_q)
    tidy(ax_q, grid_axis="both")
    tidy(ax_l, grid_axis="both")

    cx = [xpos[d] for d in sorted(ceiling)]
    cy = [ceiling[d]["oracle_hit@1"] for d in sorted(ceiling)]
    ax_q.plot(cx, cy, color=ORACLE, linestyle=(0, (1, 1.6)), linewidth=2.0, marker="_",
              markersize=13, markeredgewidth=2.0, zorder=4,
              label="oracle Hit@1 (perfect reranking\nof a pool this deep)")
    for x, y, d in zip(cx, cy, sorted(ceiling)):
        ax_q.annotate(f"{y:.3f}", xy=(x, y), xytext=(0, 7), textcoords="offset points",
                      fontsize=8, color=ORACLE, ha="center", va="bottom")

    if not multi:
        ax_q.annotate("no reranker in this run set was scored at more than one candidate "
                      "depth,\nso only the oracle ceiling can be drawn against depth",
                      xy=(0.5, 0.45), xycoords="axes fraction", fontsize=10,
                      color=INK_MUTED, ha="center", va="center")
    free = []
    for name, depth_map in sorted(multi.items()):
        ds = sorted(depth_map)
        keys = [depth_map[d] for d in ds]
        colour = style.color.get(keys[0], NEUTRAL)
        marker = style.marker.get(keys[0], "o")
        dash = style.dash.get(keys[0], "-")
        lw = 2.6 if style.winner in keys else 2.0
        ax_q.plot([xpos[d] for d in ds], [entries[k].metrics["hit@1"] for k in keys],
                  color=colour, marker=marker, markersize=7, linestyle=dash, linewidth=lw,
                  markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=3, label=name)
        lat = [entries[k].metrics["latency_p95_ms"] for k in keys]
        # A no-op reranker costs ~1e-4 ms. On a shared log axis it would add five empty
        # decades and flatten every real curve, so it is named in the footnote instead.
        if max(lat) < NO_COST_MS:
            free.append(name)
            continue
        ax_l.plot([xpos[d] for d in ds], lat, color=colour, marker=marker, markersize=7,
                  linestyle=dash, linewidth=lw, markeredgecolor=SURFACE, markeredgewidth=1.0,
                  zorder=3, label=name)

    ax_q.set_ylim(0, 1.0)
    ax_q.set_ylabel("Hit@1 (fraction of queries)")
    ax_q.set_title("Is a deeper candidate pool worth its cost?")
    ax_q.tick_params(labelbottom=False)

    ax_l.set_yscale("log")
    ax_l.axhline(LATENCY_BUDGET_MS, color=C_SEVERE, linewidth=1.5, linestyle=(0, (4, 2)),
                 zorder=2)
    ax_l.annotate(f"{LATENCY_BUDGET_MS:.0f} ms production budget",
                  xy=(0.995, LATENCY_BUDGET_MS), xycoords=("axes fraction", "data"),
                  xytext=(0, 4), textcoords="offset points", fontsize=8.5, color=C_SEVERE,
                  ha="right", va="bottom", fontweight="bold")
    ax_l.set_ylabel("P95 latency per query\n(ms, LOG scale)")
    ax_l.set_xticks(list(xpos.values()), [str(d) for d in depths_all])
    ax_l.set_xlim(-0.25, len(depths_all) - 0.75)
    ax_l.set_xlabel("Candidate depth (documents handed to the reranker)")

    handles, labels = ax_q.get_legend_handles_labels()
    side_legend(fig, handles=handles, labels=labels)
    ceiling_gain = (max(r["oracle_hit@1"] for r in ceiling.values())
                    - min(r["oracle_hit@1"] for r in ceiling.values()))
    note = (f"Top: the ceiling rises only {ceiling_gain:+.3f} Hit@1 across the whole depth "
            f"range in candidate_ceiling.csv, so deeper pools add very little headroom; the "
            f"measured rerankers (coloured) show what is actually realised. Bottom: the same "
            f"depths on a log latency axis. A depth with no marker for a given reranker was "
            f"not scored at that depth in these run dirs. Latency excludes one-off model "
            f"load.")
    if free:
        note += (f" Omitted from the latency panel because their P95 is below "
                 f"{NO_COST_MS} ms at every depth (they would add five empty decades to the "
                 f"log axis): {', '.join(free)}.")
    footnote(fig, note)
    save(fig, out_dir, "fig9_candidate_depth", written)


# --------------------------------------------------------------------------------------
# figure 10 -- remaining headroom against the oracle
# --------------------------------------------------------------------------------------


def fig10_oracle_gap(entries, style: Style, finalists, ceiling_row, out_dir, written) -> None:
    pairs = [("hit@1", "oracle_hit@1", "Hit@1"), ("recall@3", "oracle_recall@3", "Recall@3")]
    usable = [(m, o, lab) for m, o, lab in pairs if o in ceiling_row]
    if not usable:
        die("the ceiling file carries neither oracle_hit@1 nor oracle_recall@3; fig10 has "
            "nothing to measure headroom against")

    order = sorted(finalists, key=lambda k: -entries[k].metrics["hit@1"])
    x = np.arange(len(order))
    width = min(0.8 / len(usable), 0.34)
    fig, ax = plt.subplots(figsize=(max(10.5, 1.5 * len(order) + 6.0), 5.8),
                           layout="constrained")
    tidy(ax, grid_axis="y")

    hatches = (None, "///")
    for i, (mcol, ocol, lab) in enumerate(usable):
        offs = (i - (len(usable) - 1) / 2) * width
        gaps = [ceiling_row[ocol] - entries[k].metrics[mcol] for k in order]
        ax.bar(x + offs, gaps, width * 0.9, zorder=3,
               color=[style.color.get(k, NEUTRAL) for k in order],
               hatch=hatches[i % len(hatches)],
               edgecolor=[INK if k == style.winner else SURFACE for k in order],
               linewidth=[1.4 if k == style.winner else 0.8 for k in order])
        for xi, g, k in zip(x + offs, gaps, order):
            ax.annotate(f"{g:.3f}\n({entries[k].metrics[mcol]:.3f}→{ceiling_row[ocol]:.3f})",
                        xy=(xi, g), xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, color=INK)

    ax.set_xticks(x, ["\n".join(textwrap.wrap(style.label(k), 24, break_long_words=False))
                      for k in order])
    ax.set_ylim(0, max(0.02, max(ceiling_row[o] - entries[k].metrics[m]
                                 for m, o, _ in usable for k in order) * 1.30))
    ax.set_ylabel("Oracle − achieved (absolute, lower is better)")
    ax.set_xlabel("Reranker")
    ax.set_title("Headroom left on the table: how far each reranker sits below the oracle")
    # The bars carry the hatch in the surface colour, so the swatch has to as well -- a
    # swatch whose edge matches its fill shows no hatch at all.
    handles = [Patch(facecolor="#9E9E9E", edgecolor=SURFACE, linewidth=0.8,
                     hatch=hatches[i % len(hatches)], label=lab)
               for i, (_m, _o, lab) in enumerate(usable)]
    side_legend(fig, handles=handles, labels=None)
    depth = int(ceiling_row["candidate_depth"])
    footnote(fig, f"Oracle = the best achievable by reordering the SAME depth-{depth} "
                  f"candidate pool (candidate_ceiling.csv), so this gap is the part a better "
                  f"reranker could still close; the part that needs a better first stage is "
                  f"not counted here. Bar colour = reranker (shared across figures); the "
                  f"hatch distinguishes the two metrics. Each label shows achieved → "
                  f"oracle.")
    save(fig, out_dir, "fig10_oracle_gap", written)


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------


def default_out_dir(run_dirs: list[Path]) -> Path:
    if len(run_dirs) == 1:
        return run_dirs[0] / "figures"
    try:
        common = Path(os.path.commonpath([str(p) for p in run_dirs]))
    except ValueError:
        return run_dirs[0] / "figures"
    return common / "figures"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Draw the reranking-benchmark figures from one or more run directories.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("run_dirs", nargs="+",
                    help="reranking run directories written by run_reranking_lab.py")
    ap.add_argument("--ceiling", required=True,
                    help="path to candidate_ceiling.csv (oracle_metrics.json must sit "
                         "beside it)")
    ap.add_argument("--baseline", default="original_ranking@10",
                    help="entry used as the comparison baseline "
                         "(default: original_ranking@10)")
    ap.add_argument("--out-dir", default=None,
                    help="default: <run_dir>/figures, or <common parent>/figures")
    ap.add_argument("--highlight", default=None,
                    help="entry to mark as the winner (default: best Hit@1)")
    ap.add_argument("--max-finalists", type=int, default=5,
                    help="max series in the line figures 1, 2 and 10 (default: 5)")
    ap.add_argument("--min-queries-frac", type=float, default=0.5,
                    help="drop entries scored on fewer than this fraction of the largest "
                         "run's queries from the aggregate figures (default: 0.5)")
    args = ap.parse_args()

    run_dirs = [Path(p).expanduser().resolve() for p in args.run_dirs]
    entries = merge_runs(run_dirs)
    ceiling, oracle_metrics = load_ceiling(Path(args.ceiling).expanduser().resolve())

    if args.baseline not in entries:
        die(f"baseline '{args.baseline}' is not among the loaded entries.\n"
            f"  Available: {', '.join(sorted(entries))}\n"
            f"  Pass --baseline with one of those, or add the run dir that contains it.")
    primary_depth = entries[args.baseline].depth
    if primary_depth not in ceiling:
        die(f"candidate_ceiling.csv has no row for candidate_depth={primary_depth} (the "
            f"baseline's depth). It has {sorted(ceiling)}. Re-run candidate_ceiling.py with "
            f"--depths including {primary_depth}.")
    ceiling_row = ceiling[primary_depth]

    # Cross-check the ceiling against the un-reranked baseline: oracle_metrics.json's
    # 'achieved' block IS the first-stage ranking, so if the two disagree the ceiling files
    # came from a different first-stage run and every headroom number would be wrong. Only
    # meaningful when the baseline really is the passthrough ranking.
    if entries[args.baseline].family == "baseline":
        achieved = oracle_metrics.get("achieved", {})
        for metric in ("hit@1", "recall@3"):
            got = entries[args.baseline].metrics.get(metric)
            exp = achieved.get(metric)
            if exp is not None and got is not None and not math.isclose(got, exp,
                                                                        abs_tol=5e-4):
                print(f"  WARNING: baseline {metric}={got:.4f} but oracle_metrics.json "
                      f"reports achieved {metric}={exp:.4f}. The ceiling files look like "
                      f"they come from a different first-stage run than these reranking "
                      f"runs; headroom numbers would be misleading. Continuing, but check "
                      f"the inputs.")

    # A run scored on a fraction of the benchmark cannot be drawn next to a full one without
    # reading as a result, so it is dropped from every figure and named on the console.
    n_ref = max(e.n_queries for e in entries.values())
    partial = {k for k, e in entries.items() if e.n_queries < args.min_queries_frac * n_ref}
    for k in sorted(partial):
        print(f"  note: '{k}' was scored on {entries[k].n_queries} queries (largest run has "
              f"{n_ref}); excluded from the figures. Use --min-queries-frac 0 to include it.")
    usable = canonical_order(entries, [k for k in entries if k not in partial])
    primary = [k for k in usable if entries[k].depth == primary_depth]
    if args.baseline not in primary:
        die(f"baseline '{args.baseline}' was excluded as a partial run; cannot continue")

    winner = pick_winner(entries, primary, args.baseline, args.highlight)
    if entries[winner].depth != primary_depth:
        print(f"  note: --highlight '{winner}' is at depth {entries[winner].depth}, while the "
              f"baseline is at depth {primary_depth}; it will appear only in fig9.")
    finalists = select_finalists(entries, primary, args.baseline, winner, args.max_finalists)
    style = Style(entries, list(entries), finalists, args.baseline, winner, n_ref)
    out_dir = (Path(args.out_dir).expanduser().resolve() if args.out_dir
               else default_out_dir(run_dirs))

    for key in set(primary) | {winner}:
        load_rows(entries[key])

    print(f"runs:        {', '.join(str(p) for p in run_dirs)}")
    print(f"entries:     {len(entries)} ({len(primary)} at the primary depth "
          f"{primary_depth})")
    print(f"ceiling:     {args.ceiling}  (depths {sorted(ceiling)})")
    print(f"baseline:    {args.baseline}  (Hit@1 = {entries[args.baseline].metrics['hit@1']:.4f})")
    print(f"winner:      {winner}  (Hit@1 = {entries[winner].metrics['hit@1']:.4f})"
          f"{'  [--highlight]' if args.highlight else ''}")
    print(f"finalists:   {', '.join(finalists)}")
    print(f"out:         {out_dir}\n")

    apply_rc()
    boot = Bootstrap()
    improvements = compute_improvements(entries, primary, args.baseline, boot)

    written: list[Path] = []
    fig1_rerank_recall_at_k(entries, style, finalists, ceiling_row, out_dir, written)
    fig2_gold_rank_ecdf(entries, style, finalists, out_dir, written)
    fig3_hit1_recall3_improvement(entries, style, improvements, args.baseline, out_dir,
                                  written)
    fig4_rank_shift(entries, style, winner, out_dir, written)
    fig5_win_tie_loss(entries, style, primary, args.baseline, out_dir, written)
    fig6_quality_latency(entries, style, primary, args.baseline, winner, out_dir, written)
    fig7_subgroup_delta(entries, style, primary, args.baseline, out_dir, written)
    fig9_candidate_depth(entries, style, ceiling, usable, args.baseline, out_dir, written)
    fig10_oracle_gap(entries, style, finalists, ceiling_row, out_dir, written)

    print(f"wrote {len(written)} files to {out_dir}:")
    for path in written:
        print(f"  {path.name:38s} {path.stat().st_size:>9,d} bytes")


if __name__ == "__main__":
    main()
