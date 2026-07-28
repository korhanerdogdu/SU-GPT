#!/usr/bin/env python3
from __future__ import annotations

"""
Publication figures for a retrieval-lab run.

Reads ONLY the machine-readable artifacts that analyze_retrieval_lab.py writes into a run
directory and draws seven figures from them. No metric is typed in here: every plotted
number is parsed out of a CSV/JSON file, and a missing file or column is a hard error
rather than a zero, so a figure can never quietly show a placeholder.

INPUTS (all under <run_dir>/, produced by analyze_retrieval_lab.py)
    retrieval_metrics.csv     one row per method: recall@K, mrr@10, ndcg@10, hit@1,
                              latency_{mean,p50,p95}_ms, index build cost
    subgroup_metrics.csv      method x subgroup x metric
    significance_results.json {method: {metric: {delta, ci_low, ci_high, p_value,
                              significant, improved, harmed, tied, n, ...}}}
    per_query_results.csv     one row per (method, query): per-query metrics + the
                              pipe-separated `subgroups` tag list

OUTPUTS (under <out_dir>/, default <run_dir>/figures/), each as .png (200 dpi) + .svg + .pdf
    fig1_recall_at_k        Recall@K curves for the finalist methods
    fig2_delta_vs_bm25      paired delta in recall@10 vs the baseline, with 95% CIs
    fig3_quality_latency    recall@10 vs P95 latency, with the Pareto frontier
    fig4_subgroups          recall@10 heatmap, method x subgroup
    fig5_ranking_metrics    MRR@10 / nDCG@10 / Hit@1 for the finalists
    fig6_win_tie_loss       per-query improved / tied / harmed counts vs the baseline
    fig7_error_categories   why the baseline and the winner fail, by category

USAGE
    python server/evaluation/make_retrieval_figures.py <run_dir> \
        [--baseline bm25_full_corpus] [--out-dir <run_dir>/figures] [--highlight <method>]


--- HOW FIGURE 7's CATEGORIES ARE DERIVED (documented because it is the one figure whose
    numbers are not read straight out of a column) ---

Population: rows of per_query_results.csv with recall@10 == 0 ("the method failed this
query"), restricted to two methods: the baseline and the winner.

A failed query is counted into every category it matches, so the bars sum to at least the
number of failures -- they are overlapping tags, not a partition. Categories:

  * one bucket per tag in FAILURE_TAGS that appears in that row's `subgroups` field
    (pipe-separated). FAILURE_TAGS = cross_language, hard_negatives, multi_evidence,
    has_catalog_year, minor, has_course_code. A tag that occurs nowhere in the file is
    dropped from the chart instead of being drawn as an empty bar.
  * no_gold_in_top50 -- added when recall@50 is also 0, i.e. no gold chunk appears anywhere
    in the 50-document candidate list. This one is computed from a column, not a tag,
    because it separates "ranked badly" from "never retrieved at all" (the latter cannot be
    fixed by reranking).
  * uncategorized -- a failure that matched none of the above, so that every failed query is
    visible somewhere and the totals can be audited against the raw CSV.


--- DESIGN CHOICES ---

Palette: Okabe-Ito, ordered blue / vermillion / bluish-green / orange / reddish-purple /
sky-blue. That specific ORDER matters: it was run through a colour-blind-separation check
and is the ordering whose worst adjacent pair stays above the deuteranopia dE floor (the
canonical ordering puts reddish-purple next to bluish-green, which does not). Black and
yellow from the Okabe-Ito set are unused: black reads as an axis, yellow is illegible on
white. Every series is also given its own marker and dash pattern, so identity never rests
on hue alone.

Colour follows the method, not its rank: one map is built once and reused by every figure,
so a method keeps its colour whether or not other methods are in the frame. Only the
finalists are given a hue; the remaining methods share one neutral grey. Past ~6 hues the
reader cannot hold the mapping anyway, and in the figures where every method appears
(2, 4, 6) identity is carried by the axis label, which is unambiguous.

Figures 6 and 7 are the deliberate exception to "colour = method": there, colour encodes
outcome class (improved/tied/harmed) and method identity comes from the axis label or from
a two-series legend that still uses the shared map.

Sequential colour (figure 4) is viridis: perceptually uniform and colour-blind-safe, unlike
rainbow/jet, whose bright bands invent boundaries that are not in the data.

Recall and every other 0-1 metric is drawn on an axis anchored at 0, and the heatmap colour
scale spans the full 0-1 proportion range, so a regression cannot be hidden by a cropped
axis or a clipped colour ramp.
"""

import argparse
import csv
import json
import math
import textwrap
from collections import Counter, OrderedDict
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

K_VALUES: tuple[int, ...] = (1, 3, 5, 10, 20, 50)
HEADLINE_K = 10  # the K the report leads with; marked explicitly in fig 1

# Okabe-Ito in the validated order (see module docstring).
PALETTE: tuple[str, ...] = ("#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9")
NEUTRAL = "#7A7A7A"          # non-finalist methods
INK = "#1A1A1A"              # primary text
INK_MUTED = "#5A5A5A"        # secondary text
GRID = "#DCDCDC"             # hairline grid, one shade off the surface
SURFACE = "#FFFFFF"

# outcome classes for fig 6 (status colours, used only where they mean status)
C_IMPROVED = "#009E73"
C_TIED = "#C4C4C4"
C_HARMED = "#D55E00"

MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")
DASHES = ("-", "--", "-.", ":", (0, (5, 1, 1, 1)), (0, (3, 1, 3, 1, 1, 1)))

# Family detection is substring-based on the method name. Order matters: a
# "hybrid_rerank" method is a reranker first and a hybrid second, so rerank is tested first.
FAMILY_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("rerank", ("rerank", "cross_encoder", "crossencoder", "monot5", "_ce", "ce_")),
    ("hybrid", ("hybrid", "rrf", "fusion", "fuse", "ensemble", "combo", "blend", "+dense")),
    ("dense", ("dense", "embed", "vector", "e5", "bge", "gte", "minilm", "mpnet",
               "openai", "ada", "nomic", "jina", "qwen", "labse", "sbert", "colbert")),
    ("sparse", ("bm25", "tfidf", "tf_idf", "splade", "sparse", "lexical", "keyword")),
)
FAMILY_ORDER = ("sparse", "dense", "hybrid", "rerank", "other")

FAILURE_TAGS: tuple[str, ...] = (
    "cross_language", "hard_negatives", "multi_evidence",
    "has_catalog_year", "minor", "has_course_code",
)

REQUIRED_FILES = (
    "retrieval_metrics.csv",
    "subgroup_metrics.csv",
    "significance_results.json",
    "per_query_results.csv",
)


# --------------------------------------------------------------------------------------
# loading -- every failure here is fatal and says exactly what is wrong
# --------------------------------------------------------------------------------------


def die(msg: str) -> NoReturn:
    raise SystemExit(f"make_retrieval_figures: ERROR: {msg}")


def _require_file(run_dir: Path, name: str) -> Path:
    path = run_dir / name
    if not path.is_file():
        die(
            f"required input '{name}' not found in {run_dir}.\n"
            f"  Generate it first:\n"
            f"    python server/evaluation/analyze_retrieval_lab.py {run_dir} --baseline <baseline>"
        )
    return path


def _require_columns(header: list[str] | None, required: tuple[str, ...], path: Path) -> None:
    if header is None:
        die(f"{path} is empty (no header row)")
    missing = [c for c in required if c not in header]
    if missing:
        die(f"{path} is missing required column(s): {missing}. Found: {header}")


def _num(row: dict, col: str, path: Path, *, allow_blank: bool = False) -> float:
    """Parse a numeric cell, refusing to silently substitute 0 for junk."""
    raw = row.get(col, "")
    if raw is None or str(raw).strip() == "":
        if allow_blank:
            return float("nan")
        die(f"{path}: empty value for required numeric column '{col}' (row: {row.get('method')})")
    try:
        return float(raw)
    except ValueError:
        die(f"{path}: value {raw!r} in column '{col}' is not a number")


def load_retrieval_metrics(run_dir: Path) -> "OrderedDict[str, dict]":
    path = _require_file(run_dir, "retrieval_metrics.csv")
    required = ("method", "n_queries", "recall@10", "mrr@10", "ndcg@10", "hit@1",
                "latency_p95_ms")
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        _require_columns(reader.fieldnames, required, path)
        rows = list(reader)
    if not rows:
        die(f"{path} has a header but no method rows")

    out: "OrderedDict[str, dict]" = OrderedDict()
    for row in rows:
        method = (row.get("method") or "").strip()
        if not method:
            die(f"{path}: a row has an empty 'method' value")
        rec: dict = {"method": method}
        for k in K_VALUES:
            col = f"recall@{k}"
            # recall@K columns beyond the required @10 may legitimately be absent in an
            # older run; carry NaN and let the figure drop that K rather than invent it.
            rec[col] = _num(row, col, path, allow_blank=True) if col in row else float("nan")
        for col in ("mrr@10", "ndcg@10", "hit@1", "latency_mean_ms", "latency_p50_ms",
                    "latency_p95_ms", "index_build_seconds", "index_bytes"):
            rec[col] = _num(row, col, path, allow_blank=True) if col in row else float("nan")
        rec["n_queries"] = int(_num(row, "n_queries", path))
        rec["build_notes"] = (row.get("build_notes") or "").strip()
        out[method] = rec
    return out


def load_subgroup_metrics(run_dir: Path) -> list[dict]:
    path = _require_file(run_dir, "subgroup_metrics.csv")
    required = ("method", "subgroup", "n_queries", "recall@10")
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        _require_columns(reader.fieldnames, required, path)
        rows = []
        for row in reader:
            rows.append({
                "method": (row["method"] or "").strip(),
                "subgroup": (row["subgroup"] or "").strip(),
                "n_queries": int(_num(row, "n_queries", path)),
                "recall@10": _num(row, "recall@10", path),
            })
    if not rows:
        die(f"{path} has a header but no data rows")
    return rows


def load_significance(run_dir: Path) -> dict:
    path = _require_file(run_dir, "significance_results.json")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        die(f"{path} is not valid JSON: {exc}")
    if not isinstance(data, dict):
        die(f"{path} should be a mapping of method -> metric -> comparison")
    return data


def load_per_query(run_dir: Path) -> list[dict]:
    path = _require_file(run_dir, "per_query_results.csv")
    required = ("method", "query_id", "recall@10", "subgroups")
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        _require_columns(reader.fieldnames, required, path)
        fields = list(reader.fieldnames or [])
        rows = []
        for row in reader:
            parsed: dict = {
                "method": (row["method"] or "").strip(),
                "query_id": (row["query_id"] or "").strip(),
                "subgroups": [t for t in (row.get("subgroups") or "").split("|") if t],
            }
            for col in [f"recall@{k}" for k in K_VALUES] + ["mrr@10", "ndcg@10", "hit@1"]:
                if col in fields:
                    parsed[col] = _num(row, col, path, allow_blank=True)
            rows.append(parsed)
    if not rows:
        die(f"{path} has a header but no data rows")
    return rows


# --------------------------------------------------------------------------------------
# method selection & shared styling
# --------------------------------------------------------------------------------------


def family_of(method: str) -> str:
    name = method.lower()
    for family, needles in FAMILY_PATTERNS:
        if any(n in name for n in needles):
            return family
    return "other"


def canonical_order(methods: list[str]) -> list[str]:
    """Stable display order: grouped by family, alphabetical inside a family.

    Deliberately NOT sorted by score -- the colour map is derived from this order, and a
    score-derived order would repaint every method whenever the numbers shift slightly.
    """
    return sorted(methods, key=lambda m: (FAMILY_ORDER.index(family_of(m)), m))


def pick_winner(metrics: dict, highlight: str | None) -> str:
    if highlight:
        if highlight not in metrics:
            die(f"--highlight '{highlight}' is not a method in retrieval_metrics.csv "
                f"(have: {sorted(metrics)})")
        return highlight
    # Winner = best recall@10; ties broken by mrr@10 then name, so the choice is reproducible.
    return max(metrics, key=lambda m: (metrics[m]["recall@10"], metrics[m]["mrr@10"], m))


def select_finalists(metrics: dict, baseline: str, winner: str, max_n: int) -> list[str]:
    """Baseline + winner + the best method of each family, topped up by recall@10.

    Chosen from the data, never from a hardcoded name list, so a run whose families differ
    (e.g. a sparse-only dev run) simply yields fewer picks instead of failing.
    """
    # dict keys, not a list: baseline and winner can be the same method, and a duplicate
    # here would draw the same series (and legend entry) twice.
    picks: dict[str, None] = dict.fromkeys([baseline, winner])
    by_family: dict[str, list[str]] = {}
    for m in metrics:
        by_family.setdefault(family_of(m), []).append(m)
    for family in FAMILY_ORDER:
        members = by_family.get(family)
        if not members:
            continue  # e.g. a sparse-only dev run has no dense/hybrid entry: skip it
        best = max(members, key=lambda m: (metrics[m]["recall@10"], m))
        picks.setdefault(best)
    # Top up with the next best methods so a run with one dominant family still gets a
    # readable multi-line figure rather than two lines.
    for m in sorted(metrics, key=lambda m: (-metrics[m]["recall@10"], m)):
        if len(picks) >= max_n:
            break
        picks.setdefault(m)
    return [m for m in canonical_order(list(picks)) if m in metrics][:max_n]


class Style:
    """One method -> one colour/marker/dash, built once and shared by every figure."""

    def __init__(self, all_methods: list[str], finalists: list[str], baseline: str, winner: str):
        self.baseline = baseline
        self.winner = winner
        self.order = canonical_order(all_methods)
        self.color: dict[str, str] = {}
        self.marker: dict[str, str] = {}
        self.dash: dict[str, object] = {}

        # Baseline and winner take the two highest-contrast slots: they are the pair the
        # reader must never confuse, and the brief requires them separable in every figure.
        reserved = [baseline, winner] if winner != baseline else [baseline]
        rest = [m for m in canonical_order(finalists) if m not in reserved]
        for i, m in enumerate(reserved + rest):
            self.color[m] = PALETTE[i % len(PALETTE)]
        for m in all_methods:
            self.color.setdefault(m, NEUTRAL)
        for i, m in enumerate(self.order):
            self.marker[m] = MARKERS[i % len(MARKERS)]
            self.dash[m] = DASHES[i % len(DASHES)]
        # The baseline is always dashed and the winner always solid-and-thick, so the two
        # are told apart in print and by a colour-blind reader without reading the legend.
        self.dash[baseline] = (0, (6, 2))
        self.dash[winner] = "-"

    def label(self, method: str) -> str:
        if method == self.baseline == self.winner:
            return f"{method}  (baseline, best)"
        if method == self.baseline:
            return f"{method}  (baseline)"
        if method == self.winner:
            return f"{method}  ★ winner"
        return method

    def linewidth(self, method: str) -> float:
        return 2.6 if method == self.winner else 2.0


# --------------------------------------------------------------------------------------
# confidence intervals computed from per-query data (never invented)
# --------------------------------------------------------------------------------------


def wilson_ci(successes: float, n: int, z: float = 1.959963985) -> tuple[float, float]:
    """95% Wilson interval for a 0/1 metric. Preferred over the normal approximation
    because recall@K sits near 0 or 1 for several methods here, where the normal
    interval runs outside [0, 1]."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def mean_ci(values: list[float], z: float = 1.959963985) -> tuple[float, float, float]:
    """Mean and normal-approximation 95% CI for a continuous [0,1] metric (MRR, nDCG)."""
    n = len(values)
    if n == 0:
        return (float("nan"),) * 3
    mean = sum(values) / n
    if n < 2:
        return (mean, mean, mean)
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    half = z * math.sqrt(var / n)
    return (mean, max(0.0, mean - half), min(1.0, mean + half))


def per_query_index(rows: list[dict]) -> dict[str, list[dict]]:
    idx: dict[str, list[dict]] = {}
    for r in rows:
        idx.setdefault(r["method"], []).append(r)
    return idx


def binary_ci(rows: list[dict], col: str) -> tuple[float, float] | None:
    vals = [r[col] for r in rows if col in r and not math.isnan(r[col])]
    if not vals:
        return None
    return wilson_ci(sum(vals), len(vals))


# --------------------------------------------------------------------------------------
# figure plumbing
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
        "grid.linestyle": "-",   # solid hairline: a dashed grid reads as a threshold
        "legend.frameon": False,
        "figure.dpi": 110,
    })


def tidy(ax, *, grid_axis: str = "y") -> None:
    """Recessive chrome: no top/right spines, one hairline grid behind the marks."""
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
    shrinking the layout rect first -- otherwise the caption lands on top of the x label.
    """
    wrap_at = max(60, int(fig.get_figwidth() * 17))   # ~17 chars per inch at 8 pt
    lines = textwrap.wrap(text, wrap_at)
    strip = (0.16 * len(lines) + 0.10) / fig.get_figheight()
    engine = fig.get_layout_engine()
    if engine is not None:
        engine.set(rect=(0.0, strip, 1.0, 1.0 - strip))
    fig.text(0.006, 0.006, "\n".join(lines), fontsize=8, color=INK_MUTED,
             ha="left", va="bottom")


def side_legend(fig, ax, handles=None, labels=None, **kwargs):
    """Legend in a reserved column to the right of the plot.

    Every figure here has at least one dense corner, and a legend box parked inside the
    axes eventually lands on a bar. 'outside right upper' makes constrained layout reserve
    the space instead, so the legend can never occlude data whatever the run contains.
    """
    if handles is None:
        handles, labels = ax.get_legend_handles_labels()
    if labels is None:
        return fig.legend(handles=handles, loc="outside right upper", **kwargs)
    return fig.legend(handles, labels, loc="outside right upper", **kwargs)


# --------------------------------------------------------------------------------------
# figure 1 -- Recall@K
# --------------------------------------------------------------------------------------


def fig1_recall_at_k(metrics, per_q, style: Style, finalists, out_dir, written) -> None:
    ks = [k for k in K_VALUES
          if any(not math.isnan(metrics[m].get(f"recall@{k}", float("nan"))) for m in finalists)]
    if not ks:
        die("retrieval_metrics.csv has no usable recall@K columns for the selected methods")

    fig, ax = plt.subplots(figsize=(11.0, 5.4), layout="constrained")
    # Evenly spaced categorical x: the K values (1,3,5,10,20,50) are far from uniform, and
    # on a true numeric axis the interesting low-K region collapses into the left margin.
    xs = list(range(len(ks)))
    tidy(ax, grid_axis="both")

    # Draw the winner last so its line sits on top where curves converge.
    draw_order = sorted(finalists, key=lambda m: (m == style.winner, metrics[m]["recall@10"]))
    for m in draw_order:
        ys = [metrics[m].get(f"recall@{k}", float("nan")) for k in ks]
        rows = per_q.get(m, [])
        lo, hi = [], []
        for k in ks:
            ci = binary_ci(rows, f"recall@{k}")
            lo.append(ci[0] if ci else float("nan"))
            hi.append(ci[1] if ci else float("nan"))
        if not all(math.isnan(v) for v in lo):
            ax.fill_between(xs, lo, hi, color=style.color[m], alpha=0.13, linewidth=0, zorder=1)
        ax.plot(xs, ys, color=style.color[m], marker=style.marker[m], markersize=7,
                linestyle=style.dash[m], linewidth=style.linewidth(m),
                markeredgecolor=SURFACE, markeredgewidth=1.0,
                label=style.label(m), zorder=3)

    if HEADLINE_K in ks:
        xk = ks.index(HEADLINE_K)
        ax.axvline(xk, color=INK_MUTED, linewidth=1.0, linestyle=(0, (2, 3)), zorder=2)
        # Annotated at the top: the curves converge there least often, and the bottom of
        # the plot is where a weak method's line lives.
        ax.annotate("K = 10\n(headline metric)", xy=(xk, 0.995), xytext=(6, -2),
                    textcoords="offset points",
                    fontsize=9, color=INK_MUTED, ha="left", va="top")

    ax.set_xticks(xs, [str(k) for k in ks])
    ax.set_xlabel("K (number of retrieved chunks considered)")
    ax.set_ylabel("Recall@K (fraction of queries)")
    ax.set_ylim(0, 1.0)  # proportion axis anchored at 0 -- never cropped to widen gaps
    ax.set_xlim(-0.15, len(ks) - 0.85)
    ax.set_title("Recall@K by retrieval method")
    handles, labels = ax.get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: -metrics[draw_order[i]]["recall@10"])
    side_legend(fig, ax, [handles[i] for i in order], [labels[i] for i in order])
    footnote(fig, "Recall@K = fraction of benchmark queries with at least one gold chunk in "
                  "the top K. Shaded band: 95% Wilson interval per method (unpaired); for "
                  "paired significance between methods see fig2.")
    save(fig, out_dir, "fig1_recall_at_k", written)


# --------------------------------------------------------------------------------------
# figure 2 -- delta vs baseline
# --------------------------------------------------------------------------------------


def _sig_rows(sig: dict, metric: str, baseline: str) -> list[dict]:
    rows = []
    for method, per_metric in sig.items():
        entry = (per_metric or {}).get(metric)
        if not entry:
            continue
        for field in ("delta", "ci_low", "ci_high"):
            if field not in entry:
                die(f"significance_results.json: {method}/{metric} has no '{field}'")
        rows.append({"method": method, **entry})
    if not rows:
        die(f"significance_results.json contains no '{metric}' comparison against "
            f"baseline '{baseline}'. Re-run analyze_retrieval_lab.py with {metric} in --metrics.")
    return sorted(rows, key=lambda r: r["delta"])


def fig2_delta_vs_baseline(sig, style: Style, baseline, out_dir, written) -> None:
    rows = _sig_rows(sig, "recall@10", baseline)
    ys = np.arange(len(rows))
    height = max(3.2, 0.52 * len(rows) + 2.3)
    fig, ax = plt.subplots(figsize=(11.4, height), layout="constrained")
    tidy(ax, grid_axis="x")

    lo_err, hi_err, clipped = [], [], False
    for r in rows:
        lo = r["delta"] - r["ci_low"]
        hi = r["ci_high"] - r["delta"]
        if lo < 0 or hi < 0:
            clipped = True  # would raise in matplotlib; surfaced in the footnote, not hidden
        lo_err.append(max(0.0, lo))
        hi_err.append(max(0.0, hi))

    for i, r in enumerate(rows):
        m = r["method"]
        colour = style.color.get(m, NEUTRAL)
        significant = bool(r.get("significant"))
        # Fill = significant, hatched outline = not. Texture rather than a second hue keeps
        # the significance channel readable in greyscale and under colour-blindness.
        ax.barh(i, r["delta"], height=0.62, zorder=3,
                color=colour if significant else SURFACE,
                edgecolor=colour, linewidth=1.6,
                hatch=None if significant else "///")
    ax.errorbar([r["delta"] for r in rows], ys, xerr=[lo_err, hi_err], fmt="none",
                ecolor=INK, elinewidth=1.3, capsize=4, capthick=1.3, zorder=4)

    ax.axvline(0, color=INK, linewidth=1.6, zorder=5)
    ax.set_yticks(ys, [style.label(r["method"]) for r in rows])
    ax.set_xlabel("Change in Recall@10 vs baseline (absolute difference, +/- 95% bootstrap CI)")
    ax.set_title(f"Recall@10 delta vs baseline '{baseline}'")

    span = max(abs(min(r["ci_low"] for r in rows)), abs(max(r["ci_high"] for r in rows)), 0.01)
    ax.set_xlim(-span * 1.35, span * 1.35)
    ax.set_ylim(-0.65, len(rows) - 0.25)
    # Anchored to x=0 in data coords but to the top edge in axes coords, so it labels the
    # zero rule wherever that rule sits and never drifts outside the frame.
    ax.annotate(f"0 = baseline ({baseline})", xy=(0, 1.0),
                xycoords=("data", "axes fraction"), xytext=(5, -4),
                textcoords="offset points",
                fontsize=9, color=INK_MUTED, ha="left", va="top")

    for i, r in enumerate(rows):
        offset = 6 if r["delta"] >= 0 else -6
        ax.annotate(f"{r['delta']:+.3f}", xy=(r["ci_high"] if r["delta"] >= 0 else r["ci_low"], i),
                    xytext=(offset, 0), textcoords="offset points", fontsize=9, color=INK,
                    ha="left" if r["delta"] >= 0 else "right", va="center")

    # Legend swatches are deliberately neutral: the channel they explain is fill-vs-hatch,
    # and a coloured swatch would read as "significant == this method's colour".
    swatch = "#9E9E9E"
    side_legend(fig, ax, handles=[
        Patch(facecolor=swatch, edgecolor=swatch,
              label="significant\n(95% CI excludes 0)"),
        Patch(facecolor=SURFACE, edgecolor=swatch, hatch="///",
              label="not significant\n(CI spans 0)"),
        Line2D([0], [0], color=INK, lw=1.3, label="95% paired bootstrap CI"),
    ], labels=None)
    note = "Paired bootstrap over queries; bar colour = method (shared across figures)."
    if clipped:
        note += " NOTE: a CI bound fell on the wrong side of its delta; error bar clipped at 0."
    footnote(fig, note)
    save(fig, out_dir, "fig2_delta_vs_bm25", written)


# --------------------------------------------------------------------------------------
# figure 3 -- quality vs latency
# --------------------------------------------------------------------------------------


def pareto_frontier(points: list[tuple[str, float, float]]) -> list[tuple[str, float, float]]:
    """Methods not dominated on (minimise latency, maximise recall).

    A method is dominated when another is at least as fast AND at least as accurate, with
    at least one of the two strictly better. Ties in both coordinates are kept (identical
    trade-offs are equally on the frontier) so a duplicated configuration is not hidden.
    """
    keep = []
    for name, x, y in points:
        dominated = any(
            (ox <= x and oy >= y) and (ox < x or oy > y)
            for oname, ox, oy in points if oname != name
        )
        if not dominated:
            keep.append((name, x, y))
    return sorted(keep, key=lambda p: p[1])


def _place_labels(ax, points, style: Style) -> None:
    """Greedy label placement: try four offsets per point, take the first that does not
    collide with an already-placed label. Cheap alternative to an adjustText dependency.

    Points are labelled with the bare method name; baseline/winner identity is carried by
    the legend and the ink ring, which keeps these labels short enough to place.
    """
    placed: list[tuple[float, float, float, float]] = []
    candidates = ((10, 9), (10, -13), (-10, 9), (-10, -13))
    fig = ax.figure
    fig.canvas.draw()  # transforms must be current before measuring text extents
    for name, x, y in points:
        for dx, dy in candidates:
            ha = "left" if dx > 0 else "right"
            txt = ax.annotate(name, xy=(x, y),
                              xytext=(dx, dy), textcoords="offset points",
                              fontsize=9, color=INK, ha=ha, va="center", zorder=6)
            bb = txt.get_window_extent(renderer=fig.canvas.get_renderer())
            box = (bb.x0, bb.y0, bb.x1, bb.y1)
            if not any(box[0] < p[2] and p[0] < box[2] and box[1] < p[3] and p[1] < box[3]
                       for p in placed):
                placed.append(box)
                break
            txt.remove()
        else:
            # Every candidate collided; keep the label anyway rather than dropping a method.
            txt = ax.annotate(name, xy=(x, y),
                              xytext=(9, 8), textcoords="offset points",
                              fontsize=8, color=INK_MUTED, ha="left", va="center", zorder=6)
            placed.append(tuple(txt.get_window_extent(
                renderer=fig.canvas.get_renderer()).extents))


def fig3_quality_latency(metrics, style: Style, baseline, winner, out_dir, written) -> None:
    points = [(m, metrics[m]["latency_p95_ms"], metrics[m]["recall@10"]) for m in metrics
              if not math.isnan(metrics[m]["latency_p95_ms"])]
    if not points:
        die("retrieval_metrics.csv has no usable latency_p95_ms values; cannot draw fig3")

    fig, ax = plt.subplots(figsize=(11.4, 5.8), layout="constrained")
    tidy(ax, grid_axis="both")

    frontier = pareto_frontier(points)
    fx = [p[1] for p in frontier]
    fy = [p[2] for p in frontier]
    ax.plot(fx, fy, drawstyle="steps-post", color=INK_MUTED, linewidth=1.4,
            linestyle=(0, (5, 3)), zorder=2, label="Pareto frontier")

    for name, x, y in points:
        # Baseline and winner get a larger mark with an ink ring; every other method keeps
        # the 2px surface ring that stops overlapping points from merging.
        emphasised = name in (baseline, winner)
        ax.scatter(x, y, s=190 if emphasised else 110,
                   color=style.color.get(name, NEUTRAL),
                   marker=style.marker.get(name, "o"),
                   edgecolor=INK if emphasised else SURFACE,
                   linewidth=1.8 if emphasised else 1.4, zorder=5)

    xs = [p[1] for p in points]
    # Log x only when the latencies span orders of magnitude; on a linear axis a 3 ms and a
    # 2000 ms method would otherwise pile up on the left edge.
    if min(xs) > 0 and max(xs) / min(xs) > 20:
        ax.set_xscale("log")
        ax.set_xlabel("P95 retrieval latency (ms, log scale)")
    else:
        ax.set_xlim(0, max(xs) * 1.18)
        ax.set_xlabel("P95 retrieval latency (ms)")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Recall@10 (fraction of queries)")
    ax.set_title("Retrieval quality vs latency cost")
    legend_handles = [
        Line2D([0], [0], color=INK_MUTED, lw=1.4, linestyle=(0, (5, 3)),
               label="Pareto frontier:\nnothing is both faster\nand more accurate"),
        Line2D([0], [0], marker=style.marker.get(baseline, "o"), color="none",
               markerfacecolor=style.color.get(baseline, NEUTRAL), markeredgecolor=INK,
               markersize=11, label=f"{baseline}\n(baseline)"),
    ]
    if winner != baseline:
        legend_handles.append(
            Line2D([0], [0], marker=style.marker.get(winner, "o"), color="none",
                   markerfacecolor=style.color.get(winner, NEUTRAL), markeredgecolor=INK,
                   markersize=11, label=f"{winner}\n(winner)"))
    side_legend(fig, ax, handles=legend_handles)
    footnote(fig, "Latency is single-query P95 from the run's own latency sample; index "
                  "build cost is reported separately in latency_metrics.csv. Marks with an "
                  "ink ring are the baseline and the winner.")
    # Labels last: constrained layout and the axis scale must be settled before text
    # extents are measured, or the collision test is done against stale positions.
    _place_labels(ax, points, style)
    save(fig, out_dir, "fig3_quality_latency", written)


# --------------------------------------------------------------------------------------
# figure 4 -- subgroup heatmap
# --------------------------------------------------------------------------------------


def fig4_subgroups(sub_rows, metrics, style: Style, baseline, winner, max_methods,
                   out_dir, written) -> None:
    methods_present = [m for m in canonical_order(sorted({r["method"] for r in sub_rows}))]
    if len(methods_present) > max_methods:
        keep = {baseline, winner}
        for m in sorted(methods_present,
                        key=lambda m: -metrics.get(m, {}).get("recall@10", 0.0)):
            if len(keep) >= max_methods:
                break
            keep.add(m)
        methods_present = [m for m in methods_present if m in keep]
        print(f"  fig4: showing top {max_methods} methods by recall@10 (of "
              f"{len({r['method'] for r in sub_rows})}) to keep the cells legible")

    subgroups = sorted({r["subgroup"] for r in sub_rows})
    cell: dict[tuple[str, str], float] = {}
    n_by_subgroup: dict[str, int] = {}
    for r in sub_rows:
        cell[(r["method"], r["subgroup"])] = r["recall@10"]
        n_by_subgroup[r["subgroup"]] = max(n_by_subgroup.get(r["subgroup"], 0), r["n_queries"])

    grid = np.full((len(methods_present), len(subgroups)), np.nan)
    for i, m in enumerate(methods_present):
        for j, g in enumerate(subgroups):
            if (m, g) in cell:
                grid[i, j] = cell[(m, g)]

    fig_w = max(7.5, 0.85 * len(subgroups) + 4.2)
    fig_h = max(3.6, 0.46 * len(methods_present) + 2.6)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), layout="constrained")

    # viridis: perceptually uniform and colour-blind-safe. vmin/vmax pinned to the full
    # 0-1 proportion range so a low cell stays visibly low instead of being stretched
    # back into the middle of the ramp by an auto-scaled colour bar.
    im = ax.imshow(grid, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(subgroups)),
                  [f"{g}\n(n={n_by_subgroup.get(g, 0)})" for g in subgroups],
                  rotation=40, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(methods_present)), [style.label(m) for m in methods_present])
    for tick, m in zip(ax.get_yticklabels(), methods_present):
        if m in (baseline, winner):
            tick.set_fontweight("bold")

    for i in range(len(methods_present)):
        for j in range(len(subgroups)):
            v = grid[i, j]
            if math.isnan(v):
                ax.text(j, i, "-", ha="center", va="center", fontsize=9, color=INK_MUTED)
                continue
            # White text on the dark end of viridis, black on the light end.
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                    color="#FFFFFF" if v < 0.55 else "#12241F")

    # Thin surface gaps between cells instead of a heavy border on each cell.
    ax.set_xticks(np.arange(-0.5, len(subgroups), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(methods_present), 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.015)
    # Short label: on a run with few method rows the colour bar is short, and a long
    # rotated label overflows the right edge of the figure.
    cbar.set_label("Recall@10")
    cbar.outline.set_visible(False)
    ax.set_title("Recall@10 by method and query subgroup")
    ax.set_xlabel("Query subgroup")
    footnote(fig, "Cell = fraction of that subgroup's queries with a gold chunk in the top "
                  "10; n = queries in the subgroup. The colour scale spans the full 0-1 "
                  "range (not clipped), so regressions stay visible. A subgroup a method "
                  "was not scored on is shown as '-'.")
    save(fig, out_dir, "fig4_subgroups", written)


# --------------------------------------------------------------------------------------
# figure 5 -- ranking metrics
# --------------------------------------------------------------------------------------


def fig5_ranking_metrics(metrics, per_q, style: Style, finalists, out_dir, written) -> None:
    cols = ("mrr@10", "ndcg@10", "hit@1")
    labels = ("MRR@10", "nDCG@10", "Hit@1")
    usable = [m for m in finalists
              if all(not math.isnan(metrics[m].get(c, float("nan"))) for c in cols)]
    if not usable:
        die("no method has all of mrr@10 / ndcg@10 / hit@1 in retrieval_metrics.csv")

    x = np.arange(len(cols))
    width = min(0.8 / len(usable), 0.22)
    fig, ax = plt.subplots(figsize=(11.0, 5.2), layout="constrained")
    tidy(ax, grid_axis="y")

    for i, m in enumerate(usable):
        offs = (i - (len(usable) - 1) / 2) * width
        vals = [metrics[m][c] for c in cols]
        rows = per_q.get(m, [])
        lo_err, hi_err, have_ci = [], [], False
        for c in cols:
            if c == "hit@1":
                ci = binary_ci(rows, c)          # 0/1 metric -> Wilson
            else:
                series = [r[c] for r in rows if c in r and not math.isnan(r[c])]
                ci = mean_ci(series)[1:] if series else None   # continuous -> normal approx
            if ci is None:
                lo_err.append(0.0)
                hi_err.append(0.0)
            else:
                have_ci = True
                v = metrics[m][c]
                lo_err.append(max(0.0, v - ci[0]))
                hi_err.append(max(0.0, ci[1] - v))
        ax.bar(x + offs, vals, width * 0.88, color=style.color[m], zorder=3,
               label=style.label(m),
               edgecolor=INK if m == style.winner else SURFACE,
               linewidth=1.4 if m == style.winner else 0.8)
        if have_ci:
            ax.errorbar(x + offs, vals, yerr=[lo_err, hi_err], fmt="none",
                        ecolor=INK, elinewidth=1.1, capsize=3, capthick=1.1, zorder=4)
        # Selective direct labels only: a number over all 18 bars is unreadable, and the
        # exact values are in retrieval_metrics.csv. The winner's row is the one the
        # report quotes, so that is the one labelled.
        if m == style.winner:
            for xi, v, hi in zip(x + offs, vals, hi_err):
                ax.annotate(f"{v:.2f}", xy=(xi, v + hi), xytext=(0, 5),
                            textcoords="offset points", ha="center", va="bottom",
                            fontsize=9, color=INK, fontweight="bold")

    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.0)   # one shared 0-1 axis: all three are proportions, so no second axis
    ax.set_ylabel("Metric value (0-1, higher is better)")
    ax.set_xlabel("Ranking metric")
    ax.set_title("Ranking quality of the finalist methods")
    side_legend(fig, ax)
    footnote(fig, "Error bars: 95% CI (Wilson for Hit@1; normal approximation for MRR@10 "
                  "and nDCG@10), computed from per_query_results.csv. Values are labelled "
                  "on the winner only; the full table is retrieval_metrics.csv.")
    save(fig, out_dir, "fig5_ranking_metrics", written)


# --------------------------------------------------------------------------------------
# figure 6 -- win / tie / loss
# --------------------------------------------------------------------------------------


def fig6_win_tie_loss(sig, style: Style, baseline, out_dir, written) -> None:
    rows = _sig_rows(sig, "recall@10", baseline)
    for r in rows:
        for field in ("improved", "tied", "harmed"):
            if field not in r:
                die(f"significance_results.json: {r['method']}/recall@10 has no '{field}'")

    height = max(3.2, 0.5 * len(rows) + 2.4)
    fig, ax = plt.subplots(figsize=(11.4, height), layout="constrained")
    tidy(ax, grid_axis="x")
    ys = np.arange(len(rows))

    imp = np.array([r["improved"] for r in rows], dtype=float)
    tie = np.array([r["tied"] for r in rows], dtype=float)
    har = np.array([r["harmed"] for r in rows], dtype=float)

    # Colour here encodes OUTCOME, not method: each bar is one method, already named on the
    # y axis, and the reader's question is "how many queries got better vs worse".
    # The surface-coloured edge is a 2px gap between segments, not a border.
    seg = dict(height=0.6, zorder=3, edgecolor=SURFACE, linewidth=2)
    ax.barh(ys, imp, color=C_IMPROVED, label="improved", **seg)
    ax.barh(ys, tie, left=imp, color=C_TIED, label="tied", **seg)
    ax.barh(ys, har, left=imp + tie, color=C_HARMED, label="harmed", **seg)

    total = imp + tie + har
    for i in range(len(rows)):
        for value, left, colour in ((imp[i], 0.0, "#FFFFFF"),
                                    (tie[i], imp[i], INK),
                                    (har[i], imp[i] + tie[i], "#FFFFFF")):
            if value <= 0 or value / max(total[i], 1) < 0.05:
                continue  # too narrow for a legible label; the axis and legend carry it
            ax.text(left + value / 2, i, f"{int(value)}", ha="center", va="center",
                    fontsize=9, color=colour, zorder=5)

    ax.set_yticks(ys, [style.label(r["method"]) for r in rows])
    for tick, r in zip(ax.get_yticklabels(), rows):
        if r["method"] == style.winner:
            tick.set_fontweight("bold")
    ax.set_xlim(0, max(total) * 1.02 if len(total) else 1)
    ax.set_xlabel(f"Number of benchmark queries (n = {int(max(total)) if len(total) else 0})")
    ax.set_title(f"Per-query outcome vs baseline '{baseline}' (Recall@10)")
    side_legend(fig, ax)
    footnote(fig, "improved / harmed = the query's Recall@10 rose / fell against the "
                  "baseline; tied = unchanged. The baseline itself has no bar (it is the "
                  "reference).")
    save(fig, out_dir, "fig6_win_tie_loss", written)


# --------------------------------------------------------------------------------------
# figure 7 -- failure categories
# --------------------------------------------------------------------------------------


def categorise_failure(row: dict) -> list[str]:
    """Categories for one failed query. See the module docstring for the full contract."""
    tags = set(row["subgroups"])
    cats = [t for t in FAILURE_TAGS if t in tags]
    r50 = row.get("recall@50")
    if r50 is not None and not math.isnan(r50) and r50 == 0.0:
        cats.append("no_gold_in_top50")
    if not cats:
        cats.append("uncategorized")
    return cats


def fig7_error_categories(per_q, metrics, style: Style, baseline, winner, out_dir,
                          written) -> None:
    compare = [baseline]
    contrast_role = "winner"
    if winner != baseline:
        compare.append(winner)
    else:
        contrast_role = "runner-up"
        # The baseline is also the best method: compare it with the runner-up instead of
        # drawing the same series twice, so the figure still contrasts two systems.
        others = [m for m in metrics if m != baseline]
        if others:
            runner_up = max(others, key=lambda m: (metrics[m]["recall@10"], m))
            compare.append(runner_up)
            print(f"  fig7: winner == baseline; contrasting with runner-up '{runner_up}'")

    counts: dict[str, Counter] = {}
    totals: dict[str, int] = {}
    for m in compare:
        rows = [r for r in per_q.get(m, [])
                if "recall@10" in r and not math.isnan(r["recall@10"]) and r["recall@10"] == 0.0]
        c = Counter()
        for r in rows:
            c.update(categorise_failure(r))
        counts[m] = c
        totals[m] = len(rows)

    present = [c for c in list(FAILURE_TAGS) + ["no_gold_in_top50", "uncategorized"]
               if any(counts[m].get(c, 0) for m in compare)]
    if not present:
        # Nothing failed at Recall@10 for either method: draw the fact rather than an
        # empty grid, so the figure set stays complete and self-explaining.
        fig, ax = plt.subplots(figsize=(8.0, 3.2), layout="constrained")
        ax.axis("off")
        ax.text(0.5, 0.5, "No Recall@10 failures for "
                          + " or ".join(compare)
                          + "\n(nothing to categorise)",
                ha="center", va="center", fontsize=12, color=INK)
        ax.set_title("Failure categories at Recall@10")
        save(fig, out_dir, "fig7_error_categories", written)
        return

    x = np.arange(len(present))
    width = 0.8 / len(compare)
    fig, ax = plt.subplots(figsize=(max(10.0, 1.25 * len(present) + 5.5), 5.2),
                           layout="constrained")
    tidy(ax, grid_axis="y")

    for i, m in enumerate(compare):
        offs = (i - (len(compare) - 1) / 2) * width
        vals = [counts[m].get(c, 0) for c in present]
        label = style.label(m)
        if m != baseline and contrast_role == "runner-up":
            label += "  (runner-up)"
        ax.bar(x + offs, vals, width * 0.9, color=style.color.get(m, NEUTRAL), zorder=3,
               edgecolor=INK if m == winner and m != baseline else SURFACE,
               linewidth=1.4 if m == winner and m != baseline else 0.8,
               label=f"{label}\n({totals[m]} failed queries)")
        for xi, v in zip(x + offs, vals):
            if v:
                ax.annotate(str(v), xy=(xi, v), xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=9, color=INK_MUTED)

    ax.set_xticks(x, ["\n".join(textwrap.wrap(c.replace("_", " "), 12)) for c in present])
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Failure category (overlapping tags -- a query can appear in several)")
    ax.set_ylabel("Failed queries (Recall@10 = 0)")
    ax.set_title(f"Where retrieval fails: baseline vs {contrast_role}")
    side_legend(fig, ax)
    footnote(fig, "A query counts once per category it matches, so bars sum to more than "
                  "the failure total. no_gold_in_top50 = no gold chunk anywhere in the "
                  "top 50, i.e. unrecoverable by reranking.")
    save(fig, out_dir, "fig7_error_categories", written)


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Draw the seven retrieval-benchmark figures from a run directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("run_dir", help="run directory written by run_retrieval_lab.py + "
                                    "analyze_retrieval_lab.py")
    ap.add_argument("--baseline", default="bm25_full_corpus",
                    help="method used as the comparison baseline (default: bm25_full_corpus)")
    ap.add_argument("--out-dir", default=None, help="default: <run_dir>/figures")
    ap.add_argument("--highlight", default=None,
                    help="method to mark as the winner (default: best recall@10)")
    ap.add_argument("--max-methods", type=int, default=6,
                    help="max methods in the line/grouped-bar figures (default: 6)")
    ap.add_argument("--max-heatmap-methods", type=int, default=14,
                    help="max method rows in the subgroup heatmap (default: 14)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        die(f"run directory not found: {run_dir}")
    missing = [f for f in REQUIRED_FILES if not (run_dir / f).is_file()]
    if missing:
        die(f"{run_dir} is missing {missing}.\n"
            f"  Generate them first:\n"
            f"    python server/evaluation/analyze_retrieval_lab.py {run_dir} "
            f"--baseline {args.baseline}")

    metrics = load_retrieval_metrics(run_dir)
    sub_rows = load_subgroup_metrics(run_dir)
    sig = load_significance(run_dir)
    per_q = per_query_index(load_per_query(run_dir))

    if args.baseline not in metrics:
        die(f"baseline '{args.baseline}' is not in retrieval_metrics.csv "
            f"(have: {sorted(metrics)})")

    winner = pick_winner(metrics, args.highlight)
    finalists = select_finalists(metrics, args.baseline, winner, args.max_methods)
    style = Style(list(metrics), finalists, args.baseline, winner)
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else run_dir / "figures"

    print(f"run:       {run_dir}")
    print(f"methods:   {len(metrics)}  ({', '.join(canonical_order(list(metrics)))})")
    print(f"baseline:  {args.baseline}  (recall@10 = {metrics[args.baseline]['recall@10']:.4f})")
    print(f"winner:    {winner}  (recall@10 = {metrics[winner]['recall@10']:.4f})"
          f"{'  [--highlight]' if args.highlight else ''}")
    print(f"finalists: {', '.join(finalists)}")
    print(f"out:       {out_dir}\n")

    apply_rc()
    written: list[Path] = []
    fig1_recall_at_k(metrics, per_q, style, finalists, out_dir, written)
    fig2_delta_vs_baseline(sig, style, args.baseline, out_dir, written)
    fig3_quality_latency(metrics, style, args.baseline, winner, out_dir, written)
    fig4_subgroups(sub_rows, metrics, style, args.baseline, winner,
                   args.max_heatmap_methods, out_dir, written)
    fig5_ranking_metrics(metrics, per_q, style, finalists, out_dir, written)
    fig6_win_tie_loss(sig, style, args.baseline, out_dir, written)
    fig7_error_categories(per_q, metrics, style, args.baseline, winner, out_dir, written)

    print(f"wrote {len(written)} files to {out_dir}:")
    for path in written:
        print(f"  {path.name:34s} {path.stat().st_size:>9,d} bytes")


if __name__ == "__main__":
    main()
