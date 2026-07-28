from __future__ import annotations

"""
Build the expanded adviSU retrieval benchmark (train / dev / test).

WHY THIS EXISTS
data/benchmark/questions.jsonl holds 22 questions, only 16 of which carry gold chunks.
At n=16 a single query is worth 6.25 points of Recall@10, a paired bootstrap CI spans most
of the [0,1] interval, and the 13 subgroups the task requires would hold 1-3 queries each.
No honest significance claim is possible at that size, so the benchmark is widened here.

GROUND TRUTH IS DERIVED, NOT WRITTEN
Every answerable question is generated from one real row of data/degree_requirements/** or
data/minors/**. The gold `expected_chunk_ids` are that row's own `chunk_id`, and the
reference answer is built from that row's own fields, so nothing is fabricated and
`verify()` re-checks every id against the corpus. Re-scrape the corpus and re-run this and
the benchmark stays true.

KNOWN LIMITATION (stated here because it bounds every number downstream)
The question *wording* is templated. Templated phrasing reuses corpus vocabulary, which
favours lexical retrievers; BM25 numbers on this benchmark are therefore an optimistic
bound on BM25's real-world behaviour, and any margin a dense/hybrid system wins here is
correspondingly a conservative estimate. Paraphrase pools and Turkish renderings reduce but
do not remove that bias. The 6 hand-written unanswerable/misleading items from the original
benchmark are preserved verbatim.

LEAKAGE CONTROL
Splitting is by gold chunk, not by question: a chunk that is gold for a train question can
never be gold for a dev or test question. Course/program/term entities are also partitioned
so the same (course, program, term) fact cannot be learned in train and scored in test.

Usage:
    python server/evaluation/build_retrieval_benchmark.py [--out-dir data/benchmark] [--seed 20260727]
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retrieval_lab.corpus import PROJECT_ROOT, load_corpus

SEED = 20260727

# --- paraphrase pools -----------------------------------------------------------------
# Several surface forms per fact type so the benchmark is not a single lexical pattern.

CATEGORY_EN = [
    "In the {prog} {term} curriculum, which requirement category does {course} belong to?",
    "Which requirement group does {course} count towards for {prog} students on the {term} catalog?",
    "For the {term} {prog} curriculum, under which category is {course} listed?",
    "Where does {course} fit in the {prog} {term} degree requirements?",
]
CATEGORY_TR = [
    "{prog} {term} müfredatında {course} hangi kategoriye ait?",
    "{term} {prog} müfredatı için {course} dersi hangi gereklilik grubunda yer alıyor?",
    "{course} dersi {prog} {term} müfredatında hangi başlık altında sayılıyor?",
]
CREDITS_EN = [
    "How many SU credits and ECTS does {course} carry in the {prog} {term} curriculum?",
    "What is the credit load of {course} for {prog} {term}?",
    "How many SU credits is {course} worth on the {prog} {term} catalog?",
]
CREDITS_TR = [
    "{course} dersi {prog} {term} müfredatında kaç SU kredisidir?",
    "{prog} {term} müfredatında {course} dersinin kredisi nedir?",
]
TOTAL_EN = [
    "What is the minimum total number of SU credits required to graduate from the {prog} {term} curriculum?",
    "How many SU credits in total does a {prog} student on the {term} catalog need to graduate?",
    "What is the graduation credit requirement for {prog} {term}?",
]
TOTAL_TR = [
    "{prog} {term} müfredatında mezun olmak için gereken minimum toplam SU kredisi kaçtır?",
    "{term} {prog} müfredatından mezun olmak için toplam kaç SU kredisi gerekir?",
]
ECTS_EN = [
    "What is the minimum total ECTS required for the {prog} {term} curriculum?",
    "How many ECTS does a {prog} {term} student need in total?",
]
CATMIN_EN = [
    "How many SU credits of {category} are required in the {prog} {term} curriculum?",
    "What is the minimum {category} credit requirement for {prog} {term}?",
    "How many {category} credits must a {prog} student on the {term} catalog complete?",
]
CATMIN_TR = [
    "{prog} {term} müfredatında {category} için kaç SU kredisi gereklidir?",
]
# Minor questions must name the catalog term: the same minor has a different requirement
# row per term, so an unqualified "what does X require?" would have 4 equally valid answers.
MINOR_EN = [
    "What are the requirements of the {name} minor programme on the {term} catalog?",
    "What does the {name} minor require for {term}?",
    "Which conditions must be met to complete the {name} minor under the {term} curriculum?",
]
MINOR_COURSE_EN = [
    "Does {course} count towards the {name} minor on the {term} catalog?",
    "Is {course} part of the {term} {name} minor requirements?",
]
RULE_EN = [
    "In the {prog} {term} curriculum, what rule applies to {category}?",
    "What is the {category} rule for {prog} {term}?",
]


# --- natural-phrasing pools -----------------------------------------------------------
# The templates above name the program code, the 6-digit term and the course code verbatim,
# which is exactly how the corpus spells them. That turns retrieval into a token-match
# exercise and produced a ceiling effect (BM25F scored 1.000 on the first dev smoke run).
# Real advising questions do not talk like that, so a fraction of every family is rendered
# with the surface forms a student would actually use: the program's full name, the human
# term label ("Fall 2024-2025"), and colloquial framing. The gold chunk is unchanged - only
# the wording differs - so this measures robustness to paraphrase, not a different fact.

NATURAL_CATEGORY_EN = [
    "I'm a {progname} student who started in {termlabel}. Does {course} count as an elective for me, and under what heading?",
    "As a {progname} student admitted in {termlabel}, where would {course} count in my degree?",
    "I started {progname} in {termlabel} - what does {course} count towards?",
]
NATURAL_CATEGORY_TR = [
    "{termlabel} döneminde {progname} programına başladım. {course} dersi bende hangi başlıkta sayılıyor?",
    "{progname} öğrencisiyim, {termlabel} girişliyim. {course} dersi neye sayılır?",
]
NATURAL_CREDITS_EN = [
    "I'm in {progname} ({termlabel} entry) - how much credit do I get for {course}?",
    "How many credits would {course} give me as a {progname} student who started in {termlabel}?",
]
NATURAL_TOTAL_EN = [
    "I'm studying {progname} and started in {termlabel}. How many credits do I need in total to graduate?",
    "What's the total credit requirement for someone doing {progname} from {termlabel}?",
]
NATURAL_TOTAL_TR = [
    "{progname} okuyorum, {termlabel} girişliyim. Mezun olmak için toplam kaç kredi gerekiyor?",
]

# Fraction of generated questions rendered in natural rather than literal form.
NATURAL_SHARE = 0.45


def _fmt(template: str, **kw) -> str:
    return template.format(**kw)


def _pretty(raw: str) -> str:
    return (raw or "").replace("_", " ").strip()


class Builder:
    def __init__(self, corpus, rng: random.Random) -> None:
        self.corpus = corpus
        self.rng = rng
        self.items: list[dict] = []
        self.by_type: dict[str, list] = defaultdict(list)
        for c in corpus.chunks:
            self.by_type[c.document_type].append(c)

    def surface(self, chunk) -> tuple[str, str]:
        """Human surface forms for a chunk's program and term, for natural phrasing."""
        progname = str(chunk.meta.get("program_name") or chunk.program)
        termlabel = str(chunk.meta.get("admit_term_label") or chunk.curriculum_term)
        return progname, termlabel

    def natural(self) -> bool:
        return self.rng.random() < NATURAL_SHARE

    def add(self, **kw) -> None:
        item = {
            "id": "",
            "language": "en",
            "answerable": True,
            "program": None,
            "curriculum_term": None,
            "intent": "ders_ayrintisi",
            "expected_sources": [],
            "expected_chunk_ids": [],
            "reference_answer": "",
            "notes": "",
            "origin": "generated",
            "multi_evidence": False,
            "phrasing": "literal",
        }
        item.update(kw)
        item["multi_evidence"] = len(item["expected_chunk_ids"]) > 1
        self.items.append(item)

    # --- families ---------------------------------------------------------------------

    def course_category(self, limit: int) -> None:
        """course -> requirement category. Only courses with ONE category in that (prog, term)."""
        groups: dict[tuple[str, str, str], list] = defaultdict(list)
        for c in self.by_type["degree_requirement_pool_course"]:
            if c.course_id and c.requirement_category:
                groups[(c.program, c.curriculum_term, c.course_id)].append(c)
        keys = sorted(k for k, v in groups.items() if len({x.requirement_category for x in v}) == 1)
        self.rng.shuffle(keys)
        for prog, term, course in keys[:limit]:
            rows = groups[(prog, term, course)]
            tr = self.rng.random() < 0.18
            progname, termlabel = self.surface(rows[0])
            if self.natural():
                tpl = self.rng.choice(NATURAL_CATEGORY_TR if tr else NATURAL_CATEGORY_EN)
                phrasing = "natural"
            else:
                tpl = self.rng.choice(CATEGORY_TR if tr else CATEGORY_EN)
                phrasing = "literal"
            self.add(
                phrasing=phrasing,
                question=_fmt(tpl, prog=prog, term=term, course=course,
                              progname=progname, termlabel=termlabel),
                question_type="turkish" if tr else "factual_lookup",
                language="tr" if tr else "en",
                program=prog,
                curriculum_term=term,
                expected_sources=[rows[0].source_document],
                expected_chunk_ids=sorted({r.chunk_id for r in rows}),
                reference_answer=f"{course} belongs to the {_pretty(rows[0].requirement_category)} pool of the {prog} {term} curriculum.",
                notes=f"degree_requirement_pool_course; category={rows[0].requirement_category}",
            )

    def course_credits(self, limit: int) -> None:
        rows = [
            c for c in self.by_type["degree_requirement_pool_course"]
            if c.course_id and c.meta.get("su_credits") is not None and c.meta.get("ects") is not None
        ]
        self.rng.shuffle(rows)
        for r in rows[:limit]:
            tr = self.rng.random() < 0.18
            progname, termlabel = self.surface(r)
            if self.natural() and not tr:
                tpl, phrasing = self.rng.choice(NATURAL_CREDITS_EN), "natural"
            else:
                tpl = self.rng.choice(CREDITS_TR if tr else CREDITS_EN)
                phrasing = "literal"
            self.add(
                phrasing=phrasing,
                question=_fmt(tpl, prog=r.program, term=r.curriculum_term, course=r.course_id,
                              progname=progname, termlabel=termlabel),
                question_type="turkish" if tr else "factual_lookup",
                language="tr" if tr else "en",
                program=r.program,
                curriculum_term=r.curriculum_term,
                expected_sources=[r.source_document],
                expected_chunk_ids=[r.chunk_id],
                reference_answer=f"{r.course_id} carries {r.meta['su_credits']} SU credits and {r.meta['ects']} ECTS.",
                notes="su_credits/ects fields",
            )

    def program_totals(self) -> None:
        for p in self.by_type["degree_requirement_profile"]:
            progname, termlabel = self.surface(p)
            if p.meta.get("total_min_su_credits"):
                tr = self.rng.random() < 0.25
                if self.natural():
                    tpl = self.rng.choice(NATURAL_TOTAL_TR if tr else NATURAL_TOTAL_EN)
                    phrasing = "natural"
                else:
                    tpl = self.rng.choice(TOTAL_TR if tr else TOTAL_EN)
                    phrasing = "literal"
                self.add(
                    phrasing=phrasing,
                    question=_fmt(tpl, prog=p.program, term=p.curriculum_term,
                                  progname=progname, termlabel=termlabel),
                    question_type="turkish" if tr else "course_requirement",
                    language="tr" if tr else "en",
                    intent="mezuniyet_durumu",
                    program=p.program,
                    curriculum_term=p.curriculum_term,
                    expected_sources=[p.source_document],
                    expected_chunk_ids=[p.chunk_id],
                    reference_answer=f"{p.meta['total_min_su_credits']} SU credits.",
                    notes="degree_requirement_profile.total_min_su_credits",
                )
            if p.meta.get("total_min_ects"):
                self.add(
                    question=_fmt(self.rng.choice(ECTS_EN), prog=p.program, term=p.curriculum_term),
                    question_type="course_requirement",
                    intent="mezuniyet_durumu",
                    program=p.program,
                    curriculum_term=p.curriculum_term,
                    expected_sources=[p.source_document],
                    expected_chunk_ids=[p.chunk_id],
                    reference_answer=f"{p.meta['total_min_ects']} ECTS.",
                    notes="degree_requirement_profile.total_min_ects",
                )

    def category_minimums(self) -> None:
        """Multi-evidence: the category_pool row AND any rule row restating the same minimum."""
        rules = self.by_type["degree_requirement_rule"]
        rules_by = defaultdict(list)
        for r in rules:
            rules_by[(r.program, r.curriculum_term, r.requirement_category)].append(r)
        for pool in self.by_type["degree_requirement_category_pool"]:
            if not pool.meta.get("min_su_credits"):
                continue
            cat = _pretty(pool.requirement_category)
            gold = [pool.chunk_id] + [
                r.chunk_id for r in rules_by[(pool.program, pool.curriculum_term, pool.requirement_category)]
                if r.meta.get("min_su_credits") == pool.meta["min_su_credits"]
            ]
            tr = self.rng.random() < 0.15
            tpl = self.rng.choice(CATMIN_TR if tr else CATMIN_EN)
            self.add(
                question=_fmt(tpl, prog=pool.program, term=pool.curriculum_term, category=cat),
                question_type="turkish" if tr else "course_requirement",
                language="tr" if tr else "en",
                intent="mezuniyet_durumu",
                program=pool.program,
                curriculum_term=pool.curriculum_term,
                expected_sources=[pool.source_document],
                expected_chunk_ids=sorted(set(gold)),
                reference_answer=f"{pool.meta['min_su_credits']} SU credits of {cat}.",
                notes=f"category_pool.min_su_credits (+restating rules); category={pool.requirement_category}",
            )

    def rules(self, limit: int) -> None:
        rows = [r for r in self.by_type["degree_requirement_rule"] if r.requirement_category]
        self.rng.shuffle(rows)
        for r in rows[:limit]:
            self.add(
                question=_fmt(self.rng.choice(RULE_EN), prog=r.program, term=r.curriculum_term,
                              category=_pretty(r.requirement_category)),
                question_type="course_requirement",
                intent="mezuniyet_durumu",
                program=r.program,
                curriculum_term=r.curriculum_term,
                expected_sources=[r.source_document],
                expected_chunk_ids=[r.chunk_id],
                reference_answer=(r.text or "")[:400],
                notes=f"degree_requirement_rule; category={r.requirement_category}",
            )

    def minors(self, course_limit: int) -> None:
        for p in self.by_type["minor_requirement_profile"]:
            name = str(p.meta.get("program_name") or p.program)
            self.add(
                question=_fmt(self.rng.choice(MINOR_EN), name=name, term=p.curriculum_term),
                question_type="course_requirement",
                intent="minor",
                program=p.program,
                curriculum_term=p.curriculum_term,
                expected_sources=[p.source_document],
                expected_chunk_ids=[p.chunk_id],
                reference_answer=(p.text or "")[:400],
                notes="minor_requirement_profile; also checks data_role isolation from majors",
            )
        rows = [c for c in self.by_type["minor_requirement_pool_course"] if c.course_id]
        self.rng.shuffle(rows)
        for r in rows[:course_limit]:
            name = str(r.meta.get("program_name") or r.program)
            self.add(
                question=_fmt(self.rng.choice(MINOR_COURSE_EN), course=r.course_id, name=name,
                              term=r.curriculum_term),
                question_type="course_requirement",
                intent="minor",
                program=r.program,
                curriculum_term=r.curriculum_term,
                expected_sources=[r.source_document],
                expected_chunk_ids=[r.chunk_id],
                reference_answer=f"Yes - {r.course_id} appears in the {name} minor requirements.",
                notes="minor_requirement_pool_course",
            )

    # --- hard-negative aware families --------------------------------------------------

    def catalog_year_discrimination(self, limit: int) -> None:
        """Same course + same program in >=3 catalog terms: the term is the only disambiguator."""
        groups: dict[tuple[str, str], dict[str, list]] = defaultdict(lambda: defaultdict(list))
        for c in self.by_type["degree_requirement_pool_course"]:
            if c.course_id and c.curriculum_term:
                groups[(c.program, c.course_id)][c.curriculum_term].append(c)
        keys = sorted(k for k, v in groups.items() if len(v) >= 3)
        self.rng.shuffle(keys)
        for prog, course in keys[:limit]:
            terms = sorted(groups[(prog, course)])
            term = self.rng.choice(terms)
            rows = groups[(prog, course)][term]
            if len({r.requirement_category for r in rows}) != 1:
                continue
            self.add(
                question=f"On the {term} {prog} catalog specifically, which requirement category is {course} in?",
                question_type="catalog_year",
                program=prog,
                curriculum_term=term,
                expected_sources=[rows[0].source_document],
                expected_chunk_ids=sorted({r.chunk_id for r in rows}),
                reference_answer=f"On the {term} {prog} catalog, {course} is in the {_pretty(rows[0].requirement_category)} pool.",
                notes=f"hard negatives: same course in terms {terms} of the same program",
            )

    def program_discrimination(self, limit: int) -> None:
        """Same course + same term across >=3 programs: the program is the only disambiguator."""
        groups: dict[tuple[str, str], dict[str, list]] = defaultdict(lambda: defaultdict(list))
        for c in self.by_type["degree_requirement_pool_course"]:
            if c.course_id and c.program:
                groups[(c.course_id, c.curriculum_term)][c.program].append(c)
        keys = sorted(k for k, v in groups.items() if len(v) >= 3)
        self.rng.shuffle(keys)
        for course, term in keys[:limit]:
            progs = sorted(groups[(course, term)])
            prog = self.rng.choice(progs)
            rows = groups[(course, term)][prog]
            if len({r.requirement_category for r in rows}) != 1:
                continue
            name = self.corpus.program_names.get(prog, prog)
            self.add(
                question=f"For {name} ({prog}) students on the {term} catalog, which requirement category does {course} fall under?",
                question_type="program_specific",
                program=prog,
                curriculum_term=term,
                expected_sources=[rows[0].source_document],
                expected_chunk_ids=sorted({r.chunk_id for r in rows}),
                reference_answer=f"For {prog} {term}, {course} is in the {_pretty(rows[0].requirement_category)} pool.",
                notes=f"hard negatives: same course+term in programs {progs}",
            )


def load_existing(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            row["origin"] = "original_benchmark"
            row.setdefault("multi_evidence", len(row.get("expected_chunk_ids") or []) > 1)
            out.append(row)
    return out


def split_items(items: list[dict], rng: random.Random) -> dict[str, list[dict]]:
    """Partition by gold chunk so no chunk is gold in two splits (leakage control).

    Unanswerable items carry no gold, so they are stratified by hand across the splits.
    """
    # Group questions into components that share at least one gold chunk.
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    answerable = [i for i in items if i["expected_chunk_ids"]]
    unanswerable = [i for i in items if not i["expected_chunk_ids"]]
    for it in answerable:
        gold = it["expected_chunk_ids"]
        for cid in gold[1:]:
            union(gold[0], cid)

    comps: dict[str, list[dict]] = defaultdict(list)
    for it in answerable:
        comps[find(it["expected_chunk_ids"][0])].append(it)

    keys = sorted(comps)
    rng.shuffle(keys)
    # Original-benchmark items are pinned to test: they are the closest thing to a
    # manually-curated set and the task requires the final test set to preserve them.
    pinned = {k for k in keys if any(i["origin"] == "original_benchmark" for i in comps[k])}
    free = [k for k in keys if k not in pinned]

    n = len(free)
    n_train, n_dev = int(0.40 * n), int(0.20 * n)
    buckets = {
        "train": free[:n_train],
        "dev": free[n_train:n_train + n_dev],
        "test": free[n_train + n_dev:] + sorted(pinned),
    }
    out = {s: [i for k in ks for i in comps[k]] for s, ks in buckets.items()}

    rng.shuffle(unanswerable)
    for idx, it in enumerate(unanswerable):
        out[("train", "dev", "test")[idx % 3] if it["origin"] != "original_benchmark" else "test"].append(it)
    for s in out:
        out[s].sort(key=lambda i: i["id"] or "")
    return out


def verify(items: list[dict], corpus) -> None:
    known = set(corpus.by_id)
    bad = [(i["id"], c) for i in items for c in i["expected_chunk_ids"] if c not in known]
    if bad:
        sys.exit(f"ground truth references unknown chunk_ids: {bad[:10]}")
    print(f"verified: all gold chunk ids exist in the corpus ({len(known):,} chunks)")


def check_leakage(splits: dict[str, list[dict]]) -> None:
    gold = {s: {c for i in its for c in i["expected_chunk_ids"]} for s, its in splits.items()}
    qs = {s: {i["question"] for i in its} for s, its in splits.items()}
    fail = False
    for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
        g, q = gold[a] & gold[b], qs[a] & qs[b]
        print(f"  {a:5s} vs {b:5s}: shared gold chunks={len(g)}  identical questions={len(q)}")
        if g or q:
            fail = True
    if fail:
        sys.exit("LEAKAGE DETECTED - splits share gold chunks or questions")
    print("  leakage check passed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "data" / "benchmark"))
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    corpus = load_corpus()
    b = Builder(corpus, rng)

    b.course_category(limit=260)
    b.course_credits(limit=200)
    b.program_totals()
    b.category_minimums()
    b.rules(limit=110)
    b.minors(course_limit=110)
    b.catalog_year_discrimination(limit=130)
    b.program_discrimination(limit=130)

    generated = b.items
    existing = load_existing(PROJECT_ROOT / "data" / "benchmark" / "questions.jsonl")
    # Drop generated duplicates of a preserved original question.
    original_qs = {i["question"] for i in existing}
    generated = [i for i in generated if i["question"] not in original_qs]

    items = existing + generated

    # Ambiguity guard: if one wording still maps to several different gold sets, the question
    # is under-specified. Every one of those rows is a legitimate answer to it, so union the
    # gold rather than pretending one is uniquely correct (and keep a single copy).
    by_question: dict[str, list[dict]] = defaultdict(list)
    for it in items:
        by_question[it["question"]].append(it)
    collapsed, merged = [], 0
    for _, dupes in by_question.items():
        head = dupes[0]
        if len(dupes) > 1:
            gold, srcs = set(), set()
            for d in dupes:
                gold.update(d["expected_chunk_ids"])
                srcs.update(d["expected_sources"])
            head["expected_chunk_ids"] = sorted(gold)
            head["expected_sources"] = sorted(srcs)
            head["multi_evidence"] = len(gold) > 1
            head["notes"] = (head["notes"] + f" | merged {len(dupes)} equally-valid variants").strip()
            merged += len(dupes) - 1
        collapsed.append(head)
    items = collapsed
    if merged:
        print(f"ambiguity guard: merged {merged} duplicate-wording questions into their union gold")

    for n, it in enumerate(items, 1):
        if it["origin"] != "original_benchmark":
            it["id"] = f"g{n:04d}"

    verify(items, corpus)
    splits = split_items(items, rng)

    print(f"\ntotal questions: {len(items)}  (preserved original: {len(existing)}, generated: {len(generated)})")
    print("\nleakage audit:")
    check_leakage(splits)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        path = out_dir / f"retrieval_{split}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for r in rows:
                r["split"] = split
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        ans = sum(1 for r in rows if r["expected_chunk_ids"])
        multi = sum(1 for r in rows if r["multi_evidence"])
        tr = sum(1 for r in rows if r["language"] == "tr")
        types = defaultdict(int)
        for r in rows:
            types[r["question_type"]] += 1
        print(f"\n{split:5s} n={len(rows):4d}  scorable={ans:4d}  multi-evidence={multi:3d}  turkish={tr:3d}")
        print(f"       types: {dict(sorted(types.items()))}")
        print(f"       -> {path}")

    meta = {
        "seed": args.seed,
        "corpus_fingerprint": corpus.fingerprint,
        "corpus_chunks": len(corpus),
        "counts": {s: len(r) for s, r in splits.items()},
        "scorable": {s: sum(1 for i in r if i["expected_chunk_ids"]) for s, r in splits.items()},
        "preserved_original": len(existing),
        "split_policy": "by gold-chunk connected component; original benchmark pinned to test",
        "limitation": "question wording is templated; BM25 numbers are an optimistic bound",
    }
    (out_dir / "retrieval_splits_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nwrote {out_dir / 'retrieval_splits_meta.json'}")


if __name__ == "__main__":
    main()
