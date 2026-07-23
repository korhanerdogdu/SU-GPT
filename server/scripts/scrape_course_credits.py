from __future__ import annotations

"""
Fill per-course engineering / basic-science ECTS in data/course_catalog/current.jsonl.

The degree-requirement corpus only has SU/ECTS/faculty per course. The engineering vs
basic-science ECTS split lives on each course's Sabancı catalog page, e.g. NS 101 shows
"6 ECTS (ENGINEERING:0 / BASIC:6)" and CS 303 shows "ENGINEERING:6 / BASIC:1". This scraper
fetches each course page (we already have every course code) and parses that split, so the
graduation audit can check the Engineering (>=90 ECTS) and Basic Science (>=60 ECTS) rules
against the student's course history.

Idempotent: results are cached in data/course_catalog/credits_cache.json, so re-runs only
fetch courses that aren't cached yet. Courses with no split shown are recorded as 0/0.

Usage: python server/scripts/scrape_course_credits.py [--data-dir DIR] [--limit N] [--workers 8]
"""

import argparse
import json
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.config import DEGREE_DATA_DIR

URL = ("https://www.sabanciuniv.edu/en/prospective-students/degree-detail?"
       "sabanci_www.p_get_courses?levl_code=UG&subj_code={subj}&crse_numb={num}&lang=eng")
SPLIT_RE = re.compile(r"ENGINEERING:\s*([0-9.]+)\s*/\s*BASIC:\s*([0-9.]+)", re.IGNORECASE)


def _fetch(subj: str, num: str) -> tuple[float, float] | None:
    try:
        req = urllib.request.Request(URL.format(subj=subj, num=num), headers={"User-Agent": "adviSU/1.0"})
        with urllib.request.urlopen(req, timeout=25) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception:
        return None
    m = SPLIT_RE.search(html)
    if not m:
        return (0.0, 0.0)  # page loaded but no eng/basic split -> genuinely 0/0
    return (float(m.group(1)), float(m.group(2)))


def _num(x: float) -> int | float:
    return int(x) if float(x).is_integer() else x


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DEGREE_DATA_DIR))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    catalog_path = data_dir / "course_catalog" / "current.jsonl"
    cache_path = data_dir / "course_catalog" / "credits_cache.json"
    rows = [json.loads(l) for l in catalog_path.open(encoding="utf-8") if l.strip()]
    cache: dict[str, list] = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}

    todo = [r for r in rows if r["course_id"] not in cache]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(rows)} courses, {len(cache)} cached, fetching {len(todo)}...")

    def work(r: dict) -> tuple[str, tuple[float, float] | None]:
        return r["course_id"], _fetch(r["subject"], r["number"])

    fetched = failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, (code, res) in enumerate(pool.map(work, todo), 1):
            if res is None:
                failed += 1
            else:
                cache[code] = list(res)
                fetched += 1
            if i % 50 == 0:
                cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
                print(f"  {i}/{len(todo)} (ok={fetched} fail={failed})", flush=True)
    cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")

    # write cached splits back into the catalog
    filled = 0
    with catalog_path.open("w", encoding="utf-8") as fh:
        for r in rows:
            if r["course_id"] in cache:
                eng, basic = cache[r["course_id"]]
                r["engineering_ects"] = _num(eng)
                r["basic_science_ects"] = _num(basic)
                filled += 1
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Done. fetched={fetched} failed={failed} | catalog rows filled={filled}/{len(rows)}")
    if failed:
        print("Some fetches failed (transient) — re-run to fill them (cache makes it cheap).")


if __name__ == "__main__":
    main()
