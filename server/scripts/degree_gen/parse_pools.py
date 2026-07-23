#!/usr/bin/env python3
"""Parse downloaded SU degree pool HTML pages into a pools+catalog JSON.

Table row shape (confirmed):
  <td>marker</td>
  <td><a ...subj_code=XX&crse_numb=YY...>XX YY</a></td>
  <td><a ...>Course Name</a></td>
  <td>ECTS</td>
  <td>SU</td>
  <td>Faculty</td>  (FASS/SBS pool has an extra <td>Area</td>)
"""
import html
import json
import os
import re
import sys

PAGES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "pages")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), "parsed_pools.json")
TERMS = ["202201", "202301", "202401", "202501"]
POOLS = {"CEL": "core_electives", "AEL": "area_electives", "FEL": "free_electives",
         "FCFENS": "faculty_fens", "FCFASS": "faculty_fass", "FCSBS": "faculty_sbs"}

# Match a course row: code link (with subj/crse), then title link, then ECTS td, then SU td.
ROW = re.compile(
    r'subj_code=([A-Za-z]+)&crse_numb=([0-9A-Za-z]+)&lang=eng"[^>]*>\s*([^<]+?)\s*</a>\s*</td>\s*'
    r'<td>\s*<a[^>]*>\s*(.*?)\s*</a>\s*</td>\s*'
    r'<td>\s*([0-9]+)\s*</td>\s*'
    r'<td>\s*([0-9]+)\s*</td>'
    r'(?:\s*<td>\s*([A-Za-z/]*)\s*</td>)?',
    re.DOTALL,
)

def clean(s):
    s = re.sub(r"<[^>]+>", "", s)          # strip any stray tags
    s = html.unescape(s)
    return re.sub(r"\s+", " ", s).strip()

def norm_code(display):
    # display like "ACC 201" / "CIP 101N" / "CS 48006" -> normalized single space
    return re.sub(r"\s+", " ", display).strip().upper()

def parse_file(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        html_txt = f.read()
    rows = []
    for m in ROW.finditer(html_txt):
        code = norm_code(m.group(3))
        title = clean(m.group(4))
        ects = int(m.group(5))
        su = int(m.group(6))
        faculty = (m.group(7) or "").strip()
        rows.append((code, title, su, ects, faculty))
    return rows

def main():
    catalog = {}          # code -> {title, su, ects}
    pools = {t: {} for t in TERMS}
    conflicts = []
    for t in TERMS:
        for key, cat in POOLS.items():
            path = os.path.join(PAGES, f"{key}_{t}.html")
            rows = parse_file(path)
            seen = []
            for code, title, su, ects, faculty in rows:
                if code not in seen:
                    seen.append(code)
                if code in catalog:
                    old = catalog[code]
                    if (old["su"], old["ects"]) != (su, ects):
                        conflicts.append((code, (old["su"], old["ects"]), (su, ects), f"{key}_{t}"))
                    if faculty and not old.get("faculty"):
                        old["faculty"] = faculty
                else:
                    catalog[code] = {"title": title, "su": su, "ects": ects, "faculty": faculty}
            pools[t][cat] = seen
    out = {"catalog": catalog, "pools": pools}
    outpath = OUT
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    # report
    print("catalog size:", len(catalog))
    for t in TERMS:
        counts = {k: len(v) for k, v in pools[t].items()}
        print(t, counts)
    if conflicts:
        print("\nCREDIT CONFLICTS (same code, different su/ects across pages):")
        for c in conflicts[:40]:
            print(" ", c)
    print("\nwrote", outpath)

if __name__ == "__main__":
    main()
