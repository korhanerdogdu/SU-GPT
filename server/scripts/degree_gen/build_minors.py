#!/usr/bin/env python3
"""
Parse Sabanci Minor degree-detail pages and emit RAG JSONL.

Minors are small and self-contained (Required + Core/Area Elective sections enumerated
inline on the p_degree_detail page; no separate p_list_courses pool pages).

Usage: python build_minors.py <pages_dir> <out_root>
  pages_dir contains files named <CODE>_<TERM>.html  (e.g. MATH-MINOR_202501.html)
  out_root gets  <out_root>/<CODE>/<TERM>.jsonl

Records use data_role="minor_requirement" (kept SEPARATE from major
data_role="curriculum_requirement" so minor courses never leak into a graduation audit).
"""
import html as htmllib
import hashlib
import json
import os
import re
import sys

SOURCE_AUTHORITY = "Sabanci University"
EXTRACTED_AT = "2026-07-20"
DATA_VERSION = "2026-07-20"
BASE_URL = "https://www.sabanciuniv.edu/en/prospective-students/degree-detail"
TERM_LABELS = {"202201":"Fall 2022-2023","202301":"Fall 2023-2024",
               "202401":"Fall 2024-2025","202501":"Fall 2025-2026"}

ROW = re.compile(
    r'subj_code=([A-Za-z]+)&crse_numb=([0-9A-Za-z]+)&lang=eng"[^>]*>\s*([^<]+?)\s*</a>\s*</td>\s*'
    r'<td>\s*<a[^>]*>\s*(.*?)\s*</a>\s*</td>\s*'
    r'<td>\s*([0-9]+)\s*</td>\s*'
    r'<td>\s*([0-9]+)\s*</td>'
    r'(?:\s*<td>\s*([A-Za-z/]*)\s*</td>)?',
    re.DOTALL)

SECTION_HDR = re.compile(
    r'<b>\s*(Required Courses|Core Elective Courses|Core Electives|Core Elective|'
    r'Area Elective Courses|Area Electives|Area Elective|Elective Courses|Electives|'
    r'University Courses|Free Electives|Free Elective)\s*</b>', re.IGNORECASE)

def strip(s):
    s = re.sub(r"<[^>]+>", " ", s)
    s = htmllib.unescape(s)
    return re.sub(r"\s+", " ", s).strip()

def norm_cat(label):
    l = label.lower()
    if "required" in l: return "required_courses"
    if "core" in l: return "core_electives"
    if "area" in l: return "area_electives"
    if "free" in l: return "free_electives"
    if "university" in l: return "university_courses"
    if "elective" in l: return "electives"
    return re.sub(r"[^a-z0-9]+","_",l).strip("_")

def parse_rows(chunk):
    out = []
    for m in ROW.finditer(chunk):
        code = re.sub(r"\s+"," ", m.group(3)).strip().upper()
        title = strip(m.group(4)); ects = int(m.group(5)); su = int(m.group(6))
        fac = (m.group(7) or "").strip()
        if code not in [c[0] for c in out]:
            out.append((code, title, su, ects, fac))
    return out

def parse_minor(html_txt):
    # program name + code
    name, code = "", ""
    m = re.search(r'<h2>\s*([^<]*?MINOR[^<]*?)\(\s*([A-Za-z0-9 _-]+?)\s*\)\s*</h2>', html_txt, re.IGNORECASE)
    if m:
        name = re.sub(r"\s+"," ", m.group(1)).strip().title()
        code = m.group(2).strip().upper()
    # advisor(s)
    advisors = []
    am = re.search(r'Diploma Area Advisor[s]?\s*:?\s*</b>\s*([^<]+)', html_txt)
    if am:
        advisors = [a.strip() for a in re.split(r",|;", strip(am.group(1))) if a.strip()]
    # summary table (first table after the SUMMARY heading)
    total_su = total_ects = None
    cat_meta = {}   # normalized category -> {min_su, min_ects, min_courses}
    sm = re.search(r'SUMMARY OF DEGREE REQUIREMENTS.*?</table>', html_txt, re.DOTALL|re.IGNORECASE)
    if sm:
        for tr in re.findall(r'<tr>(.*?)</tr>', sm.group(0), re.DOTALL):
            cells = [strip(c) for c in re.findall(r'<td[^>]*>(.*?)</td>', tr, re.DOTALL)]
            if len(cells) < 4:
                continue
            label = cells[0]
            def num(x):
                x = x.strip()
                return int(x) if x.isdigit() else None
            ects, su, mc = num(cells[1]), num(cells[2]), num(cells[3])
            if label.lower().startswith("total"):
                total_ects, total_su = ects, su
            elif label:
                cat_meta[norm_cat(label)] = {"min_ects":ects, "min_su":su, "min_courses":mc}
    # sections: slice body after the summary table by section headers
    body_start = sm.end() if sm else 0
    body = html_txt[body_start:]
    hdrs = [(mm.start(), norm_cat(mm.group(1)), mm.group(1)) for mm in SECTION_HDR.finditer(body)]
    sections = []
    for i,(pos,cat,raw) in enumerate(hdrs):
        end = hdrs[i+1][0] if i+1 < len(hdrs) else len(body)
        chunk = body[pos:end]
        rows = parse_rows(chunk)
        # external pool link? ("Click For ..." -> p_list_courses P_AREA=..&P_PROGRAM=..)
        ext = None
        em = re.search(r'p_list_courses\?P_TERM=[0-9]+&P_AREA=([A-Za-z0-9_]+)&P_PROGRAM=([A-Za-z0-9-]+)', chunk)
        if em:
            ext = {"area": em.group(1), "program": em.group(2)}
        if not rows and not ext:
            continue
        sc = chunk.find("subj_code=")
        row_start = chunk.rfind("<tr", 0, sc) if sc > 0 else -1
        note = strip(chunk[:row_start]) if row_start > 0 else strip(chunk)
        note = re.sub(r"^\s*(Required Courses|Core Elective(?:s| Courses)?|Area Elective(?:s| Courses)?|"
                      r"Free Electives?|University Courses|Electives?)\s*", "", note, flags=re.IGNORECASE)
        note = re.split(r"(?i)Course\s+Name\s+ECTS", note)[0].strip()
        note = re.sub(r"(?i)\s*Click For.*$", "", note).strip()
        if not re.search(r"(?i)minimum|at least|must|maximum|exclud|graduate|coded|second language|pool|required to take", note):
            note = ""
        sections.append({"category":cat, "raw_label":raw, "courses":rows, "note":note,
                         "external":ext, **cat_meta.get(cat, {})})
    return {"code":code, "name":name, "advisors":advisors,
            "total_su":total_su, "total_ects":total_ects, "sections":sections}

def cid(code, term, dtype, cat, key):
    raw = "|".join([code, term, dtype, cat, key])
    h = hashlib.sha1(raw.encode()).hexdigest()[:16]
    return f"minor_requirement:{code}:{term}:{dtype}/{cat}/{key}:{h}"

def build_records(parsed, code, term):
    label = TERM_LABELS.get(term, term)
    name = parsed["name"] or code
    tag = f"{name} ({code}), admit/curriculum term {label} ({term})"
    url = f"{BASE_URL}?SU_DEGREE.p_degree_detail?P_TERM={term}&P_PROGRAM={code}&P_SUBMIT=&P_LANG=EN&P_LEVEL=UG"
    def meta():
        return {"data_role":"minor_requirement","authority_level":"official","is_minor":True,
                "program":code,"program_name":name,"curriculum_term":term,"admit_term":term,
                "admit_term_label":label,"source_authority":SOURCE_AUTHORITY,"source_url":url,
                "source_document":f"minors/{code}/{term}.jsonl","extracted_at":EXTRACTED_AT,
                "data_version":DATA_VERSION}
    def rec(dtype, text, extra):
        r = meta(); r["document_type"]=dtype; r["text"]=text; r.update(extra); return r
    recs = []
    cats = [{"category":s["category"],"min_su_credits":s.get("min_su"),"min_courses":s.get("min_courses")}
            for s in parsed["sections"]]
    ptext = (f"Minor degree requirement profile for {tag}. Total minimum SU credits: {parsed['total_su']}. "
             f"Total minimum ECTS: {parsed['total_ects']}. Sections: " +
             ", ".join(f"{s['category'].replace('_',' ')} ({s.get('min_su')} SU, {len(s['courses'])} courses in pool)"
                       for s in parsed["sections"]) + ". "
             + (f"Diploma area advisors: {', '.join(parsed['advisors'])}." if parsed["advisors"] else ""))
    recs.append(rec("minor_requirement_profile", ptext,
                    {"total_min_su_credits":parsed["total_su"],"total_min_ects":parsed["total_ects"],
                     "advisors":parsed["advisors"],"categories":cats}))
    for s in parsed["sections"]:
        cat = s["category"]; ids = [c[0] for c in s["courses"]]
        summ = f"The {cat.replace('_',' ')} pool for the {tag} contains {len(ids)} courses"
        if s.get("min_su") is not None: summ += f"; minimum {s['min_su']} SU credits required"
        if s.get("min_courses") is not None: summ += f"; minimum {s['min_courses']} courses required"
        summ += ". Courses: " + ", ".join(f"{c[0]} {c[1]}" for c in s["courses"]) + "."
        if s["note"]: summ += " " + s["note"]
        recs.append(rec("minor_requirement_category_pool", summ,
                        {"requirement_category":cat,"course_ids":ids,
                         "min_su_credits":s.get("min_su"),"min_courses":s.get("min_courses"),
                         "constraint_note":s["note"]}))
        for code_, title, su, ects, fac in s["courses"]:
            ct = (f"For the {tag}, the course {code_} - {title} ({su} SU credits, {ects} ECTS"
                  f"{', ' + fac if fac else ''}) belongs to the minor's {cat.replace('_',' ')} pool.")
            recs.append(rec("minor_requirement_pool_course", ct,
                            {"requirement_category":cat,"course_id":code_,"course_title":title,
                             "su_credits":su,"ects":ects,"faculty":fac}))
        if s["note"]:
            recs.append(rec("minor_requirement_rule", f"Constraint for {cat.replace('_',' ')} of {tag}: {s['note']}",
                            {"requirement_category":cat,"rule_id":f"{cat}_constraint","constraint_note":s["note"]}))
    # unique chunk ids
    seen=set()
    for r in recs:
        key = r.get("course_id") or r.get("rule_id") or ""
        c = cid(code, term, r["document_type"], r.get("requirement_category",""), key); base=c; n=1
        while c in seen: n+=1; c=f"{base}#{n}"
        seen.add(c); r["chunk_id"]=c
    return recs

def load_external_pools(pool_dir):
    # files named <PROGRAM>__<AREA>__<term>.html
    ext = {}
    if not os.path.isdir(pool_dir):
        return ext
    for fn in os.listdir(pool_dir):
        if not fn.endswith(".html"):
            continue
        program, area, term = fn[:-5].split("__")
        rows = parse_rows(open(os.path.join(pool_dir, fn), encoding="utf-8", errors="replace").read())
        ext[(program, area, term)] = rows
    return ext

def main(pages_dir, out_root, pool_dir=None):
    ext_pools = load_external_pools(pool_dir) if pool_dir else {}
    files = sorted(f for f in os.listdir(pages_dir) if f.endswith(".html"))
    summary = {}
    for fn in files:
        code, term = fn[:-5].rsplit("_", 1)
        parsed = parse_minor(open(os.path.join(pages_dir, fn), encoding="utf-8", errors="replace").read())
        pcode = parsed["code"] or code
        # fill sections that point to an external pool
        for s in parsed["sections"]:
            if not s["courses"] and s.get("external"):
                key = (s["external"]["program"], s["external"]["area"], term)
                s["courses"] = ext_pools.get(key, [])
        recs = build_records(parsed, pcode, term)
        outdir = os.path.join(out_root, pcode); os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(outdir, f"{term}.jsonl"), "w", encoding="utf-8") as fh:
            for r in recs: fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        summary.setdefault(pcode, {})[term] = {
            "total_su":parsed["total_su"],
            "sections":{s["category"]:{"min_su":s.get("min_su"),"n":len(s["courses"])} for s in parsed["sections"]},
            "records":len(recs)}
    print(json.dumps(summary, indent=1, ensure_ascii=False))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
