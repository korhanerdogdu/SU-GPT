#!/usr/bin/env python3
"""
Generic SU degree-requirement JSONL generator (program-agnostic).

Usage:
  python gen_degree_reqs.py <config.json> <parsed_pools.json> <out_dir>

config.json shape:
{
  "program": "IE", "degree_code": "BSMS", "p_program": "BSMS",
  "program_name": "...", "advisors": ["...","..."],
  "hardcoded": { "IF 100": ["Computational...", 3, 5, "FENS"], ... },
  "mandatory_uni": ["IF 100", ...],
  "hum_pool": { "202201": ["HUM 201", ...], ... },
  "required": { "202201": ["CS 201", ...], ... },
  "terms": { "202201": {"label":"Fall 2022-2023","total_su":125,...,
                          "math_choice":true, "prog_choice":["CS 201","DSA 201"]}, ... }
}
Pools (core/area/free electives + FENS/FASS/SBS faculty) come from parsed_pools.json.
"""
import hashlib
import json
import os
import sys

SOURCE_AUTHORITY = "Sabanci University"
EXTRACTED_AT = "2026-07-20"
DATA_VERSION = "2026-07-20"
BASE_URL = "https://www.sabanciuniv.edu/en/prospective-students/degree-detail"

def load(path):
    return json.load(open(path, encoding="utf-8"))

def main(cfg_path, parsed_path, out_dir):
    cfg = load(cfg_path)
    parsed = load(parsed_path)
    CATALOG = parsed["catalog"]
    POOLS = parsed["pools"]
    PROGRAM = cfg["program"]; DEGREE_CODE = cfg["degree_code"]; PPROG = cfg["p_program"]
    PROGRAM_NAME = cfg["program_name"]; ADVISORS = cfg["advisors"]
    HARDCODED = cfg["hardcoded"]; MANDATORY_UNI = cfg["mandatory_uni"]
    HUM_POOL = cfg["hum_pool"]; REQUIRED = cfg["required"]; TERMS = cfg["terms"]
    SUF = cfg.get("pool_suffix", {"core":"CEL","area":"AEL","free":"FEL"})
    CORE_AREA = f"{PPROG}_{SUF['core']}"; AREA_AREA = f"{PPROG}_{SUF['area']}"; FREE_AREA = f"{PPROG}_{SUF['free']}"
    HUM_MIN = cfg.get("hum_min_courses", 1)
    EXTRA_POOLS = cfg.get("extra_pools", [])          # e.g. Psychology "philosophy_requirement"
    FACULTY_RULE = cfg.get("faculty_rule")            # {"text": "... {tag} ...", "meta": {...}}
    CORE_NOTE = cfg.get("core_note", ""); AREA_NOTE = cfg.get("area_note", ""); FREE_NOTE = cfg.get("free_note", "")

    def degree_url(term):
        return f"{BASE_URL}?SU_DEGREE.p_degree_detail?P_TERM={term}&P_PROGRAM={PPROG}&P_SUBMIT=&P_LANG=EN&P_LEVEL=UG"
    def pool_url(term, area, extra=""):
        return f"{BASE_URL}?SU_DEGREE.p_list_courses?P_TERM={term}&P_AREA={area}&P_PROGRAM={PPROG}{extra}&P_LANG=EN&P_LEVEL=UG"
    def info(code):
        if code in CATALOG:
            c = CATALOG[code]; return c["title"], c["su"], c["ects"], c.get("faculty", "")
        t, su, ects, fac = HARDCODED[code]; return t, su, ects, fac
    def base_meta(term):
        t = TERMS[term]
        return {"data_role":"curriculum_requirement","authority_level":"official",
                "program":PROGRAM,"degree_code":DEGREE_CODE,"program_name":PROGRAM_NAME,
                "curriculum_term":term,"admit_term":term,"admit_term_label":t["label"],
                "source_authority":SOURCE_AUTHORITY,"source_url":degree_url(term),
                "source_document":f"degree_requirements/{PROGRAM}/{term}.jsonl",
                "extracted_at":EXTRACTED_AT,"data_version":DATA_VERSION}
    def rec(term, dtype, text, extra):
        r = base_meta(term); r["document_type"] = dtype; r["text"] = text; r.update(extra); return r
    def cid(term, dtype, cat, key):
        raw = "|".join([PROGRAM, DEGREE_CODE, term, dtype, cat, key])
        h = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
        return f"curriculum_requirement:{PROGRAM}:{term}:{dtype}/{cat}/{key}:{h}"

    def build_term(term):
        t = TERMS[term]; label = t["label"]
        tag = f"{PROGRAM_NAME} ({DEGREE_CODE}), admit/curriculum term {label} ({term})"
        records = []
        has_eng = t.get("eng_ects") is not None
        has_bsci = t.get("bsci_ects") is not None
        categories = [
            {"category":"university_courses","min_su_credits":t["uni_su"],"min_courses":t["uni_courses"]},
            {"category":"required_courses","min_su_credits":t["req_su"],"min_courses":t["req_courses"]},
        ]
        for ep in EXTRA_POOLS:
            categories.append({"category":ep["category"],"min_su_credits":ep.get("min_su"),"min_courses":ep.get("min_courses")})
        categories += [
            {"category":"core_electives","min_su_credits":t["core_su"]},
            {"category":"area_electives","min_su_credits":t["area_su"]},
            {"category":"free_electives","min_su_credits":t["free_su"]},
        ]
        if has_eng: categories.append({"category":"engineering","min_ects":t["eng_ects"]})
        if has_bsci: categories.append({"category":"basic_science","min_ects":t["bsci_ects"]})
        categories.append({"category":"faculty_courses","min_courses":t["faculty_courses"]})
        eng_txt = f"Engineering {t['eng_ects']} ECTS, " if has_eng else ""
        bsci_txt = f"Basic Science {t['bsci_ects']} ECTS, " if has_bsci else ""
        extra_txt = "".join(f"{ep['category'].replace('_',' ').title()} {ep.get('min_su')} SU, " for ep in EXTRA_POOLS)
        ptext = (f"Degree requirement profile for {tag}. Total minimum SU credits: {t['total_su']}. "
                 f"Total minimum ECTS: {t['total_ects']}. Category minimums: University Courses {t['uni_su']} SU "
                 f"({t['uni_courses']} courses), Required Courses {t['req_su']} SU ({t['req_courses']} courses), "
                 f"{extra_txt}Core Electives {t['core_su']} SU, Area Electives {t['area_su']} SU, Free Electives {t['free_su']} SU, "
                 f"{eng_txt}{bsci_txt}Faculty Courses at least "
                 f"{t['faculty_courses']} courses. No explicit GPA requirement is published on the degree-detail page. "
                 f"Diploma area advisors: {', '.join(ADVISORS)}.")
        records.append(rec(term, "degree_requirement_profile", ptext, {
            "total_min_su_credits":t["total_su"],"total_min_ects":t["total_ects"],
            "min_program_gpa":None,"min_cumulative_gpa":None,"advisors":ADVISORS,"categories":categories}))

        def emit_pool(category, codes, min_su=None, min_courses=None, note="", src=None):
            ids = list(codes)
            summ = f"The {category.replace('_',' ')} pool for {tag} contains {len(ids)} courses"
            if min_su is not None: summ += f"; minimum {min_su} SU credits required"
            if min_courses is not None: summ += f"; minimum {min_courses} courses required"
            listed = ids if len(ids) <= 60 else ids[:40]
            summ += ". Courses: " + ", ".join(f"{c} {info(c)[0]}" for c in listed)
            if len(ids) > 60:
                summ += f", ... and {len(ids)-40} more (full list in course_ids and individual course records)"
            summ += "."
            if note: summ += " " + note
            extra = {"requirement_category":category,"course_ids":ids,"min_su_credits":min_su,"min_courses":min_courses}
            if src: extra["pool_source_url"] = src
            records.append(rec(term, "degree_requirement_category_pool", summ, extra))
            for c in ids:
                title, su, ects, fac = info(c)
                ct = (f"For {tag}, the course {c} - {title} ({su} SU credits, {ects} ECTS"
                      f"{', ' + fac if fac else ''}) belongs to the {category.replace('_',' ')} pool.")
                if min_su is not None:
                    ct += f" A minimum of {min_su} SU credits of {category.replace('_',' ')} is required."
                records.append(rec(term, "degree_requirement_pool_course", ct, {
                    "requirement_category":category,"course_id":c,"course_title":title,
                    "su_credits":su,"ects":ects,"faculty":fac}))

        emit_pool("university_courses_mandatory", MANDATORY_UNI,
                  note="All freshman (1XX) courses and SPS 303 are mandatory.")
        hum_note = ("Exactly one HUM course must be completed to satisfy the university HUM requirement."
                    if HUM_MIN == 1 else
                    f"At least {HUM_MIN} HUM courses must be completed (a 2XX-level course first, then a 3XX-level "
                    f"course) to satisfy the university HUM requirement.")
        emit_pool("university_courses_hum_pool", HUM_POOL[term], min_courses=HUM_MIN, note=hum_note)
        notes = []
        if t.get("prog_choice"):
            a, b = t["prog_choice"]; notes.append(f"Complete either {a} or {b}, not both.")
        for a, b in (t.get("choices") or []):
            notes.append(f"Complete either {a} or {b}, not both.")
        if t.get("math_choice"):
            notes.append("Complete either MATH 201 or MATH 212, not both.")
        if t.get("math_212_or_201_202"):
            notes.append("Complete either MATH 212, or both MATH 201 and MATH 202.")
        if t.get("math_exclusion_2025"):
            notes.append("MATH 212 is required; for 2025-2026 and later MATH 201 and MATH 202 are not in any pool.")
        emit_pool("required_courses", REQUIRED[term], min_su=t["req_su"], min_courses=t["req_courses"],
                  note=" ".join(notes))
        for ep in EXTRA_POOLS:
            emit_pool(ep["category"], ep["courses"], min_su=ep.get("min_su"),
                      min_courses=ep.get("min_courses"), note=ep.get("note",""))
        emit_pool("core_electives", POOLS[term]["core_electives"], min_su=t["core_su"],
                  note=("Extra Core Elective courses count toward Area Elective requirements. " + CORE_NOTE).strip(),
                  src=pool_url(term, CORE_AREA))
        emit_pool("area_electives", POOLS[term]["area_electives"], min_su=t["area_su"],
                  note=("Extra Area Elective courses count toward Free Elective requirements. " + AREA_NOTE).strip(),
                  src=pool_url(term, AREA_AREA))
        emit_pool("free_electives", POOLS[term]["free_electives"], min_su=t["free_su"],
                  note=("Free electives come from FASS, SBS and FENS courses (excluding University Courses). "
                        "School of Languages (SL) courses do not count. " + FREE_NOTE).strip(),
                  src=pool_url(term, FREE_AREA))
        emit_pool("faculty_courses_fens", POOLS[term]["faculty_fens"],
                  note="FENS faculty-course pool. Counts toward the faculty-course requirement.",
                  src=pool_url(term, "FC_FENS", extra="&P_FAC=E"))
        emit_pool("faculty_courses_fass", POOLS[term]["faculty_fass"],
                  note="FASS faculty-course pool. Counts toward the 5-course faculty requirement.",
                  src=pool_url(term, "FC_FASS", extra="&P_FAC=S"))
        emit_pool("faculty_courses_sbs", POOLS[term]["faculty_sbs"],
                  note="SBS (School of Management) faculty-course pool. Counts toward the 5-course faculty requirement.",
                  src=pool_url(term, "FC_SOM", extra="&P_FAC=M"))

        rules = [
            ("free_electives_rule","free_electives",
             f"Free Electives for {tag}: {t['free_su']} SU credits from FASS, SBS and FENS courses (excluding "
             f"University Courses). School of Languages (SL) courses do not count as free electives.",
             {"min_su_credits":t["free_su"],"pool_source_url":pool_url(term,FREE_AREA)}),
        ]
        if has_eng:
            rules.append(("engineering_ects_rule","engineering",
             f"Engineering requirement for {tag}: at least {t['eng_ects']} ECTS of Engineering credits.",
             {"min_ects":t["eng_ects"]}))
        if has_bsci:
            rules.append(("basic_science_ects_rule","basic_science",
             f"Basic Science requirement for {tag}: at least {t['bsci_ects']} ECTS of Basic Science credits.",
             {"min_ects":t["bsci_ects"]}))
        if FACULTY_RULE:
            ftext = FACULTY_RULE["text"].replace("{tag}", tag)
            fmeta = dict(FACULTY_RULE.get("meta", {}))
        else:
            ftext = (f"Faculty Courses requirement for {tag}: at least {t['faculty_courses']} courses from FENS faculty "
                     f"courses and/or FASS and SBS courses. At least 2 must be MATH-coded, and at least 3 from the FENS pool.")
            fmeta = {"min_math_courses":2,"min_fens_courses":3}
        fmeta.setdefault("min_courses", t["faculty_courses"])
        fmeta["fass_pool_source_url"] = pool_url(term,"FC_FASS",extra="&P_FAC=S")
        fmeta["sbs_pool_source_url"] = pool_url(term,"FC_SOM",extra="&P_FAC=M")
        rules += [
            ("faculty_courses_rule","faculty_courses", ftext, fmeta),
            ("core_to_area_overflow","area_electives",
             f"Overflow rule for {tag}: Core Elective SU beyond {t['core_su']} count toward Area Electives.",
             {"source_category":"core_electives","target_category":"area_electives","condition":"source_minimum_satisfied"}),
            ("area_to_free_overflow","free_electives",
             f"Overflow rule for {tag}: Area Elective SU beyond {t['area_su']} count toward Free Electives.",
             {"source_category":"area_electives","target_category":"free_electives","condition":"source_minimum_satisfied"}),
        ]
        if t.get("prog_choice"):
            a, b = t["prog_choice"]
            rules.append(("programming_course_choice","required_courses",
                f"Choice rule for {tag}: complete either {a} or {b}; only one counts toward Required Courses.",
                {"choice_courses":[a,b],"max_count":1}))
        for a, b in (t.get("choices") or []):
            rid = "course_choice_" + a.replace(" ","") + "_or_" + b.replace(" ","")
            rules.append((rid,"required_courses",
                f"Choice rule for {tag}: complete either {a} or {b}; only one counts toward Required Courses.",
                {"choice_courses":[a,b],"max_count":1}))
        if t.get("math_choice"):
            rules.append(("math201_math212_choice","required_courses",
                f"Choice rule for {tag}: complete either MATH 201 or MATH 212; only one counts toward Required Courses.",
                {"choice_courses":["MATH 201","MATH 212"],"max_count":1}))
        if t.get("math_212_or_201_202"):
            rules.append(("math212_or_math201_math202","required_courses",
                f"Math requirement for {tag}: complete either MATH 212, or BOTH MATH 201 and MATH 202. Extra math "
                f"courses beyond the requirement do not count toward core, area, or free elective pools.",
                {"option_a":["MATH 212"],"option_b":["MATH 201","MATH 202"]}))
        if t.get("math_exclusion_2025"):
            rules.append(("math201_math202_exclusion_2025","required_courses",
                f"Rule for {tag}: for 2025-2026 and later, MATH 201 and MATH 202 are not in any course pool; "
                f"MATH 212 is required.", {"excluded_courses":["MATH 201","MATH 202"]}))
        for ep in EXTRA_POOLS:
            if ep.get("is_choice"):
                n = ep.get("min_courses", 1)
                rules.append((ep["category"] + "_choice", ep["category"],
                    f"Choice rule for {tag}: complete {n} course from the {ep['category'].replace('_',' ')} pool "
                    f"({', '.join(ep['courses'])}).",
                    {"choice_courses":ep["courses"],"max_count":n}))
        for rid, cat, text, extra in rules:
            e = {"rule_id":rid,"requirement_category":cat}; e.update(extra)
            records.append(rec(term, "degree_requirement_rule", text, e))
        return records

    os.makedirs(out_dir, exist_ok=True)
    summary = {}
    for term in TERMS:
        recs = build_term(term)
        seen = set()
        for r in recs:
            key = r.get("course_id") or r.get("rule_id") or ""
            c = cid(term, r["document_type"], r.get("requirement_category",""), key)
            base = c; n = 1
            while c in seen:
                n += 1; c = f"{base}#{n}"
            seen.add(c); r["chunk_id"] = c
        path = os.path.join(out_dir, f"{term}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        by_type = {}
        for r in recs:
            by_type[r["document_type"]] = by_type.get(r["document_type"], 0) + 1
        summary[term] = {"records":len(recs),"by_type":by_type,"unique_chunk_ids":len(seen)}
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
