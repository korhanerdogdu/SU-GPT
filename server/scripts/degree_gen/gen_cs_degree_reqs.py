#!/usr/bin/env python3
"""
Generate CS (BSCS) degree-requirement JSONL for 4 admit/curriculum terms,
following ACADEMIC_ADVISING_DATA_ROADMAP.md (data_role=curriculum_requirement).

Course pools (Core / Area / Free electives, and FENS/FASS/SBS Faculty Courses) are
loaded from parsed_pools.json, produced by parse_pools.py from the raw SU degree pages
(fully enumerated, no truncation). University + Required courses are defined here from
the main degree pages.

Output: data/degree_requirements/CS/{TERM}.jsonl
Usage:  python gen_cs_degree_reqs.py <OUT_DIR> <PARSED_POOLS_JSON>
"""
import hashlib
import json
import os
import sys

PROGRAM = "CS"
DEGREE_CODE = "BSCS"
PROGRAM_NAME = "Computer Science and Engineering Undergraduate Program"
SOURCE_AUTHORITY = "Sabanci University"
EXTRACTED_AT = "2026-07-20"
DATA_VERSION = "2026-07-20"
ADVISORS = ["Cemal Yilmaz", "Onur Varol"]
BASE_URL = "https://www.sabanciuniv.edu/en/prospective-students/degree-detail"

def degree_url(term):
    return f"{BASE_URL}?SU_DEGREE.p_degree_detail?P_TERM={term}&P_PROGRAM=BSCS&P_SUBMIT=&P_LANG=EN&P_LEVEL=UG"

def pool_url(term, area, extra=""):
    return f"{BASE_URL}?SU_DEGREE.p_list_courses?P_TERM={term}&P_AREA={area}&P_PROGRAM=BSCS{extra}&P_LANG=EN&P_LEVEL=UG"

# Courses that appear only in University / Required lists (not in any elective/faculty
# pool), so not present in parsed catalog. Credits are ECTS/SU from the main pages.
HARDCODED = {
    "IF 100": ("Computational Approaches to Problem Solving", 3, 5, "FENS"),
    "MATH 101": ("Calculus I", 3, 6, "FENS"),
    "CIP 101N": ("Civic Involvement Projects I-N", 0, 1, "FASS"),
    "NS 101": ("Science of Nature I", 4, 6, "FENS"),
    "SPS 101": ("Humanity and Society I", 3, 6, "FASS"),
    "TLL 101": ("Turkish Language and Literature I", 2, 3, "SL"),
    "AL 102": ("Academic Literacies", 3, 5, "SL"),
    "MATH 102": ("Calculus II", 3, 6, "FENS"),
    "NS 102": ("Science of Nature II", 4, 6, "FENS"),
    "SPS 102": ("Humanity and Society II", 3, 6, "FASS"),
    "TLL 102": ("Turkish Language and Literature II", 2, 3, "SL"),
    "HIST 191": ("Principles of Ataturk and the History of the Turkish Revolution I", 2, 3, "FASS"),
    "HIST 192": ("Principles of Ataturk and the History of the Turkish Revolution II", 2, 3, "FASS"),
    "PROJ 201": ("Undergraduate Project Course", 1, 1, "FENS"),
    "SPS 303": ("Law and Ethics", 3, 5, "FASS"),
    "HUM 201": ("Major Works of Literature", 3, 5, "FASS"),
    "HUM 202": ("Major Works of Western Art", 3, 5, "FASS"),
    "HUM 207": ("Major Works of Western Philosophy", 3, 5, "FASS"),
    "HUM 304": ("Major Works of Classical Music", 3, 5, "FASS"),
    "HUM 311": ("Major Works of Literature: The World Before Modernity", 3, 5, "FASS"),
    "HUM 312": ("Major Works of Modern Art", 3, 5, "FASS"),
    "HUM 317": ("Major Works of Moral Philosophy", 3, 5, "FASS"),
    "HUM 321": ("Major Works of Literature: The Modern World", 3, 5, "FASS"),
    "HUM 322": ("Major Works of Art: The World Before Modernity", 3, 5, "FASS"),
    "HUM 324": ("Major Works of 20th Century Music", 3, 5, "FASS"),
    "HUM 371": ("Major Works of Literature: The Islamic World", 3, 5, "FASS"),
    "CS 300": ("Data Structures", 3, 6, "FENS"),
    "CS 301": ("Algorithms", 3, 6, "FENS"),
    "CS 303": ("Logic and Digital System Design", 4, 7, "FENS"),
    "CS 395": ("Internship Project", 0, 5, "FENS"),
    "ENS 491": ("Graduation Project (Design)", 1, 2, "FENS"),
    "ENS 492": ("Graduation Project (Implementation)", 3, 5, "FENS"),
}

MANDATORY_UNI = ["IF 100","MATH 101","CIP 101N","NS 101","SPS 101","TLL 101","AL 102",
                 "MATH 102","NS 102","SPS 102","TLL 102","HIST 191","HIST 192","PROJ 201","SPS 303"]
HUM_POOL = {
    "202201": ["HUM 201","HUM 202","HUM 207","HUM 304","HUM 311","HUM 312","HUM 317","HUM 321","HUM 322","HUM 324","HUM 371"],
    "202301": ["HUM 201","HUM 202","HUM 207","HUM 311","HUM 312","HUM 317","HUM 321","HUM 322","HUM 371"],
    "202401": ["HUM 201","HUM 202","HUM 207","HUM 311","HUM 312","HUM 317","HUM 321","HUM 322","HUM 371"],
    "202501": ["HUM 201","HUM 202","HUM 207","HUM 311","HUM 312","HUM 317","HUM 321","HUM 322","HUM 371"],
}
REQUIRED = {
    "202201": ["CS 201","CS 204","CS 300","CS 301","CS 303","CS 395","ENS 491","ENS 492","MATH 201","MATH 203","MATH 204","MATH 212"],
    "202301": ["CS 201","CS 204","CS 300","CS 301","CS 303","CS 395","ENS 491","ENS 492","MATH 201","MATH 203","MATH 204","MATH 212"],
    "202401": ["CS 201","CS 204","CS 300","CS 301","CS 303","CS 395","ENS 491","ENS 492","MATH 201","MATH 203","MATH 204","MATH 212"],
    "202501": ["CS 201","CS 204","CS 300","CS 301","CS 303","CS 306","CS 308","CS 395","ENS 491","ENS 492","MATH 203","MATH 204","MATH 212","PHYS 113"],
}
TERMS = {
    "202201": {"label":"Fall 2022-2023","total_su":125,"total_ects":240,"uni_su":41,"uni_courses":16,"req_su":29,"req_courses":11,"core_su":31,"area_su":9,"free_su":15,"eng_ects":90,"bsci_ects":60,"faculty_courses":5,"math_choice":True},
    "202301": {"label":"Fall 2023-2024","total_su":125,"total_ects":240,"uni_su":41,"uni_courses":16,"req_su":29,"req_courses":11,"core_su":31,"area_su":9,"free_su":15,"eng_ects":90,"bsci_ects":60,"faculty_courses":5,"math_choice":True},
    "202401": {"label":"Fall 2024-2025","total_su":125,"total_ects":240,"uni_su":41,"uni_courses":16,"req_su":29,"req_courses":11,"core_su":31,"area_su":9,"free_su":15,"eng_ects":90,"bsci_ects":60,"faculty_courses":5,"math_choice":True},
    "202501": {"label":"Fall 2025-2026","total_su":132,"total_ects":240,"uni_su":41,"uni_courses":16,"req_su":40,"req_courses":14,"core_su":27,"area_su":9,"free_su":15,"eng_ects":90,"bsci_ects":60,"faculty_courses":5,"math_choice":False},
}

CATALOG = {}  # populated from parsed_pools.json + HARDCODED

def info(code):
    if code in CATALOG:
        c = CATALOG[code]
        return c["title"], c["su"], c["ects"], c.get("faculty", "")
    t, su, ects, fac = HARDCODED[code]
    return t, su, ects, fac

def base_meta(term):
    t = TERMS[term]
    return {
        "data_role": "curriculum_requirement", "authority_level": "official",
        "program": PROGRAM, "degree_code": DEGREE_CODE, "program_name": PROGRAM_NAME,
        "curriculum_term": term, "admit_term": term, "admit_term_label": t["label"],
        "source_authority": SOURCE_AUTHORITY, "source_url": degree_url(term),
        "source_document": f"degree_requirements/CS/{term}.jsonl",
        "extracted_at": EXTRACTED_AT, "data_version": DATA_VERSION,
    }

def rec(term, document_type, text, extra):
    r = base_meta(term); r["document_type"] = document_type; r["text"] = text; r.update(extra); return r

def chunk_id(term, dtype, cat, key):
    raw = "|".join([PROGRAM, DEGREE_CODE, term, dtype, cat, key])
    h = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return f"curriculum_requirement:CS:{term}:{dtype}/{cat}/{key}:{h}"

def build_term(term, pools):
    t = TERMS[term]; label = t["label"]
    tag = f"{PROGRAM_NAME} ({DEGREE_CODE}), admit/curriculum term {label} ({term})"
    records = []

    categories = [
        {"category":"university_courses","min_su_credits":t["uni_su"],"min_courses":t["uni_courses"]},
        {"category":"required_courses","min_su_credits":t["req_su"],"min_courses":t["req_courses"]},
        {"category":"core_electives","min_su_credits":t["core_su"]},
        {"category":"area_electives","min_su_credits":t["area_su"]},
        {"category":"free_electives","min_su_credits":t["free_su"]},
        {"category":"engineering","min_ects":t["eng_ects"]},
        {"category":"basic_science","min_ects":t["bsci_ects"]},
        {"category":"faculty_courses","min_courses":t["faculty_courses"]},
    ]
    ptext = (f"Degree requirement profile for {tag}. Total minimum SU credits: {t['total_su']}. "
             f"Total minimum ECTS: {t['total_ects']}. Category minimums: University Courses {t['uni_su']} SU "
             f"({t['uni_courses']} courses), Required Courses {t['req_su']} SU ({t['req_courses']} courses), "
             f"Core Electives {t['core_su']} SU, Area Electives {t['area_su']} SU, Free Electives {t['free_su']} SU, "
             f"Engineering {t['eng_ects']} ECTS, Basic Science {t['bsci_ects']} ECTS, Faculty Courses at least "
             f"{t['faculty_courses']} courses. No explicit GPA requirement is published on the degree-detail page. "
             f"Diploma area advisors: {', '.join(ADVISORS)}.")
    records.append(rec(term, "degree_requirement_profile", ptext, {
        "total_min_su_credits":t["total_su"], "total_min_ects":t["total_ects"],
        "min_program_gpa":None, "min_cumulative_gpa":None, "advisors":ADVISORS, "categories":categories}))

    def emit_pool(category, codes, min_su=None, min_courses=None, note="", pool_source_url=None):
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
        extra = {"requirement_category":category, "course_ids":ids,
                 "min_su_credits":min_su, "min_courses":min_courses}
        if pool_source_url: extra["pool_source_url"] = pool_source_url
        records.append(rec(term, "degree_requirement_category_pool", summ, extra))
        for c in ids:
            title, su, ects, fac = info(c)
            ctext = (f"For {tag}, the course {c} - {title} ({su} SU credits, {ects} ECTS"
                     f"{', ' + fac if fac else ''}) belongs to the {category.replace('_',' ')} pool.")
            if min_su is not None:
                ctext += f" A minimum of {min_su} SU credits of {category.replace('_',' ')} is required."
            records.append(rec(term, "degree_requirement_pool_course", ctext, {
                "requirement_category":category, "course_id":c, "course_title":title,
                "su_credits":su, "ects":ects, "faculty":fac}))

    emit_pool("university_courses_mandatory", MANDATORY_UNI,
              note="All freshman (1XX) courses and SPS 303 are mandatory.")
    emit_pool("university_courses_hum_pool", HUM_POOL[term], min_courses=1,
              note="Exactly one HUM course must be completed to satisfy the university HUM requirement.")
    req_note = ("Students complete EITHER MATH 201 (Linear Algebra) OR MATH 212 (Linear Algebra and "
                "Differential Equations), not both." if t["math_choice"] else
                "MATH 212 is required. For 2025-2026 and later, MATH 201 and MATH 202 are not included in any "
                "course pool; PHYS 113 and CS 306, CS 308 are now required.")
    emit_pool("required_courses", REQUIRED[term], min_su=t["req_su"], min_courses=t["req_courses"], note=req_note)
    emit_pool("core_electives", pools["core_electives"], min_su=t["core_su"],
              note="Extra courses taken from the Core Elective pool count toward Area Elective requirements.",
              pool_source_url=pool_url(term,"BSCS_CEL"))
    emit_pool("area_electives", pools["area_electives"], min_su=t["area_su"],
              note="Extra courses taken from the Area Elective pool count toward Free Elective requirements.",
              pool_source_url=pool_url(term,"BSCS_AEL"))
    emit_pool("free_electives", pools["free_electives"], min_su=t["free_su"],
              note="Free electives come from FASS, SBS and FENS courses (excluding University Courses). "
                   "School of Languages (SL) courses do not count.",
              pool_source_url=pool_url(term,"BSCS_FEL"))
    emit_pool("faculty_courses_fens", pools["faculty_fens"],
              note="FENS faculty-course pool. At least 3 of the 5 required faculty courses must come from this pool.",
              pool_source_url=pool_url(term,"FC_FENS",extra="&P_FAC=E"))
    emit_pool("faculty_courses_fass", pools["faculty_fass"],
              note="FASS faculty-course pool. Counts toward the 5-course faculty requirement.",
              pool_source_url=pool_url(term,"FC_FASS",extra="&P_FAC=S"))
    emit_pool("faculty_courses_sbs", pools["faculty_sbs"],
              note="SBS (School of Management) faculty-course pool. Counts toward the 5-course faculty requirement.",
              pool_source_url=pool_url(term,"FC_SOM",extra="&P_FAC=M"))

    rules = [
        ("free_electives_rule","free_electives",
         f"Free Electives for {tag}: {t['free_su']} SU credits from courses offered by FASS, SBS and FENS "
         f"(excluding University Courses). Language courses offered by the School of Languages (SL) do not count "
         f"as free electives.", {"min_su_credits":t["free_su"], "pool_source_url":pool_url(term,"BSCS_FEL")}),
        ("engineering_ects_rule","engineering",
         f"Engineering requirement for {tag}: at least {t['eng_ects']} ECTS of Engineering credits. May be "
         f"fulfilled while completing other degree requirements.", {"min_ects":t["eng_ects"]}),
        ("basic_science_ects_rule","basic_science",
         f"Basic Science requirement for {tag}: at least {t['bsci_ects']} ECTS of Basic Science credits. May be "
         f"fulfilled while completing other degree requirements.", {"min_ects":t["bsci_ects"]}),
        ("faculty_courses_rule","faculty_courses",
         f"Faculty Courses requirement for {tag}: at least {t['faculty_courses']} courses total from FENS faculty "
         f"courses and/or FASS and SBS courses. At least 2 must be MATH-coded, and at least 3 must come from the "
         f"FENS Faculty Courses pool.",
         {"min_courses":t["faculty_courses"], "min_math_courses":2, "min_fens_courses":3,
          "fass_pool_source_url":pool_url(term,"FC_FASS",extra="&P_FAC=S"),
          "sbs_pool_source_url":pool_url(term,"FC_SOM",extra="&P_FAC=M")}),
        ("core_to_area_overflow","area_electives",
         f"Overflow rule for {tag}: SU credits earned in the Core Elective pool beyond the required {t['core_su']} "
         f"SU count directly toward the Area Elective requirement.",
         {"source_category":"core_electives","target_category":"area_electives","condition":"source_minimum_satisfied"}),
        ("area_to_free_overflow","free_electives",
         f"Overflow rule for {tag}: SU credits earned in the Area Elective pool beyond the required {t['area_su']} "
         f"SU count directly toward the Free Elective requirement.",
         {"source_category":"area_electives","target_category":"free_electives","condition":"source_minimum_satisfied"}),
    ]
    if t["math_choice"]:
        rules.append(("math201_math212_choice","required_courses",
            f"Choice rule for {tag}: complete either MATH 201 (Linear Algebra) or MATH 212 (Linear Algebra and "
            f"Differential Equations); only one counts toward the Required Courses requirement.",
            {"choice_courses":["MATH 201","MATH 212"], "max_count":1}))
    else:
        rules.append(("math201_math202_exclusion_2025","required_courses",
            f"Rule for {tag}: for students admitted in 2025-2026 and later, MATH 201 (Linear Algebra) and MATH 202 "
            f"(Differential Equations) are not included in any course pool. MATH 212 is required.",
            {"excluded_courses":["MATH 201","MATH 202"]}))
    for rule_id, cat, text, extra in rules:
        e = {"rule_id":rule_id, "requirement_category":cat}; e.update(extra)
        records.append(rec(term, "degree_requirement_rule", text, e))
    return records

def main(out_dir, parsed_path):
    global CATALOG
    parsed = json.load(open(parsed_path, encoding="utf-8"))
    CATALOG = parsed["catalog"]
    os.makedirs(out_dir, exist_ok=True)
    summary = {}
    for term in TERMS:
        recs = build_term(term, parsed["pools"][term])
        seen_ids = set()
        for r in recs:
            key = r.get("course_id") or r.get("rule_id") or ""
            cid = chunk_id(term, r["document_type"], r.get("requirement_category",""), key)
            # ensure uniqueness (a course can appear in multiple pools -> category is in key already)
            base = cid; n = 1
            while cid in seen_ids:
                n += 1; cid = f"{base}#{n}"
            seen_ids.add(cid); r["chunk_id"] = cid
        path = os.path.join(out_dir, f"{term}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        by_type = {}
        for r in recs:
            by_type[r["document_type"]] = by_type.get(r["document_type"],0)+1
        summary[term] = {"records":len(recs), "by_type":by_type, "unique_chunk_ids":len(seen_ids)}
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
