#!/usr/bin/env python3
"""Scrape official SU major requirements into the repository JSONL schema.

This scraper is intended for programs whose official page cannot be represented by
the older ``gen_degree_reqs.py`` pool model (for example PSIR has two separately
constrained Core Elective pools).  It reads a small program config, downloads the
official English degree-detail and linked course-pool pages, and writes one
deterministic JSONL file per admit term.

Usage:
  python scrape_official_degree_reqs.py CONFIG OUT_DIR [--extracted-at YYYY-MM-DD]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urljoin

import requests
from bs4 import BeautifulSoup, Tag


BASE = "https://suis.sabanciuniv.edu/prod/"
SOURCE_AUTHORITY = "Sabanci University"
COURSE_RE = re.compile(r"^[A-Z]{2,6}\s+\d{3,5}[0-9A-Z]?$")


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _number(value: str) -> int | float | None:
    value = _clean(value)
    if not value or value == "-":
        return None
    try:
        result = float(value.replace(",", "."))
    except ValueError:
        return None
    return int(result) if result.is_integer() else result


def _degree_url(term: str, p_program: str) -> str:
    return BASE + "SU_DEGREE.p_degree_detail?" + urlencode(
        {
            "P_TERM": term,
            "P_PROGRAM": p_program,
            "P_SUBMIT": "",
            "P_LANG": "EN",
            "P_LEVEL": "UG",
        }
    )


def _pool_url(term: str, p_program: str, area: str, fac: str | None = None) -> str:
    params = {"P_TERM": term, "P_AREA": area, "P_PROGRAM": p_program}
    if fac:
        params["P_FAC"] = fac
    params.update({"P_LANG": "EN", "P_LEVEL": "UG"})
    return BASE + "SU_DEGREE.p_list_courses?" + urlencode(params)


class OfficialDegreeScraper:
    def __init__(self, config: dict[str, Any], *, extracted_at: str) -> None:
        self.config = config
        self.extracted_at = extracted_at
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "AdviSU-offline-degree-corpus-builder/1.0 (+official public SU pages)"}
        )

    def _fetch(self, url: str) -> BeautifulSoup:
        response = self.session.get(url, timeout=45)
        response.raise_for_status()
        if "text/html" not in response.headers.get("content-type", "text/html").lower():
            raise RuntimeError(f"Expected HTML from {url}, got {response.headers.get('content-type')}")
        # The SUIS pages declare UTF-8 in the HTTP header.  Passing decoded text
        # avoids BeautifulSoup's legacy byte-level detector misreading Turkish
        # characters (for example Atatürk as AtatÃ¼rk).
        return BeautifulSoup(response.text, "html.parser")

    @staticmethod
    def _course_rows(table: Tag | None) -> list[dict[str, Any]]:
        if table is None:
            return []
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for tr in table.find_all("tr"):
            cells = tr.find_all("td", recursive=False)
            if len(cells) < 5:
                continue
            # Official tables use marker, code, title, ECTS, SU and faculty columns.
            offset = 1 if len(cells) >= 6 else 0
            code = _clean(cells[offset].get_text(" ", strip=True)).upper()
            if not COURSE_RE.fullmatch(code) or code in seen:
                continue
            title = _clean(cells[offset + 1].get_text(" ", strip=True))
            ects = _number(cells[offset + 2].get_text(" ", strip=True))
            su = _number(cells[offset + 3].get_text(" ", strip=True))
            faculty = _clean(cells[offset + 4].get_text(" ", strip=True)).upper() if len(cells) > offset + 4 else ""
            marker = _clean(cells[0].get_text(" ", strip=True)) if offset else ""
            # BAPSIR's Free Electives page contains ACCA 200 / ECNA 200 grouping
            # sentinels titled "Area Electives ...". They have no ECTS or faculty
            # and are not real courses, so they must not enter the course corpus.
            if ects is None and su == 0 and not faculty and title.lower().startswith("area electives "):
                continue
            if su is None or ects is None:
                raise RuntimeError(f"Official row has non-numeric credits: {code}")
            seen.add(code)
            rows.append(
                {
                    "course_id": code,
                    "course_title": title,
                    "su_credits": su,
                    "ects": ects,
                    "faculty": faculty,
                    "is_faculty_course": marker == "*",
                }
            )
        return rows

    @staticmethod
    def _course_table(container: Tag) -> Tag | None:
        for table in container.find_all("table"):
            headers = {_clean(th.get_text(" ", strip=True)).lower() for th in table.find_all("th")}
            if "course" in headers and "name" in headers and any("su credits" in h for h in headers):
                return table
        return None

    def _section(self, soup: BeautifulSoup, anchor_name: str) -> tuple[str, str, list[dict[str, Any]], list[str]]:
        anchor = soup.find("a", attrs={"name": anchor_name})
        if anchor is None:
            raise RuntimeError(f"Official page is missing expected anchor {anchor_name}")
        heading_table = anchor.find_parent("table", class_="t_kategori")
        outer_row = heading_table.find_parent("tr") if heading_table else None
        if outer_row is None:
            raise RuntimeError(f"Cannot locate category row for {anchor_name}")

        heading = _clean(heading_table.get_text(" ", strip=True))
        description = ""
        links: list[str] = []
        courses: list[dict[str, Any]] = []
        sibling = outer_row.find_next_sibling("tr")
        while sibling is not None:
            classes = set(sibling.get("class", []))
            if "t_kategori_row" in classes:
                break
            if "t_kategori_row_desc" in classes:
                description = _clean(sibling.get_text(" ", strip=True))
            for link in sibling.find_all("a", href=True):
                if "SU_DEGREE.p_list_courses" in link["href"]:
                    links.append(urljoin(BASE, link["href"]))
            if not courses:
                table = self._course_table(sibling)
                if table is not None:
                    courses = self._course_rows(table)
            sibling = sibling.find_next_sibling("tr")
        return heading, description, courses, list(dict.fromkeys(links))

    @staticmethod
    def _summary(soup: BeautifulSoup) -> dict[str, dict[str, Any]]:
        table = soup.find("table", class_="t_mezuniyet")
        if table is None:
            raise RuntimeError("Official page has no degree-requirement summary table")
        result: dict[str, dict[str, Any]] = {}
        for row in table.find_all("tr"):
            cells = [_clean(td.get_text(" ", strip=True)) for td in row.find_all("td", recursive=False)]
            if len(cells) != 4 or not cells[0]:
                continue
            result[cells[0]] = {
                "min_ects": _number(cells[1]),
                "min_su_credits": _number(cells[2]),
                "min_courses": _number(cells[3]),
            }
        if "Total" not in result:
            raise RuntimeError("Official page summary has no Total row")
        return result

    @staticmethod
    def _advisor_names(soup: BeautifulSoup) -> list[str]:
        for row in soup.find_all("tr"):
            cleaned = _clean(row.get_text(" ", strip=True))
            if cleaned.lower().startswith("diploma area advisor"):
                _, _, names = cleaned.partition(":")
                return [_clean(name) for name in names.split(",") if _clean(name)]
        return []

    def build_term(self, term: str) -> list[dict[str, Any]]:
        cfg = self.config
        program = cfg["program"]
        degree_code = cfg["degree_code"]
        p_program = cfg["p_program"]
        source_url = _degree_url(term, p_program)
        soup = self._fetch(source_url)
        summary = self._summary(soup)
        total = summary["Total"]
        advisors = self._advisor_names(soup)
        label_node = soup.find(string=re.compile(r"Admit Term:\s*"))
        label = _clean(label_node).split(":", 1)[-1].strip() if label_node else term

        section_data: dict[str, dict[str, Any]] = {}
        university_heading, university_desc, university_courses, _ = self._section(
            soup, cfg["university_anchor"]
        )
        section_data[cfg["university_anchor"]] = {
            "heading": university_heading,
            "description": university_desc,
            "courses": university_courses,
            "source_url": source_url,
        }

        for pool_cfg in cfg["pools"]:
            heading, description, courses, links = self._section(soup, pool_cfg["anchor"])
            pool_source_url = source_url
            if pool_cfg.get("external_area"):
                expected_area = pool_cfg["external_area"]
                matching = [url for url in links if f"P_AREA={expected_area}" in url]
                pool_source_url = matching[0] if matching else _pool_url(term, p_program, expected_area)
                courses = self._course_rows(self._course_table(self._fetch(pool_source_url)))
            if not courses:
                raise RuntimeError(f"No course rows found for {program}/{term}/{pool_cfg['anchor']}")
            section_data[pool_cfg["anchor"]] = {
                "heading": heading,
                "description": description,
                "courses": courses,
                "source_url": pool_source_url,
            }

        faculty_heading, faculty_desc, _, faculty_links = self._section(soup, cfg["faculty_anchor"])
        for faculty_cfg in cfg["faculty_pools"]:
            expected_area = faculty_cfg["area"]
            matching = [url for url in faculty_links if f"P_AREA={expected_area}" in url]
            pool_source_url = matching[0] if matching else _pool_url(
                term, p_program, expected_area, faculty_cfg.get("fac")
            )
            courses = self._course_rows(self._course_table(self._fetch(pool_source_url)))
            if not courses:
                raise RuntimeError(f"No course rows found for {program}/{term}/{expected_area}")
            section_data[expected_area] = {
                "heading": faculty_cfg["name"],
                "description": faculty_desc,
                "courses": courses,
                "source_url": pool_source_url,
                "faculty_heading": faculty_heading,
            }

        official_summary = [
            {"official_category_name": name, **values}
            for name, values in summary.items()
            if name != "Total"
        ]
        profile_categories: list[dict[str, Any]] = []
        for category_cfg in cfg["profile_categories"]:
            components = []
            for official_label in category_cfg["official_labels"]:
                if official_label not in summary:
                    raise RuntimeError(f"Missing summary category {official_label} for {program}/{term}")
                components.append({"official_category_name": official_label, **summary[official_label]})
            entry: dict[str, Any] = {"category": category_cfg["category"]}
            for field in ("min_ects", "min_su_credits", "min_courses"):
                values = [c[field] for c in components if c[field] is not None]
                entry[field] = sum(values) if values else None
            if category_cfg.get("min_courses_override") is not None:
                entry["min_courses"] = category_cfg["min_courses_override"]
            if len(components) > 1:
                entry["official_components"] = components
            profile_categories.append(entry)

        profile_summary = ", ".join(
            f"{component['official_category_name']} "
            + (
                f"{component['min_su_credits']} SU"
                if component["min_su_credits"] is not None
                else f"{component['min_courses']} courses"
            )
            for component in official_summary
        )
        tag = f"{cfg['program_name']} ({degree_code}), admit/curriculum term {label} ({term})"

        def base_meta() -> dict[str, Any]:
            return {
                "data_role": "curriculum_requirement",
                "authority_level": "official",
                "program": program,
                "degree_code": degree_code,
                "program_name": cfg["program_name"],
                "curriculum_term": term,
                "admit_term": term,
                "admit_term_label": label,
                "source_authority": SOURCE_AUTHORITY,
                "source_url": source_url,
                "source_document": f"degree_requirements/{program}/{term}.jsonl",
                "extracted_at": self.extracted_at,
                "data_version": self.extracted_at,
            }

        def record(document_type: str, text: str, **extra: Any) -> dict[str, Any]:
            result = base_meta()
            result.update({"document_type": document_type, "text": text})
            result.update(extra)
            return result

        records: list[dict[str, Any]] = [
            record(
                "degree_requirement_profile",
                f"Degree requirement profile for {tag}. Total minimum SU credits: "
                f"{total['min_su_credits']}. Total minimum ECTS: {total['min_ects']}. "
                f"Official category minimums: {profile_summary}. "
                f"Diploma area advisors: {', '.join(advisors) if advisors else 'not listed'}.",
                total_min_su_credits=total["min_su_credits"],
                total_min_ects=total["min_ects"],
                min_program_gpa=None,
                min_cumulative_gpa=None,
                advisors=advisors,
                categories=profile_categories,
                official_summary=official_summary,
            )
        ]

        profile_by_category = {entry["category"]: entry for entry in profile_categories}

        def emit_pool(
            *,
            pool_key: str,
            category: str,
            official_name: str,
            courses: list[dict[str, Any]],
            note: str,
            pool_source_url: str,
            min_su: int | float | None = None,
            min_courses: int | float | None = None,
            subcategory: str | None = None,
        ) -> None:
            course_ids = [course["course_id"] for course in courses]
            listed = courses[:40]
            course_text = ", ".join(f"{c['course_id']} {c['course_title']}" for c in listed)
            if len(courses) > len(listed):
                course_text += f", ... and {len(courses) - len(listed)} more"
            summary_text = (
                f"The official {official_name} pool for {tag} contains {len(courses)} courses"
            )
            if min_su is not None:
                summary_text += f"; minimum {min_su} SU credits required"
            if min_courses is not None:
                summary_text += f"; minimum {min_courses} courses required"
            summary_text += f". Courses: {course_text}."
            if note:
                summary_text += " " + note
            common: dict[str, Any] = {
                "requirement_category": category,
                "official_category_name": official_name,
                "pool_key": pool_key,
                "pool_source_url": pool_source_url,
            }
            if subcategory:
                common["requirement_subcategory"] = subcategory
            records.append(
                record(
                    "degree_requirement_category_pool",
                    summary_text,
                    course_ids=course_ids,
                    min_su_credits=min_su,
                    min_courses=min_courses,
                    constraint_note=note,
                    **common,
                )
            )
            for course in courses:
                course_text = (
                    f"For {tag}, {course['course_id']} - {course['course_title']} "
                    f"({course['su_credits']} SU credits, {course['ects']} ECTS"
                    f"{', ' + course['faculty'] if course['faculty'] else ''}) belongs to the "
                    f"official {official_name} pool."
                )
                records.append(
                    record(
                        "degree_requirement_pool_course",
                        course_text,
                        course_id=course["course_id"],
                        course_title=course["course_title"],
                        su_credits=course["su_credits"],
                        ects=course["ects"],
                        faculty=course["faculty"],
                        is_faculty_course=course["is_faculty_course"],
                        **common,
                    )
                )

        university = section_data[cfg["university_anchor"]]
        hum_courses = [c for c in university["courses"] if c["course_id"].startswith("HUM ")]
        mandatory_courses = [c for c in university["courses"] if not c["course_id"].startswith("HUM ")]
        emit_pool(
            pool_key="university_mandatory",
            category="university_courses_mandatory",
            official_name="University Courses (mandatory)",
            courses=mandatory_courses,
            note=university["description"],
            pool_source_url=university["source_url"],
        )
        emit_pool(
            pool_key="university_hum",
            category="university_courses_hum_pool",
            official_name="University Courses HUM pool",
            courses=hum_courses,
            note=(
                f"At least {cfg['hum_min_courses']} HUM courses must be completed: a 2XX-level "
                "course first, followed by a 3XX-level course."
            ),
            pool_source_url=university["source_url"],
            min_courses=cfg["hum_min_courses"],
        )

        for pool_cfg in cfg["pools"]:
            section = section_data[pool_cfg["anchor"]]
            profile_min = profile_by_category[pool_cfg["category"]]
            official_min = summary[pool_cfg["official_name"]]
            min_su = official_min["min_su_credits"]
            min_courses = official_min["min_courses"]
            if pool_cfg.get("subcategory"):
                min_su = official_min["min_su_credits"]
            elif pool_cfg["category"] == "core_electives" and len(
                [p for p in cfg["pools"] if p["category"] == "core_electives"]
            ) == 1:
                min_su = profile_min["min_su_credits"]
            emit_pool(
                pool_key=pool_cfg["anchor"],
                category=pool_cfg["category"],
                official_name=pool_cfg["official_name"],
                courses=section["courses"],
                note=section["description"],
                pool_source_url=section["source_url"],
                min_su=min_su,
                min_courses=min_courses,
                subcategory=pool_cfg.get("subcategory"),
            )
            if pool_cfg.get("overflow_to"):
                records.append(
                    record(
                        "degree_requirement_rule",
                        f"Overflow rule for {tag}: after satisfying {pool_cfg['official_name']}, "
                        f"extra courses from that pool count toward {pool_cfg['overflow_to'].replace('_', ' ')}. "
                        f"Official wording: {section['description']}",
                        rule_id=f"{pool_cfg['anchor'].lower()}_overflow",
                        requirement_category=pool_cfg["overflow_to"],
                        source_category=pool_cfg["category"],
                        source_subcategory=pool_cfg.get("subcategory"),
                        target_category=pool_cfg["overflow_to"],
                        condition="source_minimum_satisfied",
                    )
                )

        for faculty_cfg in cfg["faculty_pools"]:
            section = section_data[faculty_cfg["area"]]
            emit_pool(
                pool_key=faculty_cfg["area"],
                category=faculty_cfg["category"],
                official_name=faculty_cfg["name"],
                courses=section["courses"],
                note=section["description"],
                pool_source_url=section["source_url"],
            )

        for configured_rule in cfg.get("rules", []):
            rule = dict(configured_rule)
            rule_id = rule.pop("rule_id")
            category = rule.pop("requirement_category")
            text = rule.pop("text")
            records.append(
                record(
                    "degree_requirement_rule",
                    f"Rule for {tag}: {text}",
                    rule_id=rule_id,
                    requirement_category=category,
                    **rule,
                )
            )

        required_cfg = next(p for p in cfg["pools"] if p["category"] == "required_courses")
        required_count = len(section_data[required_cfg["anchor"]]["courses"])
        official_required_min = summary[required_cfg["official_name"]]["min_courses"]
        if official_required_min is not None and required_count != official_required_min:
            records.append(
                record(
                    "degree_requirement_rule",
                    f"Official-source consistency note for {tag}: the Required Courses section says all "
                    f"{required_count} listed courses are required, while the summary table reports a "
                    f"minimum of {official_required_min} courses and "
                    f"{summary[required_cfg['official_name']]['min_su_credits']} SU credits. The corpus "
                    "preserves both official statements and does not invent an undocumented equivalency.",
                    rule_id="official_required_summary_discrepancy",
                    requirement_category="required_courses",
                    listed_course_count=required_count,
                    official_min_courses=official_required_min,
                    interpretation="none; official statements preserved verbatim in normalized form",
                )
            )

        seen: set[str] = set()
        for row in records:
            category = row.get("requirement_category", "")
            key = (
                row.get("course_id")
                or row.get("rule_id")
                or row.get("pool_key")
                or "profile"
            )
            raw = "|".join([program, degree_code, term, row["document_type"], category, str(key)])
            digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
            chunk_id = (
                f"curriculum_requirement:{program}:{term}:{row['document_type']}/"
                f"{category}/{key}:{digest}"
            )
            if chunk_id in seen:
                raise RuntimeError(f"Non-unique deterministic chunk id: {chunk_id}")
            seen.add(chunk_id)
            row["chunk_id"] = chunk_id
        return records

    def write(self, out_dir: Path) -> dict[str, Any]:
        out_dir.mkdir(parents=True, exist_ok=True)
        result: dict[str, Any] = {}
        for term in self.config["terms"]:
            records = self.build_term(term)
            output = out_dir / f"{term}.jsonl"
            with output.open("w", encoding="utf-8", newline="\n") as handle:
                for row in records:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            by_type: dict[str, int] = {}
            for row in records:
                by_type[row["document_type"]] = by_type.get(row["document_type"], 0) + 1
            result[term] = {"records": len(records), "by_type": by_type, "output": str(output)}
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--extracted-at", default=date.today().isoformat())
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = OfficialDegreeScraper(config, extracted_at=args.extracted_at).write(args.out_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
