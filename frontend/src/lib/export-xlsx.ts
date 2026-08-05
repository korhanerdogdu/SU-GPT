import * as XLSX from "xlsx";
import type { Course, DegreeAudit } from "./api";
import { formatMeeting, type ScheduleDocument } from "./schedule";
import { getStoredLocale, translate, type Locale } from "@/localization/resources";

/**
 * Real .xlsx export (SheetJS) built from the SAME structured data the pages render — never from
 * the rendered DOM/markdown. Each sheet gets header row, sane column widths and an autofilter.
 */

const ELIGIBLE = new Set(["completed", "transfer", "exempted"]);

function cols(ws: XLSX.WorkSheet, widths: number[]) {
  ws["!cols"] = widths.map((wch) => ({ wch }));
}

function filterAll(ws: XLSX.WorkSheet, rows: number, colCount: number) {
  const end = XLSX.utils.encode_cell({ r: rows - 1, c: colCount - 1 });
  ws["!autofilter"] = { ref: `A1:${end}` };
}

function stamp(): string {
  return new Date().toISOString().slice(0, 10);
}

/** Course History → one sheet of completed courses + a running SU total. */
export function exportCoursesXlsx(courses: Course[], username: string, locale: Locale = getStoredLocale()) {
  const tx = (key: Parameters<typeof translate>[1]) => translate(locale, key);
  const header = [tx("export.courseCode"), tx("export.courseName"), tx("export.su"), "ECTS", tx("export.status")];
  const body = courses.map((c) => [
    c.code,
    c.title,
    c.su_credits ?? "",
    c.ects ?? "",
    c.status ?? "completed",
  ]);
  const totalSu = courses
    .filter((c) => ELIGIBLE.has(c.status ?? "completed"))
    .reduce((sum, c) => sum + (c.su_credits ?? 0), 0);
  const aoa = [header, ...body, [], [tx("export.total"), "", totalSu, "", ""]];
  const ws = XLSX.utils.aoa_to_sheet(aoa);
  cols(ws, [14, 52, 12, 8, 14]);
  filterAll(ws, body.length + 1, header.length);
  const wb = XLSX.utils.book_new();
  XLSX.utils.book_append_sheet(wb, ws, tx("export.courseHistorySheet"));
  XLSX.writeFile(wb, `advisu-ders-gecmisi-${username}-${stamp()}.xlsx`);
}

/** Degree audit → Özet + Kategoriler (+ ECTS, Eksik Zorunlu when present). */
export function exportAuditXlsx(audit: DegreeAudit, username: string, locale: Locale = getStoredLocale()) {
  const tx = (key: Parameters<typeof translate>[1]) => translate(locale, key);
  const wb = XLSX.utils.book_new();

  const summary = XLSX.utils.aoa_to_sheet([
    [tx("export.field"), tx("export.value")],
    [tx("export.program"), audit.program_name || audit.program],
    [tx("export.curriculumTerm"), audit.curriculum_term],
    [tx("export.completedSu"), audit.completed_su_credits ?? ""],
    [tx("export.requiredSu"), audit.total_min_su_credits ?? ""],
    [tx("export.remainingSu"), audit.remaining_su_credits ?? ""],
    [tx("export.status"), audit.status],
    [tx("export.reliability"), audit.reliability],
  ]);
  cols(summary, [22, 48]);
  XLSX.utils.book_append_sheet(wb, summary, tx("export.summarySheet"));

  const catHeader = [tx("profile.category"), tx("export.completedSu"), tx("export.requiredSu"), tx("export.remainingSu")];
  const catRows = (audit.categories ?? []).map((c) => [
    c.category.replace(/_/g, " "),
    c.completed_su_credits,
    c.required_su_credits ?? "",
    c.remaining_su_credits ?? "",
  ]);
  const catWs = XLSX.utils.aoa_to_sheet([catHeader, ...catRows]);
  cols(catWs, [26, 16, 14, 12]);
  filterAll(catWs, catRows.length + 1, catHeader.length);
  XLSX.utils.book_append_sheet(wb, catWs, tx("export.categoriesSheet"));

  if (audit.ects_requirements && audit.ects_requirements.length > 0) {
    const eHeader = [tx("profile.category"), tx("export.completedEcts"), tx("export.requiredEcts"), tx("export.remainingEcts")];
    const eRows = audit.ects_requirements.map((e) => [
      e.category.replace(/_/g, " "),
      e.completed_ects,
      e.required_ects,
      e.remaining_ects,
    ]);
    const eWs = XLSX.utils.aoa_to_sheet([eHeader, ...eRows]);
    cols(eWs, [26, 18, 16, 14]);
    filterAll(eWs, eRows.length + 1, eHeader.length);
    XLSX.utils.book_append_sheet(wb, eWs, "ECTS");
  }

  if (audit.missing_required_courses && audit.missing_required_courses.length > 0) {
    const mWs = XLSX.utils.aoa_to_sheet([
      [tx("export.missingCourse")],
      ...audit.missing_required_courses.map((code) => [code]),
    ]);
    cols(mWs, [26]);
    XLSX.utils.book_append_sheet(wb, mWs, tx("export.missingSheet"));
  }

  XLSX.writeFile(wb, `advisu-mezuniyet-denetimi-${username}-${stamp()}.xlsx`);
}

/** Weekly schedule → a registration-ready sheet with every selected CRN and meeting. */
export function exportScheduleXlsx(schedule: ScheduleDocument, username: string, locale: Locale = getStoredLocale()) {
  const tx = (key: Parameters<typeof translate>[1]) => translate(locale, key);
  const header = [
    tx("common.course"),
    tx("export.courseName"),
    tx("export.type"),
    "CRN",
    tx("common.section"),
    tx("export.dayTime"),
    tx("export.location"),
    tx("export.instructor"),
  ];
  const body = schedule.items.map((item) => [
    item.courseCode,
    item.title,
    item.component,
    item.crn,
    item.section,
    item.meetings.length > 0 ? item.meetings.map((meeting) => formatMeeting(meeting, locale)).join(", ") : "TBA",
    item.location,
    item.instructor,
  ]);
  const ws = XLSX.utils.aoa_to_sheet([
    [`${tx("export.term")}: ${schedule.termLabel}`],
    [],
    header,
    ...body,
  ]);
  cols(ws, [14, 40, 18, 10, 10, 32, 36, 28]);
  if (body.length > 0) {
    const end = XLSX.utils.encode_cell({ r: body.length + 2, c: header.length - 1 });
    ws["!autofilter"] = { ref: `A3:${end}` };
  }
  const wb = XLSX.utils.book_new();
  XLSX.utils.book_append_sheet(wb, ws, tx("export.weekSheet"));
  XLSX.writeFile(wb, `advisu-ders-programi-${username}-${stamp()}.xlsx`);
}
