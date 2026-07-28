import * as XLSX from "xlsx";
import type { Course, DegreeAudit } from "./api";

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
export function exportCoursesXlsx(courses: Course[], username: string) {
  const header = ["Ders Kodu", "Ders Adı", "SU Kredisi", "ECTS", "Durum"];
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
  const aoa = [header, ...body, [], ["TOPLAM (kredi sayan)", "", totalSu, "", ""]];
  const ws = XLSX.utils.aoa_to_sheet(aoa);
  cols(ws, [14, 52, 12, 8, 14]);
  filterAll(ws, body.length + 1, header.length);
  const wb = XLSX.utils.book_new();
  XLSX.utils.book_append_sheet(wb, ws, "Ders Geçmişi");
  XLSX.writeFile(wb, `advisu-ders-gecmisi-${username}-${stamp()}.xlsx`);
}

/** Degree audit → Özet + Kategoriler (+ ECTS, Eksik Zorunlu when present). */
export function exportAuditXlsx(audit: DegreeAudit, username: string) {
  const wb = XLSX.utils.book_new();

  const summary = XLSX.utils.aoa_to_sheet([
    ["Alan", "Değer"],
    ["Program", audit.program_name || audit.program],
    ["Müfredat dönemi", audit.curriculum_term],
    ["Tamamlanan SU", audit.completed_su_credits ?? ""],
    ["Gerekli SU", audit.total_min_su_credits ?? ""],
    ["Kalan SU", audit.remaining_su_credits ?? ""],
    ["Durum", audit.status],
    ["Güvenilirlik", audit.reliability],
  ]);
  cols(summary, [22, 48]);
  XLSX.utils.book_append_sheet(wb, summary, "Özet");

  const catHeader = ["Kategori", "Tamamlanan SU", "Gerekli SU", "Kalan SU"];
  const catRows = (audit.categories ?? []).map((c) => [
    c.category.replace(/_/g, " "),
    c.completed_su_credits,
    c.required_su_credits ?? "",
    c.remaining_su_credits ?? "",
  ]);
  const catWs = XLSX.utils.aoa_to_sheet([catHeader, ...catRows]);
  cols(catWs, [26, 16, 14, 12]);
  filterAll(catWs, catRows.length + 1, catHeader.length);
  XLSX.utils.book_append_sheet(wb, catWs, "Kategoriler");

  if (audit.ects_requirements && audit.ects_requirements.length > 0) {
    const eHeader = ["Kategori", "Tamamlanan ECTS", "Gerekli ECTS", "Kalan ECTS"];
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
      ["Eksik Zorunlu Ders"],
      ...audit.missing_required_courses.map((code) => [code]),
    ]);
    cols(mWs, [26]);
    XLSX.utils.book_append_sheet(wb, mWs, "Eksik Zorunlu");
  }

  XLSX.writeFile(wb, `advisu-mezuniyet-denetimi-${username}-${stamp()}.xlsx`);
}
