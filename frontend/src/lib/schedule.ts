import type { StructuredContent } from "./api";
import { getStoredLocale, localizedWeekDays, translate, type Locale } from "@/localization/resources";

export const SCHEDULE_STORAGE_PREFIX = "advisu-weekly-schedule";
export const SCHEDULE_REPLACE_EVENT = "advisu:schedule-replace";
export const DEFAULT_SCHEDULE_TERM = "202601";

/** "202601" -> "2026 Güz" / "2026 Fall". SUIS term codes end 01/02/03 for fall/spring/summer;
 *  anything else (or malformed input) falls back to the raw code rather than guessing. */
export function humanizeScheduleTerm(term: string, locale: Locale): string {
  const match = /^(\d{4})(0[1-3])$/.exec(term ?? "");
  if (!match) return term;
  const [, year, session] = match;
  const seasonKey =
    session === "01" ? "schedule.seasonFall" : session === "02" ? "schedule.seasonSpring" : "schedule.seasonSummer";
  return `${year} ${translate(locale, seasonKey)}`;
}

export const WEEK_DAYS = [
  { code: "M", short: "Pzt", label: "Pazartesi" },
  { code: "T", short: "Sal", label: "Salı" },
  { code: "W", short: "Çar", label: "Çarşamba" },
  { code: "R", short: "Per", label: "Perşembe" },
  { code: "F", short: "Cum", label: "Cuma" },
] as const;

export type ScheduleDay = (typeof WEEK_DAYS)[number]["code"];
export type ScheduleSource = "chatbot" | "manual" | "server";
export type ScheduleComponentKind = "primary" | "secondary";

export interface ScheduleMeeting {
  day: ScheduleDay;
  start: string;
  end: string;
}

export interface ScheduleItem {
  id: string;
  bundleId: string;
  courseCode: string;
  title: string;
  crn: string;
  section: string;
  component: string;
  componentCode: string;
  componentKind: ScheduleComponentKind;
  instructor: string;
  location: string;
  meetings: ScheduleMeeting[];
}

export interface ScheduleDocument {
  version: 1;
  term: string;
  termLabel: string;
  updatedAt: string;
  source: ScheduleSource;
  items: ScheduleItem[];
}

export interface ScheduleCatalogMeeting {
  day_codes?: string[];
  day?: string;
  start_time?: string | null;
  end_time?: string | null;
  start?: string | null;
  end?: string | null;
}

export interface ScheduleCatalogSection {
  course_id: string;
  title?: string;
  section_title?: string;
  crn: string;
  section: string;
  component?: string;
  component_code?: string;
  component_label?: string;
  instructors?: string;
  locations?: string;
  meetings?: ScheduleCatalogMeeting[];
  term?: string;
  term_label?: string;
}

export interface ScheduleConflict {
  firstId: string;
  secondId: string;
  day: ScheduleDay;
  start: string;
  end: string;
}

const DAY_ALIASES: Record<string, ScheduleDay> = {
  m: "M",
  mon: "M",
  monday: "M",
  pzt: "M",
  pazartesi: "M",
  t: "T",
  tue: "T",
  tuesday: "T",
  sal: "T",
  sali: "T",
  salı: "T",
  w: "W",
  wed: "W",
  wednesday: "W",
  car: "W",
  çar: "W",
  carsamba: "W",
  çarşamba: "W",
  r: "R",
  thu: "R",
  thursday: "R",
  per: "R",
  pers: "R",
  perş: "R",
  persembe: "R",
  perşembe: "R",
  f: "F",
  fri: "F",
  friday: "F",
  cuma: "F",
  cum: "F",
};

function stringValue(value: unknown): string {
  return value == null ? "" : String(value).trim();
}

function normaliseDay(value: unknown): ScheduleDay | null {
  const raw = stringValue(value);
  if (raw === "M" || raw === "T" || raw === "W" || raw === "R" || raw === "F") {
    return raw;
  }
  return DAY_ALIASES[raw.toLocaleLowerCase("tr-TR")] ?? null;
}

function validTime(value: string): boolean {
  return /^([01]\d|2[0-3]):[0-5]\d$/.test(value);
}

function makeItemId(courseCode: string, crn: string, section: string): string {
  const stable = [courseCode, crn, section].filter(Boolean).join("-").replace(/[^a-z0-9]+/gi, "-");
  return stable || `schedule-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

const SECONDARY_COMPONENT_MARKERS = [
  "laboratory",
  "lab",
  "laboratuvar",
  "recitation",
  "recit",
  "problem saati",
  "discussion",
  "tartışma",
  "studio",
] as const;

export function scheduleComponentKind(component: unknown, componentCode: unknown = ""): ScheduleComponentKind {
  const value = `${stringValue(component)} ${stringValue(componentCode)}`.toLocaleLowerCase("tr-TR");
  return SECONDARY_COMPONENT_MARKERS.some((marker) => value.includes(marker)) ? "secondary" : "primary";
}

export function scheduleBundleId(courseCode: unknown): string {
  return stringValue(courseCode).toLocaleUpperCase("tr-TR").replace(/\s+/g, " ") || "manual";
}

function sanitiseMeeting(meeting: Partial<ScheduleMeeting>): ScheduleMeeting | null {
  const day = normaliseDay(meeting.day);
  const start = stringValue(meeting.start);
  const end = stringValue(meeting.end);
  if (!day || !validTime(start) || !validTime(end) || start >= end) return null;
  return { day, start, end };
}

export function sanitiseScheduleItem(value: Partial<ScheduleItem>): ScheduleItem | null {
  const courseCode = stringValue(value.courseCode).toUpperCase();
  const title = stringValue(value.title);
  if (!courseCode && !title) return null;
  const crn = stringValue(value.crn);
  const section = stringValue(value.section);
  const component = stringValue(value.component) || translate(getStoredLocale(), "schedule.defaultComponent");
  const componentCode = stringValue(value.componentCode);
  return {
    id: stringValue(value.id) || makeItemId(courseCode, crn, section),
    bundleId: stringValue(value.bundleId) || scheduleBundleId(courseCode),
    courseCode,
    title,
    crn,
    section,
    component,
    componentCode,
    componentKind: value.componentKind === "secondary" || value.componentKind === "primary"
      ? value.componentKind
      : scheduleComponentKind(component, componentCode),
    instructor: stringValue(value.instructor),
    location: stringValue(value.location),
    meetings: (value.meetings ?? [])
      .map((meeting) => sanitiseMeeting(meeting))
      .filter((meeting): meeting is ScheduleMeeting => Boolean(meeting)),
  };
}

export function createScheduleDocument(
  items: ScheduleItem[] = [],
  options: Partial<Pick<ScheduleDocument, "term" | "termLabel" | "source" | "updatedAt">> = {},
): ScheduleDocument {
  return {
    version: 1,
    term: options.term || DEFAULT_SCHEDULE_TERM,
    termLabel: options.termLabel || translate(getStoredLocale(), "schedule.defaultTerm"),
    updatedAt: options.updatedAt || new Date().toISOString(),
    source: options.source || "manual",
    items: items
      .map((item) => sanitiseScheduleItem(item))
      .filter((item): item is ScheduleItem => Boolean(item)),
  };
}

export function normaliseScheduleDocument(value: unknown): ScheduleDocument | null {
  if (!value || typeof value !== "object") return null;
  const record = value as Record<string, unknown>;
  let rawItems = Array.isArray(record.items)
    ? record.items
    : Array.isArray(record.sections)
      ? record.sections
      : [];
  if (rawItems.length === 0 && Array.isArray(record.courses)) {
    rawItems = record.courses.flatMap((rawCourse) => {
      if (!rawCourse || typeof rawCourse !== "object") return [];
      const course = rawCourse as Record<string, unknown>;
      if (!Array.isArray(course.sections)) return [];
      return course.sections.map((rawSection) => {
        const section = (rawSection ?? {}) as Record<string, unknown>;
        const meetings = Array.isArray(section.meetings)
          ? section.meetings.flatMap((rawMeeting) => {
              const meeting = (rawMeeting ?? {}) as Record<string, unknown>;
              const days = Array.isArray(meeting.day_codes) ? meeting.day_codes : [meeting.day];
              return days.map((day) => ({
                day,
                start: meeting.start_time ?? meeting.start,
                end: meeting.end_time ?? meeting.end,
              }));
            })
          : [];
        return {
          id: section.id,
          courseCode: course.course_id ?? section.course_id,
          title: course.title ?? section.title,
          crn: section.crn,
          section: section.section,
          component: section.component_label ?? section.component,
          componentCode: section.component_code,
          instructor: section.instructors ?? section.instructor,
          location: section.locations ?? section.location,
          meetings,
        };
      });
    });
  }
  const items = rawItems
    .map((item) => {
      const recordItem = (item ?? {}) as Record<string, unknown>;
      return sanitiseScheduleItem({
        ...(recordItem as Partial<ScheduleItem>),
        componentCode: stringValue(recordItem.componentCode ?? recordItem.component_code),
        bundleId: stringValue(recordItem.bundleId ?? recordItem.bundle_id),
      });
    })
    .filter((item): item is ScheduleItem => Boolean(item));
  return createScheduleDocument(items, {
    term: stringValue(record.term) || DEFAULT_SCHEDULE_TERM,
    termLabel: stringValue(record.termLabel ?? record.term_label) || translate(getStoredLocale(), "schedule.defaultTerm"),
    updatedAt: stringValue(record.updatedAt ?? record.updated_at) || new Date().toISOString(),
    source:
      record.origin === "chatbot" || record.origin === "server" || record.origin === "manual"
        ? record.origin
        : record.source === "chatbot" || record.source === "server" || record.source === "manual"
          ? record.source
          : "server",
  });
}

export function parseScheduleText(value: unknown): ScheduleMeeting[] {
  const text = stringValue(value);
  if (!text || /\bTBA\b/i.test(text)) return [];
  const meetings: ScheduleMeeting[] = [];
  for (const part of text.split(/\s*,\s*/)) {
    const match = part.match(/^(.+?)\s+(\d{1,2}:\d{2})\s*[-–]\s*(\d{1,2}:\d{2})$/);
    if (!match) continue;
    const meeting = sanitiseMeeting({ day: normaliseDay(match[1]) ?? undefined, start: match[2], end: match[3] });
    if (meeting) meetings.push(meeting);
  }
  return meetings;
}

export function scheduleFromStructuredContent(content?: StructuredContent | null): ScheduleDocument | null {
  if (!content || content.kind !== "course_schedule") return null;
  const table = content.tables?.find((candidate) => candidate.id === "weekly-schedule") ?? content.tables?.[0];
  if (!table) return null;
  const items = table.rows
    .map((row) =>
      sanitiseScheduleItem({
        courseCode: stringValue(row.course ?? row.course_code),
        title: stringValue(row.title),
        component: stringValue(row.component ?? row.type),
        crn: stringValue(row.crn),
        section: stringValue(row.section),
        meetings: parseScheduleText(row.schedule ?? row.day_time),
        location: stringValue(row.location),
        instructor: stringValue(row.instructor),
      }),
    )
    .filter((item): item is ScheduleItem => Boolean(item));
  if (items.length === 0) return null;
  const titleTerm = table.title.split("—").pop()?.trim();
  return createScheduleDocument(items, {
    term: content.term || DEFAULT_SCHEDULE_TERM,
    termLabel: titleTerm || content.term || translate(getStoredLocale(), "schedule.defaultTerm"),
    source: "chatbot",
  });
}

export function scheduleItemFromCatalog(section: ScheduleCatalogSection): ScheduleItem {
  const meetings: ScheduleMeeting[] = [];
  for (const rawMeeting of section.meetings ?? []) {
    const dayValues = rawMeeting.day_codes?.length ? rawMeeting.day_codes : [rawMeeting.day ?? ""];
    for (const rawDay of dayValues) {
      const meeting = sanitiseMeeting({
        day: normaliseDay(rawDay) ?? undefined,
        start: stringValue(rawMeeting.start_time ?? rawMeeting.start),
        end: stringValue(rawMeeting.end_time ?? rawMeeting.end),
      });
      if (meeting) meetings.push(meeting);
    }
  }
  return sanitiseScheduleItem({
    courseCode: section.course_id,
    title: section.title || section.section_title || "",
    crn: section.crn,
    section: section.section,
    component: section.component_label || section.component || translate(getStoredLocale(), "schedule.defaultComponent"),
    componentCode: section.component_code || "",
    componentKind: scheduleComponentKind(section.component, section.component_code),
    instructor: section.instructors || "",
    location: section.locations || "",
    meetings,
  })!;
}

export function scheduleToBackendPayload(schedule: ScheduleDocument): Record<string, unknown> {
  const groups = new Map<string, { course_id: string; title: string; sections: Record<string, unknown>[] }>();
  for (const item of schedule.items) {
    const key = item.courseCode || item.id;
    const group = groups.get(key) ?? {
      course_id: item.courseCode,
      title: item.title,
      sections: [],
    };
    group.sections.push({
      id: item.id,
      course_id: item.courseCode,
      title: item.title,
      crn: item.crn,
      section: item.section,
      component: item.component,
      component_code: item.componentCode,
      component_label: item.component,
      instructors: item.instructor,
      locations: item.location,
      tba: item.meetings.length === 0,
      meetings: item.meetings.map((meeting) => ({
        day_codes: [meeting.day],
        day_names_tr: [WEEK_DAYS.find((day) => day.code === meeting.day)?.label ?? meeting.day],
        day_names_en: [
          ({ M: "Monday", T: "Tuesday", W: "Wednesday", R: "Thursday", F: "Friday" } as Record<ScheduleDay, string>)[meeting.day],
        ],
        start_time: meeting.start,
        end_time: meeting.end,
        location: item.location,
        instructors: item.instructor,
        status: "scheduled",
      })),
    });
    groups.set(key, group);
  }
  return {
    schema_version: 1,
    term: schedule.term,
    term_label: schedule.termLabel,
    generated_at: schedule.updatedAt,
    origin: schedule.source,
    source: {
      name: "AdviSU weekly schedule",
      authority: schedule.source === "chatbot" ? "AdviSU planner / official SUIS snapshot" : "User-edited AdviSU schedule",
      snapshot_date: schedule.updatedAt.slice(0, 10),
      scraped_at: schedule.updatedAt,
    },
    courses: Array.from(groups.values()),
    crns: scheduleCrns(schedule.items),
    not_offered: [],
    unplaced: [],
    conflicts: findScheduleConflicts(schedule.items),
  };
}

export function scheduleStorageKey(username?: string | null): string {
  return `${SCHEDULE_STORAGE_PREFIX}:${username || "guest"}`;
}

export function loadLocalSchedule(username?: string | null): ScheduleDocument | null {
  try {
    const raw = localStorage.getItem(scheduleStorageKey(username));
    return raw ? normaliseScheduleDocument(JSON.parse(raw)) : null;
  } catch {
    return null;
  }
}

export function storeLocalSchedule(username: string | null | undefined, schedule: ScheduleDocument): void {
  const normalised = createScheduleDocument(schedule.items, {
    term: schedule.term,
    termLabel: schedule.termLabel,
    source: schedule.source,
    updatedAt: schedule.updatedAt,
  });
  localStorage.setItem(scheduleStorageKey(username), JSON.stringify(normalised));
  window.dispatchEvent(
    new CustomEvent<ScheduleDocument>(SCHEDULE_REPLACE_EVENT, { detail: normalised }),
  );
}

function toMinutes(value: string): number {
  const [hours, minutes] = value.split(":").map(Number);
  return hours * 60 + minutes;
}

export function findScheduleConflicts(items: ScheduleItem[]): ScheduleConflict[] {
  const conflicts: ScheduleConflict[] = [];
  for (let firstIndex = 0; firstIndex < items.length; firstIndex += 1) {
    for (let secondIndex = firstIndex + 1; secondIndex < items.length; secondIndex += 1) {
      for (const first of items[firstIndex].meetings) {
        for (const second of items[secondIndex].meetings) {
          if (first.day !== second.day) continue;
          const overlapStart = Math.max(toMinutes(first.start), toMinutes(second.start));
          const overlapEnd = Math.min(toMinutes(first.end), toMinutes(second.end));
          if (overlapStart < overlapEnd) {
            conflicts.push({
              firstId: items[firstIndex].id,
              secondId: items[secondIndex].id,
              day: first.day,
              start: `${String(Math.floor(overlapStart / 60)).padStart(2, "0")}:${String(overlapStart % 60).padStart(2, "0")}`,
              end: `${String(Math.floor(overlapEnd / 60)).padStart(2, "0")}:${String(overlapEnd % 60).padStart(2, "0")}`,
            });
          }
        }
      }
    }
  }
  return conflicts;
}

export function uniqueCourseCount(items: ScheduleItem[]): number {
  return new Set(items.map((item) => item.courseCode).filter(Boolean)).size;
}

export function scheduleCrns(items: ScheduleItem[]): string[] {
  return Array.from(new Set(items.map((item) => item.crn).filter(Boolean)));
}

export function formatMeeting(meeting: ScheduleMeeting, locale: Locale = getStoredLocale()): string {
  const day = localizedWeekDays(locale).find((candidate) => candidate.code === meeting.day)?.short ?? meeting.day;
  return `${day} ${meeting.start}-${meeting.end}`;
}
