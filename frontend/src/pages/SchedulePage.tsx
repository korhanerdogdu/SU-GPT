import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import {
  AlertTriangle,
  ArrowLeft,
  CalendarDays,
  Check,
  ChevronDown,
  ChevronRight,
  Clock3,
  Copy,
  ExternalLink,
  FileSpreadsheet,
  Loader2,
  RotateCcw,
  Search,
  X,
} from "lucide-react";
import { toast } from "sonner";
import LanguageToggle from "@/components/LanguageToggle";
import ThemeToggle from "@/components/ThemeToggle";
import { Button } from "@/components/ui/button";
import { useAuth } from "@/contexts/AuthContext";
import { useLocale } from "@/contexts/LocaleContext";
import { localizedWeekDays } from "@/localization/resources";
import {
  fetchScheduleCourseSections,
  fetchScheduleSections,
  getUserSchedule,
  saveUserSchedule,
} from "@/lib/api";
import { exportScheduleXlsx } from "@/lib/export-xlsx";
import {
  DEFAULT_SCHEDULE_TERM,
  SCHEDULE_REPLACE_EVENT,
  createScheduleDocument,
  findScheduleConflicts,
  formatMeeting,
  loadLocalSchedule,
  scheduleBundleId,
  scheduleComponentKind,
  scheduleCrns,
  scheduleItemFromCatalog,
  storeLocalSchedule,
  uniqueCourseCount,
  type ScheduleCatalogSection,
  type ScheduleConflict,
  type ScheduleDay,
  type ScheduleDocument,
  type ScheduleItem,
  type ScheduleMeeting,
} from "@/lib/schedule";

const GRID_START = 8 * 60 + 40;
const GRID_END = 19 * 60 + 40;
const GRID_DURATION = GRID_END - GRID_START;

const COURSE_COLOURS = [
  "border-sky-700 bg-sky-600 text-white dark:border-sky-500 dark:bg-sky-700",
  "border-indigo-700 bg-indigo-600 text-white dark:border-indigo-500 dark:bg-indigo-700",
  "border-teal-700 bg-teal-600 text-white dark:border-teal-500 dark:bg-teal-700",
  "border-violet-700 bg-violet-600 text-white dark:border-violet-500 dark:bg-violet-700",
  "border-fuchsia-700 bg-fuchsia-600 text-white dark:border-fuchsia-500 dark:bg-fuchsia-700",
  "border-amber-600 bg-amber-400 text-slate-950 dark:border-amber-400 dark:bg-amber-500",
] as const;

type SyncState = "idle" | "saving" | "saved" | "offline";

interface GridEvent {
  key: string;
  item: ScheduleItem;
  meeting: ScheduleMeeting;
  start: number;
  end: number;
  lane: number;
  laneCount: number;
  conflict: boolean;
}

function minutes(value: string): number {
  const [hour, minute] = value.split(":").map(Number);
  return hour * 60 + minute;
}

function colourFor(code: string): string {
  let hash = 0;
  for (const character of code) hash = (hash * 31 + character.charCodeAt(0)) >>> 0;
  return COURSE_COLOURS[hash % COURSE_COLOURS.length];
}

function catalogSchedule(section: ScheduleCatalogSection, locale: "tr" | "en", noTime: string): string {
  const item = scheduleItemFromCatalog(section);
  return item.meetings.length ? item.meetings.map((meeting) => formatMeeting(meeting, locale)).join(", ") : `TBA · ${noTime}`;
}

function sectionGroup(section: string): string {
  return section.trim().match(/^[A-Za-z]+/)?.[0]?.toUpperCase() ?? "";
}

function groupsAreCompatible(primarySection: string, secondarySection: string): boolean {
  const primaryGroup = sectionGroup(primarySection);
  const secondaryGroup = sectionGroup(secondarySection);
  return !primaryGroup || !secondaryGroup || primaryGroup === secondaryGroup;
}

function meetingHasConflict(item: ScheduleItem, meeting: ScheduleMeeting, conflicts: ScheduleConflict[]): boolean {
  const start = minutes(meeting.start);
  const end = minutes(meeting.end);
  return conflicts.some((conflict) => (
    conflict.day === meeting.day
    && (conflict.firstId === item.id || conflict.secondId === item.id)
    && start < minutes(conflict.end)
    && end > minutes(conflict.start)
  ));
}

function layoutDay(items: ScheduleItem[], day: ScheduleDay, conflicts: ScheduleConflict[]): GridEvent[] {
  const events = items
    .flatMap((item) => item.meetings
      .filter((meeting) => meeting.day === day)
      .map((meeting, meetingIndex) => ({
        key: `${item.id}-${day}-${meetingIndex}-${meeting.start}`,
        item,
        meeting,
        start: Math.max(minutes(meeting.start), GRID_START),
        end: Math.min(minutes(meeting.end), GRID_END),
        lane: 0,
        laneCount: 1,
        conflict: meetingHasConflict(item, meeting, conflicts),
      })))
    .filter((event) => event.end > event.start)
    .sort((first, second) => first.start - second.start || first.end - second.end);

  let cluster: GridEvent[] = [];
  let clusterEnd = -1;

  function placeCluster() {
    if (cluster.length === 0) return;
    const laneEnds: number[] = [];
    for (const event of cluster) {
      let lane = laneEnds.findIndex((end) => end <= event.start);
      if (lane === -1) lane = laneEnds.length;
      laneEnds[lane] = event.end;
      event.lane = lane;
    }
    for (const event of cluster) event.laneCount = laneEnds.length;
  }

  for (const event of events) {
    if (cluster.length > 0 && event.start >= clusterEnd) {
      placeCluster();
      cluster = [];
      clusterEnd = -1;
    }
    cluster.push(event);
    clusterEnd = Math.max(clusterEnd, event.end);
  }
  placeCluster();
  return events;
}

export default function SchedulePage() {
  const { user } = useAuth();
  const { locale, t } = useLocale();
  const weekDays = useMemo(() => localizedWeekDays(locale), [locale]);
  const initial = useMemo(
    () => loadLocalSchedule(user?.username) ?? createScheduleDocument(),
    [user?.username],
  );
  const [schedule, setSchedule] = useState<ScheduleDocument>(initial);
  const [revision, setRevision] = useState(0);
  const [syncState, setSyncState] = useState<SyncState>("idle");
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<ScheduleCatalogSection[]>([]);
  const [searching, setSearching] = useState(false);
  const [searchError, setSearchError] = useState(false);
  const [hasMore, setHasMore] = useState(false);
  const [expandedCourse, setExpandedCourse] = useState<string | null>(null);
  const [focusedCourse, setFocusedCourse] = useState<string | null>(null);
  const [dayFilters, setDayFilters] = useState<ScheduleDay[]>([]);
  const [copied, setCopied] = useState(false);
  const hydrated = useRef(false);
  const focusRequest = useRef(0);
  const courseRefs = useRef(new Map<string, HTMLLIElement>());

  const conflicts = useMemo(() => findScheduleConflicts(schedule.items), [schedule.items]);
  const crns = useMemo(() => scheduleCrns(schedule.items), [schedule.items]);
  const tbaItems = useMemo(
    () => schedule.items.filter((item) => item.meetings.length === 0),
    [schedule.items],
  );
  const visibleResults = useMemo(
    () => results.filter((section) => (
      dayFilters.length === 0
      || section.meetings?.some((meeting) => meeting.day_codes?.some((day) => dayFilters.includes(day as ScheduleDay)))
    )),
    [dayFilters, results],
  );
  const groupedResults = useMemo(() => {
    const groups = new Map<string, { courseId: string; title: string; sections: ScheduleCatalogSection[] }>();
    for (const section of visibleResults) {
      const key = section.course_id;
      const existing = groups.get(key);
      if (existing) existing.sections.push(section);
      else groups.set(key, { courseId: key, title: section.title || section.section_title || t("common.unknownCourseTitle"), sections: [section] });
    }
    return [...groups.values()];
  }, [t, visibleResults]);

  const persist = useCallback(
    async (next: ScheduleDocument, expectedRevision?: number) => {
      storeLocalSchedule(user?.username, next);
      if (!user?.username) return;
      setSyncState("saving");
      try {
        const saved = await saveUserSchedule(user.username, next, expectedRevision);
        setRevision(saved.revision);
        setSyncState("saved");
      } catch {
        setSyncState("offline");
      }
    },
    [user?.username],
  );

  const replaceItems = useCallback(
    (items: ScheduleItem[], message?: string) => {
      const next = createScheduleDocument(items, {
        term: schedule.term,
        termLabel: schedule.termLabel,
        source: "manual",
      });
      setSchedule(next);
      void persist(next, revision);
      if (message) toast.success(message);
    },
    [persist, revision, schedule.term, schedule.termLabel],
  );

  useEffect(() => {
    if (!user?.username || hydrated.current) return;
    hydrated.current = true;
    let cancelled = false;
    void getUserSchedule(user.username)
      .then((remote) => {
        if (cancelled) return;
        setRevision(remote.revision);
        if (!remote.schedule) {
          if (initial.items.length > 0) void persist(initial, remote.revision);
          return;
        }
        const localTime = new Date(initial.updatedAt).getTime();
        const remoteTime = new Date(remote.schedule.updatedAt).getTime();
        const chosen = remoteTime >= localTime ? remote.schedule : initial;
        setSchedule(chosen);
        storeLocalSchedule(user.username, chosen);
        if (chosen === initial && initial.items.length > 0) void persist(initial, remote.revision);
        else setSyncState("saved");
      })
      .catch(() => setSyncState(initial.items.length ? "offline" : "idle"));
    return () => {
      cancelled = true;
    };
  }, [initial, persist, user?.username]);

  useEffect(() => {
    function receive(event: Event) {
      const next = (event as CustomEvent<ScheduleDocument>).detail;
      if (next?.items) setSchedule(next);
    }
    window.addEventListener(SCHEDULE_REPLACE_EVENT, receive);
    return () => window.removeEventListener(SCHEDULE_REPLACE_EVENT, receive);
  }, []);

  useEffect(() => {
    if (focusedCourse && query.trim().toLocaleUpperCase("tr-TR") === focusedCourse.toLocaleUpperCase("tr-TR")) return;
    let cancelled = false;
    const timer = window.setTimeout(() => {
      setSearching(true);
      setSearchError(false);
      void fetchScheduleSections(query, schedule.term || DEFAULT_SCHEDULE_TERM, 60)
        .then((response) => {
          if (cancelled) return;
          setResults(response.sections ?? []);
          setHasMore(Boolean(response.has_more));
          if (response.term_label && schedule.items.length === 0) {
            setSchedule((current) => ({ ...current, termLabel: response.term_label }));
          }
        })
        .catch(() => {
          if (!cancelled) {
            setResults([]);
            setSearchError(true);
          }
        })
        .finally(() => {
          if (!cancelled) setSearching(false);
        });
    }, 240);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [focusedCourse, query, schedule.items.length, schedule.term]);

  useEffect(() => {
    if (!focusedCourse) return;
    window.requestAnimationFrame(() => {
      courseRefs.current.get(focusedCourse)?.scrollIntoView({ block: "nearest", behavior: "smooth" });
    });
  }, [focusedCourse, groupedResults]);

  async function focusCourse(item: ScheduleItem) {
    const courseCode = item.courseCode;
    const requestId = focusRequest.current + 1;
    focusRequest.current = requestId;
    setFocusedCourse(courseCode);
    setExpandedCourse(courseCode);
    setQuery(courseCode);
    setSearching(true);
    setSearchError(false);
    try {
      const response = await fetchScheduleCourseSections(courseCode, schedule.term || DEFAULT_SCHEDULE_TERM);
      if (requestId !== focusRequest.current) return;
      setResults(response.sections ?? []);
      setHasMore(false);
    } catch {
      if (requestId !== focusRequest.current) return;
      setSearchError(true);
      toast.error(t("schedule.sectionLoadFailed", { course: courseCode }));
    } finally {
      if (requestId === focusRequest.current) setSearching(false);
    }
  }

  function addOrReplaceSection(section: ScheduleCatalogSection) {
    const item = scheduleItemFromCatalog(section);
    if (schedule.items.some((candidate) => candidate.crn && candidate.crn === item.crn)) {
      toast.info(t("schedule.alreadySelected"));
      return;
    }

    const bundleId = scheduleBundleId(item.courseCode);
    const otherCourses = schedule.items.filter((candidate) => candidate.bundleId !== bundleId);
    const currentBundle = schedule.items.filter((candidate) => candidate.bundleId === bundleId);

    if (item.componentKind === "primary") {
      const secondary = currentBundle.find((candidate) => candidate.componentKind === "secondary");
      const keepSecondary = secondary && groupsAreCompatible(item.section, secondary.section);
      replaceItems(
        [...otherCourses, item, ...(keepSecondary ? [secondary] : [])],
        keepSecondary
          ? t("schedule.primaryUpdated", { course: item.courseCode })
          : secondary
            ? t("schedule.incompatibleRemoved", { course: item.courseCode })
            : t("schedule.added", { course: item.courseCode }),
      );
      return;
    }

    const currentPrimary = currentBundle.find((candidate) => candidate.componentKind === "primary");
    const catalogPrimary = results
      .filter((candidate) => candidate.course_id === section.course_id)
      .find((candidate) => (
        scheduleComponentKind(candidate.component, candidate.component_code) === "primary"
        && groupsAreCompatible(candidate.section, section.section)
      ));
    const primary = currentPrimary && groupsAreCompatible(currentPrimary.section, item.section)
      ? currentPrimary
      : catalogPrimary
        ? scheduleItemFromCatalog(catalogPrimary)
        : currentPrimary;
    replaceItems(
      [...otherCourses, ...(primary ? [primary] : []), item],
      primary && !currentPrimary
        ? t("schedule.bundleAdded", { course: item.courseCode, component: item.component.toLocaleLowerCase(locale === "tr" ? "tr-TR" : "en-US") })
        : t("schedule.secondaryUpdated", { course: item.courseCode, component: item.component.toLocaleLowerCase(locale === "tr" ? "tr-TR" : "en-US") }),
    );
  }

  function removeBundle(item: ScheduleItem) {
    const bundle = schedule.items.filter((candidate) => candidate.bundleId === item.bundleId);
    replaceItems(
      schedule.items.filter((candidate) => candidate.bundleId !== item.bundleId),
      bundle.length > 1
        ? t("schedule.bundleRemoved", { course: item.courseCode })
        : t("schedule.removed", { course: item.courseCode }),
    );
  }

  async function copyCrns() {
    if (crns.length === 0) return;
    try {
      await navigator.clipboard.writeText(crns.join(" "));
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1800);
    } catch {
      toast.error(t("schedule.copyFailed"));
    }
  }

  function clearSchedule() {
    if (!window.confirm(t("schedule.clearConfirm"))) return;
    replaceItems([], t("schedule.cleared"));
  }

  function updateQuery(value: string) {
    focusRequest.current += 1;
    setFocusedCourse(null);
    setExpandedCourse(null);
    setQuery(value);
  }

  return (
    <div className="flex min-h-dvh flex-col overflow-x-hidden bg-muted/30 text-foreground lg:h-dvh lg:overflow-hidden">
      <header className="z-30 shrink-0 border-b border-white/10 bg-primary text-primary-foreground shadow-md shadow-primary/10">
        <div className="mx-auto flex h-14 max-w-[1920px] items-center gap-2.5 px-3 sm:px-4">
          <Link to="/" aria-label={t("common.backToChat")} className="grid h-9 w-9 shrink-0 place-items-center rounded-xl text-white/80 transition hover:bg-white/10 hover:text-white">
            <ArrowLeft className="h-4 w-4" />
          </Link>
          <img src="/assets/small_witihoutbg.png" alt="AdviSU" className="h-9 w-9 object-contain" />
          <div className="min-w-0 flex-1">
            <div className="flex min-w-0 items-center gap-2.5">
              <h1 className="shrink-0 text-base font-bold tracking-tight sm:text-lg">{t("schedule.title")}</h1>
              <a
                href="https://github.com/aburakayaz/suchedule"
                target="_blank"
                rel="noreferrer"
                className="hidden min-w-0 items-center gap-1 truncate rounded-full bg-white/10 px-2 py-1 text-[10px] font-semibold text-white/80 transition hover:bg-white/20 hover:text-white md:inline-flex"
              >
                {t("schedule.reference")} <ExternalLink className="h-3 w-3 shrink-0" />
              </a>
            </div>
            <p className="truncate text-[10px] text-white/70">{schedule.termLabel} · {t("schedule.term", { term: schedule.term })}</p>
          </div>
          <SyncIndicator state={syncState} />
          <div className="hidden items-center gap-1.5 md:flex">
            <Button type="button" size="sm" onClick={copyCrns} disabled={crns.length === 0} className="h-8 border border-white/15 bg-white/10 px-2.5 text-white hover:bg-white/20">
              {copied ? <Check className="mr-1.5 h-3.5 w-3.5" /> : <Copy className="mr-1.5 h-3.5 w-3.5" />}
              {copied ? t("chat.copied") : t("chat.copyCrns")}
            </Button>
            <Button type="button" size="sm" onClick={() => exportScheduleXlsx(schedule, user?.username ?? "student", locale)} disabled={schedule.items.length === 0} className="h-8 border border-white/15 bg-white/10 px-2.5 text-white hover:bg-white/20">
              <FileSpreadsheet className="mr-1.5 h-3.5 w-3.5" /> Excel
            </Button>
            <Button type="button" size="sm" onClick={clearSchedule} disabled={schedule.items.length === 0} className="h-8 bg-transparent px-2.5 text-white/80 hover:bg-white/10 hover:text-white">
              <RotateCcw className="mr-1.5 h-3.5 w-3.5" /> {t("common.clear")}
            </Button>
          </div>
          <LanguageToggle />
          <ThemeToggle />
        </div>
      </header>

      <main className="mx-auto min-h-0 w-full max-w-[1920px] flex-1 overflow-y-auto p-2 lg:overflow-hidden">
        <div className="grid min-h-0 gap-2 lg:h-full lg:grid-cols-[clamp(300px,24vw,350px)_minmax(0,1fr)]">
          <aside className="flex min-h-[560px] flex-col overflow-hidden rounded-2xl border border-border bg-card text-card-foreground shadow-sm lg:min-h-0">
            <div className="shrink-0 border-b border-border bg-primary/[0.045] p-3">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <h2 className="text-sm font-bold text-foreground">{t("schedule.pick")}</h2>
                  <p className="mt-0.5 text-[11px] text-muted-foreground">{t("schedule.official")}</p>
                </div>
                <span className="rounded-full border border-primary/15 bg-primary/10 px-2 py-1 text-[10px] font-bold text-primary">
                  {t("schedule.courseCount", { count: uniqueCourseCount(schedule.items) })}
                </span>
              </div>
              <label className="relative mt-2.5 block">
                <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                <input
                  value={query}
                  onChange={(event) => updateQuery(event.target.value)}
                  placeholder={t("schedule.search")}
                  aria-label={t("schedule.search")}
                  className="w-full rounded-xl border border-input bg-background py-2.5 pl-9 pr-9 text-sm text-foreground outline-none placeholder:text-muted-foreground focus:border-primary focus:ring-2 focus:ring-primary/15"
                />
                {query && (
                  <button type="button" onClick={() => updateQuery("")} aria-label={t("schedule.clearSearch")} className="absolute right-2.5 top-1/2 -translate-y-1/2 rounded-md p-1 text-muted-foreground transition hover:bg-muted hover:text-foreground">
                    <X className="h-3.5 w-3.5" />
                  </button>
                )}
              </label>
              <div className="mt-2.5 flex items-center gap-1" aria-label={t("schedule.filterDay")}>
                <span className="mr-auto text-[10px] font-semibold uppercase tracking-[0.12em] text-muted-foreground">{t("schedule.byDay")}</span>
                {weekDays.map((day) => {
                  const active = dayFilters.includes(day.code);
                  return (
                    <button
                      key={day.code}
                      type="button"
                      aria-pressed={active}
                      onClick={() => setDayFilters((current) => active ? current.filter((code) => code !== day.code) : [...current, day.code])}
                      className={`rounded-lg px-2 py-1 text-[10px] font-bold transition ${active ? "bg-primary text-primary-foreground shadow-sm" : "border border-border bg-background text-muted-foreground hover:bg-muted hover:text-foreground"}`}
                    >
                      {day.short.toLocaleUpperCase(locale === "tr" ? "tr-TR" : "en-US")}
                    </button>
                  );
                })}
              </div>
            </div>

            <div className="min-h-0 flex-1 overflow-y-auto p-2 scrollbar-thin">
              {searching ? (
                <div className="flex items-center justify-center gap-2 py-10 text-sm text-muted-foreground">
                  <Loader2 className="h-4 w-4 animate-spin" /> {t("schedule.loading")}
                </div>
              ) : searchError ? (
                <div className="rounded-xl border border-dashed border-border bg-muted/30 p-5 text-center">
                  <p className="text-sm font-medium">{t("schedule.loadFailed")}</p>
                  <p className="mt-1 text-xs text-muted-foreground">{t("schedule.localSafe")}</p>
                </div>
              ) : groupedResults.length === 0 ? (
                <div className="rounded-xl border border-dashed border-border bg-muted/30 p-5 text-center text-sm text-muted-foreground">
                  {t("schedule.noResults")}
                </div>
              ) : (
                <ul className="space-y-1.5">
                  {groupedResults.map((group) => {
                    const open = expandedCourse === group.courseId;
                    const selectedCount = group.sections.filter((section) => schedule.items.some((item) => item.crn === String(section.crn))).length;
                    return (
                      <li
                        key={group.courseId}
                        ref={(node) => {
                          if (node) courseRefs.current.set(group.courseId, node);
                          else courseRefs.current.delete(group.courseId);
                        }}
                        className={`overflow-hidden rounded-xl border transition ${focusedCourse === group.courseId ? "border-primary/45 bg-primary/[0.04] shadow-[0_0_0_3px_hsl(var(--primary)/0.08)]" : "border-border bg-background"}`}
                      >
                        <button type="button" onClick={() => setExpandedCourse(open ? null : group.courseId)} className="flex w-full items-center gap-2.5 px-3 py-2.5 text-left transition hover:bg-muted/70">
                          <span className="min-w-0 flex-1">
                            <span className="block text-sm font-extrabold text-foreground">{group.courseId}</span>
                            <span className="mt-0.5 block truncate text-[11px] text-muted-foreground">{group.title}</span>
                          </span>
                          {selectedCount > 0 && <span className="rounded-full bg-emerald-500/12 px-2 py-0.5 text-[10px] font-bold text-emerald-700 dark:text-emerald-300">{t("schedule.selected", { count: selectedCount })}</span>}
                          {open ? <ChevronDown className="h-4 w-4 text-muted-foreground" /> : <ChevronRight className="h-4 w-4 text-muted-foreground" />}
                        </button>
                        {open && (
                          <div className="space-y-1.5 border-t border-border bg-muted/25 p-2">
                            {group.sections.map((section) => {
                              const selected = schedule.items.some((item) => item.crn === String(section.crn));
                              return (
                                <button
                                  key={`${section.course_id}-${section.crn}-${section.section}`}
                                  type="button"
                                  aria-pressed={selected}
                                  onClick={() => selected ? toast.info(t("schedule.alreadySelected")) : addOrReplaceSection(section)}
                                  className={`group w-full rounded-xl border p-2.5 text-left transition ${selected ? "border-primary/35 bg-primary/10 text-foreground" : "border-border bg-card hover:border-primary/30 hover:bg-primary/[0.04]"}`}
                                >
                                  <span className="flex items-start gap-2">
                                    <span className="min-w-0 flex-1">
                                      <span className="flex flex-wrap items-center gap-1 text-[10px]">
                                        <span className="rounded-md bg-muted px-1.5 py-0.5 font-mono font-semibold">CRN {section.crn}</span>
                                        <span className="rounded-md bg-muted px-1.5 py-0.5 font-semibold">{t("common.section")} {section.section}</span>
                                        <span className="rounded-md bg-muted px-1.5 py-0.5 font-semibold">{section.component_label || section.component || t("schedule.defaultComponent")}</span>
                                      </span>
                                      <span className="mt-1.5 block text-[11px] font-semibold leading-relaxed text-foreground">{catalogSchedule(section, locale, t("schedule.noTime"))}</span>
                                      {section.instructors && <span className="mt-1 block truncate text-[10px] text-muted-foreground">{section.instructors}</span>}
                                    </span>
                                    {selected ? <Check className="h-4 w-4 shrink-0 text-primary" /> : <span className="grid h-5 w-5 shrink-0 place-items-center rounded-full bg-primary/10 text-sm font-bold text-primary transition group-hover:bg-primary group-hover:text-primary-foreground">+</span>}
                                  </span>
                                </button>
                              );
                            })}
                          </div>
                        )}
                      </li>
                    );
                  })}
                </ul>
              )}
              {hasMore && !searching && <p className="px-2 py-3 text-center text-[11px] text-muted-foreground">{t("schedule.moreHint")}</p>}
            </div>
          </aside>

          <section className="flex min-h-[620px] min-w-0 flex-col overflow-hidden rounded-2xl border border-border bg-card shadow-sm lg:min-h-0">
            <div className="flex shrink-0 items-center gap-2 border-b border-border px-3 py-2 sm:px-4">
              <div className="mr-auto min-w-0">
                <h2 className="truncate text-sm font-bold">{t("schedule.week")}</h2>
                <p className="truncate text-[11px] text-muted-foreground">{t("schedule.weekHint")}</p>
              </div>
              <div className="hidden items-center gap-1 xl:flex">
                <QuickStat label={t("common.course")} value={uniqueCourseCount(schedule.items)} />
                <QuickStat label="CRN" value={crns.length} />
                {conflicts.length > 0 && <ConflictStat conflicts={conflicts} />}
              </div>
              <div className="flex items-center gap-1 md:hidden">
                <button type="button" onClick={copyCrns} disabled={crns.length === 0} aria-label={t("chat.copyCrns")} className="rounded-lg border border-border p-2 text-muted-foreground hover:bg-muted"><Copy className="h-4 w-4" /></button>
                <button type="button" onClick={() => exportScheduleXlsx(schedule, user?.username ?? "student", locale)} disabled={schedule.items.length === 0} aria-label={t("schedule.downloadExcel")} className="rounded-lg border border-border p-2 text-muted-foreground hover:bg-muted"><FileSpreadsheet className="h-4 w-4" /></button>
              </div>
            </div>

            {tbaItems.length > 0 && (
              <div className="flex shrink-0 items-center gap-2 border-b border-border bg-muted/35 px-3 py-1.5">
                <Clock3 className="h-3.5 w-3.5 shrink-0 text-primary" />
                <span className="shrink-0 text-[10px] font-bold uppercase tracking-wide text-muted-foreground">TBA</span>
                <div className="flex min-w-0 flex-1 gap-1.5 overflow-x-auto scrollbar-thin">
                  {tbaItems.map((item) => (
                    <span key={item.id} className="inline-flex shrink-0 items-center gap-1 rounded-lg border border-border bg-card py-1 pl-2 pr-1 text-[11px] font-semibold">
                      <button type="button" onClick={() => void focusCourse(item)} className="hover:text-primary">{item.courseCode} · {item.section || item.component}</button>
                      <button type="button" onClick={() => removeBundle(item)} aria-label={t("schedule.removeBundle", { course: item.courseCode })} className="grid h-5 w-5 place-items-center rounded-md text-muted-foreground hover:bg-destructive/10 hover:text-destructive"><X className="h-3 w-3" /></button>
                    </span>
                  ))}
                </div>
              </div>
            )}

            <div className="min-h-0 flex-1 overflow-y-auto p-2 md:overflow-hidden">
              <DesktopWeekGrid items={schedule.items} conflicts={conflicts} onFocus={focusCourse} onRemove={removeBundle} />
              <MobileWeekList items={schedule.items} conflicts={conflicts} onFocus={focusCourse} onRemove={removeBundle} />
            </div>
          </section>
        </div>
      </main>
    </div>
  );
}

function SyncIndicator({ state }: { state: SyncState }) {
  const { t } = useLocale();
  if (state === "idle") return null;
  const label = state === "saving" ? t("schedule.saving") : state === "saved" ? t("common.saved") : t("schedule.offline");
  return (
    <span className="hidden items-center gap-1.5 text-[11px] text-white/70 sm:flex">
      {state === "saving" ? <Loader2 className="h-3 w-3 animate-spin" /> : <Check className="h-3 w-3" />}
      {label}
    </span>
  );
}

function QuickStat({ label, value }: { label: string; value: number }) {
  return (
    <span className="inline-flex items-center gap-1 rounded-lg border border-border bg-muted/50 px-2 py-1 text-[10px] font-semibold text-muted-foreground">
      {label}<strong className="text-xs tabular-nums text-foreground">{value}</strong>
    </span>
  );
}

function ConflictStat({ conflicts }: { conflicts: ScheduleConflict[] }) {
  const { locale, t } = useLocale();
  const weekDays = localizedWeekDays(locale);
  const first = conflicts[0];
  const day = weekDays.find((candidate) => candidate.code === first.day)?.label ?? first.day;
  return (
    <span title={`${day} ${first.start}-${first.end}`} className="inline-flex items-center gap-1 rounded-lg border border-rose-400/60 bg-rose-500/10 px-2 py-1 text-[10px] font-bold text-rose-700 dark:text-rose-300">
      <AlertTriangle className="h-3 w-3" /> {t("schedule.conflicts", { count: conflicts.length })}
    </span>
  );
}

function DesktopWeekGrid({
  items,
  conflicts,
  onFocus,
  onRemove,
}: {
  items: ScheduleItem[];
  conflicts: ScheduleConflict[];
  onFocus: (item: ScheduleItem) => void;
  onRemove: (item: ScheduleItem) => void;
}) {
  const { locale, t } = useLocale();
  const weekDays = localizedWeekDays(locale);
  const times = Array.from({ length: 12 }, (_, index) => GRID_START + index * 60);
  const scheduledCount = items.reduce((count, item) => count + item.meetings.length, 0);
  return (
    <div className="hidden h-full min-h-0 overflow-hidden rounded-xl border border-border bg-card md:flex md:flex-col">
      <div className="grid h-10 shrink-0 grid-cols-[72px_repeat(5,minmax(0,1fr))] border-b border-primary/20 bg-primary text-primary-foreground">
        <div className="grid place-items-center border-r border-white/15 text-[10px] font-bold uppercase tracking-[0.14em] text-white/75">{t("schedule.time")}</div>
        {weekDays.map((day) => <div key={day.code} className="grid place-items-center border-r border-white/15 px-2 text-center text-xs font-extrabold last:border-r-0">{day.label}</div>)}
      </div>
      <div className="grid min-h-0 flex-1 grid-cols-[72px_repeat(5,minmax(0,1fr))]">
        <div className="relative border-r border-border bg-muted/35">
          {times.map((time, index) => {
            const top = ((time - GRID_START) / GRID_DURATION) * 100;
            const transform = index === 0 ? "translateY(6px)" : index === times.length - 1 ? "translateY(calc(-100% - 6px))" : "translateY(-50%)";
            return (
              <span key={time} className="absolute inset-x-0 z-10 px-2 text-right font-mono text-[13px] font-extrabold tabular-nums text-foreground/80" style={{ top: `${top}%`, transform }}>
                {String(Math.floor(time / 60)).padStart(2, "0")}:{String(time % 60).padStart(2, "0")}
              </span>
            );
          })}
        </div>
        {weekDays.map((day) => {
          const dayEvents = layoutDay(items, day.code, conflicts);
          return (
            <div key={day.code} className="relative min-w-0 border-r border-border bg-card last:border-r-0">
              {times.map((time) => <span key={time} className="pointer-events-none absolute inset-x-0 border-t border-border/70" style={{ top: `${((time - GRID_START) / GRID_DURATION) * 100}%` }} />)}
              {dayEvents.map((event) => {
                const top = ((event.start - GRID_START) / GRID_DURATION) * 100;
                const height = ((event.end - event.start) / GRID_DURATION) * 100;
                const left = (event.lane / event.laneCount) * 100;
                const width = 100 / event.laneCount;
                const duration = event.end - event.start;
                return (
                  <div
                    key={event.key}
                    role="group"
                    aria-label={`${event.item.courseCode} ${event.meeting.start}-${event.meeting.end}`}
                    className={`absolute z-10 overflow-hidden rounded-lg border shadow-sm transition hover:z-20 hover:shadow-lg ${event.conflict ? "border-rose-700 bg-rose-600 text-white ring-2 ring-rose-400/80" : colourFor(event.item.courseCode)}`}
                    style={{
                      top: `calc(${top}% + 2px)`,
                      height: `max(calc(${height}% - 4px), 38px)`,
                      left: `calc(${left}% + 3px)`,
                      width: `calc(${width}% - 6px)`,
                      backgroundImage: event.conflict ? "linear-gradient(135deg, rgba(255,255,255,.10) 25%, transparent 25%, transparent 50%, rgba(255,255,255,.10) 50%, rgba(255,255,255,.10) 75%, transparent 75%)" : undefined,
                      backgroundSize: event.conflict ? "12px 12px" : undefined,
                    }}
                  >
                    <button type="button" onClick={() => void onFocus(event.item)} className="h-full w-full min-w-0 px-2 py-1.5 pr-7 text-left">
                      <span className="block truncate text-[13px] font-extrabold leading-tight tracking-tight">
                        {event.item.courseCode} <span className="font-semibold opacity-90">· {event.item.title}</span>
                      </span>
                      <span className="mt-0.5 block truncate font-mono text-[11px] font-bold leading-tight tabular-nums">
                        {event.meeting.start}–{event.meeting.end} · {event.item.section || event.item.component}
                      </span>
                      {duration >= 75 && event.item.location && <span className="mt-1 block truncate text-[10px] font-medium opacity-85">{event.item.location}</span>}
                      {event.conflict && duration >= 70 && <span className="mt-1 inline-flex items-center gap-1 rounded bg-white/20 px-1.5 py-0.5 text-[9px] font-extrabold uppercase tracking-wide"><AlertTriangle className="h-2.5 w-2.5" /> {t("schedule.conflicting")}</span>}
                    </button>
                    <button
                      type="button"
                      onClick={(eventClick) => {
                        eventClick.stopPropagation();
                        onRemove(event.item);
                      }}
                      aria-label={t("schedule.removeBundle", { course: event.item.courseCode })}
                      className="absolute right-1 top-1 grid h-6 w-6 place-items-center rounded-md bg-black/15 text-white/90 backdrop-blur-sm transition hover:bg-white hover:text-slate-950 focus-visible:bg-white focus-visible:text-slate-950"
                    >
                      <X className="h-3.5 w-3.5" />
                    </button>
                  </div>
                );
              })}
              {scheduledCount === 0 && day.code === "W" && (
                <div className="pointer-events-none absolute left-1/2 top-1/2 z-20 w-[min(300px,85vw)] -translate-x-1/2 -translate-y-1/2 rounded-2xl border border-primary/15 bg-card/95 p-5 text-center shadow-xl backdrop-blur">
                  <span className="mx-auto grid h-11 w-11 place-items-center rounded-xl bg-primary/10 text-primary"><CalendarDays className="h-5 w-5" /></span>
                  <p className="mt-3 text-sm font-bold">{t("schedule.emptyTitle")}</p>
                  <p className="mt-1 text-[11px] leading-relaxed text-muted-foreground">{t("schedule.emptyBody")}</p>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function MobileWeekList({
  items,
  conflicts,
  onFocus,
  onRemove,
}: {
  items: ScheduleItem[];
  conflicts: ScheduleConflict[];
  onFocus: (item: ScheduleItem) => void;
  onRemove: (item: ScheduleItem) => void;
}) {
  const { locale, t } = useLocale();
  const weekDays = localizedWeekDays(locale);
  return (
    <div className="space-y-3 md:hidden">
      {weekDays.map((day) => {
        const dayItems = items
          .flatMap((item) => item.meetings.filter((meeting) => meeting.day === day.code).map((meeting) => ({ item, meeting })))
          .sort((first, second) => first.meeting.start.localeCompare(second.meeting.start));
        return (
          <section key={day.code} className="rounded-2xl border border-border bg-card p-3 shadow-sm">
            <h3 className="mb-2 px-1 text-xs font-bold uppercase tracking-wider text-muted-foreground">{day.label}</h3>
            {dayItems.length === 0 ? <p className="rounded-xl bg-muted/50 px-3 py-4 text-center text-xs text-muted-foreground">{t("schedule.noClass")}</p> : (
              <div className="space-y-2">
                {dayItems.map(({ item, meeting }) => {
                  const conflict = meetingHasConflict(item, meeting, conflicts);
                  return (
                    <div key={`${item.id}-${meeting.start}`} className={`relative flex items-center rounded-xl border ${conflict ? "border-rose-500 bg-rose-500/10" : "border-border bg-background"}`}>
                      <button type="button" onClick={() => void onFocus(item)} className="flex min-w-0 flex-1 items-center gap-3 p-3 pr-10 text-left">
                        <span className={`h-10 w-1 shrink-0 rounded-full ${conflict ? "bg-rose-500" : colourFor(item.courseCode).split(" ")[1]}`} />
                        <span className="w-24 shrink-0 font-mono text-xs font-bold tabular-nums">{meeting.start}–{meeting.end}</span>
                        <span className="min-w-0 flex-1"><span className="block truncate text-sm font-extrabold">{item.courseCode} · {item.title}</span><span className="block truncate text-[11px] text-muted-foreground">{item.section || item.component} · {item.location}</span></span>
                        {conflict && <AlertTriangle className="h-4 w-4 shrink-0 text-rose-600" />}
                      </button>
                      <button type="button" onClick={() => onRemove(item)} aria-label={t("schedule.removeBundle", { course: item.courseCode })} className="absolute right-2 top-1/2 grid h-7 w-7 -translate-y-1/2 place-items-center rounded-lg text-muted-foreground hover:bg-destructive/10 hover:text-destructive"><X className="h-4 w-4" /></button>
                    </div>
                  );
                })}
              </div>
            )}
          </section>
        );
      })}
    </div>
  );
}
