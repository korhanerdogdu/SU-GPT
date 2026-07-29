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
  FileSpreadsheet,
  Loader2,
  MapPin,
  Pencil,
  Plus,
  RotateCcw,
  Search,
  Trash2,
  UserRound,
  X,
} from "lucide-react";
import { toast } from "sonner";
import ThemeToggle from "@/components/ThemeToggle";
import { Button } from "@/components/ui/button";
import { useAuth } from "@/contexts/AuthContext";
import {
  fetchScheduleSections,
  getUserSchedule,
  saveUserSchedule,
} from "@/lib/api";
import { exportScheduleXlsx } from "@/lib/export-xlsx";
import {
  DEFAULT_SCHEDULE_TERM,
  SCHEDULE_REPLACE_EVENT,
  WEEK_DAYS,
  createScheduleDocument,
  findScheduleConflicts,
  formatMeeting,
  loadLocalSchedule,
  scheduleCrns,
  scheduleItemFromCatalog,
  storeLocalSchedule,
  uniqueCourseCount,
  type ScheduleCatalogSection,
  type ScheduleDay,
  type ScheduleDocument,
  type ScheduleItem,
  type ScheduleMeeting,
} from "@/lib/schedule";

const GRID_START = 8 * 60 + 40;
const GRID_END = 19 * 60 + 40;
const PIXELS_PER_MINUTE = 1;
const COURSE_COLOURS = [
  "bg-sky-600 border-sky-700 text-white",
  "bg-indigo-600 border-indigo-700 text-white",
  "bg-teal-600 border-teal-700 text-white",
  "bg-violet-600 border-violet-700 text-white",
  "bg-rose-600 border-rose-700 text-white",
  "bg-amber-500 border-amber-600 text-slate-950",
] as const;

type SyncState = "idle" | "saving" | "saved" | "offline";

function minutes(value: string): number {
  const [hour, minute] = value.split(":").map(Number);
  return hour * 60 + minute;
}

function colourFor(code: string): string {
  let hash = 0;
  for (const character of code) hash = (hash * 31 + character.charCodeAt(0)) >>> 0;
  return COURSE_COLOURS[hash % COURSE_COLOURS.length];
}

function courseLabel(item: ScheduleItem): string {
  return [item.courseCode, item.section ? `· ${item.section}` : ""].filter(Boolean).join(" ");
}

function catalogSchedule(section: ScheduleCatalogSection): string {
  const item = scheduleItemFromCatalog(section);
  return item.meetings.length ? item.meetings.map(formatMeeting).join(", ") : "TBA · Saat açıklanmadı";
}

function blankItem(): ScheduleItem {
  return {
    id: `manual-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
    courseCode: "",
    title: "",
    component: "Ders",
    crn: "",
    section: "",
    instructor: "",
    location: "",
    meetings: [],
  };
}

export default function SchedulePage() {
  const { user } = useAuth();
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
  const [dayFilters, setDayFilters] = useState<ScheduleDay[]>([]);
  const [editor, setEditor] = useState<ScheduleItem | null>(null);
  const [copied, setCopied] = useState(false);
  const hydrated = useRef(false);

  const conflicts = useMemo(() => findScheduleConflicts(schedule.items), [schedule.items]);
  const conflictedIds = useMemo(
    () => new Set(conflicts.flatMap((conflict) => [conflict.firstId, conflict.secondId])),
    [conflicts],
  );
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
      else groups.set(key, { courseId: key, title: section.title || section.section_title || "Ders adÄ±", sections: [section] });
    }
    return [...groups.values()];
  }, [visibleResults]);

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
    }, 260);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [query, schedule.items.length, schedule.term]);

  function addSection(section: ScheduleCatalogSection) {
    const item = scheduleItemFromCatalog(section);
    if (schedule.items.some((candidate) => candidate.crn && candidate.crn === item.crn)) {
      toast.info("Bu CRN zaten programında.");
      return;
    }
    replaceItems([...schedule.items, item], `${item.courseCode} programa eklendi.`);
  }

  function saveEdited(item: ScheduleItem) {
    const exists = schedule.items.some((candidate) => candidate.id === item.id);
    const items = exists
      ? schedule.items.map((candidate) => (candidate.id === item.id ? item : candidate))
      : [...schedule.items, item];
    replaceItems(items, exists ? "Ders güncellendi." : "Ders programa eklendi.");
    setEditor(null);
  }

  function removeItem(id: string) {
    replaceItems(schedule.items.filter((item) => item.id !== id), "Ders programdan kaldırıldı.");
  }

  async function copyCrns() {
    if (crns.length === 0) return;
    try {
      await navigator.clipboard.writeText(crns.join(" "));
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1800);
    } catch {
      toast.error("CRN'ler panoya kopyalanamadı.");
    }
  }

  function clearSchedule() {
    if (!window.confirm("Programındaki tüm dersleri kaldırmak istediğine emin misin?")) return;
    replaceItems([], "Ders programı temizlendi.");
  }

  return (
    <div className="flex min-h-screen flex-col bg-muted/30 text-foreground xl:h-screen xl:overflow-hidden">
      <header className="z-30 shrink-0 border-b border-white/10 bg-primary text-primary-foreground shadow-lg shadow-primary/10">
        <div className="mx-auto flex h-16 max-w-[1920px] items-center gap-3 px-3 sm:px-5">
          <Link
            to="/"
            aria-label="Sohbete dön"
            className="grid h-9 w-9 shrink-0 place-items-center rounded-xl text-white/80 transition hover:bg-white/10 hover:text-white"
          >
            <ArrowLeft className="h-4 w-4" />
          </Link>
          <img src="/assets/small_witihoutbg.png" alt="AdviSU" className="h-10 w-10 object-contain" />
          <div className="flex min-w-0 flex-1 items-center gap-2.5">
            <div className="min-w-0">
              <div className="flex items-center gap-2">
                <h1 className="truncate text-base font-bold tracking-tight sm:text-lg">Ders Programı</h1>
                <span className="hidden rounded-full bg-white/10 px-2 py-0.5 text-[10px] font-semibold text-white/80 sm:inline">SUchedule, yeniden tasarlandı</span>
              </div>
              <p className="truncate text-[11px] text-white/70">{schedule.termLabel}</p>
            </div>
          </div>
          <SyncIndicator state={syncState} inverse />
          <div className="hidden items-center gap-1.5 lg:flex">
            <Button type="button" size="sm" onClick={copyCrns} disabled={crns.length === 0} className="border border-white/15 bg-white/10 text-white hover:bg-white/20">
              {copied ? <Check className="mr-1.5 h-3.5 w-3.5" /> : <Copy className="mr-1.5 h-3.5 w-3.5" />}
              {copied ? "Kopyalandı" : "CRN'leri kopyala"}
            </Button>
            <Button type="button" size="sm" onClick={() => exportScheduleXlsx(schedule, user?.username ?? "student")} disabled={schedule.items.length === 0} className="border border-white/15 bg-white/10 text-white hover:bg-white/20">
              <FileSpreadsheet className="mr-1.5 h-3.5 w-3.5" /> Excel
            </Button>
            <Button type="button" size="sm" onClick={clearSchedule} disabled={schedule.items.length === 0} className="bg-transparent text-white/80 hover:bg-white/10 hover:text-white">
              <RotateCcw className="mr-1.5 h-3.5 w-3.5" /> Temizle
            </Button>
          </div>
          <ThemeToggle />
        </div>
      </header>

      <main className="mx-auto min-h-0 w-full max-w-[1920px] flex-1 p-2.5 sm:p-3 xl:overflow-hidden">
        <section className="mb-3 flex flex-wrap items-center gap-2 xl:hidden">
          <SummaryCard label="Ders" value={uniqueCourseCount(schedule.items)} detail={`${schedule.items.length} section`} />
          <SummaryCard label="CRN" value={crns.length} detail="kayıt için hazır" />
          <SummaryCard label="Saat bekleyen" value={tbaItems.length} detail="TBA section" />
          <SummaryCard
            label="Çakışma"
            value={conflicts.length}
            detail={conflicts.length ? "düzenleme gerekiyor" : "program dengeli"}
            warning={conflicts.length > 0}
          />
        </section>

        {conflicts.length > 0 && (
          <div className="mb-3 flex items-start gap-3 rounded-xl border border-amber-500/35 bg-amber-500/10 px-4 py-2.5 text-sm xl:hidden">
            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-600" />
            <div>
              <p className="font-semibold">Programda {conflicts.length} saat çakışması var.</p>
              <p className="mt-0.5 text-xs text-muted-foreground">
                Turuncu çerçeveli derslerden birini düzenleyebilir veya farklı bir section seçebilirsin.
              </p>
            </div>
          </div>
        )}

        <div className="grid min-h-0 gap-3 xl:h-full xl:grid-cols-[370px_minmax(0,1fr)]">
          <aside className="flex min-h-[620px] flex-col overflow-hidden rounded-2xl border border-slate-800 bg-slate-950 text-slate-100 shadow-xl xl:min-h-0">
            <div className="shrink-0 border-b border-white/10 p-3.5">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <h2 className="text-sm font-bold">Ders ve section seç</h2>
                  <p className="mt-0.5 text-[11px] text-slate-400">202601 · resmî SUIS verisi</p>
                </div>
                <button type="button" onClick={() => setEditor(blankItem())} className="inline-flex items-center rounded-lg border border-white/15 bg-white/5 px-2.5 py-1.5 text-xs font-semibold text-white transition hover:bg-white/10"><Plus className="mr-1.5 h-3.5 w-3.5" /> Özel ders</button>
              </div>
              <label className="relative mt-3 block">
                <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
                <input
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  placeholder="Ders kodu, adı veya CRN ara"
                  aria-label="Ders veya section ara"
                  className="w-full rounded-xl border border-white/10 bg-white py-2.5 pl-9 pr-9 text-sm text-slate-950 outline-none placeholder:text-slate-400 focus:border-cyan-400 focus:ring-2 focus:ring-cyan-400/20"
                />
                {query && (
                  <button
                    type="button"
                    onClick={() => setQuery("")}
                    aria-label="Aramayı temizle"
                    className="absolute right-2.5 top-1/2 -translate-y-1/2 rounded-md p-1 text-slate-500 hover:bg-slate-100"
                  >
                    <X className="h-3.5 w-3.5" />
                  </button>
                )}
              </label>
              <div className="mt-3 flex items-center gap-1.5" aria-label="Güne göre filtrele">
                <span className="mr-auto text-[10px] font-semibold uppercase tracking-[0.12em] text-slate-400">Güne göre</span>
                {WEEK_DAYS.map((day) => {
                  const active = dayFilters.includes(day.code);
                  return (
                    <button key={day.code} type="button" aria-pressed={active} onClick={() => setDayFilters((current) => active ? current.filter((code) => code !== day.code) : [...current, day.code])} className={`rounded-md px-2 py-1 text-[10px] font-bold transition ${active ? "bg-cyan-400 text-slate-950" : "bg-white/10 text-slate-300 hover:bg-white/15"}`}>
                      {day.short.slice(0, 3).toLocaleUpperCase("tr-TR")}
                    </button>
                  );
                })}
              </div>
            </div>

            <div className="min-h-0 flex-1 overflow-y-auto p-2.5 scrollbar-thin">
              {searching ? (
                <div className="flex items-center justify-center gap-2 py-10 text-sm text-slate-400">
                  <Loader2 className="h-4 w-4 animate-spin" /> Dersler yükleniyor
                </div>
              ) : searchError ? (
                <div className="rounded-xl border border-dashed border-white/15 p-5 text-center">
                  <p className="text-sm font-medium">Ders listesine ulaşılamadı.</p>
                  <p className="mt-1 text-xs text-slate-400">Mevcut programın cihazında güvende.</p>
                </div>
              ) : groupedResults.length === 0 ? (
                <div className="rounded-xl border border-dashed border-white/15 p-5 text-center text-sm text-slate-400">
                  Arama veya gün filtresine uygun ders bulunamadı.
                </div>
              ) : (
                <ul className="space-y-2">
                  {groupedResults.map((group) => {
                    const open = expandedCourse === group.courseId;
                    const selectedCount = group.sections.filter((section) => schedule.items.some((item) => item.crn === String(section.crn))).length;
                    return (
                      <li key={group.courseId} className="overflow-hidden rounded-xl border border-white/10 bg-white/[0.04]">
                        <button type="button" onClick={() => setExpandedCourse(open ? null : group.courseId)} className="flex w-full items-center gap-3 bg-white/[0.06] px-3 py-2.5 text-left transition hover:bg-white/10">
                          <span className="min-w-0 flex-1">
                            <span className="block text-sm font-bold text-white">{group.courseId}</span>
                            <span className="mt-0.5 block truncate text-[11px] text-slate-400">{group.title}</span>
                          </span>
                          {selectedCount > 0 && <span className="rounded-full bg-emerald-400/15 px-2 py-0.5 text-[10px] font-bold text-emerald-300">{selectedCount} seçili</span>}
                          {open ? <ChevronDown className="h-4 w-4 text-slate-400" /> : <ChevronRight className="h-4 w-4 text-slate-400" />}
                        </button>
                        {open && (
                          <div className="space-y-2 border-t border-white/10 p-2">
                            {group.sections.map((section) => {
                              const selected = schedule.items.some((item) => item.crn === String(section.crn));
                              return (
                                <button key={`${section.course_id}-${section.crn}-${section.section}`} type="button" onClick={() => addSection(section)} disabled={selected} className={`group w-full rounded-lg border p-2.5 text-left transition ${selected ? "border-emerald-400/30 bg-emerald-400/10" : "border-white/10 bg-slate-900 hover:border-cyan-400/40 hover:bg-slate-800"}`}>
                                  <span className="flex items-start gap-2">
                                    <span className="min-w-0 flex-1">
                                      <span className="flex flex-wrap items-center gap-1.5 text-[10px]"><span className="rounded bg-white/10 px-1.5 py-0.5 font-mono">CRN {section.crn}</span><span className="rounded bg-white/10 px-1.5 py-0.5">Section {section.section}</span><span className="rounded bg-white/10 px-1.5 py-0.5">{section.component_label || section.component || "Ders"}</span></span>
                                      <span className="mt-2 block text-[11px] font-medium leading-relaxed text-slate-200">{catalogSchedule(section)}</span>
                                      {section.instructors && <span className="mt-1 block truncate text-[10px] text-slate-400">{section.instructors}</span>}
                                    </span>
                                    {selected ? <Check className="h-4 w-4 shrink-0 text-emerald-300" /> : <Plus className="h-4 w-4 shrink-0 text-slate-400 group-hover:text-cyan-300" />}
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
              {hasMore && !searching && (
                <p className="px-2 py-3 text-center text-[11px] text-slate-500">
                  Daha net sonuçlar için ders kodu veya CRN yaz.
                </p>
              )}
            </div>
          </aside>

          <section className="flex min-h-0 min-w-0 flex-col overflow-hidden rounded-2xl border border-border bg-card shadow-sm">
            <div className="flex shrink-0 flex-wrap items-center gap-2 border-b border-border px-3 py-2.5 sm:px-4">
              <div className="mr-auto min-w-[180px]">
                <h2 className="text-sm font-bold">Haftalık görünüm</h2>
                <p className="text-[11px] text-muted-foreground">Derse tıkla; saatini veya section'ını düzenle.</p>
              </div>
              <div className="flex items-center gap-1.5">
                <QuickStat label="Ders" value={uniqueCourseCount(schedule.items)} />
                <QuickStat label="CRN" value={crns.length} />
                <QuickStat label="TBA" value={tbaItems.length} />
                <QuickStat label="Çakışma" value={conflicts.length} warning={conflicts.length > 0} />
              </div>
              <div className="flex flex-wrap items-center gap-2 lg:hidden">
                <Button type="button" variant="outline" size="sm" onClick={copyCrns} disabled={crns.length === 0}>
                  {copied ? <Check className="mr-1.5 h-3.5 w-3.5" /> : <Copy className="mr-1.5 h-3.5 w-3.5" />}
                  {copied ? "Kopyalandı" : "CRN'leri kopyala"}
                </Button>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={() => exportScheduleXlsx(schedule, user?.username ?? "student")}
                  disabled={schedule.items.length === 0}
                >
                  <FileSpreadsheet className="mr-1.5 h-3.5 w-3.5" /> Excel
                </Button>
                <Button type="button" variant="ghost" size="sm" onClick={clearSchedule} disabled={schedule.items.length === 0}>
                  <RotateCcw className="mr-1.5 h-3.5 w-3.5" /> Temizle
                </Button>
              </div>
            </div>

            <div className="min-h-0 flex-1 space-y-3 overflow-y-auto p-2.5 scrollbar-thin sm:p-3">
            {schedule.items.length === 0 ? (
              <EmptySchedule onCreate={() => setEditor(blankItem())} />
            ) : (
              <>
                <DesktopWeekGrid items={schedule.items} conflictedIds={conflictedIds} onEdit={setEditor} />
                <MobileWeekList items={schedule.items} conflictedIds={conflictedIds} onEdit={setEditor} />
              </>
            )}

            {tbaItems.length > 0 && (
              <div className="rounded-2xl border border-border bg-card p-4 shadow-sm">
                <div className="mb-3 flex items-center gap-2">
                  <Clock3 className="h-4 w-4 text-primary" />
                  <h3 className="text-sm font-semibold">Saati açıklanmayan dersler</h3>
                  <span className="rounded-full bg-muted px-2 py-0.5 text-[10px] font-semibold text-muted-foreground">TBA</span>
                </div>
                <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
                  {tbaItems.map((item) => (
                    <ScheduleListCard key={item.id} item={item} conflict={conflictedIds.has(item.id)} onEdit={setEditor} />
                  ))}
                </div>
              </div>
            )}

            <div className="rounded-2xl border border-border bg-card shadow-sm">
              <div className="flex items-center justify-between gap-3 border-b border-border px-4 py-3">
                <div>
                  <h3 className="text-sm font-semibold">Seçili section'lar</h3>
                  <p className="text-[11px] text-muted-foreground">Kayıt öncesi CRN ve section kontrolü</p>
                </div>
                <span className="rounded-full bg-primary/10 px-2.5 py-1 text-xs font-semibold text-primary">{schedule.items.length}</span>
              </div>
              <ul className="divide-y divide-border">
                {schedule.items.map((item) => (
                  <li key={item.id} className="flex items-center gap-3 px-4 py-3">
                    <span className={`h-9 w-1 shrink-0 rounded-full ${colourFor(item.courseCode).split(" ")[0]}`} />
                    <div className="min-w-0 flex-1">
                      <p className="truncate text-sm font-semibold">{courseLabel(item)} <span className="font-normal text-muted-foreground">{item.component}</span></p>
                      <p className="mt-0.5 truncate text-xs text-muted-foreground">{item.meetings.length ? item.meetings.map(formatMeeting).join(", ") : "TBA"}</p>
                    </div>
                    {item.crn && <span className="hidden font-mono text-xs text-muted-foreground sm:inline">CRN {item.crn}</span>}
                    <button type="button" onClick={() => setEditor(item)} aria-label={`${item.courseCode} dersini düzenle`} className="rounded-lg p-2 text-muted-foreground hover:bg-muted hover:text-foreground">
                      <Pencil className="h-4 w-4" />
                    </button>
                    <button type="button" onClick={() => removeItem(item.id)} aria-label={`${item.courseCode} dersini kaldır`} className="rounded-lg p-2 text-muted-foreground hover:bg-destructive/10 hover:text-destructive">
                      <Trash2 className="h-4 w-4" />
                    </button>
                  </li>
                ))}
              </ul>
            </div>
            </div>
          </section>
        </div>
      </main>

      {editor && <ScheduleEditor item={editor} onClose={() => setEditor(null)} onSave={saveEdited} />}
    </div>
  );
}

function SyncIndicator({ state, inverse = false }: { state: SyncState; inverse?: boolean }) {
  if (state === "idle") return null;
  const label = state === "saving" ? "Kaydediliyor" : state === "saved" ? "Kaydedildi" : "Bu cihazda kayıtlı";
  return (
    <span className={`hidden items-center gap-1.5 text-[11px] sm:flex ${inverse ? "text-white/70" : "text-muted-foreground"}`}>
      {state === "saving" ? <Loader2 className="h-3 w-3 animate-spin" /> : <Check className="h-3 w-3" />}
      {label}
    </span>
  );
}

function SummaryCard({ label, value, detail, warning = false }: { label: string; value: number; detail: string; warning?: boolean }) {
  return (
    <div className={`min-w-[150px] flex-1 rounded-xl border bg-card px-3 py-2 shadow-sm ${warning ? "border-amber-500/50" : "border-border"}`}>
      <div className="flex items-end justify-between gap-3">
        <div>
          <p className="text-[11px] font-semibold uppercase tracking-[0.12em] text-muted-foreground">{label}</p>
          <p className={`text-lg font-bold tabular-nums ${warning ? "text-amber-600" : "text-foreground"}`}>{value}</p>
        </div>
        <p className="pb-1 text-right text-[11px] text-muted-foreground">{detail}</p>
      </div>
    </div>
  );
}

function QuickStat({ label, value, warning = false }: { label: string; value: number; warning?: boolean }) {
  return (
    <span className={`inline-flex items-center gap-1 rounded-lg border px-2 py-1 text-[10px] font-semibold ${warning ? "border-amber-500/40 bg-amber-500/10 text-amber-700 dark:text-amber-300" : "border-border bg-muted/50 text-muted-foreground"}`}>
      {label}<strong className="text-xs tabular-nums text-foreground">{value}</strong>
    </span>
  );
}

function EmptySchedule({ onCreate }: { onCreate: () => void }) {
  return (
    <div className="grid min-h-[420px] place-items-center rounded-2xl border border-dashed border-border bg-card/60 p-8 text-center">
      <div className="max-w-sm">
        <span className="mx-auto grid h-14 w-14 place-items-center rounded-2xl bg-primary/10 text-primary">
          <CalendarDays className="h-7 w-7" />
        </span>
        <h2 className="mt-4 text-lg font-bold">Programını oluşturmaya başla</h2>
        <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
          Soldan bir ders ve section seç veya chatbot'ta “Bu dönem hangi dersleri alayım?” seçeneğini kullan. Hazırlanan program buraya otomatik gelir.
        </p>
        <Button type="button" className="mt-5" onClick={onCreate}><Plus className="mr-2 h-4 w-4" /> Manuel ders ekle</Button>
      </div>
    </div>
  );
}

function DesktopWeekGrid({ items, conflictedIds, onEdit }: { items: ScheduleItem[]; conflictedIds: Set<string>; onEdit: (item: ScheduleItem) => void }) {
  const times = Array.from({ length: 12 }, (_, index) => GRID_START + index * 60);
  const slots = times.slice(0, -1);
  return (
    <div className="hidden overflow-x-auto rounded-xl border border-border bg-card shadow-sm scrollbar-thin md:block">
      <div className="min-w-[880px]">
        <div className="grid grid-cols-[76px_repeat(5,minmax(150px,1fr))] border-b border-white/15 bg-primary text-primary-foreground">
          <div className="border-r border-white/15 p-3 text-center text-[10px] font-semibold uppercase tracking-widest text-white/70">Saat</div>
          {WEEK_DAYS.map((day) => <div key={day.code} className="border-r border-white/15 p-3 text-center text-xs font-bold last:border-r-0">{day.label}</div>)}
        </div>
        <div className="grid grid-cols-[76px_repeat(5,minmax(150px,1fr))]">
          <div className="relative border-r border-border bg-card" style={{ height: GRID_END - GRID_START }}>
            {slots.map((time, index) => <div key={`slot-${time}`} className={`absolute inset-x-0 border-t border-border/70 ${index % 2 === 0 ? "bg-muted/45" : "bg-card"}`} style={{ top: time - GRID_START, height: 60 }} />)}
            {times.map((time) => (
              <span key={time} className="absolute right-3 z-10 -translate-y-1/2 font-mono text-[10px] font-semibold text-foreground/75" style={{ top: time - GRID_START }}>
                {String(Math.floor(time / 60)).padStart(2, "0")}:{String(time % 60).padStart(2, "0")}
              </span>
            ))}
          </div>
          {WEEK_DAYS.map((day) => (
            <div key={day.code} className="relative border-r border-border bg-card last:border-r-0" style={{ height: GRID_END - GRID_START }}>
              {slots.map((time, index) => <div key={time} className={`absolute inset-x-0 border-t border-border/70 ${index % 2 === 0 ? "bg-muted/45" : "bg-card"}`} style={{ top: time - GRID_START, height: 60 }} />)}
              {items.flatMap((item) => item.meetings.filter((meeting) => meeting.day === day.code).map((meeting, index) => {
                const start = Math.max(minutes(meeting.start), GRID_START);
                const end = Math.min(minutes(meeting.end), GRID_END);
                if (end <= start) return null;
                return (
                  <button
                    type="button"
                    key={`${item.id}-${day.code}-${index}`}
                    onClick={() => onEdit(item)}
                    className={`absolute inset-x-1 z-10 overflow-hidden rounded-lg border px-2 py-1.5 text-left shadow-sm transition-transform hover:z-20 hover:scale-[1.015] ${colourFor(item.courseCode)} ${conflictedIds.has(item.id) ? "ring-2 ring-amber-400 ring-offset-1 ring-offset-card" : ""}`}
                    style={{ top: (start - GRID_START) * PIXELS_PER_MINUTE, height: Math.max((end - start) * PIXELS_PER_MINUTE, 34) }}
                    title={`${item.courseCode} · ${formatMeeting(meeting)}`}
                  >
                    <span className="block truncate text-[11px] font-bold leading-tight">{item.courseCode}</span>
                    <span className="mt-0.5 block truncate text-[9px] font-medium opacity-90">{meeting.start}-{meeting.end} · {item.section || item.component}</span>
                    {(end - start) >= 55 && item.location && <span className="mt-1 block truncate text-[9px] opacity-80">{item.location}</span>}
                  </button>
                );
              }))}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function MobileWeekList({ items, conflictedIds, onEdit }: { items: ScheduleItem[]; conflictedIds: Set<string>; onEdit: (item: ScheduleItem) => void }) {
  return (
    <div className="space-y-3 md:hidden">
      {WEEK_DAYS.map((day) => {
        const dayItems = items.flatMap((item) => item.meetings.filter((meeting) => meeting.day === day.code).map((meeting) => ({ item, meeting }))).sort((a, b) => a.meeting.start.localeCompare(b.meeting.start));
        return (
          <section key={day.code} className="rounded-2xl border border-border bg-card p-3 shadow-sm">
            <h3 className="mb-2 px-1 text-xs font-bold uppercase tracking-wider text-muted-foreground">{day.label}</h3>
            {dayItems.length === 0 ? <p className="rounded-xl bg-muted/50 px-3 py-4 text-center text-xs text-muted-foreground">Ders yok</p> : (
              <div className="space-y-2">{dayItems.map(({ item, meeting }) => (
                <button key={`${item.id}-${meeting.start}`} type="button" onClick={() => onEdit(item)} className={`flex w-full items-center gap-3 rounded-xl border p-3 text-left ${conflictedIds.has(item.id) ? "border-amber-500/50 bg-amber-500/5" : "border-border bg-background"}`}>
                  <span className={`h-10 w-1 shrink-0 rounded-full ${colourFor(item.courseCode).split(" ")[0]}`} />
                  <span className="w-24 shrink-0 font-mono text-xs font-semibold">{meeting.start}-{meeting.end}</span>
                  <span className="min-w-0 flex-1"><span className="block truncate text-sm font-bold">{item.courseCode}</span><span className="block truncate text-[11px] text-muted-foreground">{item.location || item.component}</span></span>
                  <ChevronRight className="h-4 w-4 text-muted-foreground" />
                </button>
              ))}</div>
            )}
          </section>
        );
      })}
    </div>
  );
}

function ScheduleListCard({ item, conflict, onEdit }: { item: ScheduleItem; conflict: boolean; onEdit: (item: ScheduleItem) => void }) {
  return (
    <button type="button" onClick={() => onEdit(item)} className={`flex items-center gap-3 rounded-xl border bg-background p-3 text-left ${conflict ? "border-amber-500/50" : "border-border"}`}>
      <span className={`h-10 w-1 shrink-0 rounded-full ${colourFor(item.courseCode).split(" ")[0]}`} />
      <span className="min-w-0 flex-1"><span className="block truncate text-sm font-bold">{item.courseCode}</span><span className="block truncate text-[11px] text-muted-foreground">{item.title || item.component}</span></span>
      <Pencil className="h-3.5 w-3.5 text-muted-foreground" />
    </button>
  );
}

function ScheduleEditor({ item, onClose, onSave }: { item: ScheduleItem; onClose: () => void; onSave: (item: ScheduleItem) => void }) {
  const [draft, setDraft] = useState<ScheduleItem>(() => ({ ...item, meetings: item.meetings.map((meeting) => ({ ...meeting })) }));

  function field(key: keyof Omit<ScheduleItem, "meetings">, value: string) {
    setDraft((current) => ({ ...current, [key]: key === "courseCode" ? value.toUpperCase() : value }));
  }

  function updateMeeting(index: number, key: keyof ScheduleMeeting, value: string) {
    setDraft((current) => ({
      ...current,
      meetings: current.meetings.map((meeting, meetingIndex) => meetingIndex === index ? { ...meeting, [key]: value } : meeting),
    }));
  }

  function submit(event: React.FormEvent) {
    event.preventDefault();
    if (!draft.courseCode.trim()) {
      toast.error("Ders kodunu yazmalısın.");
      return;
    }
    if (draft.meetings.some((meeting) => !meeting.start || !meeting.end || meeting.start >= meeting.end)) {
      toast.error("Bitiş saati başlangıç saatinden sonra olmalı.");
      return;
    }
    onSave({ ...draft, courseCode: draft.courseCode.trim().toUpperCase(), title: draft.title.trim() });
  }

  return (
    <div className="fixed inset-0 z-50 flex justify-end bg-slate-950/50 backdrop-blur-sm" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }}>
      <div role="dialog" aria-modal="true" aria-labelledby="schedule-editor-title" className="h-full w-full max-w-xl overflow-y-auto border-l border-border bg-card shadow-2xl scrollbar-thin">
        <form onSubmit={submit}>
          <div className="sticky top-0 z-10 flex items-center justify-between border-b border-border bg-card/95 px-5 py-4 backdrop-blur-xl">
            <div><h2 id="schedule-editor-title" className="font-bold">Section düzenle</h2><p className="text-xs text-muted-foreground">Ders, saat ve konum bilgilerini güncelle</p></div>
            <button type="button" onClick={onClose} aria-label="Düzenleyiciyi kapat" className="rounded-lg p-2 text-muted-foreground hover:bg-muted"><X className="h-5 w-5" /></button>
          </div>
          <div className="space-y-6 p-5">
            <div className="grid gap-4 sm:grid-cols-2">
              <EditorField label="Ders kodu" value={draft.courseCode} onChange={(value) => field("courseCode", value)} placeholder="CS 204" required />
              <EditorField label="Ders adı" value={draft.title} onChange={(value) => field("title", value)} placeholder="Advanced Programming" />
              <EditorField label="CRN" value={draft.crn} onChange={(value) => field("crn", value)} placeholder="10218" inputMode="numeric" />
              <EditorField label="Section" value={draft.section} onChange={(value) => field("section", value)} placeholder="A1" />
              <EditorField label="Tür" value={draft.component} onChange={(value) => field("component", value)} placeholder="Ders / Laboratuvar" />
              <EditorField label="Öğretim üyesi" value={draft.instructor} onChange={(value) => field("instructor", value)} placeholder="Ad Soyad" icon={<UserRound className="h-3.5 w-3.5" />} />
              <div className="sm:col-span-2"><EditorField label="Yer" value={draft.location} onChange={(value) => field("location", value)} placeholder="FENS G077" icon={<MapPin className="h-3.5 w-3.5" />} /></div>
            </div>

            <section>
              <div className="flex items-center justify-between gap-3"><div><h3 className="text-sm font-semibold">Gün ve saatler</h3><p className="text-[11px] text-muted-foreground">Boş bırakırsan ders TBA olarak görünür.</p></div><Button type="button" variant="outline" size="sm" onClick={() => setDraft((current) => ({ ...current, meetings: [...current.meetings, { day: "M", start: "08:40", end: "09:30" }] }))}><Plus className="mr-1.5 h-3.5 w-3.5" /> Zaman ekle</Button></div>
              <div className="mt-3 space-y-2">
                {draft.meetings.length === 0 ? <div className="rounded-xl border border-dashed border-border bg-muted/30 p-5 text-center text-xs text-muted-foreground">Saat açıklanmadı (TBA)</div> : draft.meetings.map((meeting, index) => (
                  <div key={`${index}-${meeting.day}`} className="grid grid-cols-[1fr_1fr_1fr_auto] items-end gap-2 rounded-xl border border-border bg-background p-3">
                    <label className="text-[11px] font-medium text-muted-foreground">Gün<select value={meeting.day} onChange={(event) => updateMeeting(index, "day", event.target.value as ScheduleDay)} className="mt-1 block w-full rounded-lg border border-input bg-card px-2 py-2 text-sm text-foreground">{WEEK_DAYS.map((day) => <option key={day.code} value={day.code}>{day.short}</option>)}</select></label>
                    <EditorTime label="Başlangıç" value={meeting.start} onChange={(value) => updateMeeting(index, "start", value)} />
                    <EditorTime label="Bitiş" value={meeting.end} onChange={(value) => updateMeeting(index, "end", value)} />
                    <button type="button" onClick={() => setDraft((current) => ({ ...current, meetings: current.meetings.filter((_, meetingIndex) => meetingIndex !== index) }))} aria-label="Zamanı kaldır" className="rounded-lg p-2.5 text-muted-foreground hover:bg-destructive/10 hover:text-destructive"><Trash2 className="h-4 w-4" /></button>
                  </div>
                ))}
              </div>
            </section>
          </div>
          <div className="sticky bottom-0 flex justify-end gap-2 border-t border-border bg-card/95 px-5 py-4 backdrop-blur-xl"><Button type="button" variant="outline" onClick={onClose}>Vazgeç</Button><Button type="submit"><Check className="mr-2 h-4 w-4" /> Programı güncelle</Button></div>
        </form>
      </div>
    </div>
  );
}

function EditorField({ label, value, onChange, placeholder, required = false, inputMode, icon }: { label: string; value: string; onChange: (value: string) => void; placeholder?: string; required?: boolean; inputMode?: React.HTMLAttributes<HTMLInputElement>["inputMode"]; icon?: React.ReactNode }) {
  return <label className="text-xs font-medium text-muted-foreground">{label}{required && <span className="text-destructive"> *</span>}<span className="relative mt-1.5 block">{icon && <span className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2">{icon}</span>}<input value={value} onChange={(event) => onChange(event.target.value)} placeholder={placeholder} required={required} inputMode={inputMode} className={`w-full rounded-xl border border-input bg-background px-3 py-2.5 text-sm text-foreground placeholder:text-muted-foreground ${icon ? "pl-9" : ""}`} /></span></label>;
}

function EditorTime({ label, value, onChange }: { label: string; value: string; onChange: (value: string) => void }) {
  return <label className="text-[11px] font-medium text-muted-foreground">{label}<input type="time" value={value} onChange={(event) => onChange(event.target.value)} className="mt-1 block w-full rounded-lg border border-input bg-card px-2 py-2 text-sm text-foreground" /></label>;
}
