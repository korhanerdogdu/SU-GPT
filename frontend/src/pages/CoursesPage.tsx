import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { ArrowLeft, BookOpen, Check, FileSpreadsheet, FileUp, GraduationCap, Inbox, ListChecks, Loader2, PlusCircle, Save, Search, Trash2, Upload, X } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import ConfirmDialog from "@/components/ui/confirm-dialog";
import { useAuth } from "@/contexts/AuthContext";
import ThemeToggle from "@/components/ThemeToggle";
import LanguageToggle from "@/components/LanguageToggle";
import HelpButton from "@/components/HelpButton";
import { useLocale } from "@/contexts/LocaleContext";
import { exportCoursesXlsx } from "@/lib/export-xlsx";
import {
  ALL_GRADES,
  fetchCourses,
  fetchGpa,
  fetchUserCourses,
  saveUserCourses,
  SUPPORTED_EXTENSIONS,
  uploadDocuments,
  uploadTranscript,
  type Course,
} from "@/lib/api";

function formatSize(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/** Colour-codes a saved course's status so it reads at a glance, not just by label text. */
function statusBadgeCls(status: string | undefined): string {
  switch (status) {
    case "completed":
      return "bg-success/15 text-success";
    case "transfer":
      return "bg-primary/10 text-primary-emphasis";
    case "exempted":
      return "bg-warning/15 text-warning";
    default:
      return "bg-muted text-muted-foreground";
  }
}

/**
 * Course history lives on its own page rather than buried in the chat sidebar: saving a course
 * history is setup, and — more importantly — after saving you need to SEE what the system now
 * believes you have completed.
 *
 * Styled to match ProfilePage: both are "set up your record" screens, so they share the same
 * primary/card/border tokens as the rest of the app and follow the active theme.
 */
export default function CoursesPage() {
  const { user } = useAuth();
  const { t } = useLocale();
  const [search, setSearch] = useState("");
  const [pool, setPool] = useState<Course[]>([]);
  const [saved, setSaved] = useState<Course[]>([]);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [loadingPool, setLoadingPool] = useState(false);
  const [saving, setSaving] = useState(false);

  const [files, setFiles] = useState<File[]>([]);
  const [uploading, setUploading] = useState(false);

  const [gpa, setGpa] = useState<number | null>(null);
  const [transcriptBusy, setTranscriptBusy] = useState(false);
  const [confirmClearAll, setConfirmClearAll] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setLoadingPool(true);
    const timer = window.setTimeout(async () => {
      try {
        const data = await fetchCourses(search);
        if (!cancelled) setPool(data);
      } catch {
        if (!cancelled) toast.error(t("courses.catalogFailed"));
      } finally {
        if (!cancelled) setLoadingPool(false);
      }
    }, 180);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [search]);

  useEffect(() => {
    if (!user?.username) return;
    let cancelled = false;
    (async () => {
      try {
        const courses = await fetchUserCourses(user.username);
        if (cancelled) return;
        setSaved(courses);
        setSelectedIds(new Set(courses.map((c) => c.id)));
      } catch {
        if (!cancelled) toast.error(t("courses.historyFailed"));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [user?.username]);

  async function refreshGpa() {
    if (!user?.username) return;
    try {
      const result = await fetchGpa(user.username);
      setGpa(result.gpa);
    } catch {
      // GPA is informational; a failed read must not block the rest of the page.
    }
  }

  useEffect(() => {
    void refreshGpa();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [user?.username]);

  const totalSu = useMemo(
    () =>
      saved
        .filter((course) =>
          ["completed", "transfer", "exempted"].includes(course.status ?? "completed"),
        )
        .reduce((sum, course) => sum + (course.su_credits ?? 0), 0),
    [saved]
  );
  const dirty = useMemo(() => {
    const savedIds = new Set(saved.map((c) => c.id));
    if (savedIds.size !== selectedIds.size) return true;
    for (const id of selectedIds) if (!savedIds.has(id)) return true;
    return false;
  }, [saved, selectedIds]);

  function toggle(id: string) {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  function currentStatuses(courseList: Course[]) {
    return Object.fromEntries(
      courseList.filter((course) => course.status).map((course) => [course.id, course.status]),
    );
  }

  function currentGrades(courseList: Course[]) {
    return Object.fromEntries(
      courseList.filter((course) => course.grade).map((course) => [course.id, course.grade as string]),
    );
  }

  async function save() {
    if (!user?.username) return;
    setSaving(true);
    try {
      const result = await saveUserCourses(
        user.username,
        Array.from(selectedIds),
        currentStatuses(saved),
        currentGrades(saved),
      );
      setSaved(result);
      setSelectedIds(new Set(result.map((c) => c.id)));
      toast.success(t("courses.savedToast", { count: result.length }));
      void refreshGpa();
    } catch {
      toast.error(t("courses.saveFailed"));
    } finally {
      setSaving(false);
    }
  }

  /** Grade edits apply immediately (like a form field, not a batch save) — the course is already
   *  saved; only its grade is changing. */
  async function handleGradeChange(courseId: string, grade: string) {
    if (!user?.username) return;
    const next = saved.map((c) => (c.id === courseId ? { ...c, grade: grade || null } : c));
    setSaved(next);
    try {
      const result = await saveUserCourses(
        user.username,
        next.map((c) => c.id),
        currentStatuses(next),
        currentGrades(next),
      );
      setSaved(result);
      void refreshGpa();
    } catch {
      toast.error(t("courses.saveFailed"));
    }
  }

  /** Removing a course (single or all) is a persisted edit like a grade change, not a batch
   *  "select then Save" action -- the course is already saved, so the removal takes effect
   *  immediately. */
  async function removeCourse(courseId: string) {
    if (!user?.username) return;
    const next = saved.filter((c) => c.id !== courseId);
    try {
      const result = await saveUserCourses(
        user.username,
        next.map((c) => c.id),
        currentStatuses(next),
        currentGrades(next),
      );
      setSaved(result);
      setSelectedIds(new Set(result.map((c) => c.id)));
      void refreshGpa();
    } catch {
      toast.error(t("courses.saveFailed"));
    }
  }

  async function clearAllCourses() {
    if (!user?.username) return;
    setConfirmClearAll(false);
    try {
      const result = await saveUserCourses(user.username, [], {}, {});
      setSaved(result);
      setSelectedIds(new Set());
      void refreshGpa();
    } catch {
      toast.error(t("courses.saveFailed"));
    }
  }

  async function handleTranscriptFile(file: File) {
    if (!user?.username) return;
    setTranscriptBusy(true);
    try {
      const result = await uploadTranscript(user.username, file);
      setSaved(result.courses);
      setSelectedIds(new Set(result.courses.map((c) => c.id)));
      toast.success(
        t("courses.transcriptImported", {
          count: result.matched_course_codes.length,
          gpa: result.summary.gpa ?? "—",
        }),
      );
      if (result.unmatched_course_codes.length > 0) {
        toast.warning(t("courses.transcriptUnmatched", { codes: result.unmatched_course_codes.join(", ") }));
      }
      void refreshGpa();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : t("courses.transcriptFailed"));
    } finally {
      setTranscriptBusy(false);
    }
  }

  function onPickTranscript(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (file) void handleTranscriptFile(file);
  }

  function onPickFiles(e: React.ChangeEvent<HTMLInputElement>) {
    const picked = e.target.files ? Array.from(e.target.files) : [];
    if (picked.length === 0) return;
    setFiles((prev) => {
      const seen = new Set(prev.map((f) => f.name + f.size));
      return [...prev, ...picked.filter((p) => !seen.has(p.name + p.size))];
    });
    e.target.value = "";
  }

  async function handleUpload() {
    if (files.length === 0) return;
    setUploading(true);
    try {
      const result = await uploadDocuments(files);
      toast.success(
        t("courses.uploaded", { files: result.accepted_files.length, chunks: result.chunks })
      );
      if (result.skipped_files.length > 0) {
        toast.warning(t("courses.skipped", { files: result.skipped_files.join(", ") }));
      }
      setFiles([]);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : t("courses.uploadFailed"));
    } finally {
      setUploading(false);
    }
  }

  return (
    <div className="flex min-h-dvh flex-col overflow-x-hidden bg-background text-foreground lg:h-dvh lg:overflow-hidden">
      <header className="flex shrink-0 items-center gap-3 border-b border-white/10 bg-sabanci-header px-6 py-4 text-primary-foreground shadow-md shadow-black/20">
        <Link to="/" className="flex items-center gap-1 text-sm opacity-90 hover:opacity-100">
          <ArrowLeft size={16} /> {t("common.chat")}
        </Link>
        <div className="ml-2 flex items-center gap-2">
          <BookOpen size={18} />
          <span className="font-semibold">{t("courses.title")}</span>
        </div>
        <HelpButton className="ml-auto" />
        <LanguageToggle />
        <ThemeToggle />
      </header>

      <main className="relative min-h-0 flex-1 overflow-y-auto lg:overflow-hidden">
        {/* Ambient brand wash — purely decorative, keeps the page from reading as two bare
            cards on an empty canvas at wide desktop scale. */}
        <div aria-hidden="true" className="pointer-events-none absolute inset-0 -z-10">
          <div className="absolute -top-32 right-0 h-80 w-80 rounded-full bg-[radial-gradient(circle,_hsl(var(--primary)/0.10)_0%,_transparent_70%)] blur-2xl" />
          <div className="absolute -bottom-24 left-0 h-72 w-72 rounded-full bg-[radial-gradient(circle,_rgba(214,161,58,0.09)_0%,_transparent_70%)] blur-2xl" />
        </div>

        <div className="mx-auto flex h-full max-w-5xl flex-col px-6 pb-6 pt-6 lg:pb-4">
          <div className="mb-6 max-w-xl shrink-0">
            <p className="font-ledger text-xs uppercase tracking-[0.18em] text-muted-foreground">
              {t("courses.eyebrow")}
            </p>
            <h2 className="mt-2 text-xl font-bold tracking-tight text-foreground sm:text-[1.375rem]">
              {t("courses.heroTitle")}
            </h2>
            <p className="mt-1.5 text-sm leading-relaxed text-muted-foreground sm:whitespace-nowrap">
              {t("courses.heroBody")}
            </p>
          </div>

          <div className="grid gap-6 md:grid-cols-2 lg:min-h-0 lg:flex-1">
        {/* Completed courses — the answer to "what did I select?" */}
        <section className="flex flex-col rounded-2xl border border-border bg-card p-5 shadow-sm transition-shadow hover:shadow-md lg:min-h-0">
          <div className="mb-4 flex shrink-0 flex-wrap items-center justify-between gap-3">
            <h2 className="flex items-center gap-2.5 text-lg font-semibold text-foreground">
              <span className="grid h-8 w-8 shrink-0 place-items-center rounded-lg bg-primary/10 text-primary-emphasis">
                <ListChecks className="h-4 w-4" />
              </span>
              {t("courses.completed")}
            </h2>
            <div className="flex items-center gap-3">
              <span className="text-sm text-muted-foreground">
                <span className="font-semibold text-primary-emphasis">{t("courses.savedCount", { count: saved.length })}</span> ·{" "}
                <span className="font-semibold text-primary-emphasis">{totalSu}</span> SU
                {gpa != null && (
                  <>
                    {" "}· <span className="font-semibold text-primary-emphasis">{t("courses.gpa")} {gpa.toFixed(2)}</span>
                  </>
                )}
              </span>
              {saved.length > 0 && (
                <>
                  <Button
                    onClick={() => exportCoursesXlsx(saved, user?.username ?? "student")}
                    variant="outline"
                    size="sm"
                    className="border-success text-success hover:bg-success/10"
                  >
                    <FileSpreadsheet className="mr-1.5" size={15} /> Excel
                  </Button>
                  <Button
                    onClick={() => setConfirmClearAll(true)}
                    variant="outline"
                    size="sm"
                    className="border-destructive-emphasis text-destructive-emphasis hover:bg-destructive/10"
                  >
                    <Trash2 className="mr-1.5" size={15} /> {t("courses.clearAll")}
                  </Button>
                </>
              )}
            </div>
          </div>

          {saved.length === 0 ? (
            <div className="flex shrink-0 flex-col items-center gap-2.5 rounded-xl border border-dashed border-border bg-muted/30 px-5 py-8 text-center">
              <span className="grid h-11 w-11 place-items-center rounded-xl bg-primary/10 text-primary-emphasis">
                <Inbox className="h-5 w-5" />
              </span>
              <p className="text-sm font-semibold text-foreground">{t("courses.emptyTitle")}</p>
              <p className="max-w-xs text-xs leading-relaxed text-muted-foreground">
                {t("courses.empty")}
              </p>
            </div>
          ) : (
            <ul className="min-h-0 flex-1 space-y-2 overflow-y-auto pr-1 scrollbar-thin lg:max-h-none">
              {saved.map((c) => (
                <li
                  key={c.id}
                  className="flex items-start gap-3 rounded-xl border border-border bg-muted/40 px-3 py-2.5 transition-colors hover:border-primary/25 hover:bg-muted/60"
                >
                  <Check className="mt-0.5 h-4 w-4 shrink-0 text-success" />
                  <div className="min-w-0 flex-1">
                    <p className="text-sm font-semibold text-foreground">{c.code}</p>
                    <p className="text-xs text-muted-foreground">{c.title}</p>
                    {/* Only shown for a non-default status: the checkmark plus this list's own
                        "Completed courses" heading already say "completed" for every other row,
                        so repeating it as a badge on each one was pure visual noise. */}
                    {c.status && c.status !== "completed" && (
                      <span className={`mt-1.5 inline-flex rounded-full px-2 py-0.5 text-[11px] font-bold uppercase tracking-wide ${statusBadgeCls(c.status)}`}>
                        {c.status}
                      </span>
                    )}
                  </div>
                  <div className="flex shrink-0 flex-col items-end gap-1">
                    <span className="text-xs font-medium text-muted-foreground">
                      {c.su_credits ?? "—"} SU
                    </span>
                    <select
                      value={c.grade ?? ""}
                      onChange={(e) => void handleGradeChange(c.id, e.target.value)}
                      aria-label={t("courses.grade")}
                      title={t("courses.grade")}
                      className="w-11 rounded-full border border-border bg-background py-0.5 text-center text-[11px] font-bold text-muted-foreground outline-none focus:border-primary focus:ring-1 focus:ring-primary/40"
                    >
                      <option value="">{t("courses.gradeNone")}</option>
                      {ALL_GRADES.map((g) => (
                        <option key={g} value={g}>{g}</option>
                      ))}
                    </select>
                  </div>
                  <button
                    type="button"
                    onClick={() => void removeCourse(c.id)}
                    aria-label={t("courses.removeCourse")}
                    title={t("courses.removeCourse")}
                    className="grid h-6 w-6 shrink-0 place-items-center self-start rounded-full text-muted-foreground transition-colors hover:bg-destructive/10 hover:text-destructive-emphasis"
                  >
                    <X className="h-3.5 w-3.5" />
                  </button>
                </li>
              ))}
            </ul>
          )}

          <p className="mt-4 shrink-0 border-t border-border pt-3 text-xs text-muted-foreground">
            {t("courses.countNote")}
          </p>
        </section>

        {/* Picker + upload — its own scroll container so the outer page never scrolls */}
        <div className="min-h-0 space-y-6 lg:overflow-y-auto lg:pr-1">
          <section className="rounded-2xl border border-border bg-card p-4 shadow-sm transition-shadow hover:shadow-md">
            <h2 className="mb-3 flex items-center gap-2.5 text-sm font-semibold text-foreground">
              <span className="grid h-7 w-7 shrink-0 place-items-center rounded-lg bg-primary/10 text-primary-emphasis">
                <GraduationCap className="h-3.5 w-3.5" />
              </span>
              {t("courses.transcriptTitle")}
            </h2>
            <label
              className={`flex cursor-pointer items-center justify-center gap-2.5 rounded-xl border border-dashed border-border bg-muted/40 px-3 py-3 text-center text-sm text-muted-foreground transition-colors hover:border-primary hover:bg-primary/5 ${transcriptBusy ? "pointer-events-none opacity-60" : ""}`}
            >
              {transcriptBusy ? (
                <>
                  <Loader2 className="h-4 w-4 shrink-0 animate-spin text-primary-emphasis" />
                  <span className="font-medium text-foreground">{t("courses.transcriptParsing")}</span>
                </>
              ) : (
                <>
                  <Upload className="h-4 w-4 shrink-0 text-primary-emphasis" />
                  <span className="font-medium text-foreground">{t("courses.transcriptBrowse")}</span>
                </>
              )}
              <input
                type="file"
                accept="application/pdf,.pdf"
                className="hidden"
                disabled={transcriptBusy}
                onChange={onPickTranscript}
              />
            </label>
          </section>

          <section className="rounded-2xl border border-border bg-card p-5 shadow-sm transition-shadow hover:shadow-md">
            <h2 className="mb-4 flex items-center gap-2.5 text-lg font-semibold text-foreground">
              <span className="grid h-8 w-8 shrink-0 place-items-center rounded-lg bg-primary/10 text-primary-emphasis">
                <PlusCircle className="h-4 w-4" />
              </span>
              {t("courses.add")}
            </h2>

            <div className="relative">
              <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
              <input
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder={t("courses.search")}
                className="w-full rounded-xl border border-input bg-background py-2.5 pl-9 pr-3 text-sm text-foreground placeholder:text-muted-foreground outline-none focus:border-primary focus:ring-2 focus:ring-primary/15"
              />
            </div>

            <div className="mt-3 max-h-[15rem] space-y-1.5 overflow-y-auto pr-1 scrollbar-thin">
              {loadingPool ? (
                <div className="flex items-center gap-2 rounded-xl border border-border bg-muted/40 px-3 py-2.5 text-sm text-muted-foreground">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  {t("courses.loading")}
                </div>
              ) : pool.length === 0 ? (
                <div className="rounded-xl border border-border bg-muted/40 px-3 py-2.5 text-sm text-muted-foreground">
                  {t("courses.none")}
                </div>
              ) : (
                pool.slice(0, 120).map((course) => {
                  const checked = selectedIds.has(course.id);
                  return (
                    <label
                      key={course.id}
                      className={`flex cursor-pointer items-start gap-3 rounded-xl border px-3 py-2.5 transition-colors ${
                        checked
                          ? "border-primary/50 bg-primary/10"
                          : "border-border bg-card hover:border-primary/30 hover:bg-muted/40"
                      }`}
                    >
                      <input
                        type="checkbox"
                        checked={checked}
                        onChange={() => toggle(course.id)}
                        className="mt-1 h-3.5 w-3.5 accent-primary"
                      />
                      <span className="min-w-0 flex-1">
                        <span className="block text-sm font-semibold text-foreground">
                          {course.code}
                        </span>
                        <span className="line-clamp-2 text-xs text-muted-foreground">{course.title}</span>
                      </span>
                      <span className="shrink-0 text-xs font-medium text-muted-foreground">
                        {course.su_credits ?? "—"} SU
                      </span>
                    </label>
                  );
                })
              )}
            </div>

            <Button
              onClick={save}
              disabled={saving || !dirty}
              className="mt-4 w-full"
            >
              {saving ? (
                <Loader2 className="mr-2 animate-spin" size={16} />
              ) : (
                <Save className="mr-2" size={16} />
              )}
              {dirty ? t("courses.saveCount", { count: selectedIds.size }) : t("common.saved")}
            </Button>
          </section>

          {user?.role === "admin" && (
          <section className="rounded-2xl border border-border bg-card p-5 shadow-sm transition-shadow hover:shadow-md">
            <h2 className="mb-4 flex items-center gap-2.5 text-lg font-semibold text-foreground">
              <span className="grid h-8 w-8 shrink-0 place-items-center rounded-lg bg-primary/10 text-primary-emphasis">
                <FileUp className="h-4 w-4" />
              </span>
              {t("courses.documents")}
            </h2>
            <label className="flex cursor-pointer flex-col items-center justify-center gap-2 rounded-xl border border-dashed border-border bg-muted/40 px-3 py-6 text-center text-sm text-muted-foreground transition-colors hover:border-primary hover:bg-primary/5">
              <span className="grid h-10 w-10 place-items-center rounded-xl bg-primary/10 text-primary-emphasis">
                <Upload className="h-4 w-4" />
              </span>
              <span>
                <span className="font-medium text-foreground">{t("courses.browse")}</span>
                <br />
                <span className="text-xs">PDF, PPTX, DOCX, MD, TXT</span>
              </span>
              <input
                type="file"
                multiple
                accept={SUPPORTED_EXTENSIONS.join(",")}
                className="hidden"
                onChange={onPickFiles}
              />
            </label>

            {files.length > 0 && (
              <>
                <ul className="mt-3 space-y-2">
                  {files.map((f, i) => (
                    <li
                      key={f.name + i}
                      className="flex items-center gap-2.5 rounded-xl border border-border bg-muted/40 px-3 py-2.5"
                    >
                      <div className="min-w-0 flex-1">
                        <p className="truncate text-sm text-foreground">{f.name}</p>
                        <p className="text-xs text-muted-foreground">{formatSize(f.size)}</p>
                      </div>
                      <button
                        type="button"
                        onClick={() => setFiles((prev) => prev.filter((_, idx) => idx !== i))}
                        aria-label={t("courses.removeFile")}
                        className="grid h-6 w-6 shrink-0 place-items-center rounded-full text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
                      >
                        <X className="h-4 w-4" />
                      </button>
                    </li>
                  ))}
                </ul>
                <Button
                  onClick={handleUpload}
                  disabled={uploading}
                  variant="outline"
                  className="mt-3 w-full border-primary-emphasis text-primary-emphasis"
                >
                  {uploading ? <Loader2 className="mr-2 animate-spin" size={16} /> : null}
                  {uploading ? t("courses.indexing") : t("courses.upload")}
                </Button>
              </>
            )}
          </section>
          )}
        </div>
          </div>
        </div>
      </main>

      <ConfirmDialog
        open={confirmClearAll}
        title={t("courses.clearAllConfirmTitle")}
        description={t("courses.clearAllConfirmBody", { count: saved.length })}
        confirmLabel={t("courses.clearAllConfirmAction")}
        destructive
        onConfirm={() => void clearAllCourses()}
        onCancel={() => setConfirmClearAll(false)}
      />
    </div>
  );
}
