import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { ArrowLeft, BookOpen, Check, FileSpreadsheet, Loader2, Save, Search, Upload, X } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { useAuth } from "@/contexts/AuthContext";
import ThemeToggle from "@/components/ThemeToggle";
import LanguageToggle from "@/components/LanguageToggle";
import HelpButton from "@/components/HelpButton";
import { useLocale } from "@/contexts/LocaleContext";
import { exportCoursesXlsx } from "@/lib/export-xlsx";
import {
  fetchCourses,
  fetchUserCourses,
  saveUserCourses,
  SUPPORTED_EXTENSIONS,
  uploadDocuments,
  type Course,
} from "@/lib/api";

function formatSize(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
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

  async function save() {
    if (!user?.username) return;
    setSaving(true);
    try {
      const statuses = Object.fromEntries(
        saved
          .filter((course) => course.status)
          .map((course) => [course.id, course.status]),
      );
      const result = await saveUserCourses(
        user.username,
        Array.from(selectedIds),
        statuses,
      );
      setSaved(result);
      setSelectedIds(new Set(result.map((c) => c.id)));
      toast.success(t("courses.savedToast", { count: result.length }));
    } catch {
      toast.error(t("courses.saveFailed"));
    } finally {
      setSaving(false);
    }
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
    <div className="min-h-screen bg-background text-foreground">
      <header className="flex items-center gap-3 border-b border-border bg-primary px-6 py-4 text-primary-foreground">
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

      <main className="mx-auto grid max-w-5xl gap-6 p-6 md:grid-cols-2">
        {/* Completed courses — the answer to "what did I select?" */}
        <section className="rounded-xl border border-border bg-card p-5 shadow-sm">
          <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
            <h2 className="text-lg font-semibold text-foreground">{t("courses.completed")}</h2>
            <div className="flex items-center gap-3">
              <span className="text-sm text-muted-foreground">
                <span className="font-semibold text-primary">{t("courses.savedCount", { count: saved.length })}</span> ·{" "}
                <span className="font-semibold text-primary">{totalSu}</span> SU
              </span>
              {saved.length > 0 && (
                <Button
                  onClick={() => exportCoursesXlsx(saved, user?.username ?? "student")}
                  variant="outline"
                  size="sm"
                  className="border-success text-success hover:bg-success/10"
                >
                  <FileSpreadsheet className="mr-1.5" size={15} /> Excel
                </Button>
              )}
            </div>
          </div>

          {saved.length === 0 ? (
            <p className="text-sm leading-relaxed text-muted-foreground">
              {t("courses.empty")}
            </p>
          ) : (
            <ul className="space-y-2">
              {saved.map((c) => (
                <li
                  key={c.id}
                  className="flex items-start gap-3 rounded-md border border-border bg-muted/40 px-3 py-2"
                >
                  <Check className="mt-0.5 h-4 w-4 shrink-0 text-primary" />
                  <div className="min-w-0 flex-1">
                    <p className="text-sm font-semibold text-foreground">{c.code}</p>
                    <p className="text-xs text-muted-foreground">{c.title}</p>
                    <span className="mt-1 inline-flex rounded-full bg-muted px-2 py-0.5 text-[11px] font-bold uppercase tracking-wide text-primary">
                      {c.status ?? "completed"}
                    </span>
                  </div>
                  <span className="shrink-0 text-xs font-medium text-muted-foreground">
                    {c.su_credits ?? "—"} SU
                  </span>
                </li>
              ))}
            </ul>
          )}

          <p className="mt-4 border-t border-border pt-3 text-xs text-muted-foreground">
            {t("courses.countNote")}
          </p>
        </section>

        {/* Picker + upload */}
        <div className="space-y-6">
          <section className="rounded-xl border border-border bg-card p-5 shadow-sm">
            <h2 className="mb-4 text-lg font-semibold text-foreground">{t("courses.add")}</h2>

            <div className="relative">
              <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
              <input
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder={t("courses.search")}
                className="w-full rounded-md border border-border bg-background py-2 pl-9 pr-3 text-sm text-foreground placeholder:text-muted-foreground focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary"
              />
            </div>

            <div className="mt-3 max-h-[24rem] space-y-1.5 overflow-y-auto pr-1 scrollbar-thin">
              {loadingPool ? (
                <div className="flex items-center gap-2 rounded-md border border-border bg-muted/40 px-3 py-2 text-sm text-muted-foreground">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  {t("courses.loading")}
                </div>
              ) : pool.length === 0 ? (
                <div className="rounded-md border border-border bg-muted/40 px-3 py-2 text-sm text-muted-foreground">
                  {t("courses.none")}
                </div>
              ) : (
                pool.slice(0, 120).map((course) => {
                  const checked = selectedIds.has(course.id);
                  return (
                    <label
                      key={course.id}
                      className={`flex cursor-pointer items-start gap-3 rounded-md border px-3 py-2 transition-colors ${
                        checked
                          ? "border-primary bg-primary/10"
                          : "border-border bg-card hover:border-primary/40 hover:bg-muted/40"
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
                      <span className="shrink-0 text-xs text-muted-foreground">
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
          <section className="rounded-xl border border-border bg-card p-5 shadow-sm">
            <h2 className="mb-4 text-lg font-semibold text-foreground">{t("courses.documents")}</h2>
            <label className="flex cursor-pointer flex-col items-center justify-center gap-2 rounded-md border border-dashed border-border bg-muted/40 px-3 py-5 text-center text-sm text-muted-foreground transition-colors hover:border-primary hover:bg-primary/5">
              <Upload className="h-5 w-5 text-primary" />
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
                      className="flex items-center gap-2.5 rounded-md border border-border bg-muted/40 px-3 py-2"
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
                  className="mt-3 w-full border-primary text-primary"
                >
                  {uploading ? <Loader2 className="mr-2 animate-spin" size={16} /> : null}
                  {uploading ? t("courses.indexing") : t("courses.upload")}
                </Button>
              </>
            )}
          </section>
          )}
        </div>
      </main>
    </div>
  );
}
