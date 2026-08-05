import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { ArrowLeft, BookOpen, Check, FileSpreadsheet, Loader2, Save, Search, Upload, X } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { useAuth } from "@/contexts/AuthContext";
import ThemeToggle from "@/components/ThemeToggle";
import LanguageToggle from "@/components/LanguageToggle";
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
 * Styled to match ProfilePage: both are "set up your record" screens, so they share the light
 * Sabancı palette (#F5F8FC page, #004B93 header, white cards on #D8E6F3 borders) rather than the
 * dark chat chrome.
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
    <div className="min-h-screen bg-[#F5F8FC] text-[#1a2b45]">
      <header className="flex items-center gap-3 border-b border-[#D8E6F3] bg-[#004B93] px-6 py-4 text-white">
        <Link to="/" className="flex items-center gap-1 text-sm opacity-90 hover:opacity-100">
          <ArrowLeft size={16} /> {t("common.chat")}
        </Link>
        <div className="ml-2 flex items-center gap-2">
          <BookOpen size={18} />
          <span className="font-semibold">{t("courses.title")}</span>
        </div>
        <LanguageToggle className="ml-auto" />
        <ThemeToggle />
      </header>

      <main className="mx-auto grid max-w-5xl gap-6 p-6 md:grid-cols-2">
        {/* Completed courses — the answer to "what did I select?" */}
        <section className="rounded-xl border border-[#D8E6F3] bg-white p-5 shadow-sm">
          <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
            <h2 className="text-lg font-semibold text-[#003B73]">{t("courses.completed")}</h2>
            <div className="flex items-center gap-3">
              <span className="text-sm text-[#4A5568]">
                <span className="font-semibold text-[#004B93]">{t("courses.savedCount", { count: saved.length })}</span> ·{" "}
                <span className="font-semibold text-[#004B93]">{totalSu}</span> SU
              </span>
              {saved.length > 0 && (
                <Button
                  onClick={() => exportCoursesXlsx(saved, user?.username ?? "student")}
                  variant="outline"
                  size="sm"
                  className="border-[#0f7a3d] text-[#0f7a3d] hover:bg-[#0f7a3d]/5"
                >
                  <FileSpreadsheet className="mr-1.5" size={15} /> Excel
                </Button>
              )}
            </div>
          </div>

          {saved.length === 0 ? (
            <p className="text-sm leading-relaxed text-[#4A5568]">
              {t("courses.empty")}
            </p>
          ) : (
            <ul className="space-y-2">
              {saved.map((c) => (
                <li
                  key={c.id}
                  className="flex items-start gap-3 rounded-md border border-[#eef4fa] bg-[#F5F8FC] px-3 py-2"
                >
                  <Check className="mt-0.5 h-4 w-4 shrink-0 text-[#004B93]" />
                  <div className="min-w-0 flex-1">
                    <p className="text-sm font-semibold text-[#003B73]">{c.code}</p>
                    <p className="text-xs text-[#4A5568]">{c.title}</p>
                    <span className="mt-1 inline-flex rounded-full bg-[#eef4fa] px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide text-[#004B93]">
                      {c.status ?? "completed"}
                    </span>
                  </div>
                  <span className="shrink-0 text-xs font-medium text-[#4A5568]">
                    {c.su_credits ?? "—"} SU
                  </span>
                </li>
              ))}
            </ul>
          )}

          <p className="mt-4 border-t border-[#D8E6F3] pt-3 text-xs text-[#4A5568]">
            {t("courses.countNote")}
          </p>
        </section>

        {/* Picker + upload */}
        <div className="space-y-6">
          <section className="rounded-xl border border-[#D8E6F3] bg-white p-5 shadow-sm">
            <h2 className="mb-4 text-lg font-semibold text-[#003B73]">{t("courses.add")}</h2>

            <div className="relative">
              <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-[#4A5568]" />
              <input
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder={t("courses.search")}
                className="w-full rounded-md border border-[#D8E6F3] bg-white py-2 pl-9 pr-3 text-sm text-[#1a2b45] placeholder:text-[#94a3b8] focus:border-[#004B93] focus:outline-none focus:ring-1 focus:ring-[#004B93]"
              />
            </div>

            <div className="mt-3 max-h-[24rem] space-y-1.5 overflow-y-auto pr-1 scrollbar-thin">
              {loadingPool ? (
                <div className="flex items-center gap-2 rounded-md border border-[#D8E6F3] bg-[#F5F8FC] px-3 py-2 text-sm text-[#4A5568]">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  {t("courses.loading")}
                </div>
              ) : pool.length === 0 ? (
                <div className="rounded-md border border-[#D8E6F3] bg-[#F5F8FC] px-3 py-2 text-sm text-[#4A5568]">
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
                          ? "border-[#004B93] bg-[#eef4fa]"
                          : "border-[#eef4fa] bg-white hover:border-[#D8E6F3] hover:bg-[#F5F8FC]"
                      }`}
                    >
                      <input
                        type="checkbox"
                        checked={checked}
                        onChange={() => toggle(course.id)}
                        className="mt-1 h-3.5 w-3.5 accent-[#004B93]"
                      />
                      <span className="min-w-0 flex-1">
                        <span className="block text-sm font-semibold text-[#003B73]">
                          {course.code}
                        </span>
                        <span className="line-clamp-2 text-xs text-[#4A5568]">{course.title}</span>
                      </span>
                      <span className="shrink-0 text-xs text-[#4A5568]">
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
              className="mt-4 w-full bg-[#004B93] hover:bg-[#003B73]"
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
          <section className="rounded-xl border border-[#D8E6F3] bg-white p-5 shadow-sm">
            <h2 className="mb-4 text-lg font-semibold text-[#003B73]">{t("courses.documents")}</h2>
            <label className="flex cursor-pointer flex-col items-center justify-center gap-2 rounded-md border border-dashed border-[#D8E6F3] bg-[#F5F8FC] px-3 py-5 text-center text-sm text-[#4A5568] transition-colors hover:border-[#004B93] hover:bg-[#eef4fa]">
              <Upload className="h-5 w-5 text-[#004B93]" />
              <span>
                <span className="font-medium text-[#003B73]">{t("courses.browse")}</span>
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
                      className="flex items-center gap-2.5 rounded-md border border-[#eef4fa] bg-[#F5F8FC] px-3 py-2"
                    >
                      <div className="min-w-0 flex-1">
                        <p className="truncate text-sm text-[#1a2b45]">{f.name}</p>
                        <p className="text-xs text-[#4A5568]">{formatSize(f.size)}</p>
                      </div>
                      <button
                        type="button"
                        onClick={() => setFiles((prev) => prev.filter((_, idx) => idx !== i))}
                        aria-label={t("courses.removeFile")}
                        className="rounded-full p-1 text-[#4A5568] transition-colors hover:bg-[#D8E6F3] hover:text-[#003B73]"
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
                  className="mt-3 w-full border-[#004B93] text-[#004B93]"
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
