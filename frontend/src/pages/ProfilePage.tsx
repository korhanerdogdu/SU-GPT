import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { toast } from "sonner";
import { ArrowLeft, ClipboardCheck, FileSpreadsheet, GraduationCap, Layers, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useAuth } from "@/contexts/AuthContext";
import ThemeToggle from "@/components/ThemeToggle";
import LanguageToggle from "@/components/LanguageToggle";
import HelpButton from "@/components/HelpButton";
import { useLocale } from "@/contexts/LocaleContext";
import { exportAuditXlsx } from "@/lib/export-xlsx";
import {
  getCurricula,
  getDegreeAudit,
  getProfile,
  saveProfile,
  type CurriculumRow,
  type DegreeAudit,
} from "@/lib/api";

const selectCls =
  "w-full rounded-xl border border-input bg-background px-3 py-2.5 text-sm text-foreground outline-none " +
  "focus:border-primary focus:ring-2 focus:ring-primary/15";

export default function ProfilePage() {
  const { user } = useAuth();
  const { t } = useLocale();
  const username = user?.username ?? "admin";

  const [majors, setMajors] = useState<CurriculumRow[]>([]);
  const [major, setMajor] = useState("");
  const [curriculumTerm, setCurriculumTerm] = useState("");
  const [academicYear, setAcademicYear] = useState("");
  const [saving, setSaving] = useState(false);
  const [audit, setAudit] = useState<DegreeAudit | null>(null);
  const [auditing, setAuditing] = useState(false);

  useEffect(() => {
    getCurricula()
      .then((c) => setMajors(c.majors))
      .catch(() => toast.error(t("profile.loadFailed")));
    getProfile(username)
      .then((p) => {
        setMajor(p.major ?? "");
        setCurriculumTerm(p.curriculum_term ?? "");
        setAcademicYear(p.academic_year ? String(p.academic_year) : "");
      })
      .catch(() => {});
  }, [username]);

  const programs = useMemo(
    () => Array.from(new Set(majors.map((m) => m.program))).sort(),
    [majors],
  );
  const termsForMajor = useMemo(
    () => majors.filter((m) => m.program === major).sort((a, b) => a.curriculum_term.localeCompare(b.curriculum_term)),
    [majors, major],
  );
  const degreeCode = termsForMajor.find((t) => t.curriculum_term === curriculumTerm)?.degree_code ?? null;

  async function handleSave() {
    if (!major || !curriculumTerm) {
      toast.error(t("profile.pickFirst"));
      return;
    }
    setSaving(true);
    try {
      await saveProfile(username, {
        major,
        degree_code: degreeCode,
        curriculum_term: curriculumTerm,
        academic_year: academicYear ? Number(academicYear) : null,
      });
      toast.success(t("profile.saved"));
    } catch (e) {
      toast.error(e instanceof Error ? e.message : t("profile.saveFailed"));
    } finally {
      setSaving(false);
    }
  }

  async function handleAudit() {
    setAuditing(true);
    setAudit(null);
    try {
      setAudit(await getDegreeAudit(username));
    } catch (e) {
      toast.error(e instanceof Error ? e.message : t("profile.auditFailed"));
    } finally {
      setAuditing(false);
    }
  }

  return (
    <div className="min-h-screen bg-background text-foreground">
      <header className="flex items-center gap-3 border-b border-white/10 bg-sabanci-header px-6 py-4 text-primary-foreground shadow-md shadow-black/20">
        <Link to="/" className="flex items-center gap-1 text-sm opacity-90 hover:opacity-100">
          <ArrowLeft size={16} /> {t("common.chat")}
        </Link>
        <div className="ml-2 flex items-center gap-2">
          <GraduationCap size={20} />
          <span className="font-semibold">{t("profile.title")}</span>
        </div>
        <HelpButton className="ml-auto" />
        <LanguageToggle />
        <ThemeToggle />
      </header>

      <main className="relative overflow-hidden">
        {/* Ambient brand wash — purely decorative, keeps the page from reading as two bare
            cards on an empty canvas at wide desktop scale. */}
        <div aria-hidden="true" className="pointer-events-none absolute inset-0 -z-10">
          <div className="absolute -top-32 right-0 h-80 w-80 rounded-full bg-[radial-gradient(circle,_hsl(var(--primary)/0.10)_0%,_transparent_70%)] blur-2xl" />
          <div className="absolute -bottom-24 left-0 h-72 w-72 rounded-full bg-[radial-gradient(circle,_rgba(214,161,58,0.09)_0%,_transparent_70%)] blur-2xl" />
        </div>

        <div className="mx-auto max-w-5xl px-6 pb-12 pt-10">
          <div className="mb-8 max-w-xl">
            <p className="font-ledger text-xs uppercase tracking-[0.18em] text-muted-foreground">
              {t("profile.eyebrow")}
            </p>
            <h2 className="mt-2 text-2xl font-bold tracking-tight text-foreground sm:text-[1.75rem]">
              {t("profile.heroTitle")}
            </h2>
            <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
              {t("profile.heroBody")}
            </p>
          </div>

          <div className="grid gap-6 md:grid-cols-2">
        {/* Profile form */}
        <section className="rounded-2xl border border-border bg-card p-5 shadow-sm transition-shadow hover:shadow-md">
          <h2 className="mb-4 flex items-center gap-2.5 text-lg font-semibold text-foreground">
            <span className="grid h-8 w-8 shrink-0 place-items-center rounded-lg bg-primary/10 text-primary-emphasis">
              <Layers className="h-4 w-4" />
            </span>
            {t("profile.curriculum")}
          </h2>
          <div className="space-y-4">
            <div>
              <label className="mb-1 block text-sm font-medium">{t("profile.major")}</label>
              <select className={selectCls} value={major} onChange={(e) => { setMajor(e.target.value); setCurriculumTerm(""); }}>
                <option value="">{t("profile.selectMajor")}</option>
                {programs.map((p) => (
                  <option key={p} value={p}>{p}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="mb-1 block text-sm font-medium">{t("profile.term")}</label>
              <select className={selectCls} value={curriculumTerm} onChange={(e) => setCurriculumTerm(e.target.value)} disabled={!major}>
                <option value="">{t("profile.selectTerm")}</option>
                {termsForMajor.map((t) => (
                  <option key={t.curriculum_term} value={t.curriculum_term}>
                    {t.admit_term_label || t.curriculum_term} · {t.total_min_su_credits} SU
                  </option>
                ))}
              </select>
              {degreeCode && <p className="mt-1 text-xs text-muted-foreground">{t("profile.degreeCode", { code: degreeCode })}</p>}
            </div>
            <div>
              <label className="mb-1 block text-sm font-medium">{t("profile.year")}</label>
              <select
                className={selectCls}
                value={academicYear}
                onChange={(e) => setAcademicYear(e.target.value)}
              >
                <option value="">{t("profile.yearUnknown")}</option>
                {[1, 2, 3, 4, 5, 6, 7, 8].map((year) => (
                  <option key={year} value={year}>{year}</option>
                ))}
              </select>
            </div>
            <Button onClick={handleSave} disabled={saving} className="w-full">
              {saving ? <Loader2 className="mr-2 animate-spin" size={16} /> : null} {t("profile.save")}
            </Button>
            <p className="text-xs text-muted-foreground">
              {t("profile.contract")}
            </p>
          </div>
        </section>

        {/* Audit */}
        <section className="rounded-2xl border border-border bg-card p-5 shadow-sm transition-shadow hover:shadow-md">
          <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
            <h2 className="flex items-center gap-2.5 text-lg font-semibold text-foreground">
              <span className="grid h-8 w-8 shrink-0 place-items-center rounded-lg bg-primary/10 text-primary-emphasis">
                <ClipboardCheck className="h-4 w-4" />
              </span>
              {t("profile.audit")}
            </h2>
            <div className="flex items-center gap-2">
              {audit && audit.categories && (
                <Button
                  onClick={() => exportAuditXlsx(audit, username)}
                  variant="outline"
                  size="sm"
                  className="border-success text-success hover:bg-success/10"
                >
                  <FileSpreadsheet className="mr-1.5" size={15} /> Excel
                </Button>
              )}
              <Button onClick={handleAudit} disabled={auditing} variant="outline" className="border-primary-emphasis text-primary-emphasis">
                {auditing ? <Loader2 className="mr-2 animate-spin" size={16} /> : null} {t("profile.runAudit")}
              </Button>
            </div>
          </div>

          {!audit && (
            <p className="text-sm text-muted-foreground">
              {t("profile.auditHelpBefore")}{" "}
              <Link to="/courses" className="font-medium text-primary-emphasis underline">
                {t("profile.auditHelpLink")}
              </Link>{" "}
              {t("profile.auditHelpAfter")}
            </p>
          )}

          {audit && audit.reliability === "unavailable" && (
            <p className="rounded-md bg-warning/10 p-3 text-sm text-warning">{audit.message}</p>
          )}

          {audit && audit.categories && (
            <div className="space-y-3">
              <div className="flex items-baseline gap-2">
                <span className="text-2xl font-bold text-primary-emphasis">{audit.completed_su_credits}</span>
                <span className="text-sm text-muted-foreground">{t("profile.suCredits", { total: audit.total_min_su_credits ?? "—" })}</span>
                <span className={`ml-auto rounded-full px-2 py-0.5 text-xs font-bold ${audit.status === "complete" ? "bg-success/15 text-success" : "bg-primary/10 text-primary-emphasis"}`}>
                  {audit.status}
                </span>
              </div>
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border text-left text-xs text-muted-foreground">
                    <th className="py-1">{t("profile.category")}</th><th>{t("profile.done")}</th><th>{t("profile.need")}</th><th>{t("profile.remaining")}</th>
                  </tr>
                </thead>
                <tbody>
                  {audit.categories.map((c) => (
                    <tr key={c.category} className="border-b border-border/60">
                      <td className="py-1 capitalize">{c.category.replace(/_/g, " ")}</td>
                      <td>{c.completed_su_credits}</td>
                      <td>{c.required_su_credits ?? "—"}</td>
                      <td className={c.remaining_su_credits ? "font-medium text-primary-emphasis" : "text-success"}>
                        {c.remaining_su_credits ?? "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {audit.ects_requirements && audit.ects_requirements.length > 0 && (
                <div className="rounded-md bg-muted/40 p-2 text-sm">
                  {audit.ects_requirements.map((e) => (
                    <div key={e.category} className="flex justify-between">
                      <span className="capitalize">{e.category.replace(/_/g, " ")} ECTS</span>
                      <span>
                        {e.completed_ects} / {e.required_ects}
                        <span className={e.remaining_ects ? "ml-2 text-primary-emphasis" : "ml-2 text-success"}>
                          ({t("profile.left", { count: e.remaining_ects })})
                        </span>
                      </span>
                    </div>
                  ))}
                </div>
              )}
              {audit.missing_required_courses && audit.missing_required_courses.length > 0 && (
                <p className="text-sm"><span className="font-medium">{t("profile.missing")}</span> {audit.missing_required_courses.join(", ")}</p>
              )}
              <p className="text-xs text-muted-foreground">{t("profile.reliability", { value: audit.reliability })}</p>
            </div>
          )}
        </section>
          </div>
        </div>
      </main>
    </div>
  );
}
