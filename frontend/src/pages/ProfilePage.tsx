import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { toast } from "sonner";
import { ArrowLeft, FileSpreadsheet, GraduationCap, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useAuth } from "@/contexts/AuthContext";
import ThemeToggle from "@/components/ThemeToggle";
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
  "w-full rounded-md border border-[#D8E6F3] bg-white px-3 py-2 text-sm text-[#1a2b45] " +
  "focus:outline-none focus:ring-2 focus:ring-[#004B93]/40";

export default function ProfilePage() {
  const { user } = useAuth();
  const username = user?.username ?? "admin";

  const [majors, setMajors] = useState<CurriculumRow[]>([]);
  const [major, setMajor] = useState("");
  const [curriculumTerm, setCurriculumTerm] = useState("");
  const [saving, setSaving] = useState(false);
  const [audit, setAudit] = useState<DegreeAudit | null>(null);
  const [auditing, setAuditing] = useState(false);

  useEffect(() => {
    getCurricula()
      .then((c) => setMajors(c.majors))
      .catch(() => toast.error("Could not load curricula"));
    getProfile(username)
      .then((p) => {
        setMajor(p.major ?? "");
        setCurriculumTerm(p.curriculum_term ?? "");
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
      toast.error("Pick your major and curriculum term first.");
      return;
    }
    setSaving(true);
    try {
      await saveProfile(username, {
        major,
        degree_code: degreeCode,
        curriculum_term: curriculumTerm,
      });
      toast.success("Profile saved.");
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Save failed");
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
      toast.error(e instanceof Error ? e.message : "Audit failed — save your profile first.");
    } finally {
      setAuditing(false);
    }
  }

  return (
    <div className="min-h-screen bg-[#F5F8FC] text-[#1a2b45]">
      <header className="flex items-center gap-3 border-b border-[#D8E6F3] bg-[#004B93] px-6 py-4 text-white">
        <Link to="/" className="flex items-center gap-1 text-sm opacity-90 hover:opacity-100">
          <ArrowLeft size={16} /> Chat
        </Link>
        <div className="ml-2 flex items-center gap-2">
          <GraduationCap size={20} />
          <span className="font-semibold">Academic Profile &amp; Degree Audit</span>
        </div>
        <ThemeToggle className="ml-auto" />
      </header>

      <main className="mx-auto grid max-w-5xl gap-6 p-6 md:grid-cols-2">
        {/* Profile form */}
        <section className="rounded-xl border border-[#D8E6F3] bg-white p-5 shadow-sm">
          <h2 className="mb-4 text-lg font-semibold text-[#003B73]">Your curriculum</h2>
          <div className="space-y-4">
            <div>
              <label className="mb-1 block text-sm font-medium">Major (program)</label>
              <select className={selectCls} value={major} onChange={(e) => { setMajor(e.target.value); setCurriculumTerm(""); }}>
                <option value="">Select major…</option>
                {programs.map((p) => (
                  <option key={p} value={p}>{p}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="mb-1 block text-sm font-medium">Curriculum (admit) term</label>
              <select className={selectCls} value={curriculumTerm} onChange={(e) => setCurriculumTerm(e.target.value)} disabled={!major}>
                <option value="">Select curriculum term…</option>
                {termsForMajor.map((t) => (
                  <option key={t.curriculum_term} value={t.curriculum_term}>
                    {t.admit_term_label || t.curriculum_term} · {t.total_min_su_credits} SU
                  </option>
                ))}
              </select>
              {degreeCode && <p className="mt-1 text-xs text-[#4A5568]">Degree code: {degreeCode}</p>}
            </div>
            <Button onClick={handleSave} disabled={saving} className="w-full bg-[#004B93] hover:bg-[#003B73]">
              {saving ? <Loader2 className="mr-2 animate-spin" size={16} /> : null} Save profile
            </Button>
            <p className="text-xs text-[#4A5568]">
              Your curriculum term is your graduation contract — the audit uses the official requirements for exactly this program and term.
            </p>
          </div>
        </section>

        {/* Audit */}
        <section className="rounded-xl border border-[#D8E6F3] bg-white p-5 shadow-sm">
          <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
            <h2 className="text-lg font-semibold text-[#003B73]">Degree audit</h2>
            <div className="flex items-center gap-2">
              {audit && audit.categories && (
                <Button
                  onClick={() => exportAuditXlsx(audit, username)}
                  variant="outline"
                  size="sm"
                  className="border-[#0f7a3d] text-[#0f7a3d] hover:bg-[#0f7a3d]/5"
                >
                  <FileSpreadsheet className="mr-1.5" size={15} /> Excel
                </Button>
              )}
              <Button onClick={handleAudit} disabled={auditing} variant="outline" className="border-[#004B93] text-[#004B93]">
                {auditing ? <Loader2 className="mr-2 animate-spin" size={16} /> : null} Run audit
              </Button>
            </div>
          </div>

          {!audit && (
            <p className="text-sm text-[#4A5568]">
              Save your profile, add your completed courses on the{" "}
              <Link to="/courses" className="font-medium text-[#004B93] underline">
                Course History
              </Link>{" "}
              page, then run the audit.
            </p>
          )}

          {audit && audit.reliability === "unavailable" && (
            <p className="rounded-md bg-amber-50 p-3 text-sm text-amber-800">{audit.message}</p>
          )}

          {audit && audit.categories && (
            <div className="space-y-3">
              <div className="flex items-baseline gap-2">
                <span className="text-2xl font-bold text-[#004B93]">{audit.completed_su_credits}</span>
                <span className="text-sm text-[#4A5568]">/ {audit.total_min_su_credits} SU credits</span>
                <span className={`ml-auto rounded-full px-2 py-0.5 text-xs font-medium ${audit.status === "complete" ? "bg-green-100 text-green-700" : "bg-blue-100 text-[#004B93]"}`}>
                  {audit.status}
                </span>
              </div>
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-[#D8E6F3] text-left text-xs text-[#4A5568]">
                    <th className="py-1">Category</th><th>Done</th><th>Need</th><th>Remaining</th>
                  </tr>
                </thead>
                <tbody>
                  {audit.categories.map((c) => (
                    <tr key={c.category} className="border-b border-[#eef4fa]">
                      <td className="py-1 capitalize">{c.category.replace(/_/g, " ")}</td>
                      <td>{c.completed_su_credits}</td>
                      <td>{c.required_su_credits ?? "—"}</td>
                      <td className={c.remaining_su_credits ? "font-medium text-[#004B93]" : "text-green-600"}>
                        {c.remaining_su_credits ?? "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {audit.ects_requirements && audit.ects_requirements.length > 0 && (
                <div className="rounded-md bg-[#F5F8FC] p-2 text-sm">
                  {audit.ects_requirements.map((e) => (
                    <div key={e.category} className="flex justify-between">
                      <span className="capitalize">{e.category.replace(/_/g, " ")} ECTS</span>
                      <span>
                        {e.completed_ects} / {e.required_ects}
                        <span className={e.remaining_ects ? "ml-2 text-[#004B93]" : "ml-2 text-green-600"}>
                          ({e.remaining_ects} left)
                        </span>
                      </span>
                    </div>
                  ))}
                </div>
              )}
              {audit.missing_required_courses && audit.missing_required_courses.length > 0 && (
                <p className="text-sm"><span className="font-medium">Missing required:</span> {audit.missing_required_courses.join(", ")}</p>
              )}
              <p className="text-xs text-[#4A5568]">Reliability: {audit.reliability} · computed deterministically from official requirements.</p>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}
