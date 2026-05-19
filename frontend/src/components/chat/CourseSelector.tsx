import { useEffect, useMemo, useState } from "react";
import { BookOpen, Loader2, Save } from "lucide-react";
import { toast } from "sonner";
import { Checkbox } from "@/components/ui/checkbox";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  fetchCourses,
  fetchUserCourses,
  saveUserCourses,
  type Course,
} from "@/lib/api";
import { useAuth } from "@/contexts/AuthContext";

export default function CourseSelector() {
  const { user } = useAuth();
  const [search, setSearch] = useState("");
  const [courses, setCourses] = useState<Course[]>([]);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    const timer = window.setTimeout(async () => {
      try {
        const data = await fetchCourses(search);
        if (!cancelled) setCourses(data);
      } catch {
        if (!cancelled) toast.error("Could not load course pool from MongoDB.");
      } finally {
        if (!cancelled) setLoading(false);
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
    async function loadSelected() {
      try {
        const selected = await fetchUserCourses(user!.username);
        if (!cancelled) setSelectedIds(new Set(selected.map((course) => course.id)));
      } catch {
        if (!cancelled) toast.error("Could not load saved course history.");
      }
    }
    loadSelected();
    return () => {
      cancelled = true;
    };
  }, [user?.username]);

  const selectedCount = selectedIds.size;
  const visibleCourses = useMemo(() => courses.slice(0, 80), [courses]);

  function toggle(courseId: string) {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(courseId)) next.delete(courseId);
      else next.add(courseId);
      return next;
    });
  }

  async function save() {
    if (!user?.username) return;
    setSaving(true);
    try {
      const saved = await saveUserCourses(user.username, Array.from(selectedIds));
      setSelectedIds(new Set(saved.map((course) => course.id)));
      toast.success(`Saved ${saved.length} course(s) to MongoDB.`);
    } catch {
      toast.error("Could not save selected courses.");
    } finally {
      setSaving(false);
    }
  }

  return (
    <section className="mt-6">
      <div className="mb-3 flex items-center justify-between gap-2">
        <div className="flex items-center gap-2 text-sm font-semibold uppercase tracking-wider text-muted-foreground">
          <BookOpen className="h-4 w-4" />
          Courses
        </div>
        <span className="rounded bg-secondary px-2 py-0.5 text-xs text-muted-foreground">
          {selectedCount} saved
        </span>
      </div>

      <Input
        value={search}
        onChange={(e) => setSearch(e.target.value)}
        placeholder="Search CS 300, BERT..."
        className="h-9"
      />

      <div className="mt-3 max-h-56 space-y-1 overflow-y-auto pr-1 scrollbar-thin">
        {loading ? (
          <div className="flex items-center gap-2 rounded border border-border bg-secondary/40 px-3 py-2 text-sm text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" />
            Loading courses
          </div>
        ) : visibleCourses.length === 0 ? (
          <div className="rounded border border-border bg-secondary/40 px-3 py-2 text-sm text-muted-foreground">
            No courses found
          </div>
        ) : (
          visibleCourses.map((course) => (
            <label
              key={course.id}
              className="flex cursor-pointer items-start gap-2 rounded border border-border bg-secondary/35 px-2.5 py-2 transition hover:bg-secondary/60"
            >
              <Checkbox
                checked={selectedIds.has(course.id)}
                onCheckedChange={() => toggle(course.id)}
                className="mt-0.5"
              />
              <span className="min-w-0">
                <span className="block text-xs font-semibold text-foreground">
                  {course.code}
                </span>
                <span className="line-clamp-2 text-xs text-muted-foreground">
                  {course.title}
                </span>
              </span>
            </label>
          ))
        )}
      </div>

      <Button
        type="button"
        variant="outline"
        size="sm"
        className="mt-3 w-full"
        onClick={save}
        disabled={saving}
      >
        {saving ? <Loader2 className="h-4 w-4 animate-spin" /> : <Save className="h-4 w-4" />}
        Save course history
      </Button>
    </section>
  );
}
