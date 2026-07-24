import { useEffect, useState, type FormEvent } from "react";
import { useNavigate } from "react-router-dom";
import { Eye, EyeOff, Lock, User } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useAuth } from "@/contexts/AuthContext";
import { SAMPLE_QUESTIONS } from "@/lib/sample-questions";
import ThemeToggle from "@/components/ThemeToggle";

/** Short lines, typed one word at a time. Written from the student's side of the screen. */
const TAGLINES = [
  "See what you have left.",
  "Check electives and minors.",
  "Answers from official curriculum data.",
];


function usePrefersReducedMotion() {
  const [reduced, setReduced] = useState(
    () => window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false
  );
  useEffect(() => {
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    const onChange = () => setReduced(mq.matches);
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);
  return reduced;
}

function TypedTagline() {
  const reduced = usePrefersReducedMotion();
  const [line, setLine] = useState(0);
  const [shown, setShown] = useState(0);

  const words = TAGLINES[line].split(" ");
  const complete = shown >= words.length;

  // `shown` MUST be a dependency: without it the effect never re-runs after the first word
  // and the line stalls one word in.
  useEffect(() => {
    if (reduced) return; // honour reduced motion: first line, fully shown, no cycling
    if (!complete) {
      const t = setTimeout(() => setShown((n) => n + 1), 130);
      return () => clearTimeout(t);
    }
    const t = setTimeout(() => {
      setLine((i) => (i + 1) % TAGLINES.length);
      setShown(0);
    }, 2400);
    return () => clearTimeout(t);
  }, [reduced, complete, line, shown]);

  const visible = reduced ? words : words.slice(0, shown);

  return (
    // Height is reserved for the longest line so cycling never shifts the layout.
    <p className="flex min-h-[4.5rem] flex-wrap items-start gap-x-[0.32em] text-2xl font-semibold leading-snug tracking-tight text-white sm:text-3xl">
      {visible.map((word, i) => (
        <span
          key={`${line}-${i}`}
          className="inline-block animate-in fade-in slide-in-from-bottom-1 duration-300"
        >
          {word}
        </span>
      ))}
      <span
        aria-hidden="true"
        className="ml-0.5 inline-block h-[1.05em] w-[3px] translate-y-[0.16em] rounded-sm bg-sabanci-gold animate-pulse"
      />
      <span className="sr-only">{TAGLINES[line]}</span>
    </p>
  );
}

export default function LoginPage() {
  const navigate = useNavigate();
  const { signIn } = useAuth();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [submitting, setSubmitting] = useState(false);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (!username.trim() || !password.trim()) {
      toast.error("Please enter both a username and a password.");
      return;
    }
    setSubmitting(true);
    try {
      await signIn(username.trim(), password);
      navigate("/", { replace: true });
    } catch {
      toast.error("Invalid credentials. Use admin / admin for this demo.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    // min-h-screen rather than a hard h-screen + overflow-hidden: on a short laptop the brand
    // column is taller than the viewport, and clipping it silently is worse than letting the
    // page scroll. Both panels still stretch to full height on normal screens.
    <div className="flex min-h-screen w-screen flex-col lg:flex-row">
      {/* ── Left: paper panel. The lockup was drawn for a light ground, so it sits here natively. */}
      <section
        className="relative flex flex-1 flex-col justify-center overflow-hidden bg-cover bg-center px-8 py-12 sm:px-14 lg:items-center lg:px-16 lg:py-0"
        style={{
          backgroundImage:
            // Heavy navy wash: the campus stays readable as place, never as competing detail.
            "linear-gradient(180deg, rgba(5,12,24,0.90) 0%, rgba(0,32,66,0.93) 55%, rgba(4,10,22,0.95) 100%), url(/assets/campus.jpg)",
        }}
      >
        <div className="relative w-full max-w-2xl">
          {/* Reversed (knockout) lockup — the navy type is knocked out to white so the mark sits
              directly on the photograph with no plate behind it, while the gold and teal accents
              survive. Generated from adviSU-logo.png; see adviSU-logo-reversed.png. The soft
              drop-shadow is what holds it against the brighter parts of the campus image. */}
          <img
            src="/assets/adviSU-logo-reversed.png"
            alt="adviSU — Sabancı University Academic Advisor"
            className="w-[min(100%,33rem)] drop-shadow-[0_4px_24px_rgba(0,0,0,0.55)]"
          />

          <div className="mt-9 max-w-lg">
            <TypedTagline />
          </div>

          {/* Sample questions, set as quotations: what the product is for, in the student's own
              words. Plain UI sans (not italic) at a larger size and higher contrast, so they read
              as real questions with weight rather than as decorative pull-quotes. */}
          {/* pl-10 on the whole block so the label and every quote share one left edge; only the
              ornament hangs into the margin, which is how a pulled quote should sit. */}
          {/* Wide enough that every question sits on one line — a wrapped question reads as a
              paragraph and loses its punch. */}
          <div className="relative mt-9 max-w-2xl pl-10">
            <span
              aria-hidden="true"
              className="pointer-events-none absolute left-0 top-[0.42em] select-none font-serif text-[4.5rem] leading-none text-sabanci-gold/40"
            >
              &ldquo;
            </span>
            <p className="font-mono text-[11px] uppercase tracking-[0.22em] text-sabanci-light/50">
              Students ask
            </p>
            <div className="mt-4">
              <ul className="space-y-3 [text-shadow:0_1px_12px_rgba(0,0,0,0.6)]">
                {SAMPLE_QUESTIONS.map((question, i) => (
                  <li
                    key={question}
                    style={{ animationDelay: `${300 + i * 120}ms` }}
                    className="animate-in fade-in slide-in-from-bottom-2 fill-mode-both duration-500 text-[1.15rem] font-medium leading-snug tracking-tight text-white/95"
                  >
                    &ldquo;{question}&rdquo;
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>

      </section>

      {/* ── Right: the form. Deep navy, deliberately quiet — all the character lives on the left. */}
      <section className="relative flex w-full flex-col justify-center border-t border-border bg-card px-8 py-14 text-card-foreground sm:px-14 lg:w-[30rem] lg:shrink-0 lg:border-l lg:border-t-0 lg:px-12 xl:w-[34rem]">
        <ThemeToggle className="absolute right-5 top-5" />
        <div className="mx-auto w-full max-w-sm">
          <h1 className="text-2xl font-bold tracking-tight text-foreground">Welcome back</h1>
          <p className="mt-1.5 text-sm text-muted-foreground">
            Sign in to continue to adviSU.
          </p>

          <form onSubmit={handleSubmit} className="mt-9 space-y-5">
            <div className="space-y-1.5">
              <Label htmlFor="username" className="text-foreground">
                Username
              </Label>
              <div className="relative">
                <User className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                <Input
                  id="username"
                  autoComplete="username"
                  placeholder="admin"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  className="border-border bg-background pl-9 text-foreground placeholder:text-muted-foreground"
                />
              </div>
            </div>

            <div className="space-y-1.5">
              <Label htmlFor="password" className="text-foreground">
                Password
              </Label>
              <div className="relative">
                <Lock className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                <Input
                  id="password"
                  type={showPassword ? "text" : "password"}
                  autoComplete="current-password"
                  placeholder="admin"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="border-border bg-background pl-9 pr-9 text-foreground placeholder:text-muted-foreground"
                />
                <button
                  type="button"
                  onClick={() => setShowPassword((s) => !s)}
                  aria-label={showPassword ? "Hide password" : "Show password"}
                    className="absolute right-3 top-1/2 -translate-y-1/2 rounded text-muted-foreground transition-colors hover:text-foreground"
                >
                  {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                </button>
              </div>
            </div>

            <Button
              type="submit"
              size="xl"
              className="w-full bg-primary text-primary-foreground hover:opacity-90"
              disabled={submitting}
            >
              {submitting ? "Signing in…" : "Sign in"}
            </Button>
          </form>

          <p className="mt-7 border-t border-border pt-5 text-sm text-muted-foreground">
            Student demo <span className="font-mono text-foreground">student / student</span>
            {" · "}Admin <span className="font-mono text-foreground">admin / admin</span>
          </p>
        </div>
      </section>
    </div>
  );
}
