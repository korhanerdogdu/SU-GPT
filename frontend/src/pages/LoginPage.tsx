import { useEffect, useMemo, useRef, useState, type FormEvent } from "react";
import { useNavigate } from "react-router-dom";
import { ArrowRight, ArrowUp, Eye, EyeOff, KeyRound, UserRound } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useAuth } from "@/contexts/AuthContext";
import { useLocale } from "@/contexts/LocaleContext";
import { useTheme } from "@/contexts/ThemeContext";
import ThemeToggle from "@/components/ThemeToggle";
import LanguageToggle from "@/components/LanguageToggle";
import HelpButton from "@/components/HelpButton";
import { sampleQuestions } from "@/lib/sample-questions";

/**
 * Sign-in screen.
 *
 * The hero is a mock composer that types out the questions students actually bring here. Those
 * questions are the product — showing one being asked says more than any description of it — and
 * typing them keeps the screen alive without decoration that means nothing.
 *
 * The page follows the app theme rather than pinning its own colours, and it is sized to fit the
 * viewport on desktop: a login screen that scrolls has failed at its one job.
 */

const TYPE_MS = 42;
const DELETE_MS = 18;
const HOLD_MS = 2000;

/** Types one question, holds it, clears it, moves to the next. Cycles forever. */
function useTypedQuestion(questions: readonly string[], enabled: boolean) {
  const [index, setIndex] = useState(0);
  const [text, setText] = useState("");
  const [deleting, setDeleting] = useState(false);
  const timer = useRef<number>();

  // Restart cleanly when the interface language changes mid-cycle.
  useEffect(() => {
    setIndex(0);
    setText("");
    setDeleting(false);
  }, [questions]);

  useEffect(() => {
    if (!enabled) {
      setText(questions[index] ?? "");
      return;
    }
    const target = questions[index] ?? "";

    if (!deleting && text === target) {
      timer.current = window.setTimeout(() => setDeleting(true), HOLD_MS);
    } else if (deleting && text === "") {
      setDeleting(false);
      setIndex((current) => (current + 1) % questions.length);
    } else {
      timer.current = window.setTimeout(
        () =>
          setText((current) =>
            deleting ? current.slice(0, -1) : target.slice(0, current.length + 1),
          ),
        deleting ? DELETE_MS : TYPE_MS,
      );
    }
    return () => window.clearTimeout(timer.current);
  }, [text, deleting, index, questions, enabled]);

  // Reduced motion: no typing, just swap the whole question on a slow timer.
  useEffect(() => {
    if (enabled) return;
    const id = window.setInterval(
      () => setIndex((current) => (current + 1) % questions.length),
      5000,
    );
    return () => window.clearInterval(id);
  }, [enabled, questions.length]);

  return { text, index };
}

function usePrefersReducedMotion() {
  const [reduced, setReduced] = useState(
    () => window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false,
  );
  useEffect(() => {
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    const onChange = () => setReduced(mq.matches);
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);
  return reduced;
}

/** The signature: the product's own composer, asking the product's own questions. */
function AskDemo() {
  const { locale, t } = useLocale();
  const reduced = usePrefersReducedMotion();
  const questions = useMemo(() => sampleQuestions(locale), [locale]);
  const { text, index } = useTypedQuestion(questions, !reduced);
  const upNext = useMemo(
    () => [1, 2].map((step) => questions[(index + step) % questions.length]),
    [questions, index],
  );

  return (
    <div className="mt-9 w-full max-w-xl">
      <p className="font-ledger text-[0.68rem] uppercase tracking-[0.18em] text-muted-foreground">
        {t("login.askLabel")}
      </p>

      <div className="mt-3 flex items-center gap-3 rounded-2xl border border-border bg-card px-4 py-3.5 shadow-lg shadow-black/5">
        <p className="flex min-h-[2.75rem] min-w-0 flex-1 items-center text-[1.02rem] leading-snug text-foreground">
          {text}
          <span
            aria-hidden="true"
            className="ml-[3px] inline-block h-[1.15em] w-[2px] translate-y-[0.22em] bg-sabanci-gold motion-safe:animate-pulse"
          />
        </p>
        <span
          aria-hidden="true"
          className="grid h-9 w-9 shrink-0 place-items-center rounded-xl bg-primary text-primary-foreground"
        >
          <ArrowUp className="h-4 w-4" />
        </span>
      </div>

      {/* Position in the cycle, then the two questions queued up next. The chips are part of the
          same mock as the composer — deliberately not buttons, because nothing here can be
          clicked until you are signed in, and a control that does nothing is worse than a label. */}
      <div className="mt-4 flex items-center gap-1.5" aria-hidden="true">
        {questions.map((question, dot) => (
          <span
            key={question}
            className={`h-[3px] rounded-full transition-all duration-300 ${
              dot === index ? "w-6 bg-sabanci-gold" : "w-2.5 bg-border"
            }`}
          />
        ))}
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-2" aria-hidden="true">
        {upNext.map((question) => (
          <span
            key={question}
            className="max-w-full truncate rounded-full border border-border bg-card/60 px-3 py-1.5 text-[0.78rem] text-muted-foreground backdrop-blur-sm"
          >
            {question}
          </span>
        ))}
      </div>

      {/* The live region carries the whole question, so a screen reader never reads a half-typed
          string letter by letter. */}
      <p className="sr-only" aria-live="polite">
        {questions[index]}
      </p>
    </div>
  );
}

export default function LoginPage() {
  const navigate = useNavigate();
  const { signIn } = useAuth();
  const { t } = useLocale();
  const { resolvedTheme } = useTheme();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [submitting, setSubmitting] = useState(false);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (!username.trim() || !password.trim()) {
      toast.error(t("login.missingFields"));
      return;
    }
    setSubmitting(true);
    try {
      await signIn(username.trim(), password);
      navigate("/", { replace: true });
    } catch {
      toast.error(t("login.invalid"));
    } finally {
      setSubmitting(false);
    }
  }

  // The supplied dark-ground lockup (new.png, matte removed by
  // server/scripts/make_logo_transparent.py) on dark; the navy original on light, where its
  // white "SU" and white descriptor would disappear into the background.
  const logo =
    resolvedTheme === "dark" ? "/assets/adviSU-logo-dark.png" : "/assets/adviSU-logo.png";

  return (
    // Fits the viewport on desktop — no scrolling to reach the form. Mobile stacks and scrolls,
    // which is the only honest option for two panels on a phone.
    <div className="flex min-h-screen w-full flex-col bg-background text-foreground lg:h-screen lg:min-h-0 lg:flex-row lg:overflow-hidden">
      {/* ── Left: what students ask. */}
      <section className="relative flex flex-1 flex-col overflow-hidden px-6 pb-12 pt-10 sm:px-12 lg:overflow-y-auto lg:px-14 lg:py-10 xl:px-20">
        {/* Campus photograph across the outer half of the panel, cropped to its centre. It is
            masked to nothing at its inner edge so it reads as the panel receding into place
            rather than as a picture pasted next to the text. */}
        <div
          aria-hidden="true"
          className="pointer-events-none absolute inset-y-0 right-0 w-1/2"
          style={{
            backgroundImage: "url(/assets/campus.jpg)",
            backgroundSize: "cover",
            backgroundPosition: "10% center",
            filter: "saturate(0.62) contrast(0.94)",
            maskImage:
              "linear-gradient(90deg, transparent 0%, rgba(0,0,0,0.45) 30%, rgba(0,0,0,0.85) 65%, #000 100%)",
            WebkitMaskImage:
              "linear-gradient(90deg, transparent 0%, rgba(0,0,0,0.45) 30%, rgba(0,0,0,0.85) 65%, #000 100%)",
          }}
        />

        {/* Veil + the two brand washes. Built from `--background`, so it darkens the photo in the
            dark theme and lightens it in the light one without branching on the theme — which the
            `dark:` variant could not do here anyway, since index.html pins `class="dark"`. */}
        <div
          aria-hidden="true"
          className="pointer-events-none absolute inset-0"
          style={{
            background:
              "linear-gradient(90deg, hsl(var(--background)) 0%, hsl(var(--background)) 34%, hsl(var(--background) / 0.90) 58%, hsl(var(--background) / 0.74) 100%), " +
              "linear-gradient(180deg, hsl(var(--background) / 0.55) 0%, transparent 30%, transparent 62%, hsl(var(--background) / 0.65) 100%), " +
              "radial-gradient(ellipse 60% 55% at 15% 25%, hsl(var(--primary) / 0.14) 0%, transparent 62%), " +
              "radial-gradient(ellipse 50% 45% at 85% 90%, rgba(214,161,58,0.10) 0%, transparent 60%)",
          }}
        />

        {/* The lockup sits top-left and large, straight on the panel — no halo, no plate.
            Which file depends on the theme, and it has to: the knockout carries a white "SU" and
            a white descriptor, which is what makes it read on the dark panel, and exactly what
            would make it vanish on the light one. */}
        <img
          src={logo}
          alt="adviSU — Sabancı University Academic Advisor"
          className="relative w-[min(72vw,17rem)] shrink-0 lg:w-[min(40vw,19rem)] xl:w-[21rem]"
        />

        {/* Takes the remaining height so the copy stays optically centred under the lockup. */}
        <div className="relative flex w-full flex-1 items-center pt-8 lg:pt-0">
          <div className="w-full max-w-2xl">
            <p className="font-ledger text-[0.7rem] uppercase tracking-[0.2em] text-muted-foreground">
              {t("login.eyebrow")}
            </p>

            <h1 className="mt-3 font-display text-[2.15rem] font-extrabold leading-[1.04] tracking-[-0.028em] sm:text-[2.9rem] lg:text-[3.25rem]">
              {t("login.headlineTop")}
              <br />
              <span className="text-muted-foreground">{t("login.headlineBottom")}</span>
            </h1>

            <p className="mt-4 max-w-lg text-[1rem] leading-relaxed text-muted-foreground">
              {t("login.lede")}
            </p>

            <AskDemo />
          </div>
        </div>
      </section>

      {/* ── Right: the form. Quiet, so the one action is obvious. */}
      <section className="relative flex w-full flex-col justify-center border-t border-border bg-card px-6 py-12 sm:px-12 lg:w-[28rem] lg:shrink-0 lg:overflow-y-auto lg:border-l lg:border-t-0 lg:py-8 xl:w-[32rem] xl:px-14">
        <div className="absolute right-5 top-5 flex items-center gap-2">
          <HelpButton />
          <LanguageToggle />
          <ThemeToggle />
        </div>

        <div className="mx-auto w-full max-w-sm">
          <h2 className="font-display text-[1.65rem] font-semibold tracking-[-0.02em]">
            {t("login.welcome")}
          </h2>
          <p className="mt-1.5 text-sm leading-relaxed text-muted-foreground">
            {t("login.subtitle")}
          </p>

          <form onSubmit={handleSubmit} className="mt-8 space-y-4">
            <div className="space-y-1.5">
              <Label
                htmlFor="username"
                className="font-ledger text-[0.68rem] uppercase tracking-[0.14em] text-muted-foreground"
              >
                {t("login.username")}
              </Label>
              <Input
                id="username"
                autoComplete="username"
                placeholder="student"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                className="h-11 rounded-xl border-border bg-background px-4 focus-visible:border-ring focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-0"
              />
            </div>

            <div className="space-y-1.5">
              <Label
                htmlFor="password"
                className="font-ledger text-[0.68rem] uppercase tracking-[0.14em] text-muted-foreground"
              >
                {t("login.password")}
              </Label>
              <div className="relative">
                <Input
                  id="password"
                  type={showPassword ? "text" : "password"}
                  autoComplete="current-password"
                  placeholder={t("login.passwordHint")}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="h-11 rounded-xl border-border bg-background px-4 pr-11 focus-visible:border-ring focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-0"
                />
                <button
                  type="button"
                  onClick={() => setShowPassword((s) => !s)}
                  aria-label={showPassword ? t("login.hidePassword") : t("login.showPassword")}
                  className="absolute right-3 top-1/2 -translate-y-1/2 rounded-md p-1.5 text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                >
                  {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                </button>
              </div>
            </div>

            {/* Sabancı blue warming into the gold at the far end — the two brand colours in the
                one place the page wants you to look. */}
            <Button
              type="submit"
              style={{
                backgroundImage:
                  "linear-gradient(100deg, hsl(var(--primary)) 0%, hsl(var(--primary)) 42%, #B98A3C 88%, #D6A13A 100%)",
              }}
              className="group h-11 w-full rounded-xl text-[0.95rem] font-semibold text-primary-foreground shadow-md transition-all hover:brightness-110 focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-card disabled:opacity-60"
              disabled={submitting}
            >
              {submitting ? t("login.submitting") : t("login.submit")}
              {!submitting && (
                <ArrowRight className="ml-2 h-4 w-4 transition-transform group-hover:translate-x-0.5" />
              )}
            </Button>
          </form>

          <div className="mt-8 border-t border-border pt-5">
            <p className="font-ledger text-[0.66rem] uppercase tracking-[0.14em] text-muted-foreground">
              {t("login.demoHeading")}
            </p>
            <dl className="mt-2.5 space-y-2 font-ledger text-[0.78rem]">
              <div className="flex items-center gap-3 rounded-xl border border-border bg-muted/40 px-3 py-2.5">
                <dt className="flex w-[7.5rem] shrink-0 items-center gap-2 text-muted-foreground">
                  <UserRound className="h-3.5 w-3.5 shrink-0" />
                  {t("login.demoAccounts")}
                </dt>
                <dd className="border-l border-border pl-3">student / student</dd>
              </div>
              <div className="flex items-center gap-3 rounded-xl border border-border bg-muted/40 px-3 py-2.5">
                <dt className="flex w-[7.5rem] shrink-0 items-center gap-2 text-muted-foreground">
                  <KeyRound className="h-3.5 w-3.5 shrink-0" />
                  {t("login.adminAccount")}
                </dt>
                <dd className="border-l border-border pl-3">admin / admin</dd>
              </div>
            </dl>
          </div>
        </div>

        {/* Honest attribution: this is a course project, not a university publication, so it does
            not claim a Sabancı copyright. */}
        <p className="absolute inset-x-0 bottom-4 px-6 text-center font-ledger text-[0.66rem] text-muted-foreground/70 sm:px-12">
          {t("login.footer")}
        </p>
      </section>
    </div>
  );
}
