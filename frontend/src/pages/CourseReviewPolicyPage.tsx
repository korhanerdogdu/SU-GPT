import { useEffect, useMemo, useState, type FormEvent } from "react";
import { Link } from "react-router-dom";
import LanguageToggle from "@/components/LanguageToggle";
import ThemeToggle from "@/components/ThemeToggle";
import HelpButton from "@/components/HelpButton";
import { useAuth } from "@/contexts/AuthContext";
import { useLocale } from "@/contexts/LocaleContext";
import type { TranslationKey } from "@/localization/resources";
import {
  COURSE_REVIEW_DIMENSIONS,
  deleteMyCourseReview,
  fetchCourseReviewAggregate,
  fetchCourseReviewPolicy,
  submitCourseReview,
  type CourseReviewAggregate,
  type CourseReviewDimension,
  type CourseReviewPolicy,
} from "@/lib/api";

const DIMENSION_KEYS: Record<CourseReviewDimension, TranslationKey> = {
  difficulty: "review.difficulty",
  workload: "review.workload",
  learning_value: "review.learningValue",
  organization: "review.organization",
  overall_satisfaction: "review.satisfaction",
};

export default function CourseReviewPolicyPage() {
  const { locale, t } = useLocale();
  const { isAuthenticated } = useAuth();
  const [policy, setPolicy] = useState<CourseReviewPolicy | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState(false);
  const [courseCode, setCourseCode] = useState("");
  const [comment, setComment] = useState("");
  const [consent, setConsent] = useState(false);
  const [ratings, setRatings] = useState<Record<CourseReviewDimension, number>>(() =>
    Object.fromEntries(COURSE_REVIEW_DIMENSIONS.map((name) => [name, 3])) as Record<CourseReviewDimension, number>
  );
  const [submitting, setSubmitting] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [feedback, setFeedback] = useState<TranslationKey | null>(null);
  const [aggregate, setAggregate] = useState<CourseReviewAggregate | null>(null);
  const [aggregateState, setAggregateState] = useState<"idle" | "loading" | "error">("idle");

  useEffect(() => {
    let active = true;
    setLoading(true);
    setLoadError(false);
    fetchCourseReviewPolicy(locale)
      .then((value) => { if (active) setPolicy(value); })
      .catch(() => { if (active) setLoadError(true); })
      .finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, [locale]);

  const normalizedCourseCode = useMemo(() => courseCode.trim().toUpperCase(), [courseCode]);

  async function refreshAggregate(code: string) {
    setAggregateState("loading");
    try {
      setAggregate(await fetchCourseReviewAggregate(code, locale));
      setAggregateState("idle");
    } catch {
      setAggregate(null);
      setAggregateState("error");
    }
  }

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    if (!consent || !normalizedCourseCode) return;
    setSubmitting(true);
    setFeedback(null);
    try {
      const result = await submitCourseReview({
        course_code: normalizedCourseCode,
        ...ratings,
        comment,
        consent: true,
        consent_version: "course-review-v1",
      }, locale);
      setFeedback(`review.${result.review.moderationState}` as TranslationKey);
      await refreshAggregate(normalizedCourseCode);
    } catch {
      setFeedback("review.submitError");
    } finally {
      setSubmitting(false);
    }
  }

  async function onDelete() {
    if (!normalizedCourseCode) return;
    setDeleting(true);
    try {
      await deleteMyCourseReview(normalizedCourseCode, locale);
      setFeedback("review.deleted");
      await refreshAggregate(normalizedCourseCode);
    } catch {
      setFeedback("review.deleteError");
    } finally {
      setDeleting(false);
    }
  }

  const points = [1, 2, 3, 4, 5, 6].map((number) =>
    t(`policy.point${number}` as TranslationKey)
  );
  return (
    <main className="min-h-screen bg-background px-5 py-12 text-foreground">
      <article id="privacy" className="mx-auto max-w-2xl rounded-2xl border border-border bg-card p-7 shadow-sm">
        <div className="mb-6 flex items-center justify-between gap-3">
          <h1 className="text-2xl font-bold text-foreground">{t("policy.title")}</h1>
          <div className="flex items-center gap-2">
            <HelpButton />
            <LanguageToggle />
            <ThemeToggle />
          </div>
        </div>
        {loading && <p role="status">{t("common.loading")}</p>}
        {loadError && <p role="alert" className="rounded-md bg-destructive/10 p-3 text-destructive-emphasis">{t("review.loadError")}</p>}
        {!loading && !loadError && !policy?.enabled && (
          <p className="rounded-md bg-warning/10 p-3 font-medium text-warning">{t("policy.status")}</p>
        )}
        <p className="mt-5">{t("policy.body")}</p>
        <ul className="mt-5 list-disc space-y-2 pl-6">
          {points.map((point) => <li key={point}>{point}</li>)}
        </ul>

        {policy?.enabled && (
          <section className="mt-8 border-t border-border pt-7" aria-labelledby="review-form-title">
            <h2 id="review-form-title" className="text-xl font-bold text-foreground">{t("review.title")}</h2>
            <p className="mt-2 rounded-md bg-primary/10 p-3 text-primary-emphasis">{t("review.courseOnly")}</p>
            <a href="#privacy" className="mt-2 inline-block text-sm text-primary-emphasis underline">{t("review.privacyLink")}</a>
            {!isAuthenticated ? (
              <p className="mt-5"><Link to="/login" className="text-primary-emphasis underline">{t("review.signIn")}</Link></p>
            ) : (
              <form className="mt-5 space-y-5" onSubmit={onSubmit}>
                <label className="block font-medium">
                  {t("review.courseCode")}
                  <input
                    required
                    value={courseCode}
                    onChange={(event) => setCourseCode(event.target.value)}
                    placeholder={t("review.courseCodePlaceholder")}
                    maxLength={16}
                    pattern="[A-Za-z]{2,6}[ -]?[0-9]{3,5}[A-Za-z]?"
                    className="mt-1 block w-full rounded-lg border border-input bg-background px-3 py-2 text-foreground focus:outline-none focus:ring-2 focus:ring-primary/40"
                  />
                </label>
                {COURSE_REVIEW_DIMENSIONS.map((dimension) => (
                  <fieldset key={dimension} className="rounded-lg border border-border p-3">
                    <legend className="px-1 font-medium">{t(DIMENSION_KEYS[dimension])}</legend>
                    <div className="flex flex-wrap gap-3">
                      {[1, 2, 3, 4, 5].map((score) => (
                        <label
                          key={score}
                          className="flex min-h-[2rem] min-w-[2rem] cursor-pointer items-center justify-center gap-1.5 rounded-lg border border-border px-2.5 py-1 has-[:checked]:border-primary-emphasis has-[:checked]:bg-primary/10 has-[:checked]:text-primary-emphasis hover:bg-muted"
                        >
                          <input
                            type="radio"
                            name={dimension}
                            value={score}
                            checked={ratings[dimension] === score}
                            onChange={() => setRatings((current) => ({ ...current, [dimension]: score }))}
                            aria-label={t("review.scoreLabel", { dimension: t(DIMENSION_KEYS[dimension]), score })}
                          />
                          {score}
                        </label>
                      ))}
                    </div>
                  </fieldset>
                ))}
                <label className="block font-medium">
                  {t("review.comment")}
                  <textarea
                    value={comment}
                    onChange={(event) => setComment(event.target.value)}
                    placeholder={t("review.commentPlaceholder")}
                    maxLength={1000}
                    rows={4}
                    className="mt-1 block w-full rounded-lg border border-input bg-background px-3 py-2 text-foreground focus:outline-none focus:ring-2 focus:ring-primary/40"
                  />
                </label>
                <label className="flex items-start gap-2">
                  <input type="checkbox" checked={consent} onChange={(event) => setConsent(event.target.checked)} required />
                  <span>{t("review.consent")}</span>
                </label>
                <div className="flex flex-wrap gap-3">
                  <button
                    type="submit"
                    disabled={submitting || !consent}
                    className="rounded-lg bg-primary px-4 py-2 font-medium text-primary-foreground disabled:opacity-50"
                  >
                    {t(submitting ? "review.submitting" : "review.submit")}
                  </button>
                  {feedback && !["review.submitError", "review.deleteError", "review.deleted"].includes(feedback) && (
                    <button
                      type="button"
                      disabled={deleting}
                      onClick={onDelete}
                      className="rounded-lg border border-destructive/40 px-4 py-2 text-destructive-emphasis disabled:opacity-50"
                    >
                      {t(deleting ? "review.deleting" : "review.delete")}
                    </button>
                  )}
                </div>
                {feedback && <p role="status">{t(feedback)}</p>}
              </form>
            )}

            {isAuthenticated && aggregateState !== "idle" && (
              <p role="status" className="mt-5">
                {t(aggregateState === "loading" ? "review.aggregateLoading" : "review.aggregateError")}
              </p>
            )}
            {isAuthenticated && aggregateState === "idle" && aggregate && (
              <section className="mt-5 rounded-lg bg-muted/40 p-4" aria-labelledby="aggregate-title">
                <h3 id="aggregate-title" className="font-bold">{t("review.aggregateTitle")}</h3>
                {!aggregate.available ? (
                  <p className="mt-2">{t("review.aggregateSuppressed")}</p>
                ) : (
                  <div className="mt-2">
                    <p>{t("review.reviewCount", { count: aggregate.reviewCount ?? 0 })}</p>
                    <dl className="mt-3 grid grid-cols-1 gap-2 sm:grid-cols-2">
                      {COURSE_REVIEW_DIMENSIONS.map((dimension) => (
                        <div key={dimension} className="flex justify-between gap-3">
                          <dt>{t(DIMENSION_KEYS[dimension])}</dt>
                          <dd>{aggregate.averages?.[dimension]?.toFixed(2) ?? "—"}</dd>
                        </div>
                      ))}
                    </dl>
                  </div>
                )}
              </section>
            )}
          </section>
        )}
        <Link to="/login" className="mt-7 inline-block font-medium text-primary-emphasis underline">
          {t("policy.back")}
        </Link>
      </article>
    </main>
  );
}
