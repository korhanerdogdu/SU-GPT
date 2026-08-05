from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_locale_provider_and_language_switch_cover_every_user_facing_surface():
    main_source = (REPO_ROOT / "frontend/src/main.tsx").read_text(encoding="utf-8")
    resources = (REPO_ROOT / "frontend/src/localization/resources.ts").read_text(encoding="utf-8")
    assert "<LocaleProvider>" in main_source
    assert 'export type Locale = "tr" | "en"' in resources
    assert "const en: Record<keyof typeof tr, string>" in resources

    surfaces = [
        "frontend/src/pages/LoginPage.tsx",
        "frontend/src/pages/SignupPage.tsx",
        "frontend/src/pages/ProfilePage.tsx",
        "frontend/src/pages/CoursesPage.tsx",
        "frontend/src/pages/SchedulePage.tsx",
        "frontend/src/pages/CourseReviewPolicyPage.tsx",
        "frontend/src/components/chat/ChatHeader.tsx",
    ]
    for relative_path in surfaces:
        source = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        assert "LanguageToggle" in source, relative_path
        assert "useLocale" in source, relative_path


def test_course_review_form_is_feature_gated_accessible_and_fully_localized():
    page = (REPO_ROOT / "frontend/src/pages/CourseReviewPolicyPage.tsx").read_text(encoding="utf-8")
    resources = (REPO_ROOT / "frontend/src/localization/resources.ts").read_text(encoding="utf-8")
    api = (REPO_ROOT / "frontend/src/lib/api.ts").read_text(encoding="utf-8")

    assert "policy?.enabled" in page
    assert "COURSE_REVIEW_DIMENSIONS.map" in page
    assert '<fieldset key={dimension}' in page
    assert 'type="radio"' in page and 'type="checkbox"' in page
    assert 'aria-label={t("review.scoreLabel"' in page
    assert "review.courseOnly" in page and "review.privacyLink" in page
    assert "aggregate.available" in page and "review.aggregateSuppressed" in page
    assert "deleteMyCourseReview" in page
    assert 'consent_version: "course-review-v1"' in page
    assert "submitCourseReview" in api and "fetchCourseReviewAggregate" in api

    for key in (
        "review.title",
        "review.difficulty",
        "review.workload",
        "review.learningValue",
        "review.organization",
        "review.satisfaction",
        "review.consent",
        "review.courseOnly",
        "review.aggregateSuppressed",
        "review.delete",
    ):
        assert resources.count(f'"{key}"') >= 2, key
