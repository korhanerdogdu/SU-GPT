import { getStoredLocale, localizedContent, type Locale } from "@/localization/resources";

export function sampleQuestions(locale: Locale): readonly string[] {
  return localizedContent.sampleQuestions(locale);
}

/** @deprecated Read from the live LocaleContext; retained for compatibility with old imports. */
export const SAMPLE_QUESTIONS = localizedContent.sampleQuestions(getStoredLocale());
