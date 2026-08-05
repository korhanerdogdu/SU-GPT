import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { getStoredLocale, translate, type Locale, type TranslationKey } from "@/localization/resources";

interface LocaleContextValue {
  locale: Locale;
  setLocale: (locale: Locale) => void;
  toggleLocale: () => void;
  t: (key: TranslationKey, values?: Record<string, string | number>) => string;
}

const LocaleContext = createContext<LocaleContextValue | undefined>(undefined);

export function LocaleProvider({ children }: { children: ReactNode }) {
  const [locale, setLocale] = useState<Locale>(getStoredLocale);
  useEffect(() => {
    localStorage.setItem("advisu-locale", locale);
    document.documentElement.lang = locale;
  }, [locale]);
  const value = useMemo<LocaleContextValue>(() => ({
    locale,
    setLocale,
    toggleLocale: () => setLocale((current) => current === "tr" ? "en" : "tr"),
    t: (key, values) => translate(locale, key, values),
  }), [locale]);
  return <LocaleContext.Provider value={value}>{children}</LocaleContext.Provider>;
}

export function useLocale(): LocaleContextValue {
  const value = useContext(LocaleContext);
  if (!value) throw new Error("useLocale must be used inside LocaleProvider");
  return value;
}
