import { Languages } from "lucide-react";
import { useLocale } from "@/contexts/LocaleContext";

export default function LanguageToggle({ className = "" }: { className?: string }) {
  const { toggleLocale, t } = useLocale();
  return (
    <button
      type="button"
      onClick={toggleLocale}
      title={t("language.change")}
      aria-label={t("language.change")}
      className={`inline-flex items-center gap-1 rounded-lg border border-border bg-background px-2 py-1.5 text-xs font-bold text-muted-foreground shadow-sm hover:bg-muted hover:text-foreground ${className}`}
    >
      <Languages className="h-4 w-4" /> {t("language.short")}
    </button>
  );
}
