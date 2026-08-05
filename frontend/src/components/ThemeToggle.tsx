import { Moon, Sun } from "lucide-react";
import { useTheme } from "@/contexts/ThemeContext";
import { useLocale } from "@/contexts/LocaleContext";

export default function ThemeToggle({ className = "" }: { className?: string }) {
  const { resolvedTheme, cycleTheme, preference } = useTheme();
  const { t } = useLocale();
  return (
    <button
      type="button"
      onClick={cycleTheme}
      title={t("theme.title", { preference })}
      aria-label={t("theme.change")}
      className={`rounded-lg border border-border bg-background p-2 text-muted-foreground shadow-sm hover:bg-muted hover:text-foreground ${className}`}
    >
      {resolvedTheme === "dark" ? <Moon className="h-4 w-4" /> : <Sun className="h-4 w-4" />}
    </button>
  );
}
