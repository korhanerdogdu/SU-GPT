import { useEffect, useRef, useState } from "react";
import { HelpCircle } from "lucide-react";
import { useLocale } from "@/contexts/LocaleContext";

/**
 * Same icon, same relative position, on every page that has ThemeToggle/LanguageToggle
 * (WCAG 3.2.6 Consistent Help) — a short, generic "how this works" panel rather than
 * page-specific docs, since the report only requires the entry point to stay put.
 */
export default function HelpButton({ className = "" }: { className?: string }) {
  const { t } = useLocale();
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") setOpen(false);
    }
    function onClickOutside(e: MouseEvent) {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) setOpen(false);
    }
    window.addEventListener("keydown", onKeyDown);
    document.addEventListener("mousedown", onClickOutside);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      document.removeEventListener("mousedown", onClickOutside);
    };
  }, [open]);

  const tips = [t("help.tip1"), t("help.tip2"), t("help.tip3"), t("help.tip4")];

  return (
    <div ref={rootRef} className={`relative ${className}`}>
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-haspopup="dialog"
        aria-expanded={open}
        aria-label={t("help.button")}
        title={t("help.button")}
        className="rounded-lg border border-border bg-background p-2 text-muted-foreground shadow-sm hover:bg-muted hover:text-foreground"
      >
        <HelpCircle className="h-4 w-4" />
      </button>
      {open && (
        <div
          role="dialog"
          aria-label={t("help.title")}
          className="absolute right-0 top-full z-50 mt-2 w-72 rounded-xl border border-border bg-card p-4 text-card-foreground shadow-xl"
        >
          <div className="flex items-center justify-between gap-3">
            <h2 className="text-sm font-semibold">{t("help.title")}</h2>
            <button
              type="button"
              onClick={() => setOpen(false)}
              aria-label={t("help.close")}
              className="grid h-6 w-6 shrink-0 place-items-center rounded text-muted-foreground hover:bg-muted hover:text-foreground"
            >
              ×
            </button>
          </div>
          <ul className="mt-3 space-y-2 text-xs leading-relaxed text-muted-foreground">
            {tips.map((tip) => (
              <li key={tip} className="flex gap-2">
                <span aria-hidden="true" className="mt-1 h-1 w-1 shrink-0 rounded-full bg-primary" />
                {tip}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
