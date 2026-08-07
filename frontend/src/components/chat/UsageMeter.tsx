import { useLocale } from "@/contexts/LocaleContext";
import type { UsageStatus } from "@/lib/api";

/**
 * Daily allowance as a fill bar — the "Limit" pattern from Claude's own usage indicator, so the
 * number reads as a level (how close to the wall) rather than a bare fraction.
 *
 * Shown in the chat header so the level is visible before the student runs out, not as a
 * surprise afterwards. Admins are exempt from the quota and see nothing — a bar that never moves
 * is noise.
 */
export default function UsageMeter({ usage }: { usage: UsageStatus | null }) {
  const { t } = useLocale();
  if (!usage || usage.exempt) return null;

  const { limit, remaining } = usage;
  const ratio = limit > 0 ? remaining / limit : 0;
  const usedPct = limit > 0 ? Math.min(100, Math.round(((limit - remaining) / limit) * 100)) : 0;

  const level: "ok" | "warn" | "critical" =
    remaining === 0 ? "critical" : ratio <= 0.25 ? "warn" : "ok";
  const barCls = { ok: "bg-primary-emphasis", warn: "bg-warning", critical: "bg-destructive" }[level];
  const textCls = { ok: "text-muted-foreground", warn: "text-warning", critical: "text-destructive-emphasis" }[level];
  const borderCls = { ok: "border-border", warn: "border-warning/40", critical: "border-destructive/40" }[level];

  return (
    <div
      title={t("usage.tooltip", { limit })}
      // Fixed h-9 to match the p-2/h-4-icon height of ThemeToggle/HelpButton/LanguageToggle
      // exactly, rather than letting two stacked text rows grow the chip taller than its
      // neighbours in the same header row.
      className={`hidden h-9 w-28 shrink-0 flex-col justify-center gap-0.5 rounded-lg border bg-background px-2.5 shadow-sm sm:flex ${borderCls}`}
    >
      <div className="flex items-center justify-between gap-2">
        <span className="text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
          {t("usage.label")}
        </span>
        <span className={`text-[11px] font-bold tabular-nums ${textCls}`}>{usedPct}%</span>
      </div>
      <div className="h-1 w-full overflow-hidden rounded-full bg-muted">
        <div
          className={`h-full rounded-full transition-[width] duration-300 ${barCls}`}
          style={{ width: `${usedPct}%` }}
        />
      </div>
    </div>
  );
}
