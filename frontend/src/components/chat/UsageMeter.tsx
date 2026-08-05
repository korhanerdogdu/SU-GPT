import { MessageCircle } from "lucide-react";
import { useLocale } from "@/contexts/LocaleContext";
import type { UsageStatus } from "@/lib/api";

/**
 * "N of M questions left today".
 *
 * Shown in the chat header so the number is visible before the student runs out, not as a
 * surprise afterwards. Admins are exempt from the quota and see nothing — a counter that never
 * moves is noise.
 */
export default function UsageMeter({ usage }: { usage: UsageStatus | null }) {
  const { t } = useLocale();
  if (!usage || usage.exempt) return null;

  const { limit, remaining } = usage;
  const ratio = limit > 0 ? remaining / limit : 0;
  const tone =
    remaining === 0
      ? "border-destructive/40 bg-destructive/10 text-destructive"
      : ratio <= 0.25
        ? "border-warning/40 bg-warning/10 text-warning"
        : "border-border bg-background text-muted-foreground";

  return (
    <span
      title={t("usage.tooltip", { limit })}
      className={`hidden items-center gap-1.5 whitespace-nowrap rounded-lg border px-2.5 py-2 text-xs font-medium sm:inline-flex ${tone}`}
    >
      <MessageCircle className="h-3.5 w-3.5 shrink-0" />
      {remaining === 0 ? t("usage.none") : t("usage.remaining", { remaining, limit })}
    </span>
  );
}
