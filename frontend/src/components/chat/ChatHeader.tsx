import { AlertTriangle, Menu, Moon, Sun } from "lucide-react";
import { Link } from "react-router-dom";
import { useTheme } from "@/contexts/ThemeContext";
import { useLocale } from "@/contexts/LocaleContext";
import LanguageToggle from "@/components/LanguageToggle";
import HelpButton from "@/components/HelpButton";
import UsageMeter from "./UsageMeter";
import type { UsageStatus } from "@/lib/api";

interface ChatHeaderProps {
  profileReady: boolean;
  onOpenMenu: () => void;
  usage: UsageStatus | null;
}

export default function ChatHeader({ profileReady, onOpenMenu, usage }: ChatHeaderProps) {
  const { resolvedTheme, cycleTheme, preference } = useTheme();
  const { t } = useLocale();
  return (
    <header className="border-b border-border bg-card/90 backdrop-blur-xl">
      <div className="flex items-center gap-3 px-4 py-3 md:px-6">
        {/* Sidebar toggle sits at the leading edge — the one place every chat UI convention
            (and Fitts's law) says an edge-anchored menu control belongs. */}
        <button
          type="button"
          onClick={onOpenMenu}
          className="rounded-lg p-2 text-muted-foreground hover:bg-muted hover:text-foreground md:hidden"
          aria-label={t("chat.openHistory")}
        >
          <Menu className="h-5 w-5" />
        </button>
        <div className="min-w-0 flex-1">
          <h1 className="truncate text-sm font-semibold tracking-tight text-foreground">
            {t("chat.header")}
          </h1>
          <p className="mt-0.5 truncate text-xs text-muted-foreground">
            {t("chat.subtitle")}
          </p>
        </div>
        <UsageMeter usage={usage} />
        <button
          type="button"
          onClick={cycleTheme}
          title={t("theme.title", { preference })}
          aria-label={t("theme.change")}
          className="rounded-lg border border-border bg-background p-2 text-muted-foreground hover:bg-muted hover:text-foreground"
        >
          {resolvedTheme === "dark" ? (
            <Moon className="h-4 w-4" />
          ) : (
            <Sun className="h-4 w-4" />
          )}
        </button>
        <HelpButton />
        <LanguageToggle />
      </div>

      {!profileReady && (
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5 border-t border-warning/30 bg-warning/10 px-4 py-2.5 md:px-6">
          <AlertTriangle className="h-3.5 w-3.5 shrink-0 text-warning" />
          <p className="text-xs leading-relaxed text-foreground">
            {t("chat.profileNotice")}
          </p>
          <Link
            to="/profile"
            className="rounded-md border border-warning/40 px-2 py-0.5 text-xs font-medium text-foreground hover:bg-warning/15"
          >
            {t("chat.openProfile")}
          </Link>
        </div>
      )}
    </header>
  );
}
