import { AlertTriangle, Menu, Moon, Sun } from "lucide-react";
import { Link } from "react-router-dom";
import { useTheme } from "@/contexts/ThemeContext";

interface ChatHeaderProps {
  profileReady: boolean;
  onOpenMenu: () => void;
}

export default function ChatHeader({ profileReady, onOpenMenu }: ChatHeaderProps) {
  const { resolvedTheme, cycleTheme, preference } = useTheme();
  return (
    <header className="border-b border-border bg-card/90 backdrop-blur-xl">
      <div className="flex items-center gap-3 px-4 py-3 md:px-6">
        <button
          type="button"
          onClick={onOpenMenu}
          className="rounded-lg p-2 text-muted-foreground hover:bg-muted hover:text-foreground md:hidden"
          aria-label="Sohbet geçmişini aç"
        >
          <Menu className="h-5 w-5" />
        </button>
        <div className="min-w-0 flex-1">
          <h1 className="truncate text-sm font-semibold tracking-tight text-foreground">
            Akademik danışman
          </h1>
          <p className="mt-0.5 truncate text-xs text-muted-foreground">
            Resmî Sabancı müfredatı ve kişisel ders geçmişinle yanıtlar
          </p>
        </div>
        <button
          type="button"
          onClick={cycleTheme}
          title={`Tema: ${preference}`}
          aria-label="Temayı değiştir"
          className="rounded-lg border border-border bg-background p-2 text-muted-foreground hover:bg-muted hover:text-foreground"
        >
          {resolvedTheme === "dark" ? (
            <Moon className="h-4 w-4" />
          ) : (
            <Sun className="h-4 w-4" />
          )}
        </button>
      </div>

      {!profileReady && (
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5 border-t border-amber-400/30 bg-amber-400/10 px-4 py-2.5 md:px-6">
          <AlertTriangle className="h-3.5 w-3.5 shrink-0 text-amber-600" />
          <p className="text-xs leading-relaxed text-foreground">
            Kişisel mezuniyet hesabı için bölümünü ve müfredat dönemini kaydet.
          </p>
          <Link
            to="/profile"
            className="rounded-md border border-amber-500/40 px-2 py-0.5 text-xs font-medium text-foreground hover:bg-amber-400/15"
          >
            Profili aç
          </Link>
        </div>
      )}
    </header>
  );
}
