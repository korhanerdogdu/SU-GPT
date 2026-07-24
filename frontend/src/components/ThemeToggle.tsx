import { Moon, Sun } from "lucide-react";
import { useTheme } from "@/contexts/ThemeContext";

export default function ThemeToggle({ className = "" }: { className?: string }) {
  const { resolvedTheme, cycleTheme, preference } = useTheme();
  return (
    <button
      type="button"
      onClick={cycleTheme}
      title={`Tema: ${preference}`}
      aria-label="Temayı değiştir"
      className={`rounded-lg border border-border bg-background p-2 text-muted-foreground shadow-sm hover:bg-muted hover:text-foreground ${className}`}
    >
      {resolvedTheme === "dark" ? <Moon className="h-4 w-4" /> : <Sun className="h-4 w-4" />}
    </button>
  );
}
