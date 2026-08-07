import {
  ArrowUpRight,
  CalendarRange,
  GraduationCap,
  ListChecks,
  TrendingUp,
} from "lucide-react";
import ChatInput from "./ChatInput";
import { useLocale } from "@/contexts/LocaleContext";
import { localizedContent } from "@/localization/resources";

interface Props {
  name?: string;
  onSend: (text: string) => void;
  disabled?: boolean;
  onFileSelect?: (file: File) => void;
  fileBusy?: boolean;
}

/** The visible wording is also a real router-recognised prompt, so clicking a starter keeps the
 * chat transcript natural while preserving the intended deterministic response type. */
const STARTER_ICONS = [GraduationCap, CalendarRange, ListChecks, TrendingUp];

function displayName(name?: string): string {
  if (!name) return "";
  return name.charAt(0).toUpperCase() + name.slice(1);
}

export default function EmptyState({ name, onSend, disabled, onFileSelect, fileBusy }: Props) {
  const who = displayName(name);
  const { locale, t } = useLocale();
  const starters = localizedContent.starters(locale).map(([label, prompt], index) => ({
    label, prompt, icon: STARTER_ICONS[index],
  }));
  return (
    <div className="relative flex h-full flex-col items-center justify-center overflow-hidden px-4 py-10">
      {/* Soft radial glow behind the composer — the "shadow/aura" register from the reference
          apps. Purely decorative, so it is aria-hidden and never intercepts clicks. */}
      <div aria-hidden="true" className="pointer-events-none absolute inset-0">
        <div className="absolute left-1/2 top-1/2 h-[42rem] w-[42rem] -translate-x-1/2 -translate-y-1/2 rounded-full bg-[radial-gradient(circle,_rgba(214,161,58,0.16)_0%,_rgba(0,75,147,0.12)_38%,_transparent_70%)] blur-2xl" />
      </div>

      <div className="relative flex w-full max-w-xl flex-col items-center text-center">
        <img
          src="/assets/small_witihoutbg.png"
          alt="adviSU"
          className="w-20 drop-shadow-[0_10px_30px_rgba(2,20,45,0.35)] sm:w-24"
        />

        <h1 className="mt-6 text-[2rem] font-semibold leading-tight tracking-tight text-foreground sm:text-[2.5rem]">
          {t("chat.greeting", { suffix: who ? `, ${who}` : "" })}
        </h1>
        <p className="mt-3 max-w-md text-[1.02rem] leading-relaxed text-muted-foreground">
          {t("chat.help")}
        </p>

        <div className="mt-8 w-full">
          <ChatInput
            variant="hero"
            onSend={onSend}
            disabled={disabled}
            autoFocus
            onFileSelect={onFileSelect}
            fileBusy={fileBusy}
          />
        </div>

        {/* Starters as a clean vertical list (ChatGPT-style): icon chip · label · hover arrow. */}
        <div className="mt-5 w-full space-y-2 text-left">
          {starters.map(({ label, prompt, icon: Icon }) => (
            <button
              key={label}
              type="button"
              onClick={() => onSend(prompt)}
              disabled={disabled}
              className="group flex w-full items-center gap-3 rounded-2xl border border-border/70 bg-card/50 px-4 py-3 text-sm text-foreground backdrop-blur-sm transition-all hover:border-primary/30 hover:bg-card hover:shadow-md disabled:cursor-not-allowed disabled:opacity-50"
            >
              <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-primary/10 text-primary-emphasis transition-colors group-hover:bg-primary/15">
                <Icon className="h-[1.05rem] w-[1.05rem]" />
              </span>
              <span className="flex-1 font-medium">{label}</span>
              <ArrowUpRight className="h-4 w-4 shrink-0 -translate-x-1 text-muted-foreground opacity-0 transition-all group-hover:translate-x-0 group-hover:opacity-100" />
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
