import { Compass, GraduationCap, LibraryBig, CalendarRange } from "lucide-react";
import ChatInput from "./ChatInput";

interface Props {
  name?: string;
  onSend: (text: string) => void;
  disabled?: boolean;
}

/** Conversation starters. The label is sent verbatim, so each one is phrased as a real question
 *  that the router recognises (graduation audit / recommendation / major selection / minor). */
const STARTERS: { label: string; icon: typeof GraduationCap }[] = [
  { label: "Mezuniyet durumumu hesapla", icon: GraduationCap },
  { label: "Ders programı yap", icon: CalendarRange },
  { label: "Hangi bölümü seçmeliyim?", icon: Compass },
  { label: "Yandal için hangi dersler gerekli?", icon: LibraryBig },
];

function displayName(name?: string): string {
  if (!name) return "";
  return name.charAt(0).toUpperCase() + name.slice(1);
}

export default function EmptyState({ name, onSend, disabled }: Props) {
  const who = displayName(name);
  return (
    <div className="relative flex h-full flex-col items-center justify-center overflow-hidden px-4 py-10">
      {/* Soft radial glow behind the composer — the "shadow/aura" register from the reference
          apps. Purely decorative, so it is aria-hidden and never intercepts clicks. */}
      <div aria-hidden="true" className="pointer-events-none absolute inset-0">
        <div className="absolute left-1/2 top-1/2 h-[42rem] w-[42rem] -translate-x-1/2 -translate-y-1/2 rounded-full bg-[radial-gradient(circle,_rgba(214,161,58,0.16)_0%,_rgba(0,75,147,0.12)_38%,_transparent_70%)] blur-2xl" />
      </div>

      <div className="relative flex w-full max-w-2xl flex-col items-center text-center">
        <img
          src="/assets/small_witihoutbg.png"
          alt="adviSU"
          className="w-20 drop-shadow-[0_10px_30px_rgba(2,20,45,0.35)] sm:w-24"
        />

        <h1 className="mt-6 text-[2rem] font-semibold leading-tight tracking-tight text-foreground sm:text-[2.5rem]">
          Merhaba{who ? `, ${who}` : ""}
        </h1>
        <p className="mt-3 max-w-md text-[1.02rem] leading-relaxed text-muted-foreground">
          Mezuniyet yolculuğunda bugün sana nasıl yardımcı olabilirim?
        </p>

        <div className="mt-8 w-full">
          <ChatInput variant="hero" onSend={onSend} disabled={disabled} autoFocus />
        </div>

        <div className="mt-6 flex flex-wrap items-center justify-center gap-2.5">
          {STARTERS.map(({ label, icon: Icon }) => (
            <button
              key={label}
              type="button"
              onClick={() => onSend(label)}
              disabled={disabled}
              className="group flex items-center gap-2 rounded-full border border-border bg-card px-4 py-2 text-sm text-foreground shadow-sm transition-all hover:-translate-y-0.5 hover:border-primary/40 hover:shadow-md disabled:cursor-not-allowed disabled:opacity-50"
            >
              <Icon className="h-4 w-4 text-primary transition-transform group-hover:scale-110" />
              {label}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
