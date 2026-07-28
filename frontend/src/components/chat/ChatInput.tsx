import { useState, type KeyboardEvent } from "react";
import { ArrowUp } from "lucide-react";

interface Props {
  onSend: (text: string) => void;
  disabled?: boolean;
  /** "docked" pins the composer to the bottom bar; "hero" renders it plain for the centered
   *  empty state (Claude/Gemini style) with a softer, floating shadow. */
  variant?: "docked" | "hero";
  autoFocus?: boolean;
}

export default function ChatInput({ onSend, disabled, variant = "docked", autoFocus }: Props) {
  const [value, setValue] = useState("");

  function submit() {
    const text = value.trim();
    if (!text || disabled) return;
    onSend(text);
    setValue("");
  }

  function handleKey(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  }

  const hero = variant === "hero";

  const box = (
    <div
      className={`flex items-end gap-2 rounded-2xl border border-border bg-background px-4 transition-all focus-within:border-primary ${
        hero
          ? "py-3 shadow-[0_18px_50px_-18px_rgba(2,20,45,0.55)] focus-within:shadow-[0_20px_60px_-16px_rgba(0,75,147,0.5)]"
          : "py-2.5 shadow-sm"
      }`}
    >
      <textarea
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={handleKey}
        rows={1}
        autoFocus={autoFocus}
        placeholder="Mezuniyetini sor veya “CS 201’i aldım” yaz…"
        disabled={disabled}
        className={`flex-1 resize-none bg-transparent text-foreground placeholder:text-muted-foreground focus:outline-none disabled:opacity-50 ${
          hero ? "py-1.5 text-[0.95rem]" : "py-1.5 text-sm"
        }`}
        style={{ maxHeight: 160 }}
      />
      <button
        type="button"
        onClick={submit}
        disabled={!value.trim() || disabled}
        aria-label="Gönder"
        className={`flex shrink-0 items-center justify-center rounded-xl bg-primary text-primary-foreground transition-colors hover:opacity-90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-40 ${
          hero ? "h-10 w-10" : "h-9 w-9"
        }`}
      >
        <ArrowUp className={hero ? "h-5 w-5" : "h-4 w-4"} />
      </button>
    </div>
  );

  const hint = (
    <p className="mt-2 text-center font-mono text-[9px] uppercase tracking-[0.16em] text-muted-foreground">
      Enter: gönder · Shift + Enter: yeni satır
    </p>
  );

  if (hero) {
    return (
      <div className="w-full">
        {box}
        {hint}
      </div>
    );
  }

  return (
    <div className="border-t border-border bg-card/80 px-4 py-4 backdrop-blur-xl md:px-8">
      <div className="mx-auto max-w-4xl">{box}</div>
      <div className="mx-auto max-w-4xl px-1">{hint}</div>
    </div>
  );
}
