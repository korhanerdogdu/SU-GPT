import { useState, type KeyboardEvent } from "react";
import { ArrowUp } from "lucide-react";
import { useLocale } from "@/contexts/LocaleContext";

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
  const { t } = useLocale();

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

  // Focus lives on the wrapping box, not the textarea itself, so the whole composer — including
  // the send button — reads as one focused control (WCAG 2.4.13 Focus Appearance): a real
  // 2px ring in the `--ring` token, softened shadow underneath as a secondary cue.
  const box = (
    <div
      className={`flex items-end gap-2 rounded-2xl border border-border bg-background px-4 transition-shadow focus-within:border-ring focus-within:ring-2 focus-within:ring-ring/70 ${
        hero
          ? "py-3 shadow-[0_18px_50px_-18px_rgba(2,20,45,0.5)] focus-within:shadow-[0_22px_60px_-16px_rgba(0,75,147,0.4)]"
          : "py-2.5 shadow-sm focus-within:shadow-md"
      }`}
    >
      <textarea
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={handleKey}
        rows={1}
        autoFocus={autoFocus}
        placeholder={t("chat.placeholder")}
        disabled={disabled}
        className={`flex-1 resize-none bg-transparent text-foreground placeholder:text-muted-foreground focus:outline-none focus-visible:ring-0 focus-visible:ring-offset-0 disabled:opacity-50 ${
          hero ? "py-1.5 text-[0.95rem]" : "py-1.5 text-sm"
        }`}
        style={{ maxHeight: 160 }}
      />
      <button
        type="button"
        onClick={submit}
        disabled={!value.trim() || disabled}
        aria-label={t("chat.send")}
        className={`flex shrink-0 items-center justify-center rounded-xl bg-primary text-primary-foreground transition-colors hover:opacity-90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-40 ${
          hero ? "h-10 w-10" : "h-9 w-9"
        }`}
      >
        <ArrowUp className={hero ? "h-5 w-5" : "h-4 w-4"} />
      </button>
    </div>
  );

  if (hero) {
    return <div className="w-full">{box}</div>;
  }

  return (
    <div className="border-t border-border bg-card/80 px-4 py-4 backdrop-blur-xl md:px-8">
      <div className="mx-auto max-w-4xl">{box}</div>
    </div>
  );
}
