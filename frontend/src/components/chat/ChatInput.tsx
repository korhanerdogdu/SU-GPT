import { useState, type KeyboardEvent } from "react";
import { ArrowUp } from "lucide-react";

interface Props {
  onSend: (text: string) => void;
  disabled?: boolean;
}

export default function ChatInput({ onSend, disabled }: Props) {
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

  return (
    <div className="border-t border-border bg-card/80 px-4 py-4 backdrop-blur-xl md:px-8">
      <div className="mx-auto flex max-w-4xl items-end gap-2 rounded-2xl border border-border bg-background px-4 py-2.5 shadow-sm transition-colors focus-within:border-primary">
        <textarea
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKey}
          rows={1}
          placeholder="Mezuniyetini sor veya “CS 201’i aldım” yaz…"
          disabled={disabled}
          className="flex-1 resize-none bg-transparent py-1.5 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none disabled:opacity-50"
          style={{ maxHeight: 120 }}
        />
        <button
          type="button"
          onClick={submit}
          disabled={!value.trim() || disabled}
          aria-label="Send message"
          className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-primary text-primary-foreground transition-colors hover:opacity-90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-40"
        >
          <ArrowUp className="h-4 w-4" />
        </button>
      </div>
      <p className="mx-auto mt-2 max-w-4xl px-1 text-center font-mono text-[9px] uppercase tracking-[0.16em] text-muted-foreground">
        Enter: gönder · Shift + Enter: yeni satır
      </p>
    </div>
  );
}
