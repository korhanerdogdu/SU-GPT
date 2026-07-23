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
    <div className="border-t border-white/10 bg-[#0a1830]/60 px-4 py-4 backdrop-blur-xl md:px-8">
      <div className="mx-auto flex max-w-3xl items-end gap-2 rounded-2xl border border-white/12 bg-white/[0.05] px-4 py-2.5 transition-colors focus-within:border-sabanci-gold/50 focus-within:bg-white/[0.07]">
        <textarea
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKey}
          rows={1}
          placeholder="Ask about your degree…"
          disabled={disabled}
          className="flex-1 resize-none bg-transparent py-1.5 text-sm text-white placeholder:text-sabanci-light/40 focus:outline-none disabled:opacity-50"
          style={{ maxHeight: 120 }}
        />
        <button
          type="button"
          onClick={submit}
          disabled={!value.trim() || disabled}
          aria-label="Send message"
          className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-sabanci-blue text-white transition-colors hover:bg-sabanci-blue/80 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sabanci-gold disabled:cursor-not-allowed disabled:opacity-40"
        >
          <ArrowUp className="h-4 w-4" />
        </button>
      </div>
      <p className="mx-auto mt-2 max-w-3xl px-1 font-mono text-[10px] uppercase tracking-[0.18em] text-sabanci-light/30">
        Enter to send · Shift + Enter for a new line
      </p>
    </div>
  );
}
