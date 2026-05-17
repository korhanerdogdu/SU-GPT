import { useEffect, useRef } from "react";
import { Loader2, Sparkles } from "lucide-react";

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: string[];
  pending?: boolean;
}

export default function ChatMessages({ messages }: { messages: Message[] }) {
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  if (messages.length === 0) {
    return (
      <div className="flex h-full flex-col items-center justify-center px-6 text-center">
        <div className="mb-4 rounded-full border border-border bg-card/50 p-3 backdrop-blur-sm">
          <Sparkles className="h-6 w-6 text-primary" />
        </div>
        <h2 className="text-xl font-semibold">Ask about your course materials</h2>
        <p className="mt-2 max-w-md text-sm text-muted-foreground">
          Upload syllabi, lecture slides, or project descriptions in the sidebar
          and ask grounded questions. SU-GPT will cite the source.
        </p>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4 px-4 py-6 md:px-8">
      {messages.map((m) => (
        <MessageBubble key={m.id} message={m} />
      ))}
      <div ref={endRef} />
    </div>
  );
}

function MessageBubble({ message }: { message: Message }) {
  const isUser = message.role === "user";
  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-[80%] rounded-2xl border border-border px-4 py-3 text-sm leading-relaxed shadow-md backdrop-blur-sm ${
          isUser
            ? "bg-primary/95 text-primary-foreground rounded-br-sm"
            : "bg-card/85 text-card-foreground rounded-bl-sm"
        }`}
      >
        {message.pending ? (
          <div className="flex items-center gap-2 text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" />
            <span>SU-GPT is thinking…</span>
          </div>
        ) : (
          <div className="whitespace-pre-wrap">{message.content}</div>
        )}

        {!message.pending && message.sources && message.sources.length > 0 && (
          <div className="mt-3 border-t border-border/60 pt-2">
            <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
              Sources
            </div>
            <ul className="mt-1 space-y-0.5">
              {message.sources.map((s, i) => (
                <li
                  key={i}
                  className="text-xs text-muted-foreground"
                >
                  <code className="rounded bg-background/60 px-1.5 py-0.5 text-[11px] text-primary">
                    {s}
                  </code>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}
