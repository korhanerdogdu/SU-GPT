import { useEffect, useRef } from "react";
import { Loader2 } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

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
      // Deliberately spare, in the spirit of a fresh chat thread: mark, one warm question, one
      // line saying what this can answer. Nothing to read, nothing to dismiss — the invitation
      // is the input directly below.
      <div className="flex h-full items-center justify-center px-6 py-10">
        <div className="flex w-full max-w-xl flex-col items-center text-center">
          <img
            src="/assets/adviSU-logo-reversed.png"
            alt="adviSU — Sabancı University Academic Advisor"
            className="w-[min(80%,17rem)] drop-shadow-[0_4px_24px_rgba(0,0,0,0.5)]"
          />
          <h2 className="mt-8 text-[2rem] font-semibold leading-tight tracking-tight text-white sm:text-[2.35rem]">
            Are we graduating?
          </h2>
          <p className="mt-4 max-w-xl text-[1.02rem] leading-relaxed text-sabanci-light/60">
            Ask about your remaining credits, electives, minors or graduation requirements —
            answered from official Sabancı University curriculum data, with sources.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="mx-auto flex max-w-3xl flex-col gap-5 px-4 py-8 md:px-8">
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
        className={`max-w-[85%] px-4 py-3 text-sm leading-relaxed ${
          isUser
            ? "rounded-2xl rounded-br-md bg-sabanci-blue text-white shadow-lg shadow-black/20"
            : "rounded-2xl rounded-bl-md border border-white/10 bg-white/[0.05] text-white/90 backdrop-blur-sm"
        }`}
      >
        {message.pending ? (
          <div className="flex items-center gap-2 text-sabanci-light/60">
            <Loader2 className="h-4 w-4 animate-spin" />
            <span>adviSU is thinking…</span>
          </div>
        ) : isUser ? (
          <div className="whitespace-pre-wrap">{message.content}</div>
        ) : (
          <MarkdownMessage content={message.content} />
        )}

        {!message.pending && message.sources && message.sources.length > 0 && (
          <div className="mt-3.5 border-t border-white/10 pt-3">
            <div className="font-mono text-[10px] uppercase tracking-[0.2em] text-sabanci-light/45">
              Sources
            </div>
            <ul className="mt-2 flex flex-wrap gap-1.5">
              {message.sources.map((s, i) => (
                <li
                  key={i}
                  className="rounded-md border border-white/10 bg-white/[0.06] px-2 py-1 font-mono text-[11px] text-sabanci-light/75"
                >
                  {s}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}

function MarkdownMessage({ content }: { content: string }) {
  return (
    <div className="space-y-2">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          p: ({ children }) => <p className="leading-relaxed">{children}</p>,
          strong: ({ children }) => (
            <strong className="font-semibold text-foreground">{children}</strong>
          ),
          ul: ({ children }) => (
            <ul className="ml-5 list-disc space-y-1">{children}</ul>
          ),
          ol: ({ children }) => (
            <ol className="ml-5 list-decimal space-y-1">{children}</ol>
          ),
          li: ({ children }) => <li className="pl-1">{children}</li>,
          hr: () => <div className="my-3 border-t border-border/70" />,
          code: ({ children }) => (
            <code className="rounded bg-background/70 px-1.5 py-0.5 text-[0.85em] text-primary">
              {children}
            </code>
          ),
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
