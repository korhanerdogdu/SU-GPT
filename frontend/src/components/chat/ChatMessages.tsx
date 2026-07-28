import { useEffect, useRef } from "react";
import { Download, Loader2 } from "lucide-react";
import ReactMarkdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import "katex/dist/katex.min.css";
import {
  absoluteApiUrl,
  type StructuredContent,
  type StructuredTable,
} from "@/lib/api";

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: string[];
  pending?: boolean;
  summary?: string;
  structuredContent?: StructuredContent | null;
  exportLinks?: Record<string, string>;
}

interface ChatMessagesProps {
  messages: Message[];
  showSources: boolean;
}

export default function ChatMessages({ messages, showSources }: ChatMessagesProps) {
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  // The empty state (centered greeting + composer + starters) is owned by ChatPage/EmptyState;
  // this component only renders once there is at least one message.
  return (
    <div className="mx-auto flex max-w-4xl flex-col gap-6 px-4 py-8 md:px-8">
      {messages.map((message) => (
        <MessageBubble key={message.id} message={message} showSources={showSources} />
      ))}
      <div ref={endRef} />
    </div>
  );
}

function MessageBubble({
  message,
  showSources,
}: {
  message: Message;
  showSources: boolean;
}) {
  const isUser = message.role === "user";
  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <article
        className={`min-w-0 max-w-[92%] px-4 py-3 text-sm leading-relaxed sm:max-w-[86%] ${
          isUser
            ? "rounded-2xl rounded-br-md bg-primary text-primary-foreground shadow-sm"
            : "rounded-2xl rounded-bl-md border border-border bg-card text-card-foreground shadow-sm"
        }`}
      >
        {message.pending && !message.content ? (
          <div className="flex items-center gap-2 text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" />
            <span>adviSU düşünüyor…</span>
          </div>
        ) : isUser ? (
          <div className="whitespace-pre-wrap">{message.content}</div>
        ) : (
          <>
            <MarkdownMessage content={withoutEmbeddedSummary(message.content, message.summary)} />
            {message.pending && (
              <span
                className="ml-1 inline-block h-4 w-0.5 animate-pulse rounded bg-primary align-middle"
                aria-label="Yanıt yazılıyor"
              />
            )}
            {message.structuredContent && (
              <StructuredSections content={message.structuredContent} />
            )}
            {message.exportLinks && Object.keys(message.exportLinks).length > 0 && (
              <ExportLinks links={message.exportLinks} />
            )}
            {!message.pending && message.summary && (
              <section className="mt-5 border-t border-border pt-4">
                <h3 className="mb-1 font-semibold text-primary">Kısa Özet</h3>
                <p className="leading-relaxed">{message.summary}</p>
              </section>
            )}
          </>
        )}

        {!message.pending &&
          showSources &&
          message.sources &&
          message.sources.length > 0 && (
            <details className="mt-4 border-t border-border pt-3">
              <summary className="cursor-pointer font-mono text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
                Teknik kaynaklar ({message.sources.length})
              </summary>
              <ul className="mt-2 flex flex-wrap gap-1.5">
                {message.sources.map((source, index) => (
                  <li
                    key={`${source}-${index}`}
                    className="rounded-md border border-border bg-muted px-2 py-1 font-mono text-[11px] text-muted-foreground"
                  >
                    {source}
                  </li>
                ))}
              </ul>
            </details>
          )}
      </article>
    </div>
  );
}

function withoutEmbeddedSummary(content: string, summary?: string) {
  if (!summary) return content;
  const match = content.match(/\n#{1,6}\s*(?:Kısa Özet|Kisa Ozet|Short Summary)\s*\n/i);
  return match?.index == null ? content : content.slice(0, match.index).trimEnd();
}

function StructuredSections({ content }: { content: StructuredContent }) {
  const tables = content.tables ?? [];
  if (tables.length === 0) return null;
  return (
    <div className="mt-5 space-y-4 border-t border-border pt-4">
      {tables.map((table) => (
        <StructuredTableView key={table.id} table={table} />
      ))}
    </div>
  );
}

function StructuredTableView({ table }: { table: StructuredTable }) {
  function downloadCsv() {
    const escape = (value: unknown) => `"${String(value ?? "").split('"').join('""')}"`;
    const lines = [
      table.columns.map((column) => escape(column.label)).join(","),
      ...table.rows.map((row) =>
        table.columns.map((column) => escape(row[column.key])).join(","),
      ),
    ];
    const blob = new Blob(["\ufeff", lines.join("\n")], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${table.id}.csv`;
    anchor.click();
    URL.revokeObjectURL(url);
  }

  return (
    <section>
      <div className="mb-2 flex items-center justify-between gap-3">
        <h3 className="font-semibold text-foreground">{table.title}</h3>
        {table.exportable && (
          <button
            type="button"
            onClick={downloadCsv}
            className="flex items-center gap-1 rounded-md border border-border px-2 py-1 text-xs text-muted-foreground hover:bg-muted hover:text-foreground"
          >
            <Download className="h-3.5 w-3.5" /> CSV
          </button>
        )}
      </div>
      <div className="overflow-x-auto rounded-lg border border-border">
        <table className="w-full border-collapse text-left text-xs">
          <thead className="bg-muted">
            <tr>
              {table.columns.map((column) => (
                <th key={column.key} className="whitespace-nowrap px-3 py-2 font-semibold">
                  {column.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {table.rows.map((row, rowIndex) => (
              <tr key={rowIndex} className="border-t border-border">
                {table.columns.map((column) => (
                  <td key={column.key} className="px-3 py-2 align-top">
                    {row[column.key] ?? "—"}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function ExportLinks({ links }: { links: Record<string, string> }) {
  return (
    <div className="mt-4 flex flex-wrap gap-2">
      {Object.entries(links).map(([label, path]) => (
        <a
          key={label}
          href={absoluteApiUrl(path)}
          className="flex items-center gap-1 rounded-md border border-border bg-background px-2.5 py-1.5 text-xs font-medium text-primary hover:bg-muted"
        >
          <Download className="h-3.5 w-3.5" />
          {label.split("_").join(" ").toUpperCase()}
        </a>
      ))}
    </div>
  );
}

function MarkdownMessage({ content }: { content: string }) {
  return (
    <div className="advisu-markdown space-y-2">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex]}
        components={{
          h2: ({ children }) => (
            <h2 className="mt-5 border-t border-border pt-4 text-base font-semibold text-primary">
              {children}
            </h2>
          ),
          h3: ({ children }) => <h3 className="mt-3 font-semibold">{children}</h3>,
          p: ({ children }) => <p className="leading-relaxed">{children}</p>,
          strong: ({ children }) => <strong className="font-semibold">{children}</strong>,
          ul: ({ children }) => <ul className="ml-5 list-disc space-y-1">{children}</ul>,
          ol: ({ children }) => <ol className="ml-5 list-decimal space-y-1">{children}</ol>,
          li: ({ children }) => <li className="pl-1">{children}</li>,
          table: ({ children }) => (
            <div className="my-3 overflow-x-auto rounded-lg border border-border">
              <table className="w-full border-collapse text-xs">{children}</table>
            </div>
          ),
          thead: ({ children }) => <thead className="bg-muted">{children}</thead>,
          th: ({ children }) => <th className="px-3 py-2 text-left font-semibold">{children}</th>,
          td: ({ children }) => <td className="border-t border-border px-3 py-2">{children}</td>,
          code: ({ children }) => (
            <code className="rounded bg-muted px-1.5 py-0.5 text-[0.85em] text-primary">
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
