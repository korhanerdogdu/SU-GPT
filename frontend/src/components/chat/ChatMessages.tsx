import { useEffect, useRef, useState } from "react";
import { Check, Copy, Download, Loader2 } from "lucide-react";
import { Link } from "react-router-dom";
import * as XLSX from "xlsx";
import ReactMarkdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import "katex/dist/katex.min.css";
import { type StructuredContent, type StructuredTable } from "@/lib/api";
import { useLocale } from "@/contexts/LocaleContext";

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
  const { t } = useLocale();
  return (
    <div className={`flex items-end gap-2.5 ${isUser ? "justify-end" : "justify-start"}`}>
      {/* Assistant gets a fixed identity chip so a reader scanning the thread never confuses
          "who said this" — the user's own bubbles are already anchored right, so they don't
          need a mirrored avatar to read as theirs. */}
      {!isUser && (
        <span
          aria-hidden="true"
          className="mb-0.5 grid h-10 w-10 shrink-0 place-items-center rounded-full bg-primary/10 text-primary-emphasis"
        >
          <img src="/assets/small_witihoutbg.png" alt="" className="h-7 w-7 object-contain" />
        </span>
      )}
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
            <span>{t("chat.thinking")}</span>
          </div>
        ) : isUser ? (
          <div className="whitespace-pre-wrap">{message.content}</div>
        ) : (
          <>
            <MarkdownMessage content={withoutEmbeddedSummary(message.content, message.summary)} />
            {message.pending && (
              <span
                className="ml-1 inline-block h-4 w-0.5 animate-pulse rounded bg-primary align-middle"
                aria-label={t("chat.writing")}
              />
            )}
            {message.structuredContent && (
              <StructuredSections content={message.structuredContent} />
            )}
            {/* Closing summary, written directly with no "Kısa Özet" heading. Export buttons live
                on each table (top-right), never appended to the message. */}
            {!message.pending && visibleSummary(message.content, message.summary) && (
              <section className="mt-5 border-t border-border pt-4">
                <p className="leading-relaxed text-muted-foreground">{visibleSummary(message.content, message.summary)}</p>
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
                {t("chat.sources", { count: message.sources.length })}
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

function plainText(value: string) {
  return value
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/^\s*(?:#{1,6}|[-*+]|\d+\.)\s*/gm, "")
    .replace(/\[(.*?)\]\([^)]*\)/g, "$1")
    .replace(/[`*_>#.,:;!?()[\]{}"“”'’—–-]/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .toLowerCase();
}

function visibleSummary(content: string, summary?: string) {
  if (!summary) return "";
  const summaryText = plainText(summary);
  if (!summaryText) return "";
  const bodyText = plainText(withoutEmbeddedSummary(content, summary));
  if (bodyText === summaryText || bodyText.includes(summaryText) || bodyText.endsWith(summaryText)) return "";
  const summaryTokens = summaryText.split(/\s+/).filter((token) => token.length >= 3 || /^\d{3,5}$/.test(token));
  if (summaryTokens.length < 4) return "";
  const bodyTokens = new Set(bodyText.split(/\s+/));
  const overlap = summaryTokens.filter((token) => bodyTokens.has(token)).length / summaryTokens.length;
  if (overlap >= 0.78) return "";
  return summary;
}

function StructuredSections({ content }: { content: StructuredContent }) {
  const { t } = useLocale();
  const tables = content.tables ?? [];
  if (tables.length === 0) return null;
  const crns = content.crns ?? [];
  return (
    <div className="mt-5 space-y-4 border-t border-border pt-4">
      {tables.map((table) => (
        <StructuredTableView key={table.id} table={table} />
      ))}
      {crns.length > 0 && <CopyCrns crns={crns} />}
      {content.kind === "course_schedule" && (
        <Link
          to="/schedule"
          className="inline-flex items-center rounded-lg bg-primary px-3 py-2 text-xs font-semibold text-primary-foreground transition-colors hover:bg-primary/90"
        >
          {t("chat.scheduleOpen")}
        </Link>
      )}
    </div>
  );
}

/** "Copy CRNs" — one click puts the schedule's registration CRNs on the clipboard. */
function CopyCrns({ crns }: { crns: string[] }) {
  const { t } = useLocale();
  const [copied, setCopied] = useState(false);
  async function copy() {
    try {
      await navigator.clipboard.writeText(crns.join(" "));
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    } catch {
      /* clipboard blocked — no-op */
    }
  }
  return (
    <div className="flex flex-wrap items-center gap-2">
      <button
        type="button"
        onClick={copy}
        className="flex items-center gap-1.5 rounded-md border border-border bg-background px-2.5 py-1.5 text-xs font-medium text-primary-emphasis hover:bg-muted"
      >
        {copied ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
        {copied ? t("chat.copied") : t("chat.copyCrns")}
      </button>
      <span className="font-mono text-[11px] text-muted-foreground">{crns.join(" ")}</span>
    </div>
  );
}

function StructuredTableView({ table }: { table: StructuredTable }) {
  const { t } = useLocale();
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

  function downloadXlsx() {
    const header = table.columns.map((column) => column.label);
    const body = table.rows.map((row) =>
      table.columns.map((column) => row[column.key] ?? ""),
    );
    const ws = XLSX.utils.aoa_to_sheet([header, ...body]);
    ws["!cols"] = table.columns.map((column) => ({
      wch: Math.max(12, Math.min(48, column.label.length + 6)),
    }));
    const wb = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(wb, ws, t("common.dataSheet"));
    XLSX.writeFile(wb, `${table.id}.xlsx`);
  }

  return (
    <section>
      <div className="mb-2 flex items-center justify-between gap-3">
        <h3 className="font-semibold text-foreground">{table.title}</h3>
        {table.exportable && (
          <div className="flex shrink-0 items-center gap-1.5">
            <button
              type="button"
              onClick={downloadCsv}
              className="flex items-center gap-1 rounded-md border border-border px-2 py-1 text-xs text-muted-foreground hover:bg-muted hover:text-foreground"
            >
              <Download className="h-3.5 w-3.5" /> CSV
            </button>
            <button
              type="button"
              onClick={downloadXlsx}
              className="flex items-center gap-1 rounded-md border border-border px-2 py-1 text-xs text-muted-foreground hover:bg-muted hover:text-foreground"
            >
              <Download className="h-3.5 w-3.5" /> Excel
            </button>
          </div>
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

function MarkdownMessage({ content }: { content: string }) {
  return (
    <div className="advisu-markdown space-y-2">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex]}
        components={{
          h2: ({ children }) => (
            <h2 className="mt-5 border-t border-border pt-4 text-base font-semibold text-primary-emphasis">
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
            <code className="rounded bg-muted px-1.5 py-0.5 text-[0.85em] text-primary-emphasis">
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
