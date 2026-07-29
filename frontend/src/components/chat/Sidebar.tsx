import { useEffect, useState, type MouseEvent as ReactMouseEvent } from "react";
import { Link } from "react-router-dom";
import {
  BookOpen,
  CalendarDays,
  ChevronLeft,
  GraduationCap,
  LogOut,
  MessageSquarePlus,
  PanelLeftOpen,
  Pencil,
  Pin,
  PinOff,
  Trash2,
  X,
} from "lucide-react";
import { useAuth } from "@/contexts/AuthContext";
import { useTheme } from "@/contexts/ThemeContext";
import type { ConversationSummary } from "@/lib/api";

interface SidebarProps {
  conversations: ConversationSummary[];
  activeSessionId: string;
  collapsed: boolean;
  mobileOpen: boolean;
  onToggle: () => void;
  onCloseMobile: () => void;
  onNewChat: () => void;
  onOpenChat: (sessionId: string) => void;
  onDeleteChat: (sessionId: string) => void;
  onRenameChat: (sessionId: string, title: string) => void;
  onPinChat: (sessionId: string, pinned: boolean) => void;
}

function relativeTime(iso: string | null): string {
  if (!iso) return "";
  const minutes = Math.round((Date.now() - new Date(iso).getTime()) / 60000);
  if (minutes < 1) return "şimdi";
  if (minutes < 60) return `${minutes} dk`;
  if (minutes < 1440) return `${Math.round(minutes / 60)} sa`;
  return `${Math.round(minutes / 1440)} gün`;
}

export default function Sidebar({
  conversations,
  activeSessionId,
  collapsed,
  mobileOpen,
  onToggle,
  onCloseMobile,
  onNewChat,
  onOpenChat,
  onDeleteChat,
  onRenameChat,
  onPinChat,
}: SidebarProps) {
  const { user, signOut } = useAuth();
  const { resolvedTheme } = useTheme();
  const [editing, setEditing] = useState<string | null>(null);
  const [draft, setDraft] = useState("");

  // Drag-resizable width (persisted). Collapse is a separate control; this only tunes the
  // expanded width between sensible bounds.
  const MIN_W = 232;
  const MAX_W = 480;
  const [width, setWidth] = useState<number>(() => {
    const stored = Number(localStorage.getItem("advisu-sidebar-width"));
    return stored >= MIN_W && stored <= MAX_W ? stored : 288;
  });
  useEffect(() => {
    localStorage.setItem("advisu-sidebar-width", String(width));
  }, [width]);

  function startResize(e: ReactMouseEvent) {
    e.preventDefault();
    const startX = e.clientX;
    const startW = width;
    function onMove(ev: globalThis.MouseEvent) {
      setWidth(Math.min(MAX_W, Math.max(MIN_W, startW + (ev.clientX - startX))));
    }
    function onUp() {
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
      document.body.style.userSelect = "";
      document.body.style.cursor = "";
    }
    document.body.style.userSelect = "none";
    document.body.style.cursor = "col-resize";
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
  }

  function beginRename(item: ConversationSummary) {
    setEditing(item.session_id);
    setDraft(item.title);
  }

  function commitRename() {
    if (editing && draft.trim()) onRenameChat(editing, draft.trim());
    setEditing(null);
  }

  if (collapsed && !mobileOpen) {
    return (
      <aside className="hidden h-screen w-16 shrink-0 flex-col items-center border-r border-border bg-card py-4 md:flex">
        <button
          type="button"
          onClick={onToggle}
          aria-label="Sohbet geçmişini aç"
          className="rounded-lg p-2 text-muted-foreground hover:bg-muted hover:text-foreground"
        >
          <PanelLeftOpen className="h-5 w-5" />
        </button>
        <Link
          to="/schedule"
          aria-label="Ders programı"
          title="Ders programı"
          className="mt-3 rounded-lg p-2 text-muted-foreground transition-colors hover:bg-muted hover:text-primary"
        >
          <CalendarDays className="h-5 w-5" />
        </Link>
        <button
          type="button"
          onClick={onNewChat}
          aria-label="Yeni sohbet"
          className="mt-4 rounded-lg bg-primary p-2 text-primary-foreground"
        >
          <MessageSquarePlus className="h-5 w-5" />
        </button>
      </aside>
    );
  }

  return (
    <>
      {mobileOpen && (
        <button
          className="fixed inset-0 z-30 bg-black/40 md:hidden"
          onClick={onCloseMobile}
          aria-label="Menüyü kapat"
        />
      )}
      <aside
        style={{ width }}
        className={`fixed inset-y-0 left-0 z-40 flex shrink-0 flex-col border-r border-border bg-card shadow-xl transition-transform md:relative md:shadow-none ${
          mobileOpen ? "translate-x-0" : "-translate-x-full md:translate-x-0"
        }`}
      >
        <div className="flex items-center justify-between px-5 pb-4 pt-5">
          {/* mid = compact horizontal lockup (navy ink on transparent, now tightly trimmed so it
              fills its box). In dark mode a soft white glow keeps the navy "SU" and book covers
              legible without any plate. */}
          <img
            src="/assets/mid.png"
            alt="adviSU"
            className={`h-12 w-auto ${
              resolvedTheme === "dark"
                ? "[filter:drop-shadow(0_0_9px_rgba(255,255,255,0.6))]"
                : ""
            }`}
          />
          <button
            type="button"
            onClick={mobileOpen ? onCloseMobile : onToggle}
            aria-label="Sohbet geçmişini gizle"
            className="rounded-lg p-1.5 text-muted-foreground hover:bg-muted hover:text-foreground"
          >
            {mobileOpen ? <X className="h-5 w-5" /> : <ChevronLeft className="h-5 w-5" />}
          </button>
        </div>

        <div className="px-4">
          <button
            type="button"
            onClick={onNewChat}
            className="flex w-full items-center gap-2.5 rounded-xl border border-border bg-background px-3 py-2.5 text-sm font-medium text-foreground hover:bg-muted"
          >
            <MessageSquarePlus className="h-4 w-4 text-primary" />
            Yeni sohbet
          </button>
        </div>

        <div className="mt-5 flex-1 overflow-y-auto px-3 pb-4 scrollbar-thin">
          <h3 className="mb-2 px-2 text-xs font-medium text-muted-foreground/80">
            Sohbet geçmişi
          </h3>
          {conversations.length === 0 ? (
            <p className="px-2 text-xs leading-relaxed text-muted-foreground">
              İlk sorundan sonra sohbetlerin burada görünecek.
            </p>
          ) : (
            <ul className="space-y-1">
              {conversations.map((item) => {
                const active = item.session_id === activeSessionId;
                return (
                  <li
                    key={item.session_id}
                    className={`group rounded-xl transition-colors ${
                      active ? "bg-primary/10" : "hover:bg-muted/70"
                    }`}
                  >
                    <div className="flex items-start">
                      <button
                        type="button"
                        onClick={() => onOpenChat(item.session_id)}
                        className="min-w-0 flex-1 px-2.5 py-2 text-left"
                      >
                        {editing === item.session_id ? (
                          <input
                            autoFocus
                            value={draft}
                            onChange={(event) => setDraft(event.target.value)}
                            onBlur={commitRename}
                            onKeyDown={(event) => {
                              if (event.key === "Enter") commitRename();
                              if (event.key === "Escape") setEditing(null);
                            }}
                            onClick={(event) => event.stopPropagation()}
                            className="w-full rounded border border-primary bg-background px-1.5 py-0.5 text-sm text-foreground outline-none"
                          />
                        ) : (
                          <span className="flex items-center gap-1.5">
                            {item.pinned && <Pin className="h-3 w-3 shrink-0 text-primary" />}
                            <span className={`block truncate text-[0.9rem] ${active ? "font-medium text-primary" : "text-foreground"}`}>
                              {item.title}
                            </span>
                          </span>
                        )}
                        <span className="mt-0.5 block text-[11px] text-muted-foreground">
                          {relativeTime(item.updated_at)}
                        </span>
                      </button>
                      <div className="flex pt-1.5 opacity-0 transition-opacity group-hover:opacity-100 group-focus-within:opacity-100">
                        <button
                          type="button"
                          onClick={() => onPinChat(item.session_id, !item.pinned)}
                          aria-label={item.pinned ? "Sabitlemeyi kaldır" : "Sohbeti sabitle"}
                          className="rounded p-1 text-muted-foreground hover:bg-background hover:text-foreground"
                        >
                          {item.pinned ? <PinOff className="h-3.5 w-3.5" /> : <Pin className="h-3.5 w-3.5" />}
                        </button>
                        <button
                          type="button"
                          onClick={() => beginRename(item)}
                          aria-label="Sohbeti yeniden adlandır"
                          className="rounded p-1 text-muted-foreground hover:bg-background hover:text-foreground"
                        >
                          <Pencil className="h-3.5 w-3.5" />
                        </button>
                        <button
                          type="button"
                          onClick={() => onDeleteChat(item.session_id)}
                          aria-label="Sohbeti sil"
                          className="rounded p-1 text-muted-foreground hover:bg-destructive/10 hover:text-destructive"
                        >
                          <Trash2 className="h-3.5 w-3.5" />
                        </button>
                      </div>
                    </div>
                  </li>
                );
              })}
            </ul>
          )}
        </div>

        <div className="border-t border-border px-4 py-4">
          <Link to="/schedule" className="sidebar-link">
            <CalendarDays className="h-4 w-4" /> Ders Programı
          </Link>
          <Link to="/courses" className="sidebar-link">
            <BookOpen className="h-4 w-4" /> Ders geçmişi
          </Link>
          <Link to="/profile" className="sidebar-link">
            <GraduationCap className="h-4 w-4" /> Profil ve mezuniyet
          </Link>
          <div className="mt-3 flex items-center justify-between border-t border-border pt-3">
            <div className="min-w-0">
              <p className="truncate text-sm font-medium text-foreground">{user?.username}</p>
              <p className="text-[10px] uppercase tracking-wider text-muted-foreground">
                {user?.role === "admin" ? "Yönetici" : "Öğrenci"}
              </p>
            </div>
            <button
              type="button"
              onClick={signOut}
              className="rounded-lg p-2 text-muted-foreground hover:bg-muted hover:text-foreground"
              aria-label="Çıkış yap"
            >
              <LogOut className="h-4 w-4" />
            </button>
          </div>
        </div>

        {/* Drag handle: resize the sidebar width (desktop only). */}
        <div
          onMouseDown={startResize}
          role="separator"
          aria-orientation="vertical"
          aria-label="Kenar çubuğunu yeniden boyutlandır"
          title="Sürükleyerek genişliği ayarla"
          className="group absolute inset-y-0 -right-1 z-10 hidden w-2 cursor-col-resize md:block"
        >
          <div className="absolute inset-y-0 left-1/2 w-px -translate-x-1/2 bg-transparent transition-colors group-hover:bg-primary/50" />
        </div>
      </aside>
    </>
  );
}
