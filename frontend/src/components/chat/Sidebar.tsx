import { Link } from "react-router-dom";
import {
  BookOpen,
  GraduationCap,
  LogOut,
  MessageSquarePlus,
  Trash2,
} from "lucide-react";
import { useAuth } from "@/contexts/AuthContext";
import type { ConversationSummary } from "@/lib/api";

interface SidebarProps {
  conversations: ConversationSummary[];
  activeSessionId: string;
  onNewChat: () => void;
  onOpenChat: (sessionId: string) => void;
  onDeleteChat: (sessionId: string) => void;
}

function relativeTime(iso: string | null): string {
  if (!iso) return "";
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return "";
  const mins = Math.round((Date.now() - then) / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.round(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.round(hours / 24)}d ago`;
}

/**
 * Chat-focused, like a thread list: new chat, past conversations, then the places you go to set
 * things up. Course history and uploads moved to their own page — they are setup, not chat.
 */
export default function Sidebar({
  conversations,
  activeSessionId,
  onNewChat,
  onOpenChat,
  onDeleteChat,
}: SidebarProps) {
  const { user, signOut } = useAuth();

  return (
    <aside className="flex h-screen w-[17.5rem] shrink-0 flex-col border-r border-white/10 bg-[#08152b]/80 backdrop-blur-xl">
      <div className="px-5 pb-4 pt-6">
        <img
          src="/assets/adviSU-logo-reversed.png"
          alt="adviSU — Sabancı University Academic Advisor"
          className="w-[9.5rem]"
        />
      </div>

      <div className="px-5">
        <button
          type="button"
          onClick={onNewChat}
          className="flex w-full items-center gap-2.5 rounded-xl border border-white/12 bg-white/[0.05] px-3 py-2.5 text-sm font-medium text-white transition-colors hover:bg-white/[0.1] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sabanci-gold"
        >
          <MessageSquarePlus className="h-4 w-4 text-sabanci-gold" />
          New chat
        </button>
      </div>

      {/* Chat history */}
      <div className="mt-6 flex-1 overflow-y-auto scrollbar-thin px-5 pb-4">
        <h3 className="mb-3 font-mono text-[11px] uppercase tracking-[0.22em] text-sabanci-light/45">
          Chats
        </h3>

        {conversations.length === 0 ? (
          <p className="text-xs leading-relaxed text-sabanci-light/40">
            Your past conversations appear here once you ask something.
          </p>
        ) : (
          <ul className="space-y-1">
            {conversations.map((c) => {
              const active = c.session_id === activeSessionId;
              return (
                <li key={c.session_id} className="group relative">
                  <button
                    type="button"
                    onClick={() => onOpenChat(c.session_id)}
                    className={`w-full rounded-lg px-2.5 py-2 pr-8 text-left transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sabanci-gold ${
                      active
                        ? "bg-white/[0.1] text-white"
                        : "text-sabanci-light/70 hover:bg-white/[0.06] hover:text-white"
                    }`}
                  >
                    <span className="block truncate text-sm">{c.title}</span>
                    <span className="mt-0.5 block font-mono text-[10px] uppercase tracking-[0.15em] text-sabanci-light/35">
                      {relativeTime(c.updated_at)}
                    </span>
                  </button>
                  <button
                    type="button"
                    onClick={() => onDeleteChat(c.session_id)}
                    aria-label={`Delete chat: ${c.title}`}
                    className="absolute right-1.5 top-2 rounded p-1 text-sabanci-light/40 opacity-0 transition hover:bg-white/10 hover:text-white focus-visible:opacity-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sabanci-gold group-hover:opacity-100"
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </button>
                </li>
              );
            })}
          </ul>
        )}
      </div>

      {/* Setup + account */}
      <div className="border-t border-white/10 px-5 py-4">
        <Link
          to="/courses"
          className="flex items-center gap-2.5 rounded-xl px-3 py-2 text-sm text-sabanci-light/75 transition-colors hover:bg-white/[0.06] hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sabanci-gold"
        >
          <BookOpen className="h-4 w-4" />
          Course history
        </Link>
        <Link
          to="/profile"
          className="flex items-center gap-2.5 rounded-xl px-3 py-2 text-sm text-sabanci-light/75 transition-colors hover:bg-white/[0.06] hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sabanci-gold"
        >
          <GraduationCap className="h-4 w-4" />
          Profile &amp; degree audit
        </Link>

        <div className="mt-3 flex items-center justify-between gap-2 border-t border-white/10 pt-3">
          <div className="min-w-0">
            <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-sabanci-light/40">
              Signed in
            </p>
            <p className="truncate text-sm font-medium text-white">{user?.username}</p>
          </div>
          <button
            type="button"
            onClick={signOut}
            className="flex shrink-0 items-center gap-1.5 rounded-lg px-2 py-1.5 text-xs text-sabanci-light/60 transition-colors hover:bg-white/10 hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sabanci-gold"
          >
            <LogOut className="h-3.5 w-3.5" />
            Log out
          </button>
        </div>
      </div>
    </aside>
  );
}
