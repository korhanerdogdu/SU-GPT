import { useCallback, useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { toast } from "sonner";
import Sidebar from "@/components/chat/Sidebar";
import ChatHeader from "@/components/chat/ChatHeader";
import ChatMessages, { type Message } from "@/components/chat/ChatMessages";
import ChatInput from "@/components/chat/ChatInput";
import EmptyState from "@/components/chat/EmptyState";
import { useAuth } from "@/contexts/AuthContext";
import { useLocale } from "@/contexts/LocaleContext";
import {
  askQuestionStream,
  currentSessionId,
  deleteConversation,
  getConversation,
  getProfile,
  getUsage,
  listConversations,
  saveUserSchedule,
  startNewSession,
  updateConversation,
  useSession,
  type ConversationSummary,
  type UsageStatus,
} from "@/lib/api";
import {
  normaliseScheduleDocument,
  scheduleFromStructuredContent,
  storeLocalSchedule,
} from "@/lib/schedule";

function makeId() {
  return Math.random().toString(36).slice(2) + Date.now().toString(36);
}

export default function ChatPage() {
  const { user } = useAuth();
  const { t } = useLocale();
  const navigate = useNavigate();
  const [messages, setMessages] = useState<Message[]>([]);
  const [busy, setBusy] = useState(false);
  const [sessionId, setSessionId] = useState<string>(() => currentSessionId());
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [profileReady, setProfileReady] = useState(true); // assume ok until told otherwise
  const [usage, setUsage] = useState<UsageStatus | null>(null);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(
    () => localStorage.getItem("advisu-sidebar-collapsed") === "true",
  );
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);

  const refreshConversations = useCallback(async () => {
    if (!user?.username) return;
    try {
      setConversations(await listConversations(user.username));
    } catch {
      // History is a convenience; never block the chat on it.
    }
  }, [user?.username]);

  useEffect(() => {
    void refreshConversations();
  }, [refreshConversations]);

  // A saved major + curriculum term is what makes answers personal, so prompt for it when absent.
  useEffect(() => {
    if (!user?.username) return;
    let cancelled = false;
    (async () => {
      try {
        const profile = await getProfile(user.username);
        if (!cancelled) setProfileReady(Boolean(profile?.major && profile?.curriculum_term));
      } catch {
        if (!cancelled) setProfileReady(true); // don't nag when the check itself failed
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [user?.username]);

  const refreshUsage = useCallback(async () => {
    if (!user?.username) return;
    try {
      setUsage(await getUsage(user.username));
    } catch {
      // The meter is informational; a failed read must not block chat.
    }
  }, [user?.username]);

  useEffect(() => {
    void refreshUsage();
  }, [refreshUsage]);

  async function handleSend(text: string) {
    const pendingId = makeId();
    setMessages((m) => [
      ...m,
      { id: makeId(), role: "user", content: text },
      { id: pendingId, role: "assistant", content: "", pending: true },
    ]);
    setBusy(true);
    try {
      const res = await askQuestionStream(text, (token) => {
        setMessages((current) =>
          current.map((message) =>
            message.id === pendingId
              ? {
                  ...message,
                  content: message.content + token,
                  pending: true,
                }
              : message,
          ),
        );
      });
      setMessages((m) =>
        m.map((msg) =>
          msg.id === pendingId
            ? {
                ...msg,
                content: res.response,
                sources: res.sources,
                summary: res.summary,
                structuredContent: res.structured_content,
                exportLinks: res.export_links,
                pending: false,
              }
            : msg
        )
      );
      const responseSchedule = normaliseScheduleDocument(res.schedule);
      const generatedSchedule =
        (responseSchedule?.items.length ? responseSchedule : null) ??
        scheduleFromStructuredContent(res.structured_content);
      if (generatedSchedule) {
        storeLocalSchedule(user?.username, generatedSchedule);
        toast.success(t("chat.scheduleSaved"), {
          description: t("chat.scheduleSavedDescription"),
          action: {
            label: t("chat.goSchedule"),
            onClick: () => navigate("/schedule"),
          },
        });
        if (user?.username) {
          void saveUserSchedule(user.username, generatedSchedule).catch(() => {
            toast.warning(t("chat.scheduleOffline"));
          });
        }
      }
      void refreshConversations();
      void refreshUsage();
      setMobileSidebarOpen(false);
    } catch (err) {
      const message = err instanceof Error ? err.message : t("chat.requestFailed");
      toast.error(message);
      setMessages((m) =>
        m.map((msg) =>
          msg.id === pendingId
            ? {
                ...msg,
                content: t("chat.backendError", { message }),
                pending: false,
              }
            : msg
        )
      );
    } finally {
      setBusy(false);
    }
  }

  function handleNewChat() {
    setSessionId(startNewSession());
    setMessages([]);
  }

  async function handleOpenChat(id: string) {
    try {
      const conversation = await getConversation(id);
      useSession(id);
      setSessionId(id);
      setMessages(
        conversation.messages.map((m) => ({
          id: makeId(),
          role: m.role,
          content: m.content,
          sources: m.sources,
        }))
      );
    } catch {
      toast.error(t("chat.openFailed"));
    }
  }

  async function handleDeleteChat(id: string) {
    try {
      await deleteConversation(id);
      if (id === sessionId) handleNewChat();
      void refreshConversations();
    } catch {
      toast.error(t("chat.deleteFailed"));
    }
  }

  async function handleRenameChat(id: string, title: string) {
    try {
      await updateConversation(id, { title });
      void refreshConversations();
    } catch {
      toast.error(t("chat.renameFailed"));
    }
  }

  async function handlePinChat(id: string, pinned: boolean) {
    try {
      await updateConversation(id, { pinned });
      void refreshConversations();
    } catch {
      toast.error(t("chat.pinFailed"));
    }
  }

  function toggleSidebar() {
    const next = !sidebarCollapsed;
    setSidebarCollapsed(next);
    localStorage.setItem("advisu-sidebar-collapsed", String(next));
  }

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-background">
      <Sidebar
        conversations={conversations}
        activeSessionId={sessionId}
        collapsed={sidebarCollapsed}
        mobileOpen={mobileSidebarOpen}
        onToggle={toggleSidebar}
        onCloseMobile={() => setMobileSidebarOpen(false)}
        onNewChat={handleNewChat}
        onOpenChat={handleOpenChat}
        onDeleteChat={handleDeleteChat}
        onRenameChat={handleRenameChat}
        onPinChat={handlePinChat}
      />
      <main className="flex min-w-0 flex-1 flex-col">
        <ChatHeader
          profileReady={profileReady}
          usage={usage}
          onOpenMenu={() => setMobileSidebarOpen(true)}
        />
        {messages.length === 0 ? (
          // Empty state owns its own centered composer (Claude/Gemini style), so the docked
          // bar is not rendered here — there is exactly one input on screen.
          <EmptyState name={user?.username} onSend={handleSend} disabled={busy} />
        ) : (
          <>
            <div className="flex-1 overflow-y-auto scrollbar-thin">
              <ChatMessages messages={messages} showSources={user?.role === "admin"} />
            </div>
            <ChatInput variant="docked" onSend={handleSend} disabled={busy} />
          </>
        )}
      </main>
    </div>
  );
}
