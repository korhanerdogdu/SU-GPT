import { useCallback, useEffect, useState } from "react";
import { toast } from "sonner";
import Sidebar from "@/components/chat/Sidebar";
import ChatHeader from "@/components/chat/ChatHeader";
import ChatMessages, { type Message } from "@/components/chat/ChatMessages";
import ChatInput from "@/components/chat/ChatInput";
import { useAuth } from "@/contexts/AuthContext";
import {
  askQuestion,
  currentSessionId,
  deleteConversation,
  getConversation,
  getProfile,
  listConversations,
  startNewSession,
  useSession,
  type ConversationSummary,
} from "@/lib/api";

function makeId() {
  return Math.random().toString(36).slice(2) + Date.now().toString(36);
}

export default function ChatPage() {
  const { user } = useAuth();
  const [messages, setMessages] = useState<Message[]>([]);
  const [busy, setBusy] = useState(false);
  const [sessionId, setSessionId] = useState<string>(() => currentSessionId());
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [profileReady, setProfileReady] = useState(true); // assume ok until told otherwise

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

  async function handleSend(text: string) {
    const pendingId = makeId();
    setMessages((m) => [
      ...m,
      { id: makeId(), role: "user", content: text },
      { id: pendingId, role: "assistant", content: "", pending: true },
    ]);
    setBusy(true);
    try {
      const res = await askQuestion(text);
      setMessages((m) =>
        m.map((msg) =>
          msg.id === pendingId
            ? { ...msg, content: res.response, sources: res.sources, pending: false }
            : msg
        )
      );
      void refreshConversations();
    } catch (err) {
      const message = err instanceof Error ? err.message : "Request failed";
      toast.error(message);
      setMessages((m) =>
        m.map((msg) =>
          msg.id === pendingId
            ? {
                ...msg,
                content: `Sorry — the backend returned an error: ${message}`,
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
      toast.error("Could not open that conversation.");
    }
  }

  async function handleDeleteChat(id: string) {
    try {
      await deleteConversation(id);
      if (id === sessionId) handleNewChat();
      void refreshConversations();
    } catch {
      toast.error("Could not delete that conversation.");
    }
  }

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-sky-night">
      <Sidebar
        conversations={conversations}
        activeSessionId={sessionId}
        onNewChat={handleNewChat}
        onOpenChat={handleOpenChat}
        onDeleteChat={handleDeleteChat}
      />
      <main className="flex min-w-0 flex-1 flex-col">
        <ChatHeader profileReady={profileReady} />
        <div className="flex-1 overflow-y-auto scrollbar-thin">
          <ChatMessages messages={messages} />
        </div>
        <ChatInput onSend={handleSend} disabled={busy} />
      </main>
    </div>
  );
}
