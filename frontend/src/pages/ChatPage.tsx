import { useState } from "react";
import { toast } from "sonner";
import Sidebar from "@/components/chat/Sidebar";
import ChatHeader from "@/components/chat/ChatHeader";
import ChatMessages, { type Message } from "@/components/chat/ChatMessages";
import ChatInput from "@/components/chat/ChatInput";
import { askQuestion } from "@/lib/api";

function makeId() {
  return Math.random().toString(36).slice(2) + Date.now().toString(36);
}

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [busy, setBusy] = useState(false);

  async function handleSend(text: string) {
    const userMsg: Message = { id: makeId(), role: "user", content: text };
    const pendingId = makeId();
    const pending: Message = {
      id: pendingId,
      role: "assistant",
      content: "",
      pending: true,
    };
    setMessages((m) => [...m, userMsg, pending]);
    setBusy(true);
    try {
      const res = await askQuestion(text);
      setMessages((m) =>
        m.map((msg) =>
          msg.id === pendingId
            ? {
                ...msg,
                content: res.response,
                sources: res.sources,
                pending: false,
              }
            : msg
        )
      );
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

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-sky-night">
      <Sidebar />
      <main className="flex min-w-0 flex-1 flex-col">
        <ChatHeader />
        <div className="flex-1 overflow-y-auto scrollbar-thin">
          <ChatMessages messages={messages} />
        </div>
        <ChatInput onSend={handleSend} disabled={busy} />
      </main>
    </div>
  );
}
