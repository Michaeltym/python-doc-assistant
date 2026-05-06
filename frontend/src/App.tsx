import { useCallback, useState } from "react";
import { ChatBox } from "./components/ChatBox";
import { MessageList } from "./components/MessageList";
import { useAsk } from "./hooks/useAsk";
import type { DonePayload, Message } from "./types";

function makeId() {
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

export default function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const { ask, cancel, inFlight } = useAsk();

  const handleSubmit = useCallback(
    (query: string) => {
      const userMsg: Message = { id: makeId(), role: "user", text: query };
      const assistantMsg: Message = {
        id: makeId(),
        role: "assistant",
        text: "",
        streaming: true,
      };
      setMessages((prev) => [...prev, userMsg, assistantMsg]);

      const updateAssistant = (patch: Partial<Message>) => {
        setMessages((prev) =>
          prev.map((m) => (m.id === assistantMsg.id ? { ...m, ...patch } : m)),
        );
      };

      void ask(
        { query },
        {
          onToken: (text) => {
            updateAssistant({ text });
          },
          onDone: (meta: DonePayload) => {
            updateAssistant({ meta, streaming: false });
          },
          onError: (message) => {
            updateAssistant({
              text: `Error: ${message}`,
              errored: true,
              streaming: false,
            });
          },
        },
      );
    },
    [ask],
  );

  return (
    <div className="mx-auto flex h-full max-w-3xl flex-col">
      <header className="border-b border-zinc-800 px-4 py-3">
        <h1 className="text-lg font-semibold">python-doc-assistant</h1>
        <p className="text-xs text-zinc-500">
          Grounded Q&amp;A over the Python {import.meta.env.VITE_DOCS_VERSION ?? "3.12"}{" "}
          standard library docs. Answers cite the chunks they used.
        </p>
      </header>
      <main className="flex-1 overflow-y-auto px-4">
        <MessageList messages={messages} />
      </main>
      <ChatBox onSubmit={handleSubmit} disabled={inFlight} onCancel={cancel} />
    </div>
  );
}
