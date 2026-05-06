import { useCallback, useRef, useState } from "react";
import { ChatBox, type ChatBoxHandle } from "./components/ChatBox";
import { HeaderBar } from "./components/HeaderBar";
import { MessageList } from "./components/MessageList";
import { useAsk } from "./hooks/useAsk";
import { useModels } from "./hooks/useModels";
import type { DonePayload, Message } from "./types";

function makeId() {
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

export default function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const { ask, cancel, inFlight } = useAsk();
  const { models, selectedModel, setSelectedModel } = useModels();
  const chatRef = useRef<ChatBoxHandle>(null);
  const docsVersion = import.meta.env.VITE_DOCS_VERSION ?? "3.12";

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
        setMessages((prev) => prev.map((m) => (m.id === assistantMsg.id ? { ...m, ...patch } : m)));
      };

      void ask(
        { query, model: selectedModel ?? undefined },
        {
          onToken: (text) => updateAssistant({ text }),
          onDone: (meta: DonePayload) => updateAssistant({ meta, streaming: false }),
          onError: (message) =>
            updateAssistant({
              text: `Error: ${message}`,
              errored: true,
              streaming: false,
            }),
        },
      );
    },
    [ask, selectedModel],
  );

  const handlePickSuggestion = useCallback((q: string) => {
    chatRef.current?.setValue(q);
  }, []);

  return (
    <div className="flex h-full flex-col">
      <HeaderBar
        docsVersion={docsVersion}
        models={models}
        selectedModel={selectedModel}
        onSelectModel={setSelectedModel}
      />
      <main className="mx-auto w-full max-w-3xl flex-1 overflow-y-auto px-4">
        <MessageList messages={messages} onPickSuggestion={handlePickSuggestion} />
      </main>
      <div className="mx-auto w-full max-w-3xl">
        <ChatBox inputRef={chatRef} onSubmit={handleSubmit} disabled={inFlight} onCancel={cancel} />
      </div>
    </div>
  );
}
