import { useEffect, useRef } from "react";
import type { Message } from "../types";
import { MessageBubble } from "./MessageBubble";

interface MessageListProps {
  messages: Message[];
  onPickSuggestion: (q: string) => void;
}

const SUGGESTIONS: { label: string; query: string }[] = [
  { label: "How to read a file", query: "how to read a file in python" },
  { label: "pathlib.Path.read_text", query: "pathlib.Path.read_text" },
  { label: "set vs frozenset", query: "set vs frozenset" },
  { label: "Merge two dicts", query: "how to merge two dicts" },
  { label: "asyncio.gather", query: "asyncio.gather" },
  { label: "Sort dict by value", query: "how to sort a dict by value" },
];

function EmptyState({ onPick }: { onPick: (q: string) => void }) {
  return (
    <div className="flex h-full flex-col items-center justify-center py-12">
      <div className="mb-6 flex h-14 w-14 items-center justify-center rounded-2xl bg-gradient-to-br from-amber-500 to-amber-600 shadow-lg shadow-amber-500/30">
        <span className="font-mono text-xl font-bold text-slate-900">py</span>
      </div>
      <h2 className="text-xl font-semibold text-slate-100">Ask the Python docs</h2>
      <p className="mt-1 text-sm text-slate-500">
        Grounded answers from the Python 3.12 standard library.
      </p>
      <div className="mt-8 grid w-full max-w-xl grid-cols-1 gap-2 sm:grid-cols-2">
        {SUGGESTIONS.map((s) => (
          <button
            key={s.query}
            type="button"
            onClick={() => onPick(s.query)}
            className="group rounded-xl border border-slate-800 bg-slate-900/40 px-4 py-3 text-left transition hover:border-amber-500/40 hover:bg-slate-900"
          >
            <div className="text-sm font-medium text-slate-200 group-hover:text-amber-300">
              {s.label}
            </div>
            <div className="mt-0.5 truncate font-mono text-[11px] text-slate-500">{s.query}</div>
          </button>
        ))}
      </div>
    </div>
  );
}

export function MessageList({ messages, onPickSuggestion }: MessageListProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  if (messages.length === 0) {
    return <EmptyState onPick={onPickSuggestion} />;
  }

  return (
    <div className="flex flex-col gap-5 py-6">
      {messages.map((m) => (
        <MessageBubble key={m.id} message={m} />
      ))}
      <div ref={bottomRef} />
    </div>
  );
}
