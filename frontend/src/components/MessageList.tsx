import { useEffect, useRef } from "react";
import type { Message } from "../types";
import { Citation } from "./Citation";

interface MessageListProps {
  messages: Message[];
}

function MessageRow({ message }: { message: Message }) {
  const isUser = message.role === "user";
  const align = isUser ? "items-end" : "items-start";
  const bubble = isUser
    ? "bg-blue-600 text-white"
    : message.errored
      ? "bg-red-950 border border-red-800 text-red-100"
      : "bg-zinc-800 text-zinc-100";

  return (
    <div className={`flex flex-col gap-1 ${align}`}>
      <div className={`max-w-[85%] rounded-2xl px-4 py-2 leading-relaxed ${bubble}`}>
        {message.text || (message.streaming ? <span className="opacity-70">…</span> : null)}
      </div>
      {!isUser && message.meta && message.meta.cited_chunk_ids.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {message.meta.cited_chunk_ids.map((cid) => (
            <Citation key={cid} chunkId={cid} />
          ))}
        </div>
      )}
      {!isUser && message.meta && (
        <div className="text-xs text-zinc-500">
          {message.meta.refused
            ? "refused"
            : `${message.meta.latency_seconds.toFixed(1)}s`}
          {message.meta.rewritten_query && (
            <span className="ml-2 italic">
              query rewritten → <span className="font-mono">{message.meta.rewritten_query}</span>
            </span>
          )}
        </div>
      )}
    </div>
  );
}

export function MessageList({ messages }: MessageListProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  if (messages.length === 0) {
    return (
      <div className="flex h-full items-center justify-center text-zinc-500">
        Ask anything about the Python standard library.
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4 py-4">
      {messages.map((m) => (
        <MessageRow key={m.id} message={m} />
      ))}
      <div ref={bottomRef} />
    </div>
  );
}
