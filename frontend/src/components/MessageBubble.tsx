import ReactMarkdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import rehypeRaw from "rehype-raw";
import remarkGfm from "remark-gfm";
import { remarkCiteMarker } from "../lib/remarkCiteMarker";
import type { Message } from "../types";
import { Citation } from "./Citation";
import { TraceDetails } from "./TraceDetails";

interface MessageBubbleProps {
  message: Message;
  userQuery?: string;
}

function StreamingDots() {
  return (
    <span className="inline-flex items-center gap-1 align-middle">
      <span className="h-1.5 w-1.5 animate-dot-1 rounded-full bg-cream-200/70" />
      <span className="h-1.5 w-1.5 animate-dot-2 rounded-full bg-cream-200/70" />
      <span className="h-1.5 w-1.5 animate-dot-3 rounded-full bg-cream-200/70" />
    </span>
  );
}

function RefusalMessage() {
  return (
    <div className="space-y-2 text-[14px] leading-relaxed">
      <p className="text-cream-50">
        I couldn&rsquo;t find this in the Python standard library docs.
      </p>
      <p className="text-cream-200/70">
        Try a more specific query — for example a module / class / function name like{" "}
        <code className="rounded bg-forest-950 px-1 py-0.5 font-mono text-[12px] text-sand-400">
          pathlib.Path.read_text
        </code>
        , or a how-to such as{" "}
        <code className="rounded bg-forest-950 px-1 py-0.5 font-mono text-[12px] text-sand-400">
          how to merge two dicts
        </code>
        .
      </p>
    </div>
  );
}

export function MessageBubble({ message, userQuery }: MessageBubbleProps) {
  const isUser = message.role === "user";

  if (isUser) {
    return (
      <div className="flex animate-fade-up justify-end">
        <div className="max-w-[80%] rounded-2xl rounded-br-sm bg-sand-500 px-4 py-2.5 text-[15px] font-medium text-forest-900 shadow-md shadow-sand-500/15">
          {message.text}
        </div>
      </div>
    );
  }

  const errored = message.errored;
  const refused = message.meta?.refused;
  const citedChunks = message.meta?.cited_chunks ?? [];
  const isStreaming = message.streaming && !message.text;
  const showText = !refused && !!message.text;

  return (
    <div className="flex animate-fade-up flex-col gap-2">
      <div className="flex gap-3">
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-cream-50 shadow-md shadow-cream-50/10">
          <span className="font-mono text-sm font-bold text-forest-900">py</span>
        </div>
        <div className="min-w-0 flex-1">
          <div
            className={[
              "max-w-full rounded-2xl rounded-tl-sm border px-4 py-3 shadow-lg shadow-black/30",
              errored
                ? "border-red-900/60 bg-red-950/40 text-red-100"
                : refused
                  ? "border-olive-700 bg-forest-900/60 text-cream-200/80"
                  : "border-olive-700 bg-forest-900/80 text-cream-50",
            ].join(" ")}
          >
            {refused ? (
              <RefusalMessage />
            ) : showText ? (
              <div className="prose-answer">
                <ReactMarkdown
                  remarkPlugins={[remarkGfm, remarkCiteMarker]}
                  rehypePlugins={[rehypeRaw, rehypeHighlight]}
                >
                  {message.text}
                </ReactMarkdown>
              </div>
            ) : isStreaming ? (
              <StreamingDots />
            ) : null}
          </div>

          {/* Citation pills (links to docs.python.org). */}
          {!errored && citedChunks.length > 0 && (
            <div className="mt-2 flex flex-wrap gap-1.5">
              {citedChunks.map((c, i) => (
                <Citation key={c.chunk_id} chunk={c} index={i + 1} />
              ))}
            </div>
          )}

          {/* Footer metadata: latency, refusal, rewritten query, model */}
          {message.meta && (
            <div className="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-cream-200/60">
              {message.meta.refused ? (
                <span className="text-cream-200/60 italic">
                  no matching docs · {message.meta.latency_seconds.toFixed(1)}s
                </span>
              ) : (
                <span>
                  <span className="font-mono text-cream-200">
                    {message.meta.latency_seconds.toFixed(1)}s
                  </span>{" "}
                  · {citedChunks.length} {citedChunks.length === 1 ? "source" : "sources"}
                </span>
              )}
              {message.meta.model && (
                <span className="rounded-full border border-olive-700 bg-forest-950/60 px-2 py-0.5 font-mono text-[10.5px] text-cream-200/80">
                  {message.meta.model}
                </span>
              )}
              {message.meta.rewritten_query && (
                <span className="rounded-full bg-sand-500/15 px-2 py-0.5 text-sand-400">
                  query rewritten →{" "}
                  <span className="font-mono">{message.meta.rewritten_query}</span>
                </span>
              )}
            </div>
          )}

          {/* Pipeline trace — collapsed by default. */}
          {message.meta && !errored && (
            <TraceDetails meta={message.meta} query={userQuery ?? ""} />
          )}
        </div>
      </div>
    </div>
  );
}
