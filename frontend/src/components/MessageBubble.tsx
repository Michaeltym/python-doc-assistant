import ReactMarkdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import remarkGfm from "remark-gfm";
import type { Message } from "../types";
import { Citation } from "./Citation";

interface MessageBubbleProps {
  message: Message;
}

function StreamingDots() {
  return (
    <span className="inline-flex items-center gap-1 align-middle">
      <span className="h-1.5 w-1.5 animate-dot-1 rounded-full bg-slate-500" />
      <span className="h-1.5 w-1.5 animate-dot-2 rounded-full bg-slate-500" />
      <span className="h-1.5 w-1.5 animate-dot-3 rounded-full bg-slate-500" />
    </span>
  );
}

/**
 * Pre-process the model's answer:
 *  - Replace bare [N] citation markers with a span the markdown
 *    renderer leaves alone (so highlight.js doesn't try to parse them
 *    as code), styled separately.
 */
function preprocessAnswer(text: string): string {
  return text.replace(/\[(\d+)\]/g, '<span class="cite-marker">$1</span>');
}

export function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === "user";

  if (isUser) {
    return (
      <div className="flex animate-fade-up justify-end">
        <div className="max-w-[80%] rounded-2xl rounded-br-sm bg-amber-500 px-4 py-2.5 text-[15px] font-medium text-slate-900 shadow-md shadow-amber-500/10">
          {message.text}
        </div>
      </div>
    );
  }

  const errored = message.errored;
  const refused = message.meta?.refused;

  return (
    <div className="flex animate-fade-up flex-col gap-2">
      <div className="flex gap-3">
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br from-amber-500 to-amber-600 shadow-md shadow-amber-500/20">
          <span className="font-mono text-sm font-bold text-slate-900">py</span>
        </div>
        <div className="min-w-0 flex-1">
          <div
            className={[
              "max-w-full rounded-2xl rounded-tl-sm border px-4 py-3 shadow-lg shadow-black/20",
              errored
                ? "border-red-900/60 bg-red-950/40 text-red-100"
                : refused
                  ? "border-slate-800 bg-slate-900/60 text-slate-400 italic"
                  : "border-slate-800 bg-slate-900/80 text-slate-100",
            ].join(" ")}
          >
            {message.text ? (
              <div className="prose-answer">
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  rehypePlugins={[rehypeHighlight]}
                  // eslint-disable-next-line react/no-children-prop
                  children={preprocessAnswer(message.text)}
                  components={{
                    // Allow our preprocessed cite-marker spans to pass through.
                    span: ({ className, ...props }) =>
                      className === "cite-marker" ? (
                        <span className="cite-marker" {...props} />
                      ) : (
                        <span className={className} {...props} />
                      ),
                  }}
                />
              </div>
            ) : (
              <StreamingDots />
            )}
          </div>

          {/* Citation pills */}
          {!errored && message.meta && message.meta.cited_chunk_ids.length > 0 && (
            <div className="mt-2 flex flex-wrap gap-1.5">
              {message.meta.cited_chunk_ids.map((cid, i) => (
                <Citation key={cid} chunkId={cid} index={i + 1} />
              ))}
            </div>
          )}

          {/* Footer metadata: latency, refusal, rewritten query */}
          {message.meta && (
            <div className="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-slate-500">
              {message.meta.refused ? (
                <span className="text-slate-500 italic">insufficient context — refused</span>
              ) : (
                <span>
                  <span className="font-mono text-slate-400">
                    {message.meta.latency_seconds.toFixed(1)}s
                  </span>{" "}
                  · {message.meta.cited_chunk_ids.length}{" "}
                  {message.meta.cited_chunk_ids.length === 1 ? "source" : "sources"}
                </span>
              )}
              {message.meta.rewritten_query && (
                <span className="rounded-full bg-amber-500/10 px-2 py-0.5 text-amber-400">
                  query rewritten →{" "}
                  <span className="font-mono">{message.meta.rewritten_query}</span>
                </span>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
