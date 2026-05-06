import type { CitedChunk } from "../types";

interface CitationProps {
  chunk: CitedChunk;
  index?: number;
}

/**
 * Pill rendered next to assistant messages for each cited chunk.
 * Wraps an anchor that opens the canonical docs.python.org page.
 */
export function Citation({ chunk, index }: CitationProps) {
  const [type, name] = chunk.chunk_id.split(/:(.*)/);
  const isSymbol = type === "symbol";
  const label = name ?? chunk.chunk_id;

  return (
    <a
      href={chunk.url}
      target="_blank"
      rel="noopener noreferrer"
      title={`${chunk.chunk_id} — ${chunk.url}`}
      className="group inline-flex items-center gap-1.5 rounded-full border border-olive-700 bg-forest-900/80 px-2.5 py-0.5 text-xs transition hover:border-sand-500 hover:bg-forest-900"
    >
      {index !== undefined && (
        <span className="flex h-4 w-4 items-center justify-center rounded-full bg-sand-500/20 font-mono text-[10px] font-medium text-sand-400">
          {index}
        </span>
      )}
      <span className="font-mono text-[11px] text-cream-200/60">{isSymbol ? "•" : "§"}</span>
      <span className="font-mono text-[12px] text-cream-100 group-hover:text-sand-400">
        {label}
      </span>
    </a>
  );
}
