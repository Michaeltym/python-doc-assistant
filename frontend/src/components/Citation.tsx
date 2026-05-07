import type { CitedChunk } from "../types";

interface CitationProps {
  chunk: CitedChunk;
  index?: number;
}

/**
 * Pill rendered next to assistant messages for each cited chunk.
 * Wraps an anchor that opens the canonical docs.python.org page.
 *
 * Hover behaviour: when `text_preview` is present, a popover slides
 * out showing the chunk snippet so the user can verify the citation
 * without clicking through. Implemented via Tailwind's group-hover
 * — no JS state, no portal, no delay logic.
 */
export function Citation({ chunk, index }: CitationProps) {
  const [type, name] = chunk.chunk_id.split(/:(.*)/);
  const isSymbol = type === "symbol";
  const label = name ?? chunk.chunk_id;
  const preview = chunk.text_preview;

  return (
    <span className="group relative inline-block">
      <a
        href={chunk.url}
        target="_blank"
        rel="noopener noreferrer"
        title={preview ? undefined : `${chunk.chunk_id} — ${chunk.url}`}
        className="inline-flex items-center gap-1.5 rounded-full border border-olive-700 bg-forest-900/80 px-2.5 py-0.5 text-xs transition hover:border-sand-500 hover:bg-forest-900"
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

      {preview && (
        <span
          role="tooltip"
          className="pointer-events-none absolute bottom-full left-0 z-20 mb-2 hidden w-80 max-w-[min(20rem,calc(100vw-2rem))] rounded-lg border border-olive-700 bg-forest-950/95 px-3 py-2 text-[12px] leading-relaxed text-cream-100 shadow-xl shadow-black/50 backdrop-blur-sm group-hover:block group-focus-within:block"
        >
          <span className="block truncate font-mono text-[10.5px] tracking-wide text-cream-200/60">
            {chunk.title}
          </span>
          <span className="mt-1 block whitespace-normal text-cream-50">{preview}</span>
        </span>
      )}
    </span>
  );
}
