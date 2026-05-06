interface CitationProps {
  chunkId: string;
  index?: number;
}

/**
 * Pill rendered next to assistant messages for each cited chunk_id.
 * Format: a small numbered marker + the human-readable symbol.
 */
export function Citation({ chunkId, index }: CitationProps) {
  const [type, name] = chunkId.split(/:(.*)/);
  const label = name ?? chunkId;
  const isSymbol = type === "symbol";

  return (
    <span
      title={chunkId}
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
    </span>
  );
}
