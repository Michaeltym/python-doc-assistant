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
      className="group inline-flex items-center gap-1.5 rounded-full border border-slate-700/80 bg-slate-900/80 px-2.5 py-0.5 text-xs transition hover:border-amber-500/40 hover:bg-slate-900"
    >
      {index !== undefined && (
        <span className="flex h-4 w-4 items-center justify-center rounded-full bg-amber-500/15 font-mono text-[10px] font-medium text-amber-400">
          {index}
        </span>
      )}
      <span className="font-mono text-[11px] text-slate-500">{isSymbol ? "•" : "§"}</span>
      <span className="font-mono text-[12px] text-slate-200 group-hover:text-amber-300">
        {label}
      </span>
    </span>
  );
}
