interface CitationProps {
  chunkId: string;
}

/** Pill rendered next to assistant messages for each cited chunk_id. */
export function Citation({ chunkId }: CitationProps) {
  const [type, name] = chunkId.split(":", 2);
  const label = name ?? chunkId;
  return (
    <span
      title={chunkId}
      className="inline-flex items-center gap-1 rounded border border-zinc-700 bg-zinc-900 px-2 py-0.5 text-xs font-mono text-zinc-300"
    >
      <span className="text-zinc-500">{type}:</span>
      <span className="text-zinc-200">{label}</span>
    </span>
  );
}
