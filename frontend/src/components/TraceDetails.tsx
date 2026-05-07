import type { DonePayload } from "../types";

interface TraceDetailsProps {
  meta: DonePayload;
  query: string;
}

// Renders the retrieve → augment → generate stages of one ask call as
// an expandable details block. Source data comes from the `done` SSE
// event so the panel adds zero round-trips.
export function TraceDetails({ meta, query }: TraceDetailsProps) {
  const retrieved = meta.retrieved ?? [];
  if (retrieved.length === 0 && !meta.query_type) return null;

  const citedCount = retrieved.filter((r) => r.cited).length;

  return (
    <details className="mt-2 rounded-lg border border-olive-700 bg-forest-950/50 px-3 py-1.5 open:py-2">
      <summary className="cursor-pointer font-mono text-[10.5px] tracking-wide text-cream-200/70 hover:text-cream-100">
        ▸ trace · {retrieved.length} retrieved → {citedCount} cited
        {meta.query_type && (
          <span className="ml-2 text-cream-200/50">[{meta.query_type}]</span>
        )}
      </summary>

      <div className="mt-3 flex flex-col gap-3 font-mono text-[11px] text-cream-200/80">
        {/* Stage 1 — query + route */}
        <div className="flex flex-col gap-0.5">
          <span className="text-cream-200/50">[1] route</span>
          <span>
            <span className="text-cream-200/50">query:</span>{" "}
            <span className="text-cream-50">{query}</span>
          </span>
          {meta.query_type && (
            <span>
              <span className="text-cream-200/50">classified:</span>{" "}
              <span className="text-sand-400">{meta.query_type}</span>
            </span>
          )}
          {meta.rewritten_query && (
            <span>
              <span className="text-cream-200/50">rewritten:</span>{" "}
              <span className="text-sand-400">{meta.rewritten_query}</span>
            </span>
          )}
        </div>

        {/* Stage 2 — retrieved table */}
        {retrieved.length > 0 && (
          <div className="flex flex-col gap-1">
            <span className="text-cream-200/50">
              [2] retrieve · top-{retrieved.length}
            </span>
            <table className="w-full table-auto border-collapse text-[10.5px]">
              <thead>
                <tr className="text-cream-200/40">
                  <th className="w-8 px-1 py-0.5 text-left font-normal">#</th>
                  <th className="w-16 px-1 py-0.5 text-right font-normal">score</th>
                  <th className="px-1 py-0.5 text-left font-normal">title</th>
                  <th className="w-12 px-1 py-0.5 text-center font-normal">cited</th>
                </tr>
              </thead>
              <tbody>
                {retrieved.map((r) => (
                  <tr
                    key={r.chunk_id}
                    className={r.cited ? "text-cream-50" : "text-cream-200/60"}
                  >
                    <td className="px-1 py-0.5 text-cream-200/40">{r.rank}</td>
                    <td className="px-1 py-0.5 text-right tabular-nums">
                      {r.score.toFixed(2)}
                    </td>
                    <td className="truncate px-1 py-0.5">
                      <a
                        href={r.url}
                        target="_blank"
                        rel="noreferrer"
                        className="hover:underline"
                        title={r.chunk_id}
                      >
                        {r.title}
                      </a>
                    </td>
                    <td className="px-1 py-0.5 text-center">
                      {r.cited ? (
                        <span className="rounded-full bg-sand-500/30 px-1.5 py-0.5 text-[9.5px] text-sand-300">
                          ✓
                        </span>
                      ) : (
                        <span className="text-cream-200/30">—</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* Stage 3 — generate */}
        <div className="flex flex-col gap-0.5">
          <span className="text-cream-200/50">[3] generate</span>
          <span>
            <span className="text-cream-200/50">model:</span>{" "}
            <span className="text-cream-50">{meta.model ?? "default"}</span>
            <span className="ml-3 text-cream-200/50">latency:</span>{" "}
            <span className="text-cream-50">{meta.latency_seconds.toFixed(2)}s</span>
            <span className="ml-3 text-cream-200/50">refused:</span>{" "}
            <span className={meta.refused ? "text-red-300" : "text-cream-50"}>
              {meta.refused ? "true" : "false"}
            </span>
          </span>
        </div>
      </div>
    </details>
  );
}
