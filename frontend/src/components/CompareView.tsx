import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useCompare } from "../hooks/useCompare";
import type { CompareMode, CompareModelOutput, ModelInfo } from "../types";

interface CompareViewProps {
  models: ModelInfo[];
}

const STORAGE_PROMPT = "pdr.compare.prompt";
const STORAGE_MODE = "pdr.compare.mode";
const STORAGE_SELECTION = "pdr.compare.models";

function emptyOutput(): CompareModelOutput {
  return { text: "", streaming: true };
}

export function CompareView({ models }: CompareViewProps) {
  const [prompt, setPrompt] = useState<string>(() => localStorage.getItem(STORAGE_PROMPT) ?? "");
  const [mode, setMode] = useState<CompareMode>(() => {
    const stored = localStorage.getItem(STORAGE_MODE);
    return stored === "playground" ? "playground" : "ask";
  });
  const [selectedIds, setSelectedIds] = useState<Set<string>>(() => {
    const stored = localStorage.getItem(STORAGE_SELECTION);
    if (stored) {
      try {
        const parsed = JSON.parse(stored) as string[];
        return new Set(parsed);
      } catch {
        // fall through to default
      }
    }
    return new Set(models.map((m) => m.id));
  });
  const [outputs, setOutputs] = useState<Record<string, CompareModelOutput>>({});
  const promptRef = useRef<HTMLTextAreaElement>(null);

  const { run, cancel, inFlightModels } = useCompare();
  const inFlight = inFlightModels.length > 0;

  // Persist prompt + mode + selection.
  useEffect(() => {
    localStorage.setItem(STORAGE_PROMPT, prompt);
  }, [prompt]);
  useEffect(() => {
    localStorage.setItem(STORAGE_MODE, mode);
  }, [mode]);
  useEffect(() => {
    localStorage.setItem(STORAGE_SELECTION, JSON.stringify(Array.from(selectedIds)));
  }, [selectedIds]);

  // Newly registered models default to selected; stale ids are dropped.
  useEffect(() => {
    setSelectedIds((prev) => {
      const known = new Set(models.map((m) => m.id));
      const next = new Set<string>();
      for (const id of prev) if (known.has(id)) next.add(id);
      for (const m of models) if (!prev.has(m.id) && prev.size === 0) next.add(m.id);
      // If prev had any selection, preserve it; otherwise default-on all.
      return next.size > 0 ? next : new Set(models.map((m) => m.id));
    });
  }, [models]);

  // Autosize prompt textarea.
  useEffect(() => {
    const ta = promptRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = `${Math.min(ta.scrollHeight, 240)}px`;
  }, [prompt]);

  const orderedSelection = useMemo(
    () => models.filter((m) => selectedIds.has(m.id)),
    [models, selectedIds],
  );

  const toggleModel = useCallback((id: string) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  const submit = useCallback(() => {
    if (!prompt.trim() || inFlight || orderedSelection.length === 0) return;

    const initial: Record<string, CompareModelOutput> = {};
    for (const m of orderedSelection) initial[m.id] = emptyOutput();
    setOutputs(initial);

    const body =
      mode === "ask"
        ? { query: prompt }
        : { prompt, max_tokens: 256, temperature: 0 };

    void run(
      { mode, modelIds: orderedSelection.map((m) => m.id), body },
      {
        onToken: (modelId, text) => {
          setOutputs((prev) => ({
            ...prev,
            [modelId]: { ...(prev[modelId] ?? emptyOutput()), text, streaming: true },
          }));
        },
        onDone: (modelId, meta) => {
          setOutputs((prev) => ({
            ...prev,
            [modelId]: { ...(prev[modelId] ?? emptyOutput()), meta, streaming: false },
          }));
        },
        onError: (modelId, message) => {
          setOutputs((prev) => ({
            ...prev,
            [modelId]: {
              ...(prev[modelId] ?? emptyOutput()),
              errored: true,
              errorMessage: message,
              streaming: false,
            },
          }));
        },
      },
    );
  }, [prompt, mode, orderedSelection, run, inFlight]);

  const onKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        submit();
      }
    },
    [submit],
  );

  return (
    <div className="mx-auto flex w-full max-w-7xl flex-col gap-5 px-4 py-6">
      <div>
        <h2 className="font-display text-base font-bold tracking-wider text-cream-50 uppercase">
          Compare
        </h2>
        <p className="mt-1 text-[12px] text-cream-200/70">
          Send the same prompt to every selected model in parallel. Toggle between RAG (grounded
          on retrieved chunks) and Raw (no chunks) to see what RAG actually buys you.
        </p>
      </div>

      {/* Prompt + controls */}
      <div className="flex flex-col gap-3 rounded-2xl border border-olive-700 bg-forest-900/60 p-4 shadow-lg shadow-black/20">
        <label className="flex flex-col gap-1.5">
          <span className="font-mono text-[11px] uppercase tracking-wider text-cream-200/70">
            {mode === "ask" ? "Query" : "Prompt"}
          </span>
          <textarea
            ref={promptRef}
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={onKeyDown}
            rows={4}
            placeholder={
              mode === "ask"
                ? "How do I read a file with pathlib?"
                : "Once upon a time in the Python standard library…"
            }
            className="resize-none rounded-lg border border-olive-700 bg-forest-950/60 px-3 py-2 font-mono text-[13px] leading-relaxed text-cream-50 placeholder-cream-200/40 focus:border-cream-50/60 focus:outline-none"
          />
        </label>

        <div className="flex flex-wrap items-center gap-3">
          {/* Mode toggle */}
          <div className="inline-flex items-center rounded-full border border-olive-700 bg-forest-950/60 p-0.5">
            {(["ask", "playground"] as CompareMode[]).map((m) => {
              const active = m === mode;
              return (
                <button
                  key={m}
                  type="button"
                  onClick={() => setMode(m)}
                  className={[
                    "rounded-full px-3 py-1 text-[11px] font-medium tracking-wide transition",
                    active
                      ? "bg-cream-50 text-forest-900 shadow shadow-cream-50/15"
                      : "text-cream-200/70 hover:text-cream-50",
                  ].join(" ")}
                >
                  {m === "ask" ? "RAG (with chunks)" : "Raw (no chunks)"}
                </button>
              );
            })}
          </div>

          {/* Submit / cancel */}
          <div className="ml-auto flex gap-2">
            {inFlight ? (
              <button
                type="button"
                onClick={cancel}
                className="rounded-xl bg-olive-700 px-4 py-2 text-[13px] font-medium text-cream-100 transition hover:bg-red-700"
              >
                Stop ({inFlightModels.length} streaming)
              </button>
            ) : (
              <button
                type="button"
                onClick={submit}
                disabled={!prompt.trim() || orderedSelection.length === 0}
                className="rounded-xl bg-cream-50 px-5 py-2 text-[13px] font-semibold text-forest-900 shadow-md shadow-cream-50/15 transition hover:bg-sand-400 disabled:cursor-not-allowed disabled:bg-olive-700 disabled:text-cream-200/40 disabled:shadow-none"
              >
                Run on {orderedSelection.length} model{orderedSelection.length === 1 ? "" : "s"}
              </button>
            )}
          </div>
        </div>

        {/* Model checkboxes */}
        <div className="flex flex-wrap gap-2 border-t border-olive-700/60 pt-3">
          {models.map((m) => {
            const checked = selectedIds.has(m.id);
            return (
              <button
                key={m.id}
                type="button"
                onClick={() => toggleModel(m.id)}
                className={[
                  "rounded-full border px-3 py-1 text-[11px] font-mono tracking-wide transition",
                  checked
                    ? "border-cream-50/60 bg-cream-50/10 text-cream-50"
                    : "border-olive-700 bg-forest-950/60 text-cream-200/60 hover:text-cream-100",
                ].join(" ")}
                title={m.description}
              >
                {checked ? "✓ " : ""}
                {m.label}
              </button>
            );
          })}
          <p className="basis-full font-mono text-[10.5px] tracking-wide text-cream-200/50">
            ⌘/ctrl-enter to run · server serialises per-model so concurrent requests are safe
          </p>
        </div>
      </div>

      {/* Output grid */}
      {orderedSelection.length === 0 ? (
        <div className="rounded-2xl border border-olive-700 bg-forest-950/60 px-4 py-8 text-center font-mono text-[12px] text-cream-200/60">
          Pick at least one model to compare.
        </div>
      ) : (
        <div
          className={[
            "grid gap-4",
            orderedSelection.length === 1 && "grid-cols-1",
            orderedSelection.length === 2 && "grid-cols-1 md:grid-cols-2",
            orderedSelection.length === 3 && "grid-cols-1 md:grid-cols-2 lg:grid-cols-3",
            orderedSelection.length >= 4 && "grid-cols-1 md:grid-cols-2 xl:grid-cols-4",
          ]
            .filter(Boolean)
            .join(" ")}
        >
          {orderedSelection.map((m) => {
            const out = outputs[m.id];
            return <ModelColumn key={m.id} model={m} output={out} mode={mode} />;
          })}
        </div>
      )}
    </div>
  );
}

interface ModelColumnProps {
  model: ModelInfo;
  output: CompareModelOutput | undefined;
  mode: CompareMode;
}

function ModelColumn({ model, output, mode }: ModelColumnProps) {
  const meta = output?.meta;
  return (
    <section className="flex flex-col gap-2 rounded-2xl border border-olive-700 bg-forest-900/40 p-3 shadow-lg shadow-black/20">
      <header className="flex items-start justify-between gap-2 border-b border-olive-700/60 pb-2">
        <div className="min-w-0">
          <h3 className="truncate font-mono text-[12px] font-semibold text-cream-50">
            {model.label}
          </h3>
          <p className="truncate font-mono text-[10.5px] text-cream-200/60" title={model.description}>
            {model.description}
          </p>
        </div>
        {meta && (
          <span className="shrink-0 font-mono text-[10.5px] text-cream-200/70">
            {meta.latency_seconds.toFixed(1)}s
          </span>
        )}
      </header>

      <div
        className={[
          "min-h-[8rem] flex-1 overflow-auto rounded-lg border px-3 py-2 font-mono text-[12px] leading-relaxed",
          output?.errored
            ? "border-red-900/60 bg-red-950/40 text-red-100"
            : "border-olive-700 bg-forest-950/60 text-cream-50",
        ].join(" ")}
      >
        {output?.errored ? (
          <span>Error: {output.errorMessage ?? "unknown"}</span>
        ) : output?.text ? (
          <span className="whitespace-pre-wrap">{output.text}</span>
        ) : output?.streaming ? (
          <span className="text-cream-200/60 italic">streaming…</span>
        ) : (
          <span className="text-cream-200/40 italic">waiting…</span>
        )}
      </div>

      {/* Citations only meaningful in RAG mode. */}
      {mode === "ask" && meta && meta.cited_chunks.length > 0 && (
        <details className="rounded-lg border border-olive-700 bg-forest-950/40 px-2 py-1.5">
          <summary className="cursor-pointer font-mono text-[10.5px] tracking-wide text-cream-200/70">
            {meta.cited_chunks.length} citation{meta.cited_chunks.length === 1 ? "" : "s"}
            {meta.refused && " · refused"}
          </summary>
          <ul className="mt-1.5 flex flex-col gap-1">
            {meta.cited_chunks.map((c, i) => (
              <li key={c.chunk_id} className="font-mono text-[10.5px] text-cream-200/80">
                <span className="text-cream-200/40">[{i + 1}]</span>{" "}
                <a
                  href={c.url}
                  target="_blank"
                  rel="noreferrer"
                  className="text-cream-50 underline-offset-2 hover:underline"
                >
                  {c.title}
                </a>
              </li>
            ))}
          </ul>
        </details>
      )}
      {mode === "ask" && meta && meta.refused && meta.cited_chunks.length === 0 && (
        <p className="font-mono text-[10.5px] text-cream-200/60">refused</p>
      )}
    </section>
  );
}
