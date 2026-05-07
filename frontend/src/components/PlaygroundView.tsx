import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { usePlayground } from "../hooks/usePlayground";
import type { DonePayload, ModelInfo } from "../types";

interface PlaygroundViewProps {
  models: ModelInfo[];
  selectedModel: string | null;
}

// Hard ceiling on what the slider exposes regardless of model context.
// Most generators easily handle 1024 but going higher slows things
// down enough that we want the user to opt in deliberately rather
// than nudge a slider.
const SLIDER_HARD_MAX = 1024;
const PROMPT_RESERVE = 16;

const STORAGE_PROMPT = "pdr.playground.prompt";
const STORAGE_MAX = "pdr.playground.maxTokens";
const STORAGE_TEMP = "pdr.playground.temperature";

export function PlaygroundView({ models, selectedModel }: PlaygroundViewProps) {
  const currentModel = useMemo(
    () => models.find((m) => m.id === selectedModel) ?? null,
    [models, selectedModel],
  );
  const sliderMax = useMemo(() => {
    if (!currentModel) return SLIDER_HARD_MAX;
    return Math.max(32, Math.min(SLIDER_HARD_MAX, currentModel.max_seq_len - PROMPT_RESERVE));
  }, [currentModel]);

  const [prompt, setPrompt] = useState<string>(() => localStorage.getItem(STORAGE_PROMPT) ?? "");
  const [maxTokens, setMaxTokens] = useState<number>(
    () => Number(localStorage.getItem(STORAGE_MAX)) || 256,
  );
  const [temperature, setTemperature] = useState<number>(
    () => Number(localStorage.getItem(STORAGE_TEMP)) || 0.0,
  );
  const [output, setOutput] = useState<string>("");
  const [meta, setMeta] = useState<DonePayload | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const { generate, cancel, inFlight } = usePlayground();
  const promptRef = useRef<HTMLTextAreaElement>(null);

  // Persist user inputs.
  useEffect(() => {
    localStorage.setItem(STORAGE_PROMPT, prompt);
  }, [prompt]);
  useEffect(() => {
    localStorage.setItem(STORAGE_MAX, String(maxTokens));
  }, [maxTokens]);
  useEffect(() => {
    localStorage.setItem(STORAGE_TEMP, String(temperature));
  }, [temperature]);

  // Autosize the prompt textarea.
  useEffect(() => {
    const ta = promptRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = `${Math.min(ta.scrollHeight, 280)}px`;
  }, [prompt]);

  // Re-clamp max_tokens when the model (and therefore sliderMax) changes,
  // so a saved 1000 from a previous Qwen session does not stay stuck on
  // a 256-context TinyDocs.
  useEffect(() => {
    if (maxTokens > sliderMax) setMaxTokens(sliderMax);
  }, [sliderMax, maxTokens]);

  const submit = useCallback(() => {
    if (!prompt.trim() || inFlight) return;
    setOutput("");
    setMeta(null);
    setErrorMsg(null);
    void generate(
      {
        prompt,
        max_tokens: maxTokens,
        temperature,
        model: selectedModel ?? undefined,
      },
      {
        onToken: (text) => setOutput(text),
        onDone: (m) => setMeta(m),
        onError: (msg) => {
          setErrorMsg(msg);
        },
      },
    );
  }, [prompt, maxTokens, temperature, selectedModel, generate, inFlight]);

  return (
    <div className="mx-auto flex w-full max-w-3xl flex-col gap-5 px-4 py-6">
      <div>
        <h2 className="font-display text-base font-bold tracking-wider text-cream-50 uppercase">
          Playground
        </h2>
        <p className="mt-1 text-[12px] text-cream-200/70">
          Free-form text completion. No retrieval, no grounding, no citations — the model just
          continues your prompt. Use this to compare a base / SFT-light model against an
          instruction-tuned one.
        </p>
      </div>

      {/* Prompt + sliders */}
      <div className="flex flex-col gap-3 rounded-2xl border border-olive-700 bg-forest-900/60 p-4 shadow-lg shadow-black/20">
        <label className="flex flex-col gap-1.5">
          <span className="font-mono text-[11px] uppercase tracking-wider text-cream-200/70">
            Prompt
          </span>
          <textarea
            ref={promptRef}
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            rows={6}
            placeholder="Once upon a time in the Python standard library…"
            className="resize-none rounded-lg border border-olive-700 bg-forest-950/60 px-3 py-2 font-mono text-[13px] leading-relaxed text-cream-50 placeholder-cream-200/40 focus:border-cream-50/60 focus:outline-none"
          />
        </label>

        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          <label className="flex flex-col gap-1">
            <span className="font-mono text-[10.5px] uppercase tracking-wider text-cream-200/70">
              max_tokens · {maxTokens}
              {currentModel && (
                <span className="ml-1 text-cream-200/40">
                  (cap {sliderMax} for {currentModel.id})
                </span>
              )}
            </span>
            <input
              type="range"
              min={32}
              max={sliderMax}
              step={Math.max(8, Math.floor(sliderMax / 32))}
              value={Math.min(maxTokens, sliderMax)}
              onChange={(e) => setMaxTokens(Number(e.target.value))}
              className="accent-sand-500"
            />
          </label>
          <label className="flex flex-col gap-1">
            <span className="font-mono text-[10.5px] uppercase tracking-wider text-cream-200/70">
              temperature · {temperature.toFixed(2)}
            </span>
            <input
              type="range"
              min={0}
              max={1.5}
              step={0.05}
              value={temperature}
              onChange={(e) => setTemperature(Number(e.target.value))}
              className="accent-sand-500"
            />
          </label>
        </div>
        <p className="font-mono text-[10.5px] tracking-wide text-cream-200/50">
          model: pick from the header dropdown — both models accept playground prompts.
        </p>

        <div className="flex justify-end">
          {inFlight ? (
            <button
              type="button"
              onClick={cancel}
              className="rounded-xl bg-olive-700 px-4 py-2 text-[13px] font-medium text-cream-100 transition hover:bg-red-700"
            >
              Stop
            </button>
          ) : (
            <button
              type="button"
              onClick={submit}
              disabled={!prompt.trim()}
              className="rounded-xl bg-cream-50 px-5 py-2 text-[13px] font-semibold text-forest-900 shadow-md shadow-cream-50/15 transition hover:bg-sand-400 disabled:cursor-not-allowed disabled:bg-olive-700 disabled:text-cream-200/40 disabled:shadow-none"
            >
              Generate
            </button>
          )}
        </div>
      </div>

      {/* Output */}
      <section className="flex flex-col gap-2">
        <div className="flex items-center justify-between">
          <span className="font-mono text-[11px] uppercase tracking-wider text-cream-200/70">
            Output
          </span>
          {meta && (
            <span className="flex items-center gap-2 text-[11px] text-cream-200/70">
              <span className="font-mono">{meta.latency_seconds.toFixed(1)}s</span>
              {meta.model && (
                <span className="rounded-full border border-olive-700 bg-forest-950/60 px-2 py-0.5 font-mono text-[10.5px] text-cream-200/80">
                  {meta.model}
                </span>
              )}
            </span>
          )}
        </div>
        <div
          className={[
            "min-h-[10rem] whitespace-pre-wrap rounded-2xl border px-4 py-3 font-mono text-[13px] leading-relaxed shadow-lg shadow-black/30",
            errorMsg
              ? "border-red-900/60 bg-red-950/40 text-red-100"
              : "border-olive-700 bg-forest-950/60 text-cream-50",
          ].join(" ")}
        >
          {errorMsg ? (
            <span>Error: {errorMsg}</span>
          ) : output ? (
            output
          ) : inFlight ? (
            <span className="text-cream-200/60 italic">generating…</span>
          ) : (
            <span className="text-cream-200/40 italic">
              Output appears here once you hit Generate.
            </span>
          )}
        </div>
      </section>
    </div>
  );
}
