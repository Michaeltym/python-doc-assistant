import { useCallback, useRef, useState } from "react";
import { readSSEStream } from "../lib/sse";
import type {
  AskRequest,
  CompareMode,
  DonePayload,
  PlaygroundRequest,
} from "../types";

interface CompareHandlers {
  onToken?: (modelId: string, text: string) => void;
  onDone?: (modelId: string, meta: DonePayload) => void;
  onError?: (modelId: string, message: string) => void;
}

interface CompareRunRequest {
  mode: CompareMode;
  modelIds: string[];
  // Body without `model` — useCompare injects per-model id.
  body: Omit<AskRequest, "model"> | Omit<PlaygroundRequest, "model">;
}

interface CompareResult {
  run: (req: CompareRunRequest, handlers?: CompareHandlers) => Promise<void>;
  cancel: () => void;
  inFlightModels: string[];
}

// useCompare — fan a prompt out to N models in parallel, each with
// its own AbortController. Server-side per-model `asyncio.Lock`s
// already serialise concurrent requests against the same model, so
// we don't have to gate calls on the client.
export function useCompare(): CompareResult {
  const [inFlight, setInFlight] = useState<Set<string>>(new Set());
  const ctrlsRef = useRef<Map<string, AbortController>>(new Map());

  const cancel = useCallback(() => {
    for (const ctrl of ctrlsRef.current.values()) ctrl.abort();
    ctrlsRef.current.clear();
    setInFlight(new Set());
  }, []);

  const run = useCallback(
    async ({ mode, modelIds, body }: CompareRunRequest, handlers: CompareHandlers = {}) => {
      // Abort anything still in flight from a previous run before starting.
      for (const ctrl of ctrlsRef.current.values()) ctrl.abort();
      ctrlsRef.current.clear();

      const url = mode === "ask" ? "/api/ask" : "/api/playground";
      setInFlight(new Set(modelIds));

      await Promise.all(
        modelIds.map(async (modelId) => {
          const ctrl = new AbortController();
          ctrlsRef.current.set(modelId, ctrl);
          try {
            const resp = await fetch(url, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ ...body, model: modelId }),
              signal: ctrl.signal,
            });
            await readSSEStream(resp, ctrl.signal, {
              onToken: (text) => handlers.onToken?.(modelId, text),
              onDone: (meta) => handlers.onDone?.(modelId, meta),
              onError: (message) => handlers.onError?.(modelId, message),
            });
          } catch (err) {
            if ((err as Error).name === "AbortError") return;
            handlers.onError?.(modelId, (err as Error).message);
          } finally {
            if (ctrlsRef.current.get(modelId) === ctrl) {
              ctrlsRef.current.delete(modelId);
            }
            setInFlight((prev) => {
              const next = new Set(prev);
              next.delete(modelId);
              return next;
            });
          }
        }),
      );
    },
    [],
  );

  return { run, cancel, inFlightModels: Array.from(inFlight) };
}
