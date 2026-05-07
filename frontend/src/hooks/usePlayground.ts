import { useCallback, useRef, useState } from "react";
import type { DonePayload, PlaygroundRequest, SSEEvent } from "../types";

interface UsePlaygroundHandlers {
  onToken?: (text: string) => void;
  onDone?: (meta: DonePayload) => void;
  onError?: (message: string) => void;
}

interface UsePlaygroundResult {
  generate: (req: PlaygroundRequest, handlers?: UsePlaygroundHandlers) => Promise<void>;
  cancel: () => void;
  inFlight: boolean;
}

function parseBlock(block: string): SSEEvent | null {
  const lines = block.split("\n");
  let event = "message";
  const dataLines: string[] = [];
  for (const raw of lines) {
    const line = raw.trimEnd();
    if (line.startsWith("event:")) {
      event = line.slice("event:".length).trim();
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice("data:".length).replace(/^\s/, ""));
    }
  }
  if (dataLines.length === 0) return null;
  let payload: unknown;
  try {
    payload = JSON.parse(dataLines.join("\n"));
  } catch {
    return null;
  }
  if (event === "token" || event === "done" || event === "error") {
    return { type: event, payload: payload as SSEEvent["payload"] } as SSEEvent;
  }
  return null;
}

/**
 * usePlayground — POST /api/playground, parse the SSE byte stream,
 * dispatch token / done / error events to the caller.
 *
 * Mirrors useAsk's transport (fetch + ReadableStream + manual SSE
 * parser, CRLF normalised to LF) since the server uses the same
 * sse_starlette wrapping for both endpoints.
 */
export function usePlayground(): UsePlaygroundResult {
  const [inFlight, setInFlight] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const generate = useCallback(
    async (req: PlaygroundRequest, handlers: UsePlaygroundHandlers = {}) => {
      abortRef.current?.abort();
      const ctrl = new AbortController();
      abortRef.current = ctrl;
      setInFlight(true);

      try {
        const resp = await fetch("/api/playground", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(req),
          signal: ctrl.signal,
        });
        if (!resp.ok) {
          handlers.onError?.(`HTTP ${resp.status}`);
          return;
        }
        if (!resp.body) {
          handlers.onError?.("response had no body");
          return;
        }

        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");

          let sep = buffer.indexOf("\n\n");
          while (sep !== -1) {
            const block = buffer.slice(0, sep);
            buffer = buffer.slice(sep + 2);
            const ev = parseBlock(block);
            if (ev) {
              if (ev.type === "token") handlers.onToken?.(ev.payload.text);
              else if (ev.type === "done") handlers.onDone?.(ev.payload);
              else if (ev.type === "error") handlers.onError?.(ev.payload.message);
            }
            sep = buffer.indexOf("\n\n");
          }
        }
      } catch (err) {
        if ((err as Error).name === "AbortError") return;
        handlers.onError?.((err as Error).message);
      } finally {
        setInFlight(false);
        if (abortRef.current === ctrl) abortRef.current = null;
      }
    },
    [],
  );

  const cancel = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setInFlight(false);
  }, []);

  return { generate, cancel, inFlight };
}
