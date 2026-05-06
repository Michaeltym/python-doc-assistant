import { useCallback, useRef, useState } from "react";
import type { AskRequest, DonePayload, SSEEvent } from "../types";

interface UseAskHandlers {
  onToken?: (text: string) => void;
  onDone?: (meta: DonePayload) => void;
  onError?: (message: string) => void;
}

interface UseAskResult {
  ask: (req: AskRequest, handlers?: UseAskHandlers) => Promise<void>;
  cancel: () => void;
  inFlight: boolean;
}

// Parse a single SSE block of the form:
//   event: token
//   data: {"text": "..."}
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
  if (event === "token") {
    return { type: "token", payload: payload as SSEEvent["payload"] } as SSEEvent;
  }
  if (event === "done") {
    return { type: "done", payload: payload as SSEEvent["payload"] } as SSEEvent;
  }
  if (event === "error") {
    return { type: "error", payload: payload as SSEEvent["payload"] } as SSEEvent;
  }
  return null;
}

/**
 * useAsk — POST /api/ask, parse the SSE response stream, and surface
 * token / done / error events via callbacks.
 *
 * EventSource cannot do POST, so we use fetch + ReadableStream and
 * parse the SSE byte stream by hand. Each parsed event is dispatched
 * to the handler the caller provides.
 */
export function useAsk(): UseAskResult {
  const [inFlight, setInFlight] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const ask = useCallback(async (req: AskRequest, handlers: UseAskHandlers = {}) => {
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    setInFlight(true);

    try {
      const resp = await fetch("/api/ask", {
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

      // SSE events are separated by a blank line. The HTML5 SSE spec
      // permits any of \n\n, \r\n\r\n, or \r\r as the event boundary;
      // sse_starlette emits \r\n\r\n. Normalise CRLF → LF so the
      // boundary search and parseBlock both see consistent line endings.
      const findEventBoundary = (buf: string): number => buf.indexOf("\n\n");

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");

        let sep = findEventBoundary(buffer);
        while (sep !== -1) {
          const block = buffer.slice(0, sep);
          buffer = buffer.slice(sep + 2);
          const ev = parseBlock(block);
          if (ev) {
            if (ev.type === "token") handlers.onToken?.(ev.payload.text);
            else if (ev.type === "done") handlers.onDone?.(ev.payload);
            else if (ev.type === "error") handlers.onError?.(ev.payload.message);
          }
          sep = findEventBoundary(buffer);
        }
      }
    } catch (err) {
      if ((err as Error).name === "AbortError") return;
      handlers.onError?.((err as Error).message);
    } finally {
      setInFlight(false);
      if (abortRef.current === ctrl) abortRef.current = null;
    }
  }, []);

  const cancel = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setInFlight(false);
  }, []);

  return { ask, cancel, inFlight };
}
