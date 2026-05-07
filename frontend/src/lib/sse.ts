import type { SSEEvent } from "../types";

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

export interface SSEHandlers {
  onToken?: (text: string) => void;
  onDone?: (meta: import("../types").DonePayload) => void;
  onError?: (message: string) => void;
}

// Reads an SSE stream emitted by sse_starlette and dispatches parsed
// events to the handler callbacks. CRLF is normalised to LF so the
// boundary search and parser see consistent line endings regardless
// of the upstream wire format.
export async function readSSEStream(
  resp: Response,
  signal: AbortSignal,
  handlers: SSEHandlers,
): Promise<void> {
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

  try {
    while (true) {
      if (signal.aborted) return;
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
  }
}
