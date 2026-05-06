// Shared types between FastAPI backend and React frontend.
// Mirror src/python_doc_assistant/service/app.py + streaming.py.

export interface AskRequest {
  query: string;
  k?: number;
  rerank?: boolean;
  hyde?: boolean;
}

export interface DonePayload {
  refused: boolean;
  cited_chunk_ids: string[];
  latency_seconds: number;
  rewritten_query: string | null;
}

export interface TokenPayload {
  text: string;
}

export interface ErrorPayload {
  message: string;
}

export type SSEEvent =
  | { type: "token"; payload: TokenPayload }
  | { type: "done"; payload: DonePayload }
  | { type: "error"; payload: ErrorPayload };

export type Role = "user" | "assistant" | "system";

export interface Message {
  id: string;
  role: Role;
  text: string;
  // Only present for assistant messages once the `done` event arrives:
  meta?: DonePayload;
  // True while the message is still being streamed.
  streaming?: boolean;
  // True if the request errored mid-flight.
  errored?: boolean;
}
