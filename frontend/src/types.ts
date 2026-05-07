// Shared types between FastAPI backend and React frontend.
// Mirror src/python_doc_assistant/service/app.py + streaming.py.

export interface AskRequest {
  query: string;
  k?: number;
  rerank?: boolean;
  hyde?: boolean;
  model?: string;
}

export interface CitedChunk {
  chunk_id: string;
  title: string;
  url: string;
  // Single-line, whitespace-collapsed snippet (~280 chars, ellipsised
  // if truncated). Surfaced as a hover preview on citation pills.
  // Optional so older clients / playground events stay valid.
  text_preview?: string;
}

export interface TraceChunk {
  chunk_id: string;
  rank: number;
  score: number;
  title: string;
  url: string;
  cited: boolean;
}

export interface PromptMessage {
  role: string;
  content: string;
}

export interface DonePayload {
  refused: boolean;
  cited_chunks: CitedChunk[];
  latency_seconds: number;
  rewritten_query: string | null;
  model: string | null;
  // Trace fields — present on /api/ask responses, absent (or empty)
  // on /api/playground (no retrieval).
  query_type?: string | null;
  retrieved?: TraceChunk[];
  prompt_messages?: PromptMessage[];
}

export interface ModelInfo {
  id: string;
  label: string;
  description: string;
  max_seq_len: number;
}

export interface ModelsResponse {
  default: string;
  models: ModelInfo[];
}

export interface PlaygroundRequest {
  prompt: string;
  max_tokens?: number;
  temperature?: number;
  model?: string;
}

export type View = "chat" | "playground" | "compare";

export type CompareMode = "ask" | "playground";

export interface CompareModelOutput {
  text: string;
  streaming: boolean;
  meta?: DonePayload;
  errored?: boolean;
  errorMessage?: string;
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
