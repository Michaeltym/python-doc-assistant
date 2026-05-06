# frontend

React + TypeScript + Vite + Tailwind chat UI for python-doc-assistant.

## Dev

Two terminals:

```bash
# 1. Backend (FastAPI) — listens on :8000
uv run --all-extras pdr serve \
    --gguf-model data/models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf \
    --retriever dense --rerank --hyde --port 8000

# 2. Frontend (Vite dev server) — listens on :5173, proxies /api -> :8000
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 in a browser.

## Production build

```bash
cd frontend
npm install
npm run build   # writes frontend/dist/
```

The FastAPI app auto-mounts `frontend/dist/` at `/` when the directory
exists, so `pdr serve` then serves both the API (`/api/*`) and the UI
(`/`) from a single port.

## Stack

- **Vite 6** + React 19 + TypeScript 5.9
- **Tailwind CSS 3** for styling (no UI component library)
- **Native `fetch` + `ReadableStream`** for SSE (EventSource cannot do POST)
- No router, no state library — single-page chat with `useState`

## Files

- `src/App.tsx` — top-level chat container, holds the `messages` array.
- `src/components/ChatBox.tsx` — autosizing textarea + send/stop button.
- `src/components/MessageList.tsx` — message bubbles + auto-scroll.
- `src/components/Citation.tsx` — chunk_id pill rendered under assistant messages.
- `src/hooks/useAsk.ts` — POST `/api/ask`, parse SSE byte stream, dispatch token / done / error events.
- `src/types.ts` — TypeScript mirrors of the FastAPI request/response schemas.
