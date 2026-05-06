"""FastAPI app for python-doc-assistant.

Endpoints:
    POST /api/ask   — body: AskRequest. Returns SSE stream of token +
                       done events (see service/streaming.py).
    GET  /health    — liveness check.
    GET  /          — static React build (when frontend/dist/ exists).

Concurrency:
    A single QwenGGUFGenerator + retrieve_fn is shared across requests.
    `llama-cpp-python`'s `Llama` is not thread-safe, so the endpoint
    serialises ask calls behind an `asyncio.Lock` held in `AskState`.
    Multiple concurrent clients queue; throughput stays single-stream.

State injection:
    The CLI subcommand (`pdr serve`) constructs an `AskState`, then
    passes it to `build_app(state)`. Tests inject a stub state with a
    fake generator + retrieve_fn so they don't load real models.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from python_doc_assistant.evaluation.retrieval_metrics import RetrievedChunk
from python_doc_assistant.ingest.chunker import Chunk

if TYPE_CHECKING:
    from fastapi import FastAPI

    from python_doc_assistant.generation.interface import Generator


# ------------------------------------------------------------------
# Request / state schemas
# ------------------------------------------------------------------


class AskRequest(BaseModel):
    """Body schema for POST /api/ask."""

    query: str = Field(..., min_length=1, max_length=2000)
    k: int = Field(5, ge=1, le=20)
    rerank: bool = True
    hyde: bool = True


@dataclass
class AskState:
    """Shared state attached to the FastAPI app via `app.state.shared`.

    Attributes:
        generator: loaded `Generator` (typically QwenGGUFGenerator).
        retrieve_fn: rank-K retriever closure built by the CLI for the
            requested config (dense / dense+rerank / dense+rerank+HyDE).
        chunks_by_id: full chunk lookup. Used by the typo rewriter and
            to map cited indices back to chunk_ids.
        lock: `asyncio.Lock` serialising generate calls (Llama is not
            thread-safe).
        static_root: optional path to a built React frontend
            (`frontend/dist/`). When set + exists, mounted at `/`.
    """

    generator: Generator
    retrieve_fn: Callable[[str, int], list[RetrievedChunk]]
    chunks_by_id: dict[str, Chunk]
    lock: asyncio.Lock
    static_root: Path | None = None


# ------------------------------------------------------------------
# App factory
# ------------------------------------------------------------------


def build_app(state: AskState) -> FastAPI:
    """Construct and return the FastAPI application.

    Wires:
        - POST /api/ask → SSE stream (see `_ask_stream` below).
        - GET /health → `{"status": "ok"}`.
        - GET / (static) → mounted only when `state.static_root` is set
          and exists. Otherwise return a 404 with a hint to build the
          frontend.

    Args:
        state: pre-built `AskState`. The caller is responsible for
            loading the generator + retriever before constructing the
            app.

    Returns:
        FastAPI app with `app.state.shared = state`.

    Implementation outline:
        1. from fastapi import FastAPI, HTTPException, Request
           from fastapi.staticfiles import StaticFiles
           from sse_starlette.sse import EventSourceResponse
        2. app = FastAPI(title="python-doc-assistant")
        3. app.state.shared = state
        4. @app.get("/health") → return {"status": "ok"}
        5. @app.post("/api/ask") → call _ask_stream(state, body) and
           wrap in EventSourceResponse(generator).
        6. If state.static_root and state.static_root.is_dir():
               app.mount("/", StaticFiles(directory=state.static_root,
                         html=True), name="frontend")
        7. return app

    All FastAPI / sse_starlette imports MUST stay inside this function
    so importing `python_doc_assistant.service` (e.g. by tests) doesn't
    require the `service` extra to be installed at module import time.
    """
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles
    from sse_starlette.sse import EventSourceResponse

    app = FastAPI(title="python-doc-assistant")
    app.state.shared = state

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/ask")
    async def ask(request: AskRequest) -> EventSourceResponse:
        return EventSourceResponse(_ask_stream(state, request))

    # Static mount MUST come after API routes — mount("/") catches every
    # path, so registering it earlier would shadow /health and /api/ask.
    if state.static_root and state.static_root.is_dir():
        app.mount("/", StaticFiles(directory=state.static_root, html=True), name="frontend")

    return app


# ------------------------------------------------------------------
# /api/ask handler core (extracted for testability)
# ------------------------------------------------------------------


async def _ask_stream(state: AskState, request: AskRequest) -> AsyncIterator[dict[str, str]]:
    """Async generator yielding SSE event dicts for a single ask call.

    Steps:
        1. Acquire `state.lock` (serialise concurrent requests).
        2. Compute retrieved = state.retrieve_fn(request.query, request.k).
        3. gen_chunks = [state.chunks_by_id[r.chunk_id] for r in
                         retrieved if r.chunk_id in state.chunks_by_id]
        4. rewritten = maybe_rewrite_query(request.query, gen_chunks)
        5. qt = classify(request.query)
        6. start = time.perf_counter()
        7. answer = state.generator.generate(rewritten, gen_chunks,
                                              query_type=qt)
        8. yield token_event(answer.text or "[INSUFFICIENT-CONTEXT]")
        9. yield done_event(
               refused=answer.refused,
               cited_chunk_ids=answer.cited_chunk_ids,
               latency_seconds=time.perf_counter() - start,
               rewritten_query=rewritten if rewritten != request.query else None,
           )
       10. On exception: yield error_event(str(exc)) and re-raise (or
           swallow + log; the lock release happens in the finally
           clause of the async with block).

    Yields:
        dicts shaped for EventSourceResponse, e.g.
            {"event": "token", "data": "..."},
            {"event": "done",  "data": "..."}.

    Implementation imports:
        from python_doc_assistant.prompts.grounded import REFUSAL_MARKER  # if needed
        from python_doc_assistant.retrieval.query_rewriter import maybe_rewrite_query
        from python_doc_assistant.retrieval.router import classify
        from python_doc_assistant.service.streaming import (
            done_event, error_event, token_event,
        )
        import time
    """
    import time

    from python_doc_assistant.retrieval.query_rewriter import maybe_rewrite_query
    from python_doc_assistant.retrieval.router import classify
    from python_doc_assistant.service.streaming import done_event, error_event, token_event

    async with state.lock:
        start = time.perf_counter()
        try:
            retrieved = state.retrieve_fn(request.query, request.k)
            gen_chunks = [
                state.chunks_by_id[r.chunk_id]
                for r in retrieved
                if r.chunk_id in state.chunks_by_id
            ]
            rewritten = maybe_rewrite_query(request.query, gen_chunks)
            qt = classify(request.query)
            answer = state.generator.generate(rewritten, gen_chunks, query_type=qt)
        except Exception as exc:  # noqa: BLE001
            yield error_event(str(exc))
            return

        cited_chunks = tuple(
            {
                "chunk_id": cid,
                "title": state.chunks_by_id[cid].title,
                "url": (
                    f"https://docs.python.org/{state.chunks_by_id[cid].docs_version}/"
                    f"{state.chunks_by_id[cid].canonical_url}"
                ),
            }
            for cid in answer.cited_chunk_ids
            if cid in state.chunks_by_id
        )
        yield token_event(answer.text or "[INSUFFICIENT-CONTEXT]")
        yield done_event(
            refused=answer.refused,
            cited_chunks=cited_chunks,
            latency_seconds=time.perf_counter() - start,
            rewritten_query=rewritten if rewritten != request.query else None,
        )
