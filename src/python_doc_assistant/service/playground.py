"""Playground endpoint helpers — ungrounded text completion.

The v4 production stack (`/api/ask`) wraps every query in the grounded
prompt + retrieval pipeline. The playground does the opposite: takes a
free-form prompt and asks the selected generator for a raw continuation
with no system message, no chunks, no citations. Lets the UI show off
the v3.1 TinyDocs base-LM (which has no instruction tuning) side-by-side
with Qwen 7B.

Reuses ``AskState`` so the same model registry + per-model
``asyncio.Lock`` serve both endpoints. Output streams as a single
``token`` event followed by a ``done`` event, matching the SSE shape
``/api/ask`` already uses (frontend's ``useAsk``-style parser handles
both).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from python_doc_assistant.service.app import AskState


# ------------------------------------------------------------------
# Request schema
# ------------------------------------------------------------------


class PlaygroundRequest(BaseModel):
    """Body schema for ``POST /api/playground``."""

    prompt: str = Field(..., min_length=1, max_length=4000)
    max_tokens: int = Field(256, ge=8, le=1024)
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    model: str | None = None


# ------------------------------------------------------------------
# Streaming handler
# ------------------------------------------------------------------


async def _playground_stream(
    state: AskState, request: PlaygroundRequest
) -> AsyncIterator[dict[str, str]]:
    """Yield SSE event dicts for one playground call.

    Steps:
        1. Resolve model_id (request.model or state.default_model).
           Yield ``error`` event + return when unknown.
        2. Acquire ``state.models[model_id].lock`` so the playground
           and ``/api/ask`` queue behind the same per-model lock.
        3. ``import time`` and start the wall-clock timer inside the
           lock.
        4. Call ``entry.generator.generate_raw(prompt, max_tokens=...,
           temperature=...)``. Catch any exception, yield an
           ``error`` event with ``str(exc)``, and return.
        5. Yield a ``token`` event carrying the full continuation text.
        6. Yield a ``done`` event with::

               refused          = False
               cited_chunks     = ()
               latency_seconds  = time.perf_counter() - start
                                  (or completion.latency_seconds —
                                  either is acceptable; document
                                  whichever is used)
               rewritten_query  = None
               model            = model_id

    Yields:
        Dicts shaped for ``EventSourceResponse``:
            ``{"event": "token", "data": "..."}``
            ``{"event": "done",  "data": "..."}``

    Implementation imports:
        import time
        from python_doc_assistant.service.streaming import (
            done_event,
            error_event,
            token_event,
        )
    """
    import time

    from python_doc_assistant.service.streaming import done_event, error_event, token_event

    model_id = request.model or state.default_model
    entry = state.models.get(model_id)
    if entry is None:
        yield error_event(f"unknown model {model_id!r}; available: {sorted(state.models)!r}")
        return

    async with entry.lock:
        start = time.perf_counter()
        try:
            completion = entry.generator.generate_raw(
                request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )
        except Exception as exc:  # noqa: BLE001
            yield error_event(str(exc))
            return

        yield token_event(completion.text)
        yield done_event(
            refused=False,
            cited_chunks=(),
            latency_seconds=time.perf_counter() - start,
            rewritten_query=None,
            model=model_id,
        )
