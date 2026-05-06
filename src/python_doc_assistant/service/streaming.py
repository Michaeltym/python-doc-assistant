"""SSE event helpers for the /api/ask streaming endpoint.

Event types (sent over `text/event-stream`):

    event: token   — one chunk of answer text (MVP yields the full
                     answer as a single token event; sub-task 7 will
                     swap this for real per-token streaming).
    event: done    — end-of-stream marker with final metadata.
    event: error   — surfaces a server-side failure to the client.

Each helper returns a dict shaped for `sse_starlette.sse.EventSourceResponse`:

    {"event": "<event-name>", "data": "<json-encoded payload>"}

Keeping the helpers thin + pure makes the FastAPI endpoint trivial to
test (assert the dicts produced for a given Answer / state).
"""

from __future__ import annotations

import json
from typing import Final

EVENT_TOKEN: Final[str] = "token"
EVENT_DONE: Final[str] = "done"
EVENT_ERROR: Final[str] = "error"


def token_event(text: str) -> dict[str, str]:
    """Build an SSE event for a chunk of answer text.

    Args:
        text: the substring of the answer to emit.

    Returns:
        ``{"event": "token", "data": '{"text": "<text>"}'}`` — `data`
        is JSON-encoded so the client can `JSON.parse(e.data).text`.

    Implementation outline:
        1. payload = {"text": text}
        2. return {"event": EVENT_TOKEN, "data": json.dumps(payload)}
    """
    return {"event": EVENT_TOKEN, "data": json.dumps({"text": text})}


def done_event(
    *,
    refused: bool,
    cited_chunks: tuple[dict[str, str], ...],
    latency_seconds: float,
    rewritten_query: str | None = None,
    model: str | None = None,
) -> dict[str, str]:
    """Build the terminal SSE event with metadata.

    Args:
        refused: True iff the model emitted the refusal marker.
        cited_chunks: tuple of `{"chunk_id", "title", "url"}` dicts —
            one per cited chunk. The frontend renders these as link
            pills pointing at the corresponding docs.python.org page.
        latency_seconds: wall-clock seconds for the full ask call.
        rewritten_query: when the typo rewriter fired, the rewritten
            query string. None when the original query was used.
        model: id of the model that produced the answer (e.g.
            "qwen-7b-gguf" or "tinydocs"). The frontend uses this for
            the "answered by" footer chip.

    Returns:
        ``{"event": "done", "data": "<json>"}`` where the JSON payload
        contains ``refused``, ``cited_chunks``, ``latency_seconds``,
        ``rewritten_query``, and ``model``.
    """
    return {
        "event": EVENT_DONE,
        "data": json.dumps(
            {
                "refused": refused,
                "cited_chunks": list(cited_chunks),
                "latency_seconds": latency_seconds,
                "rewritten_query": rewritten_query,
                "model": model,
            }
        ),
    }


def error_event(message: str) -> dict[str, str]:
    """Build an error SSE event so the client can show a failure toast.

    Args:
        message: human-readable error text. Do NOT include stack traces
            or secrets — this string is rendered to the user.

    Returns:
        ``{"event": "error", "data": '{"message": "<message>"}'}``.

    Implementation outline:
        1. payload = {"message": message}
        2. return {"event": EVENT_ERROR, "data": json.dumps(payload)}
    """
    return {
        "event": EVENT_ERROR,
        "data": json.dumps({"message": message}),
    }


# ------------------------------------------------------------------
# Used by tests / clients to keep the JSON encoding consistent.
# ------------------------------------------------------------------


def _decode(event: dict[str, str]) -> tuple[str, dict[str, object]]:
    """Test helper: returns (event_name, decoded_payload)."""
    return event["event"], json.loads(event["data"])
