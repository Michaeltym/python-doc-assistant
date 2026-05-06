"""Tests for `python_doc_assistant.service.app` (FastAPI endpoints)."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any

import pytest

from python_doc_assistant.evaluation.retrieval_metrics import RetrievedChunk
from python_doc_assistant.generation.interface import Answer, Generator
from python_doc_assistant.ingest.chunker import Chunk
from python_doc_assistant.retrieval.router import QueryType
from python_doc_assistant.service.app import AskRequest, AskState, ModelEntry, build_app

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


def _chunk(chunk_id: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        chunk_type="symbol",
        docs_version="3.12",
        title=chunk_id.split(":", 1)[-1],
        text=f"docs for {chunk_id}",
        symbols=(chunk_id.split(":", 1)[-1],),
        canonical_url=f"library/foo.html#{chunk_id}",
        anchor=chunk_id,
        parent_module=None,
        source_path="library/foo.html",
        source_hash="sha256:abc",
    )


@dataclass
class StubGenerator(Generator):
    """Returns canned Answer; records each call."""

    canned_answer: Answer = field(
        default_factory=lambda: Answer(
            text="canned answer [1]",
            cited_chunk_ids=("symbol:foo",),
            refused=False,
            latency_seconds=0.001,
        )
    )
    calls: list[dict[str, Any]] = field(default_factory=list)
    sleep_seconds: float = 0.0
    temperature: float = 0.0
    top_p: float = 1.0
    max_new_tokens: int = 512

    def generate(
        self,
        query: str,
        retrieved_chunks: list[Chunk],
        *,
        query_type: QueryType | None = None,
        stream: bool = False,
    ) -> Answer:
        self.calls.append(
            {"query": query, "n_chunks": len(retrieved_chunks), "query_type": query_type}
        )
        if self.sleep_seconds:
            time.sleep(self.sleep_seconds)
        return self.canned_answer


def _make_state(
    *,
    generator: Generator | None = None,
    chunks: list[Chunk] | None = None,
    retrieved_ids: tuple[str, ...] = ("symbol:foo",),
) -> AskState:
    if chunks is None:
        chunks = [_chunk(cid) for cid in retrieved_ids]
    chunks_by_id = {c.chunk_id: c for c in chunks}

    def retrieve_fn(query: str, k: int) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                chunk_id=c.chunk_id,
                score=1.0 - 0.1 * i,
                rank=i + 1,
                canonical_url=c.canonical_url,
                symbols=c.symbols,
            )
            for i, c in enumerate(chunks[:k])
        ]

    gen = generator if generator is not None else StubGenerator()
    return AskState(
        models={
            "stub": ModelEntry(
                generator=gen,
                lock=asyncio.Lock(),
                label="Stub",
                description="test stub",
            )
        },
        default_model="stub",
        retrieve_fn=retrieve_fn,
        chunks_by_id=chunks_by_id,
    )


@pytest.fixture
def client():
    """TestClient over the FastAPI app for endpoint contract tests."""
    from fastapi.testclient import TestClient

    state = _make_state()
    app = build_app(state)
    return TestClient(app), state


# ------------------------------------------------------------------
# AskRequest schema
# ------------------------------------------------------------------


def test_ask_request_defaults() -> None:
    req = AskRequest(query="hi")
    assert req.k == 5
    assert req.rerank is True
    assert req.hyde is True


def test_ask_request_rejects_empty_query() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        AskRequest(query="")


def test_ask_request_rejects_query_too_long() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        AskRequest(query="x" * 2001)


def test_ask_request_rejects_k_out_of_range() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        AskRequest(query="hi", k=0)
    with pytest.raises(ValidationError):
        AskRequest(query="hi", k=21)


# ------------------------------------------------------------------
# /health
# ------------------------------------------------------------------


def test_health_returns_ok(client) -> None:
    test_client, _ = client
    resp = test_client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ------------------------------------------------------------------
# /api/models
# ------------------------------------------------------------------


def test_api_models_returns_default_and_list(client) -> None:
    test_client, _ = client
    resp = test_client.get("/api/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["default"] == "stub"
    assert len(data["models"]) == 1
    m = data["models"][0]
    assert m["id"] == "stub"
    assert m["label"] == "Stub"
    assert "description" in m


def test_api_ask_done_carries_model_field(client) -> None:
    test_client, _ = client
    resp = test_client.post("/api/ask", json={"query": "what is json.loads"})
    done_payloads = [p for n, p in _parse_sse(resp.text) if n == "done"]
    assert done_payloads[0]["model"] == "stub"


def test_api_ask_unknown_model_yields_error_event(client) -> None:
    test_client, _ = client
    resp = test_client.post("/api/ask", json={"query": "q", "model": "does-not-exist"})
    events = _parse_sse(resp.text)
    names = [name for name, _ in events]
    assert "error" in names
    err = next(p for n, p in events if n == "error")
    assert "does-not-exist" in err["message"]


def test_api_ask_explicit_default_model_works(client) -> None:
    test_client, _ = client
    resp = test_client.post("/api/ask", json={"query": "q", "model": "stub"})
    done_payloads = [p for n, p in _parse_sse(resp.text) if n == "done"]
    assert done_payloads[0]["model"] == "stub"


# ------------------------------------------------------------------
# /api/ask happy path
# ------------------------------------------------------------------


def _parse_sse(text: str) -> list[tuple[str, dict]]:
    """Parse an SSE response body into (event_name, payload) tuples."""
    events: list[tuple[str, dict]] = []
    current_event = "message"
    current_data_lines: list[str] = []
    for line in text.split("\n"):
        if line.startswith("event:"):
            current_event = line[len("event:") :].strip()
        elif line.startswith("data:"):
            current_data_lines.append(line[len("data:") :].lstrip())
        elif line.strip() == "":
            if current_data_lines:
                events.append((current_event, json.loads("\n".join(current_data_lines))))
            current_event = "message"
            current_data_lines = []
    return events


def test_api_ask_returns_token_then_done(client) -> None:
    test_client, _ = client
    resp = test_client.post("/api/ask", json={"query": "what is json.loads"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse(resp.text)
    names = [name for name, _ in events]
    assert names[0] == "token"
    assert names[-1] == "done"


def test_api_ask_token_payload_matches_generator_answer(client) -> None:
    test_client, _ = client
    resp = test_client.post("/api/ask", json={"query": "what is json.loads"})
    events = _parse_sse(resp.text)
    token_payloads = [p for n, p in events if n == "token"]
    assert any(p["text"] == "canned answer [1]" for p in token_payloads)


def test_api_ask_done_carries_metadata(client) -> None:
    test_client, _ = client
    resp = test_client.post("/api/ask", json={"query": "what is json.loads"})
    events = _parse_sse(resp.text)
    done_payloads = [p for n, p in events if n == "done"]
    assert len(done_payloads) == 1
    p = done_payloads[0]
    assert p["refused"] is False
    assert len(p["cited_chunks"]) == 1
    cited = p["cited_chunks"][0]
    assert cited["chunk_id"] == "symbol:foo"
    assert cited["url"].startswith("https://docs.python.org/")
    assert "title" in cited
    assert isinstance(p["latency_seconds"], (int, float))


def test_api_ask_includes_rewritten_query_field(client) -> None:
    test_client, _ = client
    resp = test_client.post("/api/ask", json={"query": "what is json.loads"})
    done_payloads = [p for n, p in _parse_sse(resp.text) if n == "done"]
    assert "rewritten_query" in done_payloads[0]


# ------------------------------------------------------------------
# /api/ask validation
# ------------------------------------------------------------------


def test_api_ask_rejects_empty_query(client) -> None:
    test_client, _ = client
    resp = test_client.post("/api/ask", json={"query": ""})
    assert resp.status_code == 422


def test_api_ask_rejects_missing_query(client) -> None:
    test_client, _ = client
    resp = test_client.post("/api/ask", json={})
    assert resp.status_code == 422


# ------------------------------------------------------------------
# Concurrency: lock serialises calls
# ------------------------------------------------------------------


def test_concurrent_requests_serialise_via_lock() -> None:
    """Two concurrent /api/ask requests should not interleave generator calls.

    With the lock holding for ~50 ms per call, two parallel requests
    should take ≥ 90 ms total wall-clock (serial), not ≤ 70 ms
    (concurrent).
    """
    import threading

    from fastapi.testclient import TestClient

    gen = StubGenerator(sleep_seconds=0.05)
    state = _make_state(generator=gen)
    app = build_app(state)
    test_client = TestClient(app)

    durations: list[float] = []
    barrier = threading.Barrier(2)

    def hit() -> None:
        barrier.wait()
        t0 = time.perf_counter()
        resp = test_client.post("/api/ask", json={"query": "q"})
        durations.append(time.perf_counter() - t0)
        assert resp.status_code == 200

    threads = [threading.Thread(target=hit), threading.Thread(target=hit)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    total = time.perf_counter() - t0

    assert len(gen.calls) == 2
    assert total >= 0.09, f"Two 50 ms-locked calls served in {total:.3f} s — lock did not serialise"


# ------------------------------------------------------------------
# State injection
# ------------------------------------------------------------------


def test_build_app_attaches_state(client) -> None:
    test_client, state = client
    assert test_client.app.state.shared is state


# ------------------------------------------------------------------
# Static frontend mount
# ------------------------------------------------------------------


def test_no_static_mount_when_static_root_unset(client) -> None:
    test_client, _ = client
    resp = test_client.get("/")
    # Without a frontend build, root should 404 (no static mount).
    assert resp.status_code == 404


def test_static_mount_serves_index_html(tmp_path) -> None:
    from fastapi.testclient import TestClient

    static = tmp_path / "dist"
    static.mkdir()
    (static / "index.html").write_text("<!doctype html>STUB UI")

    state = _make_state()
    state.static_root = static
    app = build_app(state)
    test_client = TestClient(app)
    resp = test_client.get("/")
    assert resp.status_code == 200
    assert "STUB UI" in resp.text
