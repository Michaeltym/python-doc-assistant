"""Tests for `python_doc_assistant.service.playground` (raw text completion)."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any

import pytest

from python_doc_assistant.evaluation.retrieval_metrics import RetrievedChunk
from python_doc_assistant.generation.interface import Answer, Generator, RawCompletion
from python_doc_assistant.ingest.chunker import Chunk
from python_doc_assistant.retrieval.router import QueryType
from python_doc_assistant.service.app import AskState, ModelEntry, build_app
from python_doc_assistant.service.playground import PlaygroundRequest

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
    """Generator stub that records `generate_raw` calls."""

    raw_text: str = "stub continuation"
    raw_calls: list[dict[str, Any]] = field(default_factory=list)
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
        return Answer(text="ignored", cited_chunk_ids=(), refused=False, latency_seconds=0.0)

    def generate_raw(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> RawCompletion:
        self.raw_calls.append(
            {"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature}
        )
        if self.sleep_seconds:
            time.sleep(self.sleep_seconds)
        return RawCompletion(text=self.raw_text, latency_seconds=0.001)


def _make_state(generator: Generator | None = None) -> AskState:
    chunks = [_chunk("symbol:foo")]
    chunks_by_id = {c.chunk_id: c for c in chunks}

    def retrieve_fn(query: str, k: int) -> list[RetrievedChunk]:
        return []

    gen = generator if generator is not None else StubGenerator()
    return AskState(
        models={
            "stub": ModelEntry(
                generator=gen,
                lock=asyncio.Lock(),
                label="Stub",
                description="test stub",
                max_seq_len=1024,
            )
        },
        default_model="stub",
        retrieve_fn=retrieve_fn,
        chunks_by_id=chunks_by_id,
    )


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    state = _make_state()
    app = build_app(state)
    return TestClient(app), state


def _parse_sse(text: str) -> list[tuple[str, dict[str, Any]]]:
    events: list[tuple[str, dict[str, Any]]] = []
    current_event = "message"
    current_data: list[str] = []
    for raw in text.replace("\r\n", "\n").split("\n"):
        if raw.startswith("event:"):
            current_event = raw[len("event:") :].strip()
        elif raw.startswith("data:"):
            current_data.append(raw[len("data:") :].lstrip())
        elif raw.strip() == "":
            if current_data:
                events.append((current_event, json.loads("\n".join(current_data))))
            current_event = "message"
            current_data = []
    return events


# ------------------------------------------------------------------
# PlaygroundRequest schema
# ------------------------------------------------------------------


def test_playground_request_defaults() -> None:
    req = PlaygroundRequest(prompt="hello")
    assert req.max_tokens == 256
    assert req.temperature == 0.0
    assert req.model is None


def test_playground_request_rejects_empty_prompt() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        PlaygroundRequest(prompt="")


def test_playground_request_rejects_max_tokens_out_of_range() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        PlaygroundRequest(prompt="x", max_tokens=0)
    with pytest.raises(ValidationError):
        PlaygroundRequest(prompt="x", max_tokens=2048)


def test_playground_request_rejects_temperature_out_of_range() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        PlaygroundRequest(prompt="x", temperature=-0.1)
    with pytest.raises(ValidationError):
        PlaygroundRequest(prompt="x", temperature=2.5)


# ------------------------------------------------------------------
# /api/playground happy path
# ------------------------------------------------------------------


def test_api_playground_returns_token_then_done(client) -> None:
    test_client, _ = client
    resp = test_client.post("/api/playground", json={"prompt": "Once upon a time"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse(resp.text)
    names = [name for name, _ in events]
    assert names[0] == "token"
    assert names[-1] == "done"


def test_api_playground_token_payload_matches_generator_continuation(client) -> None:
    test_client, _ = client
    resp = test_client.post("/api/playground", json={"prompt": "p"})
    token_payloads = [p for n, p in _parse_sse(resp.text) if n == "token"]
    assert any(p["text"] == "stub continuation" for p in token_payloads)


def test_api_playground_done_carries_metadata(client) -> None:
    test_client, _ = client
    resp = test_client.post("/api/playground", json={"prompt": "p"})
    done = [p for n, p in _parse_sse(resp.text) if n == "done"][0]
    assert done["model"] == "stub"
    assert done["refused"] is False
    assert done["cited_chunks"] == []
    assert isinstance(done["latency_seconds"], (int, float))


def test_api_playground_forwards_max_tokens_and_temperature(client) -> None:
    test_client, state = client
    test_client.post(
        "/api/playground",
        json={"prompt": "p", "max_tokens": 64, "temperature": 0.7},
    )
    gen = state.models["stub"].generator
    assert isinstance(gen, StubGenerator)
    assert gen.raw_calls[-1]["max_tokens"] == 64
    assert gen.raw_calls[-1]["temperature"] == 0.7


# ------------------------------------------------------------------
# Validation + errors
# ------------------------------------------------------------------


def test_api_playground_rejects_empty_prompt(client) -> None:
    test_client, _ = client
    resp = test_client.post("/api/playground", json={"prompt": ""})
    assert resp.status_code == 422


def test_api_playground_unknown_model_yields_error_event(client) -> None:
    test_client, _ = client
    resp = test_client.post(
        "/api/playground",
        json={"prompt": "p", "model": "does-not-exist"},
    )
    events = _parse_sse(resp.text)
    names = [n for n, _ in events]
    assert "error" in names
    err = next(p for n, p in events if n == "error")
    assert "does-not-exist" in err["message"]


def test_api_playground_generator_raises_yields_error_event(client) -> None:
    @dataclass
    class BoomGenerator(StubGenerator):
        def generate_raw(
            self,
            prompt: str,
            *,
            max_tokens: int = 256,
            temperature: float = 0.0,
        ) -> RawCompletion:
            raise RuntimeError("boom")

    from fastapi.testclient import TestClient

    state = _make_state(generator=BoomGenerator())
    app = build_app(state)
    test_client = TestClient(app)
    resp = test_client.post("/api/playground", json={"prompt": "p"})
    events = _parse_sse(resp.text)
    names = [n for n, _ in events]
    assert "error" in names


# ------------------------------------------------------------------
# Concurrency
# ------------------------------------------------------------------


def test_api_playground_serialises_per_model_lock() -> None:
    """Two concurrent playground calls on the same model serialise."""
    import threading

    from fastapi.testclient import TestClient

    gen = StubGenerator(sleep_seconds=0.05)
    state = _make_state(generator=gen)
    app = build_app(state)
    test_client = TestClient(app)

    barrier = threading.Barrier(2)

    def hit() -> None:
        barrier.wait()
        resp = test_client.post("/api/playground", json={"prompt": "p"})
        assert resp.status_code == 200

    threads = [threading.Thread(target=hit), threading.Thread(target=hit)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    total = time.perf_counter() - t0
    assert len(gen.raw_calls) == 2
    assert total >= 0.09, (
        f"two 50 ms playground calls served in {total:.3f} s — lock did not serialise"
    )
