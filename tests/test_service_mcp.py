"""Tests for `python_doc_assistant.service.mcp` (MCP tool handler)."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from python_doc_assistant.evaluation.retrieval_metrics import RetrievedChunk
from python_doc_assistant.generation.interface import Answer, Generator
from python_doc_assistant.ingest.chunker import Chunk
from python_doc_assistant.retrieval.router import QueryType
from python_doc_assistant.service.app import AskState, ModelEntry
from python_doc_assistant.service.mcp import (
    _ask_handler,
    _search_handler,
    build_mcp_app,
    build_mcp_server,
)

# ------------------------------------------------------------------
# Fixtures (mirror tests/test_service_app.py)
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
    canned_answer: Answer = field(
        default_factory=lambda: Answer(
            text="Use `json.loads(s)` to deserialize a JSON string [1].",
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
                max_seq_len=1024,
            )
        },
        default_model="stub",
        retrieve_fn=retrieve_fn,
        chunks_by_id=chunks_by_id,
    )


# ------------------------------------------------------------------
# _ask_handler — happy path
# ------------------------------------------------------------------


def test_ask_handler_returns_markdown_with_answer_body() -> None:
    state = _make_state()
    out = asyncio.run(_ask_handler(state, query="what is json.loads"))
    assert isinstance(out, str)
    assert "json.loads" in out


def test_ask_handler_appends_sources_section() -> None:
    state = _make_state()
    out = asyncio.run(_ask_handler(state, query="what is json.loads"))
    assert "**Sources**" in out
    # Cited chunk_id and its docs.python.org URL must both appear.
    assert "symbol:foo" in out
    assert "https://docs.python.org/3.12/library/foo.html#symbol:foo" in out


def test_ask_handler_drops_cited_ids_missing_from_chunks_by_id() -> None:
    """Defensive: if generator cites a chunk_id we don't know, skip it."""

    @dataclass
    class GenWithStaleCite(StubGenerator):
        canned_answer: Answer = field(
            default_factory=lambda: Answer(
                text="answer body",
                cited_chunk_ids=("symbol:does-not-exist",),
                refused=False,
                latency_seconds=0.001,
            )
        )

    state = _make_state(generator=GenWithStaleCite())
    out = asyncio.run(_ask_handler(state, query="q"))
    # No Sources section since no cited id matched chunks_by_id.
    assert "**Sources**" not in out


# ------------------------------------------------------------------
# _ask_handler — refusal
# ------------------------------------------------------------------


def test_ask_handler_unknown_model_returns_error_markdown() -> None:
    state = _make_state()
    out = asyncio.run(_ask_handler(state, query="q", model="does-not-exist"))
    assert "does-not-exist" in out
    assert "Unknown model" in out


def test_ask_handler_explicit_default_model_works() -> None:
    state = _make_state()
    out = asyncio.run(_ask_handler(state, query="json.loads", model="stub"))
    assert "json.loads" in out


def test_ask_handler_returns_refusal_message_when_model_refused() -> None:
    refused_gen = StubGenerator(
        canned_answer=Answer(text="", cited_chunk_ids=(), refused=True, latency_seconds=0.001)
    )
    state = _make_state(generator=refused_gen)
    out = asyncio.run(_ask_handler(state, query="how to train transformer"))
    assert isinstance(out, str)
    # Don't pin the exact phrase — leave wording flexible — but the
    # refusal must be communicated and no Sources section attached.
    assert "**Sources**" not in out
    assert out.strip() != ""


# ------------------------------------------------------------------
# _ask_handler — concurrency: same lock
# ------------------------------------------------------------------


def test_ask_handler_serialises_via_state_lock() -> None:
    """Two concurrent _ask_handler calls must serialise behind the AskState lock."""
    sleeping = StubGenerator(sleep_seconds=0.05)
    state = _make_state(generator=sleeping)

    async def run_two() -> None:
        await asyncio.gather(
            _ask_handler(state, query="q1"),
            _ask_handler(state, query="q2"),
        )

    t0 = time.perf_counter()
    asyncio.run(run_two())
    total = time.perf_counter() - t0
    assert len(sleeping.calls) == 2
    assert total >= 0.09, f"two 50 ms-locked handler calls in {total:.3f}s — lock did not serialise"


# ------------------------------------------------------------------
# build_mcp_app — ASGI surface
# ------------------------------------------------------------------


def test_build_mcp_app_returns_asgi_app() -> None:
    """Smoke: factory builds an ASGI-callable Starlette app."""
    state = _make_state()
    asgi = build_mcp_app(state)
    # Starlette / FastMCP HTTP apps are callables expecting (scope, receive, send).
    assert callable(asgi)


def test_build_mcp_app_can_be_mounted_on_fastapi() -> None:
    """The returned ASGI app must be mountable on a FastAPI Application."""
    from fastapi import FastAPI

    state = _make_state()
    app = FastAPI()
    app.mount("/mcp", build_mcp_app(state))
    # If mount() raises, the test fails. Otherwise we trust the routing.


def test_build_mcp_server_registers_ask_tool() -> None:
    """The FastMCP server should expose a tool named 'ask'."""
    state = _make_state()
    server = build_mcp_server(state)
    tools = asyncio.run(server.list_tools())
    names = {t.name for t in tools}
    assert "ask" in names


def test_build_mcp_server_ask_tool_input_schema() -> None:
    """The 'ask' tool should declare query/k/rerank/hyde inputs."""
    state = _make_state()
    server = build_mcp_server(state)
    tools = asyncio.run(server.list_tools())
    ask = next(t for t in tools if t.name == "ask")
    schema = ask.inputSchema
    properties = schema["properties"]
    assert "query" in properties
    assert "k" in properties
    assert "rerank" in properties
    assert "hyde" in properties
    # `query` is required.
    assert "query" in schema.get("required", [])


# ------------------------------------------------------------------
# _search_handler — retrieve-only happy path + edge cases
# ------------------------------------------------------------------


def test_search_handler_returns_one_block_per_retrieved_chunk() -> None:
    state = _make_state(retrieved_ids=("symbol:foo", "symbol:bar", "symbol:baz"))
    out = asyncio.run(_search_handler(state, query="anything", k=3))
    # Each chunk should produce a "## [N] ..." block headed by its rank.
    assert "## [1]" in out
    assert "## [2]" in out
    assert "## [3]" in out
    # All three chunk_ids should appear in inline-code form.
    assert "`symbol:foo`" in out
    assert "`symbol:bar`" in out
    assert "`symbol:baz`" in out


def test_search_handler_includes_canonical_url() -> None:
    state = _make_state()
    out = asyncio.run(_search_handler(state, query="json"))
    assert "https://docs.python.org/3.12/library/foo.html" in out


def test_search_handler_truncates_long_chunk_text() -> None:
    big_chunk = _chunk("symbol:big")
    big_chunk = Chunk(
        chunk_id=big_chunk.chunk_id,
        chunk_type=big_chunk.chunk_type,
        docs_version=big_chunk.docs_version,
        title=big_chunk.title,
        text="A" * 5000,
        symbols=big_chunk.symbols,
        canonical_url=big_chunk.canonical_url,
        anchor=big_chunk.anchor,
        parent_module=big_chunk.parent_module,
        source_path=big_chunk.source_path,
        source_hash=big_chunk.source_hash,
    )
    state = _make_state(chunks=[big_chunk], retrieved_ids=("symbol:big",))
    out = asyncio.run(_search_handler(state, query="anything", k=1))
    assert "…" in out
    # Full body must NOT appear verbatim.
    assert "A" * 5000 not in out


def test_search_handler_rejects_invalid_k() -> None:
    state = _make_state()
    too_low = asyncio.run(_search_handler(state, query="q", k=0))
    too_high = asyncio.run(_search_handler(state, query="q", k=99))
    assert "Invalid k" in too_low
    assert "Invalid k" in too_high


def test_search_handler_returns_empty_marker_when_no_chunks() -> None:
    @dataclass
    class _EmptyState:
        retrieve_fn: Any
        chunks_by_id: dict[str, Chunk]

    def empty_retrieve(_query: str, _k: int) -> list[RetrievedChunk]:
        return []

    state = _make_state()
    state.retrieve_fn = empty_retrieve  # type: ignore[misc]
    out = asyncio.run(_search_handler(state, query="nothing matches", k=5))
    assert "No chunks retrieved" in out


def test_build_mcp_server_registers_search_tool() -> None:
    state = _make_state()
    server = build_mcp_server(state)
    tools = asyncio.run(server.list_tools())
    names = {t.name for t in tools}
    assert "search" in names


def test_build_mcp_server_search_tool_input_schema() -> None:
    state = _make_state()
    server = build_mcp_server(state)
    tools = asyncio.run(server.list_tools())
    search = next(t for t in tools if t.name == "search")
    properties = search.inputSchema["properties"]
    assert "query" in properties
    assert "k" in properties
    # search does not need rerank / hyde / model.
    assert "rerank" not in properties
    assert "model" not in properties
