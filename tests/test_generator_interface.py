"""Tests for python_doc_assistant.generation.interface.

Hermetic — no real model loaded. Stubs the Generator ABC with a tiny subclass
to verify the interface contract.
"""

from __future__ import annotations

from typing import Any

import pytest

from python_doc_assistant.generation.interface import Answer, Generator
from python_doc_assistant.ingest.chunker import Chunk
from python_doc_assistant.retrieval.router import QueryType

# ------------------------------------------------------------------
# Fixture builders
# ------------------------------------------------------------------


def _chunk(chunk_id: str = "symbol:foo") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        chunk_type="symbol",
        docs_version="3.12",
        title="foo",
        text="body",
        symbols=("foo",),
        canonical_url="library/foo.html#foo",
        anchor="foo",
        parent_module=None,
        source_path="library/foo.html",
        source_hash="sha256:abc",
    )


# ------------------------------------------------------------------
# Answer dataclass
# ------------------------------------------------------------------


def test_answer_required_fields() -> None:
    a = Answer(
        text="hello",
        cited_chunk_ids=("c1",),
        refused=False,
        latency_seconds=0.5,
    )
    assert a.text == "hello"
    assert a.cited_chunk_ids == ("c1",)
    assert a.refused is False
    assert a.latency_seconds == 0.5


def test_answer_is_frozen() -> None:
    a = Answer(text="x", cited_chunk_ids=(), refused=False, latency_seconds=0.0)
    with pytest.raises(Exception):
        a.text = "y"  # type: ignore[misc]


def test_answer_refused_allows_empty_text_and_citations() -> None:
    """A refused Answer typically has empty text + no citations."""
    a = Answer(text="", cited_chunk_ids=(), refused=True, latency_seconds=0.1)
    assert a.refused is True
    assert a.text == ""
    assert a.cited_chunk_ids == ()


# ------------------------------------------------------------------
# Generator ABC
# ------------------------------------------------------------------


def test_generator_cannot_instantiate_directly() -> None:
    """ABC should prevent direct instantiation."""
    with pytest.raises(TypeError):
        Generator()  # type: ignore[abstract]


def test_subclass_without_generate_cannot_instantiate() -> None:
    class IncompleteGenerator(Generator):
        pass  # does NOT override generate

    with pytest.raises(TypeError):
        IncompleteGenerator()  # type: ignore[abstract]


def test_subclass_with_override_works() -> None:
    class StubGenerator(Generator):
        def generate(
            self,
            query: str,
            retrieved_chunks: list[Chunk],
            *,
            query_type: QueryType | None = None,
            stream: bool = False,
        ) -> Answer:
            return Answer(
                text=f"answer to {query}",
                cited_chunk_ids=tuple(c.chunk_id for c in retrieved_chunks),
                refused=False,
                latency_seconds=0.0,
            )

    gen = StubGenerator()
    answer = gen.generate("test query", [_chunk("symbol:foo")])
    assert answer.text == "answer to test query"
    assert answer.cited_chunk_ids == ("symbol:foo",)
    assert answer.refused is False


def test_subclass_receives_query_type_and_stream_kwargs() -> None:
    """query_type and stream pass through the interface untouched."""
    captured: dict[str, Any] = {}

    class StubGenerator(Generator):
        def generate(
            self,
            query: str,
            retrieved_chunks: list[Chunk],
            *,
            query_type: QueryType | None = None,
            stream: bool = False,
        ) -> Answer:
            captured["query_type"] = query_type
            captured["stream"] = stream
            return Answer(text="", cited_chunk_ids=(), refused=False, latency_seconds=0.0)

    gen = StubGenerator()
    gen.generate("q", [], query_type=QueryType.IDENTIFIER, stream=True)
    assert captured["query_type"] == QueryType.IDENTIFIER
    assert captured["stream"] is True
