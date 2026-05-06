"""Tests for `python_doc_assistant.retrieval.hyde`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from python_doc_assistant.indexes.dense_index import DenseHit
from python_doc_assistant.ingest.chunker import Chunk
from python_doc_assistant.retrieval.hyde import (
    DEFAULT_HYDE_PROMPT_TEMPLATE,
    DEFAULT_RERANK_CANDIDATES,
    DEFAULT_SKIP_QUERY_TYPES,
    HypotheticalGenerator,
    QwenHypotheticalGenerator,
    make_hyde_retrieve_fn,
)
from python_doc_assistant.retrieval.rerank import RerankedHit
from python_doc_assistant.retrieval.router import QueryType

# ------------------------------------------------------------------
# Fixtures: chunks + stubs
# ------------------------------------------------------------------


def _chunk(chunk_id: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        chunk_type="symbol",
        docs_version="3.12",
        title=chunk_id,
        text=f"docs for {chunk_id}",
        symbols=(chunk_id,),
        canonical_url=f"library/foo.html#{chunk_id}",
        anchor=chunk_id,
        parent_module=None,
        source_path="library/foo.html",
        source_hash="sha256:abc",
    )


@dataclass
class StubHypotheticalGenerator:
    """Returns canned hypothetical strings, records each query it sees."""

    canned: str = "fake answer document"
    calls: list[str] = field(default_factory=list)

    def generate(self, query: str) -> str:
        self.calls.append(query)
        return self.canned


@dataclass
class StubDenseIndex:
    """Returns canned dense hits, records each (query, k) pair."""

    hits: list[DenseHit] = field(default_factory=list)
    calls: list[tuple[str, int]] = field(default_factory=list)

    def search(self, query: str, *, k: int) -> list[DenseHit]:
        self.calls.append((query, k))
        return list(self.hits[:k])


@dataclass
class StubReranker:
    """Returns canned reranked hits, records each (query, top_k, candidate ids)."""

    reranked: list[RerankedHit] = field(default_factory=list)
    calls: list[tuple[str, int, tuple[str, ...]]] = field(default_factory=list)

    def rerank(
        self,
        query: str,
        chunks: list[Chunk],
        *,
        top_k: int,
        batch_size: int = 32,
    ) -> list[RerankedHit]:
        self.calls.append((query, top_k, tuple(c.chunk_id for c in chunks)))
        return list(self.reranked[:top_k])


def _classify_constant(qt: QueryType):
    def fn(_query: str) -> QueryType:
        return qt

    return fn


# ------------------------------------------------------------------
# Constants surface
# ------------------------------------------------------------------


def test_default_prompt_has_query_placeholder() -> None:
    assert "{query}" in DEFAULT_HYDE_PROMPT_TEMPLATE


def test_default_skip_query_types_contains_identifier() -> None:
    assert QueryType.IDENTIFIER in DEFAULT_SKIP_QUERY_TYPES


def test_default_rerank_candidates_matches_v2_ablation_constant() -> None:
    assert DEFAULT_RERANK_CANDIDATES == 20


# ------------------------------------------------------------------
# QwenHypotheticalGenerator
# ------------------------------------------------------------------


class _StubLlama:
    """Minimal Llama-shaped stub for QwenHypotheticalGenerator tests."""

    def __init__(self, return_text: str) -> None:
        self.return_text = return_text
        self.calls: list[dict[str, Any]] = []

    def create_chat_completion(
        self,
        *,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        **_: Any,
    ) -> dict[str, Any]:
        self.calls.append(
            {"messages": messages, "max_tokens": max_tokens, "temperature": temperature}
        )
        return {"choices": [{"message": {"content": self.return_text}}]}


def test_qwen_hyp_generator_formats_prompt_with_query() -> None:
    llm = _StubLlama("hyp text")
    gen = QwenHypotheticalGenerator(llm=llm)
    out = gen.generate("how to read a file")
    assert out == "hyp text"
    assert len(llm.calls) == 1
    msg = llm.calls[0]["messages"][0]
    assert msg["role"] == "user"
    assert "how to read a file" in msg["content"]


def test_qwen_hyp_generator_strips_whitespace() -> None:
    llm = _StubLlama("\n  hyp text  \n")
    gen = QwenHypotheticalGenerator(llm=llm)
    assert gen.generate("q") == "hyp text"


def test_qwen_hyp_generator_uses_decoding_params() -> None:
    llm = _StubLlama("x")
    gen = QwenHypotheticalGenerator(llm=llm, max_tokens=42, temperature=0.7)
    gen.generate("q")
    call = llm.calls[0]
    assert call["max_tokens"] == 42
    assert call["temperature"] == 0.7


def test_qwen_hyp_generator_custom_prompt_template() -> None:
    llm = _StubLlama("hyp")
    template = "ANSWER: {query} END"
    gen = QwenHypotheticalGenerator(llm=llm, prompt_template=template)
    gen.generate("test query")
    assert llm.calls[0]["messages"][0]["content"] == "ANSWER: test query END"


def test_qwen_hyp_generator_rejects_template_missing_placeholder() -> None:
    llm = _StubLlama("x")
    with pytest.raises(ValueError):
        QwenHypotheticalGenerator(llm=llm, prompt_template="no placeholder here")


# ------------------------------------------------------------------
# make_hyde_retrieve_fn — skip path (identifier query)
# ------------------------------------------------------------------


def test_skip_uses_original_query_for_dense() -> None:
    chunks_by_id = {"c1": _chunk("c1")}
    dense = StubDenseIndex(hits=[DenseHit(chunk_id="c1", score=0.9)])
    hyp = StubHypotheticalGenerator()
    retrieve = make_hyde_retrieve_fn(
        dense_index=dense,
        chunks_by_id=chunks_by_id,
        hypothetical_generator=hyp,
        reranker=None,
        classify_fn=_classify_constant(QueryType.IDENTIFIER),
    )
    out = retrieve("pathlib.Path.read_text", k=5)
    assert hyp.calls == []  # HyDE skipped
    assert dense.calls == [("pathlib.Path.read_text", 5)]
    assert [r.chunk_id for r in out] == ["c1"]


def test_skip_query_types_is_configurable() -> None:
    chunks_by_id = {"c1": _chunk("c1")}
    dense = StubDenseIndex(hits=[DenseHit(chunk_id="c1", score=0.9)])
    hyp = StubHypotheticalGenerator(canned="hyp")
    retrieve = make_hyde_retrieve_fn(
        dense_index=dense,
        chunks_by_id=chunks_by_id,
        hypothetical_generator=hyp,
        reranker=None,
        skip_query_types=(QueryType.HOWTO,),
        classify_fn=_classify_constant(QueryType.HOWTO),
    )
    retrieve("how do I X", k=5)
    assert hyp.calls == []
    assert dense.calls[0][0] == "how do I X"


# ------------------------------------------------------------------
# make_hyde_retrieve_fn — HyDE path (non-skip query)
# ------------------------------------------------------------------


def test_hyde_path_passes_hypothetical_to_dense() -> None:
    chunks_by_id = {"c1": _chunk("c1"), "c2": _chunk("c2")}
    dense = StubDenseIndex(hits=[DenseHit("c1", 0.9), DenseHit("c2", 0.8)])
    hyp = StubHypotheticalGenerator(canned="documentation passage about X")
    retrieve = make_hyde_retrieve_fn(
        dense_index=dense,
        chunks_by_id=chunks_by_id,
        hypothetical_generator=hyp,
        reranker=None,
        classify_fn=_classify_constant(QueryType.NATURAL_LANGUAGE),
    )
    out = retrieve("how to do X", k=2)
    assert hyp.calls == ["how to do X"]
    assert dense.calls == [("documentation passage about X", 2)]
    assert [r.chunk_id for r in out] == ["c1", "c2"]


def test_hyde_path_no_rerank_uses_k_for_dense() -> None:
    """Without reranker, dense.search receives k=k (no over-fetch)."""
    dense = StubDenseIndex(hits=[])
    hyp = StubHypotheticalGenerator(canned="hyp")
    retrieve = make_hyde_retrieve_fn(
        dense_index=dense,
        chunks_by_id={},
        hypothetical_generator=hyp,
        reranker=None,
        classify_fn=_classify_constant(QueryType.NATURAL_LANGUAGE),
    )
    retrieve("q", k=7)
    assert dense.calls[0][1] == 7


def test_returned_chunks_are_rank_ordered_with_metadata() -> None:
    chunks_by_id = {"c1": _chunk("c1"), "c2": _chunk("c2")}
    dense = StubDenseIndex(hits=[DenseHit("c1", 0.9), DenseHit("c2", 0.8)])
    retrieve = make_hyde_retrieve_fn(
        dense_index=dense,
        chunks_by_id=chunks_by_id,
        hypothetical_generator=StubHypotheticalGenerator(),
        reranker=None,
        classify_fn=_classify_constant(QueryType.NATURAL_LANGUAGE),
    )
    out = retrieve("q", k=2)
    assert out[0].chunk_id == "c1"
    assert out[0].rank == 1
    assert out[0].score == pytest.approx(0.9)
    assert out[0].canonical_url == "library/foo.html#c1"
    assert out[0].symbols == ("c1",)
    assert out[1].chunk_id == "c2"
    assert out[1].rank == 2


def test_unknown_chunk_ids_dropped_defensively() -> None:
    chunks_by_id = {"c1": _chunk("c1")}  # c2 missing
    dense = StubDenseIndex(hits=[DenseHit("c1", 0.9), DenseHit("c2", 0.8)])
    retrieve = make_hyde_retrieve_fn(
        dense_index=dense,
        chunks_by_id=chunks_by_id,
        hypothetical_generator=StubHypotheticalGenerator(),
        reranker=None,
        classify_fn=_classify_constant(QueryType.NATURAL_LANGUAGE),
    )
    out = retrieve("q", k=5)
    assert [r.chunk_id for r in out] == ["c1"]


# ------------------------------------------------------------------
# make_hyde_retrieve_fn — rerank path
# ------------------------------------------------------------------


def test_rerank_uses_original_query_not_hypothetical() -> None:
    """The cross-encoder must score the user's intent, not the LLM's invention."""
    chunks_by_id = {"c1": _chunk("c1"), "c2": _chunk("c2")}
    dense = StubDenseIndex(hits=[DenseHit("c1", 0.5), DenseHit("c2", 0.4)])
    rer = StubReranker(reranked=[RerankedHit("c2", 0.95), RerankedHit("c1", 0.4)])
    retrieve = make_hyde_retrieve_fn(
        dense_index=dense,
        chunks_by_id=chunks_by_id,
        hypothetical_generator=StubHypotheticalGenerator(canned="HYP"),
        reranker=rer,  # type: ignore[arg-type]
        classify_fn=_classify_constant(QueryType.NATURAL_LANGUAGE),
    )
    out = retrieve("ORIGINAL Q", k=2)
    # Dense saw the hypothetical:
    assert dense.calls[0][0] == "HYP"
    # Reranker saw the ORIGINAL query, not the hypothetical:
    assert rer.calls[0][0] == "ORIGINAL Q"
    # Reranked order is preserved in output:
    assert [r.chunk_id for r in out] == ["c2", "c1"]


def test_rerank_path_overfetches_dense() -> None:
    """Dense fetches `rerank_candidates`, not `k`, when reranker is set."""
    dense = StubDenseIndex(hits=[])
    rer = StubReranker(reranked=[])
    retrieve = make_hyde_retrieve_fn(
        dense_index=dense,
        chunks_by_id={},
        hypothetical_generator=StubHypotheticalGenerator(canned="hyp"),
        reranker=rer,  # type: ignore[arg-type]
        rerank_candidates=15,
        classify_fn=_classify_constant(QueryType.NATURAL_LANGUAGE),
    )
    retrieve("q", k=3)
    assert dense.calls[0][1] == 15  # over-fetched
    assert rer.calls[0][1] == 3  # rerank trims to k


def test_rerank_skipped_for_skip_query_type() -> None:
    """Identifier (skip) queries still use the rerank path, but with original query."""
    chunks_by_id = {"c1": _chunk("c1")}
    dense = StubDenseIndex(hits=[DenseHit("c1", 0.9)])
    rer = StubReranker(reranked=[RerankedHit("c1", 0.99)])
    retrieve = make_hyde_retrieve_fn(
        dense_index=dense,
        chunks_by_id=chunks_by_id,
        hypothetical_generator=StubHypotheticalGenerator(canned="HYP"),
        reranker=rer,  # type: ignore[arg-type]
        classify_fn=_classify_constant(QueryType.IDENTIFIER),
    )
    retrieve("foo.bar", k=1)
    # Both dense and rerank receive the ORIGINAL query for skip case.
    assert dense.calls[0][0] == "foo.bar"
    assert rer.calls[0][0] == "foo.bar"


# ------------------------------------------------------------------
# make_hyde_retrieve_fn — empties + edge cases
# ------------------------------------------------------------------


def test_empty_dense_results_returns_empty() -> None:
    retrieve = make_hyde_retrieve_fn(
        dense_index=StubDenseIndex(hits=[]),
        chunks_by_id={},
        hypothetical_generator=StubHypotheticalGenerator(),
        reranker=None,
        classify_fn=_classify_constant(QueryType.NATURAL_LANGUAGE),
    )
    assert retrieve("q", k=5) == []


def test_protocol_satisfied_by_stub() -> None:
    """StubHypotheticalGenerator satisfies the HypotheticalGenerator Protocol."""
    stub: HypotheticalGenerator = StubHypotheticalGenerator()
    assert stub.generate("q") == "fake answer document"
