"""Tests for python_doc_assistant.retrieval.factory (plan v2 §5 prereq).

Hermetic — uses minimal stub indexes / reranker. No real
sentence-transformers / BM25Okapi / SymbolIndex objects are loaded; the
factory is verified to compose them in the right shape.
"""

from __future__ import annotations

from typing import Any

import pytest

from python_doc_assistant.evaluation.retrieval_metrics import RetrievedChunk
from python_doc_assistant.indexes.bm25_index import Hit as BM25Hit
from python_doc_assistant.indexes.dense_index import DenseHit
from python_doc_assistant.indexes.symbol_index import Candidate
from python_doc_assistant.ingest.chunker import Chunk
from python_doc_assistant.retrieval.factory import (
    DEFAULT_ALPHA,
    DEFAULT_RERANK_CANDIDATES,
    VALID_RETRIEVERS,
    build_retrieve_fn,
)
from python_doc_assistant.retrieval.rerank import RerankedHit

# ------------------------------------------------------------------
# Fixture builders
# ------------------------------------------------------------------


def _chunk(chunk_id: str, *, symbols: tuple[str, ...] | None = None) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        chunk_type="symbol",
        docs_version="3.12",
        title=chunk_id,
        text=f"body of {chunk_id}",
        symbols=symbols if symbols is not None else (chunk_id,),
        canonical_url=f"library/foo.html#{chunk_id}",
        anchor=chunk_id,
        parent_module=None,
        source_path="library/foo.html",
        source_hash="sha256:abc",
    )


class _StubBM25Index:
    """Stub BM25Index — `search(query, k=)` returns canned BM25Hit list."""

    def __init__(self, hits_by_query: dict[str, list[BM25Hit]] | None = None) -> None:
        self._hits = hits_by_query or {}
        self.search_calls: list[tuple[str, int]] = []

    def search(self, query: str, *, k: int = 10) -> list[BM25Hit]:
        self.search_calls.append((query, k))
        return list(self._hits.get(query, []))[:k]


class _StubDenseIndex:
    """Stub DenseIndex — `search(query, *, k)` returns canned DenseHit list."""

    def __init__(self, hits_by_query: dict[str, list[DenseHit]] | None = None) -> None:
        self._hits = hits_by_query or {}
        self.search_calls: list[tuple[str, int]] = []

    def search(self, query: str, *, k: int = 10) -> list[DenseHit]:
        self.search_calls.append((query, k))
        return list(self._hits.get(query, []))[:k]


class _StubSymbolIndex:
    """Stub SymbolIndex — `lookup(query)` returns canned Candidate list."""

    def __init__(self, candidates_by_query: dict[str, list[Candidate]] | None = None) -> None:
        self._candidates = candidates_by_query or {}

    def lookup(self, query: str) -> list[Candidate]:
        return list(self._candidates.get(query, []))


class _StubReranker:
    """Stub CrossEncoderReranker — uses an injected score function."""

    def __init__(self, score_fn: Any = None) -> None:
        self._score_fn = score_fn or (lambda chunk: 1.0)
        self.rerank_calls: list[tuple[str, list[Chunk], int]] = []

    def rerank(
        self,
        query: str,
        chunks: list[Chunk],
        *,
        top_k: int = 5,
        batch_size: int = 32,
    ) -> list[RerankedHit]:
        self.rerank_calls.append((query, chunks, top_k))
        scored = [
            RerankedHit(chunk_id=c.chunk_id, score=float(self._score_fn(c)))
            for c in chunks
        ]
        return sorted(scored, key=lambda h: -h.score)[:top_k]


def _chunks_dict(chunk_ids: list[str]) -> dict[str, Chunk]:
    return {cid: _chunk(cid) for cid in chunk_ids}


# ------------------------------------------------------------------
# Constants + validation
# ------------------------------------------------------------------


def test_valid_retrievers_set() -> None:
    assert VALID_RETRIEVERS == frozenset(
        {"bm25", "symbol+bm25", "dense", "hybrid-rrf", "hybrid-linear"}
    )


def test_default_constants() -> None:
    assert DEFAULT_RERANK_CANDIDATES == 20
    assert DEFAULT_ALPHA == 0.5


def test_unknown_retriever_raises() -> None:
    with pytest.raises(ValueError):
        build_retrieve_fn(retriever="nonexistent", chunks_by_id={})


def test_bm25_without_index_raises() -> None:
    with pytest.raises(ValueError):
        build_retrieve_fn(retriever="bm25", chunks_by_id={})


def test_dense_without_index_raises() -> None:
    with pytest.raises(ValueError):
        build_retrieve_fn(retriever="dense", chunks_by_id={})


def test_hybrid_rrf_without_indexes_raises() -> None:
    with pytest.raises(ValueError):
        build_retrieve_fn(
            retriever="hybrid-rrf", chunks_by_id={}, bm25_index=_StubBM25Index()
        )


def test_symbol_bm25_without_symbol_index_raises() -> None:
    with pytest.raises(ValueError):
        build_retrieve_fn(
            retriever="symbol+bm25",
            chunks_by_id={},
            bm25_index=_StubBM25Index(),
        )


# ------------------------------------------------------------------
# bm25
# ------------------------------------------------------------------


def test_bm25_returns_top_k_retrieved_chunks() -> None:
    bm25 = _StubBM25Index(
        {"q": [BM25Hit(chunk_id="c1", score=10.0), BM25Hit(chunk_id="c2", score=5.0)]}
    )
    fn = build_retrieve_fn(
        retriever="bm25",
        chunks_by_id=_chunks_dict(["c1", "c2"]),
        bm25_index=bm25,
    )
    out = fn("q", 5)
    assert len(out) == 2
    assert all(isinstance(r, RetrievedChunk) for r in out)
    assert [r.chunk_id for r in out] == ["c1", "c2"]
    # 1-indexed rank
    assert out[0].rank == 1
    assert out[1].rank == 2
    # score from BM25
    assert out[0].score == 10.0


def test_bm25_passes_k_through_to_search() -> None:
    bm25 = _StubBM25Index({"q": [BM25Hit(chunk_id=f"c{i}", score=1.0) for i in range(20)]})
    fn = build_retrieve_fn(
        retriever="bm25",
        chunks_by_id=_chunks_dict([f"c{i}" for i in range(20)]),
        bm25_index=bm25,
    )
    fn("q", 7)
    assert bm25.search_calls == [("q", 7)]


# ------------------------------------------------------------------
# dense
# ------------------------------------------------------------------


def test_dense_returns_top_k_retrieved_chunks() -> None:
    dense = _StubDenseIndex(
        {"q": [DenseHit(chunk_id="c1", score=0.9), DenseHit(chunk_id="c2", score=0.7)]}
    )
    fn = build_retrieve_fn(
        retriever="dense",
        chunks_by_id=_chunks_dict(["c1", "c2"]),
        dense_index=dense,
    )
    out = fn("q", 5)
    assert [r.chunk_id for r in out] == ["c1", "c2"]
    assert out[0].score == pytest.approx(0.9)


# ------------------------------------------------------------------
# symbol+bm25 (v0 router)
# ------------------------------------------------------------------


def test_symbol_bm25_uses_symbol_when_exact_match() -> None:
    """Symbol exact-match → no BM25 fallback."""
    symbol = _StubSymbolIndex(
        {
            "pathlib.Path.read_text": [
                Candidate(
                    chunk_id="symbol:pathlib.Path.read_text",
                    fully_qualified_name="pathlib.Path.read_text",
                    role="py:method",
                    parent_module="pathlib",
                )
            ]
        }
    )
    bm25 = _StubBM25Index()
    fn = build_retrieve_fn(
        retriever="symbol+bm25",
        chunks_by_id=_chunks_dict(["symbol:pathlib.Path.read_text"]),
        bm25_index=bm25,
        symbol_index=symbol,
    )
    out = fn("pathlib.Path.read_text", 5)
    assert [r.chunk_id for r in out] == ["symbol:pathlib.Path.read_text"]
    # Symbol succeeded → BM25 not consulted
    assert bm25.search_calls == []


def test_symbol_bm25_falls_back_to_bm25_on_miss() -> None:
    """Symbol miss → BM25 search runs."""
    symbol = _StubSymbolIndex({})  # no candidates for any query
    bm25 = _StubBM25Index({"how to read": [BM25Hit(chunk_id="c1", score=3.0)]})
    fn = build_retrieve_fn(
        retriever="symbol+bm25",
        chunks_by_id=_chunks_dict(["c1"]),
        bm25_index=bm25,
        symbol_index=symbol,
    )
    out = fn("how to read", 5)
    assert [r.chunk_id for r in out] == ["c1"]
    assert len(bm25.search_calls) == 1


# ------------------------------------------------------------------
# hybrid-rrf
# ------------------------------------------------------------------


def test_hybrid_rrf_fuses_bm25_and_dense() -> None:
    """Chunks in both rankings get a higher fused score than chunks in only one."""
    bm25 = _StubBM25Index(
        {"q": [BM25Hit(chunk_id="a", score=10.0), BM25Hit(chunk_id="b", score=5.0)]}
    )
    dense = _StubDenseIndex(
        {"q": [DenseHit(chunk_id="a", score=0.9), DenseHit(chunk_id="c", score=0.4)]}
    )
    fn = build_retrieve_fn(
        retriever="hybrid-rrf",
        chunks_by_id=_chunks_dict(["a", "b", "c"]),
        bm25_index=bm25,
        dense_index=dense,
    )
    out = fn("q", 5)
    by_id = {r.chunk_id: r.score for r in out}
    # 'a' in both lists → highest fused score
    assert by_id["a"] > by_id["b"]
    assert by_id["a"] > by_id["c"]


# ------------------------------------------------------------------
# hybrid-linear
# ------------------------------------------------------------------


def test_hybrid_linear_uses_alpha() -> None:
    """alpha=1 → only BM25 ranking; alpha=0 → only dense."""
    bm25 = _StubBM25Index(
        {"q": [BM25Hit(chunk_id="bm_top", score=10.0), BM25Hit(chunk_id="other", score=1.0)]}
    )
    dense = _StubDenseIndex(
        {"q": [DenseHit(chunk_id="dn_top", score=0.9), DenseHit(chunk_id="other", score=0.1)]}
    )
    chunks = _chunks_dict(["bm_top", "dn_top", "other"])

    fn_bm = build_retrieve_fn(
        retriever="hybrid-linear",
        chunks_by_id=chunks,
        bm25_index=bm25,
        dense_index=dense,
        alpha=1.0,
    )
    fn_dn = build_retrieve_fn(
        retriever="hybrid-linear",
        chunks_by_id=chunks,
        bm25_index=bm25,
        dense_index=dense,
        alpha=0.0,
    )
    out_bm = fn_bm("q", 5)
    out_dn = fn_dn("q", 5)
    assert out_bm[0].chunk_id == "bm_top"
    assert out_dn[0].chunk_id == "dn_top"


# ------------------------------------------------------------------
# rerank wrapping
# ------------------------------------------------------------------


def test_rerank_fetches_rerank_candidates_from_inner() -> None:
    """When reranker is provided, inner retriever is asked for `rerank_candidates`."""
    bm25 = _StubBM25Index(
        {"q": [BM25Hit(chunk_id=f"c{i}", score=float(20 - i)) for i in range(20)]}
    )
    reranker = _StubReranker()
    fn = build_retrieve_fn(
        retriever="bm25",
        chunks_by_id=_chunks_dict([f"c{i}" for i in range(20)]),
        bm25_index=bm25,
        reranker=reranker,
        rerank_candidates=15,
    )
    fn("q", 5)
    # Inner retriever asked for rerank_candidates, NOT k
    assert bm25.search_calls == [("q", 15)]
    # Reranker called with full candidate list, top_k=5
    assert len(reranker.rerank_calls) == 1
    _, candidate_chunks, top_k = reranker.rerank_calls[0]
    assert top_k == 5
    assert len(candidate_chunks) == 15


def test_rerank_returns_top_k_after_rerank() -> None:
    """Final output is what the reranker returned, capped at k."""
    bm25 = _StubBM25Index(
        {"q": [BM25Hit(chunk_id="c1", score=10.0),
               BM25Hit(chunk_id="c2", score=8.0),
               BM25Hit(chunk_id="c3", score=5.0)]}
    )
    # Reranker flips the order: c3 > c1 > c2
    reranker = _StubReranker(
        score_fn=lambda c: {"c1": 0.5, "c2": 0.1, "c3": 0.9}[c.chunk_id]
    )
    fn = build_retrieve_fn(
        retriever="bm25",
        chunks_by_id=_chunks_dict(["c1", "c2", "c3"]),
        bm25_index=bm25,
        reranker=reranker,
        rerank_candidates=10,
    )
    out = fn("q", 2)
    assert [r.chunk_id for r in out] == ["c3", "c1"]
    assert out[0].rank == 1
    assert out[1].rank == 2


def test_rerank_default_candidates_is_20() -> None:
    bm25 = _StubBM25Index({"q": [BM25Hit(chunk_id=f"c{i}", score=1.0) for i in range(50)]})
    reranker = _StubReranker()
    fn = build_retrieve_fn(
        retriever="bm25",
        chunks_by_id=_chunks_dict([f"c{i}" for i in range(50)]),
        bm25_index=bm25,
        reranker=reranker,
    )
    fn("q", 5)
    # Default rerank_candidates=20
    assert bm25.search_calls == [("q", 20)]
