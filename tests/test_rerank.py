"""Tests for python_doc_assistant.retrieval.rerank (plan v2 §3).

Hermetic — no real `BAAI/bge-reranker-base` loaded. A `_StubCrossEncoder`
mimics `sentence_transformers.CrossEncoder.predict()` and is injected
via the `model=` kwarg (DI path).
"""

from __future__ import annotations

from typing import Any

import pytest

from python_doc_assistant.ingest.chunker import Chunk
from python_doc_assistant.retrieval.rerank import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MODEL_ID,
    DEFAULT_TOP_K,
    CrossEncoderReranker,
    RerankedHit,
)

# ------------------------------------------------------------------
# Fixture builders
# ------------------------------------------------------------------


def _chunk(
    chunk_id: str,
    *,
    title: str | None = None,
    text: str = "BODY",
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        chunk_type="symbol",
        docs_version="3.12",
        title=title if title is not None else chunk_id,
        text=text,
        symbols=(chunk_id,),
        canonical_url=f"library/foo.html#{chunk_id}",
        anchor=chunk_id,
        parent_module=None,
        source_path="library/foo.html",
        source_hash="sha256:abc",
    )


class _StubCrossEncoder:
    """Stub for sentence_transformers.CrossEncoder.

    Returns deterministic scores from `scores_by_doc_text`; falls back to
    a hash-based score when the doc text is unknown.

    Records every `predict()` call in `self.predict_calls` (a list of the
    raw `pairs` list passed in).
    """

    def __init__(
        self,
        scores_by_doc_text: dict[str, float] | None = None,
    ) -> None:
        self._scores = scores_by_doc_text or {}
        self.predict_calls: list[list[tuple[str, str]]] = []
        self.last_batch_size: int | None = None

    def predict(
        self,
        pairs: list[tuple[str, str]],
        *,
        batch_size: int = 32,
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
        **_: Any,
    ) -> Any:
        import numpy as np

        # Coerce iterable → list so test assertions can introspect
        pair_list = [tuple(p) for p in pairs]
        self.predict_calls.append(pair_list)  # type: ignore[arg-type]
        self.last_batch_size = batch_size
        scores = []
        for _, doc in pair_list:
            scores.append(
                self._scores.get(doc, (hash(doc) % 1000) / 1000.0)
            )
        return np.array(scores, dtype=np.float32) if convert_to_numpy else scores


# ------------------------------------------------------------------
# Constants + dataclass
# ------------------------------------------------------------------


def test_default_constants() -> None:
    assert DEFAULT_MODEL_ID == "BAAI/bge-reranker-base"
    assert DEFAULT_TOP_K == 5
    assert DEFAULT_BATCH_SIZE == 32


def test_reranked_hit_is_frozen() -> None:
    h = RerankedHit(chunk_id="c1", score=0.5)
    with pytest.raises(Exception):
        h.score = 0.9  # type: ignore[misc]


# ------------------------------------------------------------------
# __init__ — DI hook
# ------------------------------------------------------------------


def test_init_with_di_skips_from_pretrained() -> None:
    """Passing model=... must NOT trigger CrossEncoder download."""
    encoder = _StubCrossEncoder()
    # If from_pretrained were called, this would attempt network → fail in CI.
    CrossEncoderReranker(model=encoder)


# ------------------------------------------------------------------
# rerank()
# ------------------------------------------------------------------


def test_rerank_returns_top_k_sorted_descending() -> None:
    """Cross-encoder scores reorder the candidates; output is desc-sorted."""
    encoder = _StubCrossEncoder(
        {
            "c1\n\nBODY": 0.9,
            "c2\n\nBODY": 0.3,
            "c3\n\nBODY": 0.7,
            "c4\n\nBODY": 0.1,
            "c5\n\nBODY": 0.5,
        }
    )
    reranker = CrossEncoderReranker(model=encoder)
    chunks = [_chunk(f"c{i}") for i in range(1, 6)]
    out = reranker.rerank("query", chunks, top_k=3)
    # All RerankedHit
    assert all(isinstance(h, RerankedHit) for h in out)
    # Top-3 by score desc
    assert [h.chunk_id for h in out] == ["c1", "c3", "c5"]
    scores = [h.score for h in out]
    assert scores == sorted(scores, reverse=True)


def test_rerank_default_top_k_is_5() -> None:
    encoder = _StubCrossEncoder()
    reranker = CrossEncoderReranker(model=encoder)
    chunks = [_chunk(f"c{i}") for i in range(20)]
    out = reranker.rerank("query", chunks)  # no top_k arg
    assert len(out) == 5


def test_rerank_top_k_larger_than_candidates_returns_all() -> None:
    encoder = _StubCrossEncoder()
    reranker = CrossEncoderReranker(model=encoder)
    chunks = [_chunk("c1"), _chunk("c2"), _chunk("c3")]
    out = reranker.rerank("query", chunks, top_k=10)
    assert len(out) == 3


def test_rerank_empty_candidates_returns_empty_list() -> None:
    encoder = _StubCrossEncoder()
    reranker = CrossEncoderReranker(model=encoder)
    assert reranker.rerank("query", [], top_k=5) == []


def test_rerank_passes_query_doc_pairs_to_predict() -> None:
    """Verify the model receives (query, chunk_text) pairs in candidate order."""
    encoder = _StubCrossEncoder()
    reranker = CrossEncoderReranker(model=encoder)
    chunks = [_chunk("c1", text="alpha"), _chunk("c2", text="beta")]
    reranker.rerank("how to alpha", chunks, top_k=2)
    assert len(encoder.predict_calls) == 1
    pairs = encoder.predict_calls[0]
    assert pairs == [
        ("how to alpha", "c1\n\nalpha"),
        ("how to alpha", "c2\n\nbeta"),
    ]


def test_rerank_passes_batch_size_through_to_predict() -> None:
    """Custom batch_size reaches the cross-encoder."""
    encoder = _StubCrossEncoder()
    reranker = CrossEncoderReranker(model=encoder)
    chunks = [_chunk(f"c{i}") for i in range(10)]
    reranker.rerank("q", chunks, top_k=3, batch_size=8)
    assert encoder.last_batch_size == 8


def test_rerank_does_not_mutate_input_chunks() -> None:
    """Reranker should read chunks but not re-order the input list in-place."""
    encoder = _StubCrossEncoder(
        {
            "c1\n\nBODY": 0.1,
            "c2\n\nBODY": 0.9,  # c2 should win
        }
    )
    reranker = CrossEncoderReranker(model=encoder)
    chunks = [_chunk("c1"), _chunk("c2")]
    reranker.rerank("q", chunks, top_k=2)
    # Input still in original order
    assert [c.chunk_id for c in chunks] == ["c1", "c2"]


def test_rerank_score_field_is_python_float() -> None:
    """numpy.float32 must be cast to float for the dataclass."""
    encoder = _StubCrossEncoder({"c1\n\nBODY": 0.42})
    reranker = CrossEncoderReranker(model=encoder)
    out = reranker.rerank("q", [_chunk("c1")], top_k=1)
    assert isinstance(out[0].score, float)
