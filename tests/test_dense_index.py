"""Tests for python_doc_assistant.indexes.dense_index.

Hermetic — no real `BAAI/bge-small-en-v1.5` loaded. A `_StubEncoder`
mimics `sentence_transformers.SentenceTransformer.encode()` and is
injected via the `model=` kwarg (DI path).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from python_doc_assistant.indexes.dense_index import (
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_MODEL_ID,
    DenseHit,
    DenseIndex,
)
from python_doc_assistant.ingest.chunker import Chunk

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


class _StubEncoder:
    """Stand-in for SentenceTransformer.

    Returns deterministic small embeddings:
      - if `embeddings_by_text[text]` is set, use that vector.
      - otherwise, derive a 4-dim vector from `hash(text)` for stability.

    Records every `encode()` call in `self.encode_calls` (list of input lists).
    """

    def __init__(
        self,
        embeddings_by_text: dict[str, list[float]] | None = None,
        *,
        dim: int = 4,
    ) -> None:
        self._explicit = embeddings_by_text or {}
        self._dim = dim
        self.encode_calls: list[list[str]] = []

    def encode(
        self,
        texts: Any,
        *,
        normalize_embeddings: bool = False,
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
        **_: Any,
    ) -> Any:
        import numpy as np

        single = isinstance(texts, str)
        text_list: list[str] = [texts] if single else list(texts)
        self.encode_calls.append(text_list)

        rows = []
        for t in text_list:
            if t in self._explicit:
                vec = list(self._explicit[t])
            else:
                # Stable deterministic vector
                vec = [(hash(f"{t}::{i}") % 1000) / 1000.0 for i in range(self._dim)]
            arr = np.array(vec, dtype=np.float32)
            if normalize_embeddings:
                norm = float(np.linalg.norm(arr)) + 1e-9
                arr = arr / norm
            rows.append(arr)

        out = np.stack(rows)
        return out[0] if single else out


# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------


def test_default_constants() -> None:
    assert DEFAULT_MODEL_ID == "BAAI/bge-small-en-v1.5"
    assert DEFAULT_EMBEDDING_DIM == 384


# ------------------------------------------------------------------
# DenseHit dataclass
# ------------------------------------------------------------------


def test_dense_hit_is_frozen() -> None:
    h = DenseHit(chunk_id="c1", score=0.5)
    with pytest.raises(Exception):
        h.score = 0.9  # type: ignore[misc]


# ------------------------------------------------------------------
# __init__ — building the index
# ------------------------------------------------------------------


def test_init_encodes_all_chunks() -> None:
    """Encoder receives every chunk's text exactly once at build time."""
    encoder = _StubEncoder()
    chunks = [_chunk("c1"), _chunk("c2"), _chunk("c3")]
    DenseIndex(chunks, model=encoder)
    encoded = [t for call in encoder.encode_calls for t in call]
    # 3 chunks → 3 inputs in the build-time call
    assert len(encoded) == 3
    # Each chunk's title + text composes the input
    assert any("c1" in t and "BODY" in t for t in encoded)


def test_init_with_di_skips_from_pretrained() -> None:
    """Passing `model=...` must NOT trigger sentence-transformers download."""
    encoder = _StubEncoder()
    # If from_pretrained were called, this would attempt network → fail in CI.
    DenseIndex([_chunk("c1")], model=encoder)


def test_init_stores_chunk_ids_in_order() -> None:
    encoder = _StubEncoder()
    chunks = [_chunk("c1"), _chunk("c2"), _chunk("c3")]
    idx = DenseIndex(chunks, model=encoder)
    assert idx._chunk_ids == ["c1", "c2", "c3"]  # type: ignore[attr-defined]


def test_init_empty_chunks_produces_empty_index() -> None:
    """No chunks → empty embeddings + empty chunk_ids; search returns []."""
    encoder = _StubEncoder()
    idx = DenseIndex([], model=encoder)
    assert idx._chunk_ids == []  # type: ignore[attr-defined]
    assert idx.search("anything", k=5) == []


# ------------------------------------------------------------------
# search()
# ------------------------------------------------------------------


def test_search_returns_top_k_dense_hits() -> None:
    """Cosine similarity ranks the closest chunk first."""
    encoder = _StubEncoder(
        {
            "c1\n\nBODY": [1.0, 0.0, 0.0, 0.0],
            "c2\n\nBODY": [0.0, 1.0, 0.0, 0.0],
            "c3\n\nBODY": [0.0, 0.0, 1.0, 0.0],
            "find c1": [1.0, 0.0, 0.0, 0.0],
        }
    )
    chunks = [_chunk("c1"), _chunk("c2"), _chunk("c3")]
    idx = DenseIndex(chunks, model=encoder)
    hits = idx.search("find c1", k=2)
    assert len(hits) == 2
    assert isinstance(hits[0], DenseHit)
    assert hits[0].chunk_id == "c1"
    # Top hit should have higher score than runner-up
    assert hits[0].score >= hits[1].score


def test_search_k_caps_results() -> None:
    encoder = _StubEncoder()
    chunks = [_chunk(f"c{i}") for i in range(10)]
    idx = DenseIndex(chunks, model=encoder)
    hits = idx.search("query", k=3)
    assert len(hits) == 3


def test_search_k_larger_than_corpus_returns_all() -> None:
    encoder = _StubEncoder()
    chunks = [_chunk(f"c{i}") for i in range(3)]
    idx = DenseIndex(chunks, model=encoder)
    hits = idx.search("query", k=10)
    assert len(hits) == 3


def test_search_scores_sorted_descending() -> None:
    encoder = _StubEncoder()
    chunks = [_chunk(f"c{i}") for i in range(5)]
    idx = DenseIndex(chunks, model=encoder)
    hits = idx.search("query", k=5)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


# ------------------------------------------------------------------
# save() / load() round-trip
# ------------------------------------------------------------------


def test_save_writes_npy_and_sidecar(tmp_path: Path) -> None:
    encoder = _StubEncoder()
    idx = DenseIndex([_chunk("c1"), _chunk("c2")], model=encoder)
    path = tmp_path / "dense.npy"
    idx.save(path)
    assert path.exists()
    assert path.with_suffix(".json").exists()


def test_save_creates_parent_dirs(tmp_path: Path) -> None:
    encoder = _StubEncoder()
    idx = DenseIndex([_chunk("c1")], model=encoder)
    nested = tmp_path / "deep" / "nested" / "dir" / "dense.npy"
    idx.save(nested)
    assert nested.exists()


def test_save_sidecar_has_required_metadata(tmp_path: Path) -> None:
    """Sidecar JSON must carry model_id, chunk_ids, dim — needed by load()."""
    import json as _json

    encoder = _StubEncoder()
    idx = DenseIndex(
        [_chunk("c1"), _chunk("c2")], model=encoder, model_id="custom/model"
    )
    path = tmp_path / "dense.npy"
    idx.save(path)
    meta = _json.loads(path.with_suffix(".json").read_text())
    assert meta["model_id"] == "custom/model"
    assert meta["chunk_ids"] == ["c1", "c2"]
    assert meta["dim"] == 4  # _StubEncoder default dim


def test_save_load_round_trip_preserves_search(tmp_path: Path) -> None:
    """Loaded index returns the same top-K as the original."""
    import numpy as np

    encoder = _StubEncoder(
        {
            "c1\n\nBODY": [1.0, 0.0, 0.0, 0.0],
            "c2\n\nBODY": [0.0, 1.0, 0.0, 0.0],
            "find c1": [1.0, 0.0, 0.0, 0.0],
        }
    )
    chunks = [_chunk("c1"), _chunk("c2")]
    idx = DenseIndex(chunks, model=encoder)
    path = tmp_path / "dense.npy"
    idx.save(path)

    loaded = DenseIndex.load(path, model=encoder)
    # chunk_ids preserved
    assert loaded._chunk_ids == ["c1", "c2"]  # type: ignore[attr-defined]
    # embeddings array preserved (allowing float32 round-trip)
    np.testing.assert_array_almost_equal(
        loaded._embeddings,  # type: ignore[attr-defined]
        idx._embeddings,  # type: ignore[attr-defined]
    )
    # Search works identically
    hits_original = idx.search("find c1", k=2)
    hits_loaded = loaded.search("find c1", k=2)
    assert [h.chunk_id for h in hits_original] == [h.chunk_id for h in hits_loaded]


def test_load_missing_npy_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        DenseIndex.load(tmp_path / "missing.npy")


def test_load_missing_sidecar_raises(tmp_path: Path) -> None:
    """`.npy` exists but `.json` missing → must raise (not silently corrupt)."""
    import numpy as np

    path = tmp_path / "dense.npy"
    np.save(path, np.zeros((2, 4), dtype=np.float32))
    # Intentionally no sidecar JSON
    with pytest.raises(FileNotFoundError):
        DenseIndex.load(path)
