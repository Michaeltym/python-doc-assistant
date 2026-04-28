"""Tests for python_doc_assistant.retrieval.hybrid (plan v2 §2)."""

from __future__ import annotations

import pytest

from python_doc_assistant.retrieval.hybrid import (
    DEFAULT_LINEAR_ALPHA,
    DEFAULT_RRF_K,
    HybridHit,
    _min_max_normalize,
    linear_merge,
    rrf_merge,
)

# ------------------------------------------------------------------
# Constants + dataclass
# ------------------------------------------------------------------


def test_default_constants() -> None:
    assert DEFAULT_RRF_K == 60
    assert DEFAULT_LINEAR_ALPHA == 0.5


def test_hybrid_hit_is_frozen() -> None:
    h = HybridHit(chunk_id="c1", score=0.5)
    with pytest.raises(Exception):
        h.score = 0.9  # type: ignore[misc]


# ------------------------------------------------------------------
# _min_max_normalize
# ------------------------------------------------------------------


def test_min_max_normalize_basic() -> None:
    out = _min_max_normalize([1.0, 2.0, 3.0])
    assert out == [0.0, 0.5, 1.0]


def test_min_max_normalize_negative_to_positive() -> None:
    out = _min_max_normalize([-1.0, 0.0, 1.0])
    assert out == [0.0, 0.5, 1.0]


def test_min_max_normalize_empty_input() -> None:
    assert _min_max_normalize([]) == []


def test_min_max_normalize_constant_returns_half() -> None:
    """All-equal input → all 0.5 (no within-list signal; avoid divide by zero)."""
    out = _min_max_normalize([2.5, 2.5, 2.5])
    assert out == [0.5, 0.5, 0.5]


def test_min_max_normalize_single_element_returns_half() -> None:
    """A single element is degenerate — no ranking → 0.5."""
    out = _min_max_normalize([7.0])
    assert out == [0.5]


# ------------------------------------------------------------------
# rrf_merge
# ------------------------------------------------------------------


def test_rrf_merge_single_ranking_preserves_order() -> None:
    """One retriever → output mirrors its order."""
    out = rrf_merge([["a", "b", "c"]])
    assert [h.chunk_id for h in out] == ["a", "b", "c"]
    # Scores strictly decreasing: 1/(60+1) > 1/(60+2) > 1/(60+3)
    assert out[0].score > out[1].score > out[2].score


def test_rrf_merge_two_rankings_combines() -> None:
    """A chunk appearing in BOTH rankings gets a higher score than chunks in only one."""
    out = rrf_merge([["a", "b", "c"], ["a", "x", "y"]])
    by_id = {h.chunk_id: h.score for h in out}
    # 'a' is rank 1 in both → score = 1/61 + 1/61
    # 'b' is rank 2 in only the first → score = 1/62
    assert by_id["a"] > by_id["b"]
    assert by_id["a"] > by_id["x"]


def test_rrf_merge_score_formula() -> None:
    """Spot-check exact values to lock the formula."""
    out = rrf_merge([["a"], ["b"]], k=60)
    by_id = {h.chunk_id: h.score for h in out}
    # Each chunk is rank 1 in exactly one ranking → score = 1/(60+1)
    assert by_id["a"] == pytest.approx(1.0 / 61.0)
    assert by_id["b"] == pytest.approx(1.0 / 61.0)


def test_rrf_merge_unique_to_one_path_still_returned() -> None:
    """Chunk only in dense (not BM25) still appears in the merged output."""
    out = rrf_merge([["a", "b"], ["c"]])
    ids = {h.chunk_id for h in out}
    assert ids == {"a", "b", "c"}


def test_rrf_merge_custom_k_changes_scores() -> None:
    """k=10 weights top ranks more aggressively than k=60."""
    out_small_k = rrf_merge([["a"]], k=10)
    out_large_k = rrf_merge([["a"]], k=60)
    # 1/(10+1) > 1/(60+1)
    assert out_small_k[0].score > out_large_k[0].score


def test_rrf_merge_empty_rankings_list() -> None:
    """No retrievers → []."""
    assert rrf_merge([]) == []


def test_rrf_merge_all_empty_rankings() -> None:
    """All retrievers empty → []."""
    assert rrf_merge([[], []]) == []


def test_rrf_merge_sorted_descending() -> None:
    out = rrf_merge([["a", "b", "c", "d"], ["d", "a", "b", "c"]])
    scores = [h.score for h in out]
    assert scores == sorted(scores, reverse=True)


# ------------------------------------------------------------------
# linear_merge
# ------------------------------------------------------------------


def test_linear_merge_alpha_1_uses_only_bm25() -> None:
    """alpha=1 → dense ignored entirely."""
    bm25 = [("a", 10.0), ("b", 5.0)]
    dense = [("c", 0.99), ("d", 0.8)]  # dense should not appear in top
    out = linear_merge(bm25, dense, alpha=1.0)
    # Top is the bm25 winner
    assert out[0].chunk_id == "a"


def test_linear_merge_alpha_0_uses_only_dense() -> None:
    bm25 = [("a", 10.0), ("b", 5.0)]
    dense = [("c", 0.99), ("d", 0.8)]
    out = linear_merge(bm25, dense, alpha=0.0)
    assert out[0].chunk_id == "c"


def test_linear_merge_alpha_half_balances() -> None:
    """alpha=0.5 + same-rank in both → averaged normalized score."""
    bm25 = [("a", 10.0), ("b", 0.0)]
    dense = [("a", 1.0), ("b", 0.0)]
    out = linear_merge(bm25, dense, alpha=0.5)
    by_id = {h.chunk_id: h.score for h in out}
    # 'a' normalizes to 1.0 in both → blended = 1.0
    # 'b' normalizes to 0.0 in both → blended = 0.0
    assert by_id["a"] == pytest.approx(1.0)
    assert by_id["b"] == pytest.approx(0.0)


def test_linear_merge_chunk_in_one_list_only() -> None:
    """A chunk in only BM25 contributes 0 from dense (treated as below threshold)."""
    bm25 = [("a", 10.0), ("b", 0.0)]
    dense = [("c", 1.0)]
    out = linear_merge(bm25, dense, alpha=0.5)
    by_id = {h.chunk_id: h.score for h in out}
    # 'a': 0.5 * 1.0 + 0.5 * 0.0 = 0.5
    # 'b': 0.5 * 0.0 + 0.5 * 0.0 = 0.0
    # 'c': 0.5 * 0.0 + 0.5 * 0.5 (constant single-element dense) = 0.25
    assert by_id["a"] == pytest.approx(0.5)
    assert by_id["b"] == pytest.approx(0.0)
    assert by_id["c"] == pytest.approx(0.25)


def test_linear_merge_normalizes_independently() -> None:
    """BM25 and dense scales differ; min-max each side independently."""
    bm25 = [("a", 100.0), ("b", 50.0)]  # raw scale: 50-100
    dense = [("a", 0.5), ("b", 0.1)]  # raw scale: 0.1-0.5
    out = linear_merge(bm25, dense, alpha=0.5)
    by_id = {h.chunk_id: h.score for h in out}
    # 'a': 0.5 * 1.0 + 0.5 * 1.0 = 1.0  (top in both after norm)
    # 'b': 0.5 * 0.0 + 0.5 * 0.0 = 0.0  (bottom in both after norm)
    assert by_id["a"] == pytest.approx(1.0)
    assert by_id["b"] == pytest.approx(0.0)


def test_linear_merge_empty_inputs() -> None:
    assert linear_merge([], []) == []


def test_linear_merge_returns_hybrid_hits_sorted_desc() -> None:
    bm25 = [("a", 5.0), ("b", 3.0), ("c", 1.0)]
    dense = [("c", 0.9), ("a", 0.4), ("b", 0.1)]
    out = linear_merge(bm25, dense, alpha=0.5)
    assert all(isinstance(h, HybridHit) for h in out)
    scores = [h.score for h in out]
    assert scores == sorted(scores, reverse=True)


def test_linear_merge_alpha_boundary_clamping() -> None:
    """alpha=0 must yield only-dense; alpha=1 only-bm25 — verify both endpoints."""
    bm25 = [("only_bm", 7.0)]
    dense = [("only_dense", 0.9)]
    out_bm = linear_merge(bm25, dense, alpha=1.0)
    out_dn = linear_merge(bm25, dense, alpha=0.0)
    # alpha=1 → only_bm gets full bm25_norm (0.5 due to single-elem); only_dense gets 0
    assert out_bm[0].chunk_id == "only_bm"
    # alpha=0 → only_dense wins
    assert out_dn[0].chunk_id == "only_dense"
