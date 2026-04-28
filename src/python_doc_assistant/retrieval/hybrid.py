"""Hybrid retrieval merge functions for v2 (plan v2 §2).

Two merge strategies, each ablated separately in §5:

    rrf_merge(rankings, *, k=60)
        Reciprocal Rank Fusion. Rank-based; no score normalization needed.

    linear_merge(bm25_hits, dense_hits, *, alpha=0.5)
        Score-weighted: alpha * bm25_norm + (1 - alpha) * dense_norm.
        Each list is min-max normalized to [0, 1] independently.

API decouples from BM25Hit / DenseHit specifics:
    - RRF takes ordered chunk_id lists (one per retriever).
    - Linear takes (chunk_id, score) pairs from each side.

Caller adapts:
    bm25_hits  = [(h.chunk_id, h.score) for h in bm25_index.search(q, k=20)]
    dense_hits = [(h.chunk_id, h.score) for h in dense_index.search(q, k=20)]
    merged_rrf    = rrf_merge([
        [cid for cid, _ in bm25_hits],
        [cid for cid, _ in dense_hits],
    ])
    merged_linear = linear_merge(bm25_hits, dense_hits, alpha=0.5)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

# ------------------------------------------------------------------
# Defaults
# ------------------------------------------------------------------

DEFAULT_RRF_K: Final[int] = 60  # standard RRF smoothing constant
DEFAULT_LINEAR_ALPHA: Final[float] = 0.5  # equal weight by default


# ------------------------------------------------------------------
# Data records
# ------------------------------------------------------------------


@dataclass(frozen=True)
class HybridHit:
    """One merged-retrieval result."""

    chunk_id: str
    score: float


# ------------------------------------------------------------------
# Public: RRF
# ------------------------------------------------------------------


def rrf_merge(
    rankings: list[list[str]],
    *,
    k: int = DEFAULT_RRF_K,
) -> list[HybridHit]:
    """Reciprocal Rank Fusion across multiple ranked chunk_id lists.

    Score formula:
        score(c) = Σ_i  1 / (k + rank_i(c))
    where rank_i is the 1-indexed rank of chunk c in ranking i. Chunks
    absent from ranking i contribute 0 to the sum (no penalty).

    Args:
        rankings: list of ranked chunk_id lists, one per retriever.
        k: smoothing constant (default 60 — standard in the literature).

    Returns:
        HybridHits sorted by score descending.

    Notes:
        - Empty `rankings` (no retrievers) → return [].
        - All-empty rankings (retrievers found nothing) → return [].
        - The same chunk appearing in multiple rankings gets a boosted
          score; this is the whole point of fusion.
    """
    if len(rankings) == 0:
        return []
    chunk_scores: dict[str, float] = {}
    for ranking in rankings:
        for i, chunk_id in enumerate(ranking, start=1):
            chunk_scores.setdefault(chunk_id, 0.0)
            chunk_scores[chunk_id] += 1 / (i + k)
    hits = [HybridHit(chunk_id=chunk_id, score=score) for chunk_id, score in chunk_scores.items()]
    return sorted(hits, key=lambda h: h.score, reverse=True)


# ------------------------------------------------------------------
# Public: linear weighting
# ------------------------------------------------------------------


def linear_merge(
    bm25_hits: list[tuple[str, float]],
    dense_hits: list[tuple[str, float]],
    *,
    alpha: float = DEFAULT_LINEAR_ALPHA,
) -> list[HybridHit]:
    """Score-weighted hybrid:  alpha * bm25_norm + (1 - alpha) * dense_norm.

    Each input list is min-max normalized to [0, 1] INDEPENDENTLY (because
    BM25 raw scores and dense cosine similarities live on different scales).

    Args:
        bm25_hits:  (chunk_id, raw_bm25_score) pairs.
        dense_hits: (chunk_id, cosine_score) pairs.
        alpha: blend weight in [0, 1]. 1 → BM25 only, 0 → dense only,
               0.5 → equal contribution after normalization.

    Returns:
        HybridHits sorted by blended score descending.

    Notes:
        - A chunk in only one input list contributes 0 from the other side.
        - Constant-score input list (max == min) normalizes to all 0.5
          (avoids divide-by-zero, no within-list ranking signal).
        - Empty inputs (both lists empty) → return [].
    """
    hits: list[HybridHit] = []
    bm25_norm_scores = _min_max_normalize([score for _, score in bm25_hits])
    bm25_map = {chunk_id: score for (chunk_id, _), score in zip(bm25_hits, bm25_norm_scores)}
    dense_norm_scores = _min_max_normalize([score for _, score in dense_hits])
    dense_map = {chunk_id: score for (chunk_id, _), score in zip(dense_hits, dense_norm_scores)}
    chunk_ids = set(bm25_map) | set(dense_map)
    for chunk_id in chunk_ids:
        hits.append(
            HybridHit(
                chunk_id=chunk_id,
                score=(bm25_map.get(chunk_id, 0.0)) * alpha
                + (1 - alpha) * (dense_map.get(chunk_id, 0.0)),
            )
        )
    return sorted(hits, key=lambda h: h.score, reverse=True)


# ------------------------------------------------------------------
# Helpers (private)
# ------------------------------------------------------------------


def _min_max_normalize(scores: list[float]) -> list[float]:
    """Scale `scores` to [0, 1] via (s - min) / (max - min).

    Degenerate cases:
        - empty input → []
        - constant input (max == min) → all 0.5

    Used by linear_merge to put BM25 and cosine on the same scale before
    blending. Each retriever's list is normalized in isolation.
    """
    if len(scores) == 0:
        return []
    min_value = min(scores)
    max_value = max(scores)
    if min_value == max_value:
        return [0.5] * len(scores)
    return [(s - min_value) / (max_value - min_value) for s in scores]
