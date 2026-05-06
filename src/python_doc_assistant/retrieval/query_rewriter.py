"""Pre-generation query rewriter for typo'd identifier queries.

Background:
    The dense+rerank retriever already lifts the right symbol chunk into
    top-K when the user types a typo (e.g. `subprocess.runn` → top-2 is
    `symbol:subprocess.run`). The Qwen 7B Q4 generator, however, refuses
    such queries because the surface form does not match the chunk
    symbol exactly (v4 sub-task 1 Round 3 measurement: prompt-only typo
    coaching trades +5pp accuracy for +5pp hallucination).

    This module sits between retrieval and generation, post-processing
    the query so the generator sees the canonical symbol when one of
    the top-K chunks is an obvious near-match. Pure code, no LLM call,
    no extra dependency.

Algorithm:
    1. Scan all retrieved chunks; for each symbol_chunk, compute
       levenshtein(query.lower(), symbol.lower()).
    2. Short-circuit: if any chunk's symbol exactly equals the query,
       return the query unchanged (query is already canonical).
    3. Pick the minimum distance across the scanned candidates. If it
       exceeds MAX_LEV_DISTANCE, return query unchanged.
    4. Disambiguation: if more than one distinct symbol ties at the
       minimum distance, return query unchanged (e.g. `list.cont` is
       equidistant from `list.copy` and `list.count`).
    5. Otherwise return that single closest symbol.

Why scan all top-K rather than just top-1: dense+rerank often promotes
the parent module to rank 1 (e.g. `subprocess.runn` → top-1 is
`subprocess`, top-2 is `subprocess.run`). The typo target is usually
within top-K, not necessarily at rank 1.

Returns the canonical symbol on rewrite, otherwise returns the original
query unchanged. Caller must keep the original query for retrieval and
eval logging — this rewrite is generator-input-only.
"""

from __future__ import annotations

from typing import Final

from python_doc_assistant.ingest.chunker import Chunk

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

MAX_LEV_DISTANCE: Final[int] = 2  # >2 treated as a different identifier
SYMBOL_PREFIX: Final[str] = "symbol:"


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def maybe_rewrite_query(query: str, retrieved_chunks: list[Chunk]) -> str:
    """Rewrite query → canonical symbol if exactly one top-K chunk is an obvious typo match.

    See module docstring for the algorithm.

    Args:
        query: original user query (may contain a typo).
        retrieved_chunks: top-K chunks from the retriever, in rank order.

    Returns:
        The canonical symbol when a rewrite fires; otherwise the
        original query unchanged.
    """
    if not retrieved_chunks:
        return query
    query_lower = query.lower()
    candidates: list[tuple[int, str]] = []
    for chunk in retrieved_chunks:
        symbol = _symbol_of(chunk)
        if symbol is None:
            continue
        if symbol == query:
            return query
        candidates.append((_levenshtein(query_lower, symbol.lower()), symbol))
    if not candidates:
        return query
    min_distance = min(distance for distance, _ in candidates)
    if min_distance > MAX_LEV_DISTANCE:
        return query
    closest = []
    for distance, symbol in candidates:
        if distance == min_distance and symbol not in closest:
            closest.append(symbol)
    if len(closest) > 1:
        return query
    return closest[0]


# ------------------------------------------------------------------
# Helpers (private)
# ------------------------------------------------------------------


def _symbol_of(chunk: Chunk) -> str | None:
    """Return canonical symbol name for a symbol chunk, else None."""
    if not chunk.chunk_id.startswith(SYMBOL_PREFIX):
        return None
    if not chunk.symbols:
        return None
    return chunk.symbols[0]


def _levenshtein(a: str, b: str) -> int:
    """Iterative Levenshtein edit distance (insert / delete / substitute)."""
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            insert_cost = curr[j - 1] + 1
            delete_cost = prev[j] + 1
            substitute_cost = prev[j - 1] + (ca != cb)
            curr.append(min(insert_cost, delete_cost, substitute_cost))
        prev = curr
    return prev[-1]
