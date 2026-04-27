"""Hit logic, Recall@K, and MRR for retrieval evaluation.

See plans/v0-retrieval-eval.md §9 and PLAN.md §8 for the contract.

Hit semantics (mirrors plan §8):
    match_policy = "any":
        Hit iff at least one retrieved chunk's symbols intersects expected_symbols
        OR at least one chunk's URL matches some expected URL (per url_match mode).
    match_policy = "all":
        Hit iff EVERY expected_symbol is covered by some retrieved chunk
        AND EVERY expected_url is covered by some retrieved chunk.

MRR semantics:
    any: 1 / (rank of FIRST matching item in retrieved); 0 if no match.
    all: 1 / (rank where the LAST-needed expected item first appears);
         0 if any expected item is missing entirely.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from python_doc_assistant.evaluation.dataset import EvalQuery

# ------------------------------------------------------------------
# Data records
# ------------------------------------------------------------------


@dataclass(frozen=True)
class RetrievedChunk:
    """One ranked result for a query (1-indexed rank)."""

    chunk_id: str
    score: float
    rank: int
    canonical_url: str
    symbols: tuple[str, ...]


@dataclass(frozen=True)
class PerQueryResult:
    """Per-query detail for per_query.jsonl."""

    query: str
    query_type: str
    match_policy: str
    url_match: str
    expected_symbols: tuple[str, ...]
    expected_urls: tuple[str, ...]
    retrieved: tuple[RetrievedChunk, ...]
    hit_at_5: bool
    hit_at_10: bool
    rank_for_mrr: int | None  # None when no full match


@dataclass(frozen=True)
class EvalRunResult:
    """Aggregate output of evaluate() — what run_writer serializes."""

    queries: tuple[PerQueryResult, ...]
    recall_at_5: float
    recall_at_10: float
    mrr: float
    n_queries: int


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


RetrieveFn = Callable[[str, int], list[RetrievedChunk]]


def evaluate(
    eval_queries: list[EvalQuery],
    retrieve_fn: RetrieveFn,
    *,
    max_k: int = 10,
) -> EvalRunResult:
    """Run retrieve_fn for every query; compute Recall@5/Recall@10/MRR.

    Suggested flow:
        per_query: list[PerQueryResult] = []
        for q in eval_queries:
            retrieved = retrieve_fn(q.query, max_k)
            hit5  = is_hit(q, retrieved, k=5)
            hit10 = is_hit(q, retrieved, k=10)
            rank  = match_rank(q, retrieved)
            per_query.append(PerQueryResult(...))
        recall5  = mean(p.hit_at_5  for p in per_query)
        recall10 = mean(p.hit_at_10 for p in per_query)
        mrr      = mean(1/p.rank_for_mrr if p.rank_for_mrr else 0 for p in per_query)
        return EvalRunResult(...)
    """
    per_query: list[PerQueryResult] = []
    for q in eval_queries:
        retrieved = retrieve_fn(q.query, max_k)
        hit_at_5 = is_hit(q, retrieved, k=5)
        hit_at_10 = is_hit(q, retrieved, k=10)
        rank_for_mrr = match_rank(q, retrieved)
        per_query.append(
            PerQueryResult(
                query=q.query,
                query_type=q.query_type,
                expected_symbols=q.expected_symbols,
                expected_urls=q.expected_urls,
                match_policy=q.match_policy,
                url_match=q.url_match,
                retrieved=tuple(retrieved),
                hit_at_5=hit_at_5,
                hit_at_10=hit_at_10,
                rank_for_mrr=rank_for_mrr,
            )
        )
    total = len(per_query)
    if total == 0:
        return EvalRunResult(queries=(), recall_at_5=0.0, recall_at_10=0.0, mrr=0.0, n_queries=0)
    recall_at_5 = sum(1 for p in per_query if p.hit_at_5) / total
    recall_at_10 = sum(1 for p in per_query if p.hit_at_10) / total
    mrr = sum(1 / p.rank_for_mrr if p.rank_for_mrr else 0.0 for p in per_query) / total
    return EvalRunResult(
        queries=tuple(per_query),
        recall_at_5=recall_at_5,
        recall_at_10=recall_at_10,
        mrr=mrr,
        n_queries=total,
    )


def is_hit(query: EvalQuery, retrieved: list[RetrievedChunk], *, k: int) -> bool:
    """Apply match_policy + url_match to top-k of `retrieved`.

    any: at least one symbol-intersection OR at least one URL-match in top-k.
    all: every expected symbol covered AND every expected URL covered in top-k.
    """
    topk = retrieved[:k]
    if query.match_policy == "any":
        for r in topk:
            if _chunk_satisfies_symbol(r.symbols, query.expected_symbols) or any(
                _url_match(r.canonical_url, url, query.url_match) for url in query.expected_urls
            ):
                return True
        return False
    elif query.match_policy == "all":
        symbol_covered = all(
            any(_chunk_satisfies_symbol(r.symbols, (s,)) for r in topk)
            for s in query.expected_symbols
        )
        url_covered = all(
            any(_url_match(r.canonical_url, u, query.url_match) for r in topk)
            for u in query.expected_urls
        )
        return symbol_covered and url_covered
    else:
        raise ValueError(f"Unknown match_policy: {query.match_policy}")


def match_rank(query: EvalQuery, retrieved: list[RetrievedChunk]) -> int | None:
    """1-indexed rank used for MRR.

    any: rank of FIRST chunk that satisfies any expected_symbol or expected_url.
         None if no chunk in `retrieved` matches.
    all: rank at which the LAST expected item is first covered.
         None if any expected item is never covered anywhere in `retrieved`.
    """
    if query.match_policy == "any":
        for r in retrieved:
            if _chunk_satisfies_symbol(r.symbols, query.expected_symbols) or any(
                _url_match(r.canonical_url, url, query.url_match) for url in query.expected_urls
            ):
                return r.rank
        return None
    elif query.match_policy == "all":
        symbol_ranks: list[int] = []
        for s in query.expected_symbols:
            for r in retrieved:
                if _chunk_satisfies_symbol(r.symbols, (s,)):
                    symbol_ranks.append(r.rank)
                    break
            else:
                return None

        url_ranks: list[int] = []
        for u in query.expected_urls:
            for r in retrieved:
                if _url_match(r.canonical_url, u, query.url_match):
                    url_ranks.append(r.rank)
                    break
            else:
                return None
        combined = symbol_ranks + url_ranks
        if not combined:
            return None
        return max(combined)
    else:
        raise ValueError(f"Unknown match_policy: {query.match_policy}")


# ------------------------------------------------------------------
# Helpers (private)
# ------------------------------------------------------------------


def _url_match(chunk_url: str, expected: str, mode: str) -> bool:
    """Compare `chunk_url` against `expected` per url_match mode.

    Modes:
        exact         -- strict equality
        strip_anchor  -- both sides stripped of `#anchor` then equal
        prefix        -- chunk_url starts with `expected`

    Default mode is "strip_anchor" (caller should pass the EvalQuery.url_match value).
    """
    if mode == "exact":
        return chunk_url == expected
    elif mode == "strip_anchor":
        return chunk_url.split("#", 1)[0] == expected.split("#", 1)[0]
    elif mode == "prefix":
        return chunk_url.startswith(expected)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def _chunk_satisfies_symbol(chunk_symbols: tuple[str, ...], expected: tuple[str, ...]) -> bool:
    """True if chunk_symbols and expected share any element."""
    return not set(chunk_symbols).isdisjoint(expected)
