"""Tests for python_doc_assistant.evaluation.retrieval_metrics.

Hermetic — fixture EvalQuery + RetrievedChunk built in-memory; no real indexes.
"""

from __future__ import annotations

from typing import Any

import pytest

from python_doc_assistant.evaluation.dataset import EvalQuery
from python_doc_assistant.evaluation.retrieval_metrics import (
    RetrievedChunk,
    _chunk_satisfies_symbol,
    _url_match,
    evaluate,
    is_hit,
    match_rank,
)

# ------------------------------------------------------------------
# Fixture builders
# ------------------------------------------------------------------


def _eval_query(
    *,
    expected_symbols: tuple[str, ...] = (),
    expected_urls: tuple[str, ...] = (),
    match_policy: str = "any",
    url_match: str = "strip_anchor",
    query_type: str = "identifier",
    query: str = "test query",
) -> EvalQuery:
    return EvalQuery(
        query=query,
        query_type=query_type,
        expected_symbols=expected_symbols,
        expected_urls=expected_urls,
        match_policy=match_policy,
        url_match=url_match,
    )


def _chunk(
    chunk_id: str,
    rank: int,
    *,
    canonical_url: str = "library/foo.html#sym",
    symbols: tuple[str, ...] = (),
    score: float = 1.0,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        score=score,
        rank=rank,
        canonical_url=canonical_url,
        symbols=symbols,
    )


# ------------------------------------------------------------------
# _url_match
# ------------------------------------------------------------------


def test_url_match_exact() -> None:
    assert _url_match("library/foo.html#bar", "library/foo.html#bar", "exact") is True
    assert _url_match("library/foo.html", "library/foo.html#bar", "exact") is False


def test_url_match_strip_anchor() -> None:
    assert _url_match("library/foo.html#bar", "library/foo.html", "strip_anchor") is True
    assert _url_match("library/foo.html", "library/foo.html", "strip_anchor") is True
    assert _url_match("library/bar.html", "library/foo.html", "strip_anchor") is False


def test_url_match_prefix() -> None:
    assert _url_match("library/pathlib.html#anchor", "library/pathlib.html", "prefix") is True
    assert _url_match("library/pathlib.html", "library", "prefix") is True
    assert _url_match("library/foo.html", "library/bar", "prefix") is False


# ------------------------------------------------------------------
# _chunk_satisfies_symbol
# ------------------------------------------------------------------


def test_chunk_satisfies_symbol_intersection() -> None:
    assert _chunk_satisfies_symbol(("a", "b"), ("b", "c")) is True
    assert _chunk_satisfies_symbol(("a",), ("b",)) is False
    assert _chunk_satisfies_symbol((), ("a",)) is False


# ------------------------------------------------------------------
# is_hit — match_policy "any"
# ------------------------------------------------------------------


def test_is_hit_any_symbol_match() -> None:
    q = _eval_query(expected_symbols=("pathlib.Path",), match_policy="any")
    chunks = [_chunk("c1", 1, symbols=("pathlib.Path",))]
    assert is_hit(q, chunks, k=5) is True


def test_is_hit_any_url_match() -> None:
    q = _eval_query(expected_urls=("library/pathlib.html",), match_policy="any")
    chunks = [_chunk("c1", 1, canonical_url="library/pathlib.html#x")]
    assert is_hit(q, chunks, k=5) is True


def test_is_hit_any_no_match() -> None:
    q = _eval_query(expected_symbols=("pathlib.Path",), match_policy="any")
    chunks = [_chunk("c1", 1, symbols=("os.path",))]
    assert is_hit(q, chunks, k=5) is False


def test_is_hit_any_respects_k() -> None:
    q = _eval_query(expected_symbols=("z",), match_policy="any")
    chunks = [
        _chunk("c1", 1, symbols=("a",)),
        _chunk("c2", 2, symbols=("b",)),
        _chunk("c3", 3, symbols=("z",)),  # match at rank 3
    ]
    assert is_hit(q, chunks, k=2) is False
    assert is_hit(q, chunks, k=3) is True


# ------------------------------------------------------------------
# is_hit — match_policy "all"
# ------------------------------------------------------------------


def test_is_hit_all_symbols_covered() -> None:
    q = _eval_query(
        expected_symbols=("pathlib.Path", "os.path"),
        match_policy="all",
    )
    chunks = [
        _chunk("c1", 1, symbols=("pathlib.Path",)),
        _chunk("c2", 2, symbols=("os.path",)),
    ]
    assert is_hit(q, chunks, k=5) is True


def test_is_hit_all_one_missing() -> None:
    q = _eval_query(
        expected_symbols=("pathlib.Path", "os.path"),
        match_policy="all",
    )
    chunks = [_chunk("c1", 1, symbols=("pathlib.Path",))]
    assert is_hit(q, chunks, k=5) is False


def test_is_hit_all_urls_covered() -> None:
    q = _eval_query(
        expected_urls=("library/a.html", "library/b.html"),
        match_policy="all",
    )
    chunks = [
        _chunk("c1", 1, canonical_url="library/a.html#x"),
        _chunk("c2", 2, canonical_url="library/b.html#y"),
    ]
    assert is_hit(q, chunks, k=5) is True


def test_is_hit_all_combined_symbol_and_url() -> None:
    """all-policy must cover BOTH expected_symbols AND expected_urls (each completely)."""
    q = _eval_query(
        expected_symbols=("a",),
        expected_urls=("library/x.html",),
        match_policy="all",
    )
    # Symbol covered but URL not present in top-k
    chunks = [_chunk("c1", 1, symbols=("a",), canonical_url="library/y.html")]
    assert is_hit(q, chunks, k=5) is False


# ------------------------------------------------------------------
# match_rank — "any"
# ------------------------------------------------------------------


def test_match_rank_any_first_hit() -> None:
    q = _eval_query(expected_symbols=("z",), match_policy="any")
    chunks = [
        _chunk("c1", 1, symbols=("a",)),
        _chunk("c2", 2, symbols=("z",)),
        _chunk("c3", 3, symbols=("z",)),
    ]
    assert match_rank(q, chunks) == 2


def test_match_rank_any_no_hit_returns_none() -> None:
    q = _eval_query(expected_symbols=("z",), match_policy="any")
    chunks = [_chunk("c1", 1, symbols=("a",))]
    assert match_rank(q, chunks) is None


# ------------------------------------------------------------------
# match_rank — "all"
# ------------------------------------------------------------------


def test_match_rank_all_returns_last_needed() -> None:
    q = _eval_query(expected_symbols=("a", "b"), match_policy="all")
    chunks = [
        _chunk("c1", 1, symbols=("a",)),  # covers "a"
        _chunk("c2", 2, symbols=("x",)),
        _chunk("c3", 3, symbols=("b",)),  # covers "b" — last needed → rank 3
    ]
    assert match_rank(q, chunks) == 3


def test_match_rank_all_missing_returns_none() -> None:
    q = _eval_query(expected_symbols=("a", "b"), match_policy="all")
    chunks = [_chunk("c1", 1, symbols=("a",))]
    assert match_rank(q, chunks) is None


# ------------------------------------------------------------------
# evaluate — orchestration
# ------------------------------------------------------------------


def test_evaluate_computes_recall_and_mrr() -> None:
    queries = [
        _eval_query(expected_symbols=("a",), query="q1"),
        _eval_query(expected_symbols=("b",), query="q2"),
        _eval_query(expected_symbols=("c",), query="q3"),
    ]

    def fake_retrieve(query: str, k: int) -> list[RetrievedChunk]:
        # q1 hits at rank 1 (symbol a); q2 hits at rank 2 (symbol b); q3 misses
        if query == "q1":
            return [_chunk("c1", 1, symbols=("a",))]
        if query == "q2":
            return [_chunk("c1", 1, symbols=("x",)), _chunk("c2", 2, symbols=("b",))]
        return [_chunk("c1", 1, symbols=("z",))]

    result = evaluate(queries, fake_retrieve, max_k=10)

    assert result.n_queries == 3
    # 2 of 3 hit at top-5 (and top-10); recall = 2/3
    assert abs(result.recall_at_5 - 2 / 3) < 1e-9
    assert abs(result.recall_at_10 - 2 / 3) < 1e-9
    # MRR = (1/1 + 1/2 + 0) / 3
    assert abs(result.mrr - (1.0 + 0.5 + 0.0) / 3) < 1e-9


def test_evaluate_per_query_records_hit_flags(  # noqa: PLR0913
    monkeypatch: Any = None,
) -> None:
    queries = [_eval_query(expected_symbols=("a",), query="q")]

    def fake_retrieve(_q: str, _k: int) -> list[RetrievedChunk]:
        return [_chunk("c1", 1, symbols=("a",))]

    result = evaluate(queries, fake_retrieve, max_k=10)
    assert len(result.queries) == 1
    pq = result.queries[0]
    assert pq.hit_at_5 is True
    assert pq.hit_at_10 is True
    assert pq.rank_for_mrr == 1


def test_evaluate_empty_queries() -> None:
    """Edge: empty eval set yields zero metrics, not divide-by-zero."""
    result = evaluate([], lambda _q, _k: [], max_k=10)
    assert result.n_queries == 0
    assert result.recall_at_5 == 0.0
    assert result.recall_at_10 == 0.0
    assert result.mrr == 0.0


# ------------------------------------------------------------------
# Adversarial: recall_at_10 must use hit_at_10 (catches copy-paste bug)
# ------------------------------------------------------------------


def test_evaluate_recall_at_10_distinct_from_recall_at_5() -> None:
    """If a query hits at rank 7, hit_at_5=False but hit_at_10=True.

    recall_at_5 should NOT equal recall_at_10 in this case.
    """
    queries = [_eval_query(expected_symbols=("z",), query="q1")]

    def fake_retrieve(_query: str, _k: int) -> list[RetrievedChunk]:
        # 6 misses, then "z" at rank 7
        return [_chunk(f"c{i}", i, symbols=("x",)) for i in range(1, 7)] + [
            _chunk("c7", 7, symbols=("z",))
        ]

    result = evaluate(queries, fake_retrieve, max_k=10)
    assert result.recall_at_5 == 0.0  # not in top-5
    assert result.recall_at_10 == 1.0  # in top-10
    assert result.recall_at_5 != result.recall_at_10


# ------------------------------------------------------------------
# Adversarial: is_hit all-policy must respect k (not look at full retrieved)
# ------------------------------------------------------------------


def test_is_hit_all_respects_k_does_not_peek_beyond() -> None:
    """all-policy with k=2 must NOT count expected items that appear at rank 3+."""
    q = _eval_query(
        expected_symbols=("a", "b"),
        match_policy="all",
    )
    chunks = [
        _chunk("c1", 1, symbols=("a",)),  # within top-2
        _chunk("c2", 2, symbols=("x",)),  # within top-2
        _chunk("c3", 3, symbols=("b",)),  # OUTSIDE top-2
    ]
    # is_hit(k=2) must look only at top-2 → "b" not covered → False
    assert is_hit(q, chunks, k=2) is False
    # is_hit(k=3) sees all three → covered → True
    assert is_hit(q, chunks, k=3) is True


def test_is_hit_all_url_respects_k() -> None:
    q = _eval_query(
        expected_urls=("library/a.html", "library/b.html"),
        match_policy="all",
    )
    chunks = [
        _chunk("c1", 1, canonical_url="library/a.html#x"),
        _chunk("c2", 2, canonical_url="library/c.html#x"),
        _chunk("c3", 3, canonical_url="library/b.html#x"),
    ]
    assert is_hit(q, chunks, k=2) is False
    assert is_hit(q, chunks, k=3) is True


# ------------------------------------------------------------------
# Adversarial: match_rank all-policy with duplicate appearances
# ------------------------------------------------------------------


def test_match_rank_all_with_duplicate_expected_appearances() -> None:
    """If "a" appears at rank 2 AND rank 5, match_rank should not count it twice.

    The reciprocal of the LAST first-appearance rank — here: a@2, b@4 → max(2,4)=4.
    """
    q = _eval_query(expected_symbols=("a", "b"), match_policy="all")
    chunks = [
        _chunk("c1", 1, symbols=("x",)),
        _chunk("c2", 2, symbols=("a",)),  # a's first appearance
        _chunk("c3", 3, symbols=("y",)),
        _chunk("c4", 4, symbols=("b",)),  # b's first appearance — last needed
        _chunk("c5", 5, symbols=("a",)),  # a again, must NOT confuse logic
    ]
    assert match_rank(q, chunks) == 4


# ------------------------------------------------------------------
# Adversarial: unknown match_policy / unknown url_match should fail-fast
# ------------------------------------------------------------------


def test_is_hit_unknown_match_policy_raises() -> None:
    q = _eval_query(expected_symbols=("a",), match_policy="bogus")
    with pytest.raises(ValueError):
        is_hit(q, [_chunk("c1", 1, symbols=("a",))], k=5)


def test_match_rank_unknown_match_policy_raises() -> None:
    q = _eval_query(expected_symbols=("a",), match_policy="bogus")
    with pytest.raises(ValueError):
        match_rank(q, [_chunk("c1", 1, symbols=("a",))])


def test_url_match_unknown_mode_raises() -> None:
    with pytest.raises(ValueError):
        _url_match("library/a.html", "library/a.html", "fuzzy")


# ------------------------------------------------------------------
# Adversarial: match_rank Branch B with empty expected (max([]) edge)
# ------------------------------------------------------------------


def test_match_rank_all_empty_expected_returns_none() -> None:
    """Both expected_symbols and expected_urls empty under all-policy:
    no items to cover → semantically nothing matches → None (not max([])).
    """
    q = _eval_query(expected_symbols=(), expected_urls=(), match_policy="all")
    chunks = [_chunk("c1", 1, symbols=("a",))]
    # Should NOT raise ValueError from max([])
    assert match_rank(q, chunks) is None
