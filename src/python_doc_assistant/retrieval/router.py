"""Query router: identifier vs natural-language heuristic + retrieval dispatch.

See plans/v0-retrieval-eval.md §6 for the full contract.
"""

from __future__ import annotations

import string
from dataclasses import dataclass
from enum import Enum

from python_doc_assistant.indexes.bm25_index import BM25Index
from python_doc_assistant.indexes.symbol_index import SymbolIndex

# ------------------------------------------------------------------
# Data records
# ------------------------------------------------------------------


class QueryType(str, Enum):
    """Aligned with eval schema (PLAN.md §8): identifier / natural_language / etc."""

    IDENTIFIER = "identifier"
    NATURAL_LANGUAGE = "natural_language"


@dataclass(frozen=True)
class RouteResult:
    """Outcome of route(): which classifier branch ran and the resulting chunk_ids."""

    query_type: QueryType
    used: tuple[str, ...]  # ("symbol",), ("bm25",), or ("symbol", "bm25") if fallback ran
    chunk_ids: list[str]  # ranked, length <= k


# ------------------------------------------------------------------
# Public: classify
# ------------------------------------------------------------------


def classify(query: str) -> QueryType:
    """Heuristic classifier (no ML in v0).

    A query is IDENTIFIER if it looks like a Python symbol token:
        - no whitespace AND
        - characters limited to letters / digits / '_' / '.' (and at least one alnum)

    Everything else is NATURAL_LANGUAGE.

    Examples (plan §6 acceptance — 10 hand-labeled queries):
        "pathlib.Path.read_text"           -> IDENTIFIER
        "dict.fromKeys"                    -> IDENTIFIER
        "Path.read_text"                   -> IDENTIFIER
        "read_text"                        -> IDENTIFIER
        "open"                             -> IDENTIFIER
        "os.path.join"                     -> IDENTIFIER
        "how to iterate dict safely"       -> NATURAL_LANGUAGE
        "what is pathlib"                  -> NATURAL_LANGUAGE
        "why use Path"                     -> NATURAL_LANGUAGE
        "compare os.path and pathlib"      -> NATURAL_LANGUAGE
    """
    return QueryType.IDENTIFIER if _looks_like_identifier(query) else QueryType.NATURAL_LANGUAGE


# ------------------------------------------------------------------
# Public: route
# ------------------------------------------------------------------


def route(
    query: str,
    *,
    symbol_index: SymbolIndex,
    bm25_index: BM25Index,
    k: int = 10,
) -> RouteResult:
    """Dispatch `query` per its classified type.

    Behavior (plan §6):
        IDENTIFIER:
            1. symbol_index.lookup(query) -> list[Candidate]
            2. if any → return RouteResult with used=("symbol",)
            3. else (miss) → fallback to bm25_index.search(query, k=k);
               return RouteResult with used=("bm25",) and query_type still IDENTIFIER
               (caller can tell from `used`)
        NATURAL_LANGUAGE:
            1. bm25_index.search(query, k=k)
            2. return RouteResult with used=("bm25",)

    Returned chunk_ids are length <= k, ordered by the underlying index.
    """
    query_type = classify(query)
    if query_type == QueryType.IDENTIFIER:
        candidates = symbol_index.lookup(query)
        if candidates:
            return RouteResult(
                query_type=QueryType.IDENTIFIER,
                used=("symbol",),
                chunk_ids=[c.chunk_id for c in candidates[:k]],
            )
        hits = bm25_index.search(query, k=k)
        return RouteResult(
            query_type=QueryType.IDENTIFIER,
            used=("bm25",),
            chunk_ids=[h.chunk_id for h in hits],
        )
    hits = bm25_index.search(query, k=k)
    return RouteResult(
        query_type=QueryType.NATURAL_LANGUAGE,
        used=("bm25",),
        chunk_ids=[h.chunk_id for h in hits],
    )


# ------------------------------------------------------------------
# Helpers (private)
# ------------------------------------------------------------------


def _looks_like_identifier(query: str) -> bool:
    """True if `query` has no whitespace and only identifier-safe chars."""
    if any(c.isspace() for c in query):
        return False
    allowed = set(string.ascii_letters + string.digits + "._")
    return all(c in allowed for c in query) and any(c.isalnum() for c in query)
