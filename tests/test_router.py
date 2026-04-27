"""Tests for python_doc_assistant.retrieval.router.

Hermetic — fixture chunks + symbols built in-memory; small SymbolIndex + BM25Index.
"""

from __future__ import annotations

import pytest

from python_doc_assistant.indexes.bm25_index import BM25Index
from python_doc_assistant.indexes.symbol_index import SymbolIndex
from python_doc_assistant.ingest.chunker import Chunk
from python_doc_assistant.ingest.parse_objects_inv import SymbolEntry
from python_doc_assistant.retrieval.router import (
    QueryType,
    RouteResult,
    _looks_like_identifier,
    classify,
    route,
)

# ------------------------------------------------------------------
# Fixture builders
# ------------------------------------------------------------------


def _symbol_chunk(name: str, text: str | None = None) -> Chunk:
    parent = name.rsplit(".", 1)[0] if "." in name else None
    return Chunk(
        chunk_id=f"symbol:{name}",
        chunk_type="symbol",
        docs_version="3.12",
        title=name.rsplit(".", 1)[-1],
        text=text or f"Stub documentation for {name}",
        symbols=(name,),
        canonical_url=f"library/foo.html#{name}",
        anchor=name,
        parent_module=parent,
        source_path="library/foo.html",
        source_hash="sha256:abc",
    )


def _symbol_entry(name: str, role: str, module: str | None) -> SymbolEntry:
    return SymbolEntry(name=name, role=role, uri=f"library/foo.html#{name}", module=module)


SYMBOL_FIXTURES: list[tuple[str, str, str | None]] = [
    ("pathlib.Path", "py:class", "pathlib"),
    ("pathlib.Path.read_text", "py:method", "pathlib"),
    ("pathlib.Path.write_text", "py:method", "pathlib"),
    ("dict.fromkeys", "py:method", "dict"),
    ("os.path.join", "py:function", "os.path"),
    ("io.open", "py:function", "io"),
    ("json.loads", "py:function", "json"),
]


@pytest.fixture
def fake_chunks() -> list[Chunk]:
    chunks = [_symbol_chunk(name) for name, _, _ in SYMBOL_FIXTURES]
    # Add a section_chunk with natural-language body to test BM25 fallback
    chunks.append(
        Chunk(
            chunk_id="section:library/howto#dict-iteration",
            chunk_type="section",
            docs_version="3.12",
            title="Dict iteration",
            text="how to iterate a dict safely without mutating it during the loop",
            symbols=(),
            canonical_url="library/howto.html#dict-iteration",
            anchor="dict-iteration",
            parent_module=None,
            source_path="library/howto.html",
            source_hash="sha256:def",
        )
    )
    return chunks


@pytest.fixture
def fake_symbols() -> list[SymbolEntry]:
    return [_symbol_entry(name, role, module) for name, role, module in SYMBOL_FIXTURES]


@pytest.fixture
def symbol_index(fake_chunks: list[Chunk], fake_symbols: list[SymbolEntry]) -> SymbolIndex:
    return SymbolIndex(fake_chunks, fake_symbols)


@pytest.fixture
def bm25_index(fake_chunks: list[Chunk]) -> BM25Index:
    return BM25Index(fake_chunks)


# ------------------------------------------------------------------
# _looks_like_identifier
# ------------------------------------------------------------------


def test_looks_like_identifier_dotted() -> None:
    assert _looks_like_identifier("pathlib.Path.read_text") is True


def test_looks_like_identifier_single_word() -> None:
    assert _looks_like_identifier("open") is True


def test_looks_like_identifier_with_underscore() -> None:
    assert _looks_like_identifier("read_text") is True


def test_looks_like_identifier_rejects_whitespace() -> None:
    assert _looks_like_identifier("how to iterate") is False


def test_looks_like_identifier_rejects_punctuation_only() -> None:
    assert _looks_like_identifier("!!!") is False


def test_looks_like_identifier_rejects_empty() -> None:
    assert _looks_like_identifier("") is False


# ------------------------------------------------------------------
# classify — plan §6 acceptance: 10 hand-labeled queries
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    "query,expected",
    [
        ("pathlib.Path.read_text", QueryType.IDENTIFIER),
        ("dict.fromKeys", QueryType.IDENTIFIER),
        ("Path.read_text", QueryType.IDENTIFIER),
        ("read_text", QueryType.IDENTIFIER),
        ("open", QueryType.IDENTIFIER),
        ("os.path.join", QueryType.IDENTIFIER),
        ("how to iterate dict safely", QueryType.NATURAL_LANGUAGE),
        ("what is pathlib", QueryType.NATURAL_LANGUAGE),
        ("why use Path", QueryType.NATURAL_LANGUAGE),
        ("compare os.path and pathlib", QueryType.NATURAL_LANGUAGE),
    ],
)
def test_classify_plan_acceptance(query: str, expected: QueryType) -> None:
    assert classify(query) == expected


# ------------------------------------------------------------------
# route
# ------------------------------------------------------------------


def test_route_identifier_hit_uses_symbol_only(
    symbol_index: SymbolIndex, bm25_index: BM25Index
) -> None:
    result = route(
        "pathlib.Path.read_text",
        symbol_index=symbol_index,
        bm25_index=bm25_index,
        k=5,
    )
    assert result.query_type == QueryType.IDENTIFIER
    assert result.used == ("symbol",)
    assert "symbol:pathlib.Path.read_text" in result.chunk_ids


def test_route_identifier_miss_falls_back_to_bm25(
    symbol_index: SymbolIndex, bm25_index: BM25Index
) -> None:
    """Identifier with no symbol_index hit → fallback BM25 (used=('bm25',))."""
    result = route(
        "completely.unknown.symbol",
        symbol_index=symbol_index,
        bm25_index=bm25_index,
        k=5,
    )
    assert result.query_type == QueryType.IDENTIFIER
    assert result.used == ("bm25",)


def test_route_natural_language_uses_bm25(symbol_index: SymbolIndex, bm25_index: BM25Index) -> None:
    result = route(
        "how to iterate dict safely",
        symbol_index=symbol_index,
        bm25_index=bm25_index,
        k=5,
    )
    assert result.query_type == QueryType.NATURAL_LANGUAGE
    assert result.used == ("bm25",)
    assert "section:library/howto#dict-iteration" in result.chunk_ids


def test_route_respects_k(symbol_index: SymbolIndex, bm25_index: BM25Index) -> None:
    result = route("how to iterate", symbol_index=symbol_index, bm25_index=bm25_index, k=2)
    assert len(result.chunk_ids) <= 2


def test_route_returns_route_result_dataclass(
    symbol_index: SymbolIndex, bm25_index: BM25Index
) -> None:
    result = route("open", symbol_index=symbol_index, bm25_index=bm25_index, k=3)
    assert isinstance(result, RouteResult)
    assert isinstance(result.chunk_ids, list)


# ------------------------------------------------------------------
# Plan §6 routing acceptance — 10 queries each end-to-end
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    "query,expected_type,expected_used",
    [
        ("pathlib.Path.read_text", QueryType.IDENTIFIER, ("symbol",)),
        ("dict.fromkeys", QueryType.IDENTIFIER, ("symbol",)),
        ("Path.read_text", QueryType.IDENTIFIER, ("symbol",)),
        ("read_text", QueryType.IDENTIFIER, ("symbol",)),
        ("open", QueryType.IDENTIFIER, ("symbol",)),
        ("os.path.join", QueryType.IDENTIFIER, ("symbol",)),
        ("how to iterate dict safely", QueryType.NATURAL_LANGUAGE, ("bm25",)),
        ("what is pathlib", QueryType.NATURAL_LANGUAGE, ("bm25",)),
        ("why use Path", QueryType.NATURAL_LANGUAGE, ("bm25",)),
        ("compare os.path and pathlib", QueryType.NATURAL_LANGUAGE, ("bm25",)),
    ],
)
def test_route_plan_acceptance(
    symbol_index: SymbolIndex,
    bm25_index: BM25Index,
    query: str,
    expected_type: QueryType,
    expected_used: tuple[str, ...],
) -> None:
    """Plan §6 acceptance: 10 hand-labeled queries route correctly."""
    result = route(query, symbol_index=symbol_index, bm25_index=bm25_index, k=5)
    assert result.query_type == expected_type
    assert result.used == expected_used
