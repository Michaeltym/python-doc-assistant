"""Tests for python_doc_assistant.indexes.bm25_index.

Hermetic — fixture chunks built in-memory; persistence uses tmp_path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from python_doc_assistant.indexes.bm25_index import (
    BM25Index,
    Hit,
    _lowercase_and_drop_empty,
    _split_camelcase,
    _split_on_dots,
    _split_underscores,
    analyze,
)
from python_doc_assistant.ingest.chunker import Chunk

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


CORPUS_NAMES: list[str] = [
    "pathlib.Path.read_text",
    "pathlib.Path.write_text",
    "pathlib.Path",
    "dict.fromkeys",
    "dict.update",
    "dict.items",
    "os.path.join",
    "os.path.split",
    "io.open",
    "json.loads",
]


@pytest.fixture
def fake_chunks() -> list[Chunk]:
    """10 symbol_chunks covering identifiers used in the parametrized search test."""
    return [_symbol_chunk(name) for name in CORPUS_NAMES]


# ------------------------------------------------------------------
# analyze — plan §5 acceptance: 4 canonical examples
# ------------------------------------------------------------------


def test_analyze_pathlib_path_read_text() -> None:
    assert analyze("pathlib.Path.read_text") == ["pathlib", "path", "read_text", "read", "text"]


def test_analyze_dict_fromKeys() -> None:
    assert analyze("dict.fromKeys") == ["dict", "fromkeys", "from", "keys"]


def test_analyze_os_path_join() -> None:
    assert analyze("os.path.join") == ["os", "path", "join"]


def test_analyze_natural_language() -> None:
    assert analyze("how to iterate dict safely") == ["how", "to", "iterate", "dict", "safely"]


# ------------------------------------------------------------------
# analyze — edge cases
# ------------------------------------------------------------------


def test_analyze_uppercase_abbrev_splits_HTTPServer() -> None:
    """HTTPServer must split into HTTP + Server (and merged 'httpserver' too)."""
    tokens = analyze("HTTPServer")
    assert "http" in tokens
    assert "server" in tokens
    assert "httpserver" in tokens


def test_analyze_dunder_preserved() -> None:
    """__init__ is preserved as a token alongside the inner 'init'."""
    tokens = analyze("__init__")
    assert "__init__" in tokens
    assert "init" in tokens


def test_analyze_lowercase_runs_last() -> None:
    """Order check: if lowercase ran before CamelCase, fromKeys -> fromkeys, no boundary."""
    tokens = analyze("fromKeys")
    assert "from" in tokens
    assert "keys" in tokens
    assert "fromkeys" in tokens


def test_analyze_empty_input_returns_empty() -> None:
    assert analyze("") == []


def test_analyze_drops_punctuation_and_empties() -> None:
    """Pure-punctuation tokens (e.g. extra dots) must be filtered out."""
    tokens = analyze("a.b...c")
    assert "" not in tokens
    assert "." not in tokens
    assert "a" in tokens
    assert "b" in tokens
    assert "c" in tokens


# ------------------------------------------------------------------
# Helper unit tests (catch ordering bugs in isolation)
# ------------------------------------------------------------------


def test_split_on_dots() -> None:
    assert _split_on_dots(["pathlib.Path", "os"]) == ["pathlib", "Path", "os"]


def test_split_camelcase_keeps_merged_form() -> None:
    out = _split_camelcase(["fromKeys"])
    assert "from" in out
    assert "Keys" in out
    assert "fromKeys" in out


def test_split_underscores_keeps_merged_form() -> None:
    out = _split_underscores(["read_text"])
    assert "read" in out
    assert "text" in out
    assert "read_text" in out


def test_lowercase_and_drop_empty_removes_blanks() -> None:
    assert _lowercase_and_drop_empty(["Foo", "", "BAR", "."]) == ["foo", "bar"]


# ------------------------------------------------------------------
# BM25Index
# ------------------------------------------------------------------


def test_bm25_index_search_returns_hits(fake_chunks: list[Chunk]) -> None:
    idx = BM25Index(fake_chunks)
    hits = idx.search("read_text", k=3)
    assert len(hits) <= 3
    assert all(isinstance(h, Hit) for h in hits)
    assert all(h.score > 0 for h in hits)


def test_bm25_index_search_orders_by_score_desc(fake_chunks: list[Chunk]) -> None:
    idx = BM25Index(fake_chunks)
    hits = idx.search("path", k=5)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.parametrize(
    "query,expected_chunk_id",
    [
        ("Path.read_text", "symbol:pathlib.Path.read_text"),
        ("read_text", "symbol:pathlib.Path.read_text"),
        ("Path.write_text", "symbol:pathlib.Path.write_text"),
        ("dict.fromkeys", "symbol:dict.fromkeys"),
        ("fromkeys", "symbol:dict.fromkeys"),
        ("dict.fromKeys", "symbol:dict.fromkeys"),  # camelCase typo via analyzer
        ("os.path.join", "symbol:os.path.join"),
        ("path.split", "symbol:os.path.split"),
        ("io.open", "symbol:io.open"),
        ("json.loads", "symbol:json.loads"),
    ],
)
def test_bm25_index_symbol_queries_top5(
    fake_chunks: list[Chunk], query: str, expected_chunk_id: str
) -> None:
    """Plan §5 acceptance: 10 symbol-style queries each surface their target in Top-5."""
    idx = BM25Index(fake_chunks)
    hits = idx.search(query, k=5)
    assert expected_chunk_id in {h.chunk_id for h in hits}


# ------------------------------------------------------------------
# Persistence
# ------------------------------------------------------------------


def test_bm25_save_load_round_trip(tmp_path: Path, fake_chunks: list[Chunk]) -> None:
    idx = BM25Index(fake_chunks)
    path = tmp_path / "bm25.pkl"
    idx.save(path)
    assert path.is_file()

    restored = BM25Index.load(path)
    original_hits = [(h.chunk_id, round(h.score, 6)) for h in idx.search("Path.read_text", k=5)]
    restored_hits = [
        (h.chunk_id, round(h.score, 6)) for h in restored.search("Path.read_text", k=5)
    ]
    assert original_hits == restored_hits


def test_bm25_save_creates_parent_dirs(tmp_path: Path, fake_chunks: list[Chunk]) -> None:
    idx = BM25Index(fake_chunks)
    deep = tmp_path / "data" / "indexes" / "3.12" / "abc123" / "bm25.pkl"
    idx.save(deep)
    assert deep.is_file()
