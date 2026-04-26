"""Tests for python_doc_assistant.indexes.symbol_index.

Hermetic — fixture chunks built in-memory; no real docs / objects.inv needed.
"""

from __future__ import annotations

import pytest

from python_doc_assistant.indexes.symbol_index import (
    UNKNOWN_ROLE_PRIORITY,
    Candidate,
    SymbolIndex,
    _role_priority,
    _short_names,
    _sort_candidates,
)
from python_doc_assistant.ingest.chunker import Chunk
from python_doc_assistant.ingest.parse_objects_inv import SymbolEntry

# ------------------------------------------------------------------
# Fixture builders
# ------------------------------------------------------------------


def _symbol_chunk(name: str, role: str, module: str | None) -> Chunk:
    return Chunk(
        chunk_id=f"symbol:{name}",
        chunk_type="symbol",
        docs_version="3.12",
        title=name.rsplit(".", 1)[-1],
        text="stub body",
        symbols=(name,),
        canonical_url=f"library/foo.html#{name}",
        anchor=name,
        parent_module=module,
        source_path="library/foo.html",
        source_hash="sha256:abc",
    )


def _symbol_entry(name: str, role: str, module: str | None) -> SymbolEntry:
    return SymbolEntry(
        name=name,
        role=role,
        uri=f"library/foo.html#{name}",
        module=module,
    )


def _section_chunk(title: str) -> Chunk:
    return Chunk(
        chunk_id=f"section:library/foo#{title.lower()}",
        chunk_type="section",
        docs_version="3.12",
        title=title,
        text="example body",
        symbols=(),
        canonical_url=f"library/foo.html#{title.lower()}",
        anchor=title.lower(),
        parent_module=None,
        source_path="library/foo.html",
        source_hash="sha256:abc",
    )


SYMBOL_FIXTURES: list[tuple[str, str, str | None]] = [
    ("pathlib.Path", "py:class", "pathlib"),
    ("pathlib.Path.read_text", "py:method", "pathlib"),
    ("pathlib.Path.write_text", "py:method", "pathlib"),
    ("pathlib.PurePath", "py:class", "pathlib"),
    ("dict.fromkeys", "py:method", "dict"),
    ("dict", "py:class", None),
    ("open", "py:function", None),  # builtin
    ("io.open", "py:function", "io"),
    ("os.open", "py:function", "os"),
    ("codecs.open", "py:function", "codecs"),
]


@pytest.fixture
def fake_chunks() -> list[Chunk]:
    """Mix of symbol_chunks (10) and one section_chunk (must be ignored by the index)."""
    chunks: list[Chunk] = [
        _symbol_chunk(name, role, module) for name, role, module in SYMBOL_FIXTURES
    ]
    chunks.append(_section_chunk("Examples"))
    return chunks


@pytest.fixture
def fake_symbols() -> list[SymbolEntry]:
    """Matching SymbolEntry list for the symbol_chunks above; index pairs by name."""
    return [_symbol_entry(name, role, module) for name, role, module in SYMBOL_FIXTURES]


# ------------------------------------------------------------------
# _short_names
# ------------------------------------------------------------------


def test_short_names_three_segments() -> None:
    assert set(_short_names("pathlib.Path.read_text")) == {
        "pathlib.Path.read_text",
        "Path.read_text",
        "read_text",
    }


def test_short_names_two_segments() -> None:
    assert set(_short_names("dict.fromkeys")) == {"dict.fromkeys", "fromkeys"}


def test_short_names_single_segment() -> None:
    assert _short_names("open") == ["open"]


def test_short_names_full_name_first() -> None:
    """The full name is always present in the returned list."""
    assert "pathlib.Path.read_text" in _short_names("pathlib.Path.read_text")


# ------------------------------------------------------------------
# _role_priority
# ------------------------------------------------------------------


def test_role_priority_class_lower_than_method() -> None:
    assert _role_priority("py:class") < _role_priority("py:method")


def test_role_priority_method_lower_than_function() -> None:
    assert _role_priority("py:method") < _role_priority("py:function")


def test_role_priority_unknown_returns_sentinel() -> None:
    assert _role_priority("py:unknown") == UNKNOWN_ROLE_PRIORITY


# ------------------------------------------------------------------
# _sort_candidates
# ------------------------------------------------------------------


def test_sort_candidates_class_before_method() -> None:
    method = Candidate(
        chunk_id="x",
        fully_qualified_name="pathlib.Path.read_text",
        role="py:method",
        parent_module="pathlib",
    )
    klass = Candidate(
        chunk_id="y",
        fully_qualified_name="pathlib.Path",
        role="py:class",
        parent_module="pathlib",
    )
    out = _sort_candidates([method, klass])
    assert out[0].role == "py:class"
    assert out[1].role == "py:method"


def test_sort_candidates_same_role_sorted_by_module() -> None:
    a = Candidate(
        chunk_id="1", fully_qualified_name="os.open", role="py:function", parent_module="os"
    )
    b = Candidate(
        chunk_id="2", fully_qualified_name="io.open", role="py:function", parent_module="io"
    )
    out = _sort_candidates([a, b])
    assert [c.parent_module for c in out] == ["io", "os"]


# ------------------------------------------------------------------
# SymbolIndex.exact
# ------------------------------------------------------------------


def test_exact_full_name_returns_single(
    fake_chunks: list[Chunk], fake_symbols: list[SymbolEntry]
) -> None:
    idx = SymbolIndex(fake_chunks, fake_symbols)
    cands = idx.exact("pathlib.Path.read_text")
    assert len(cands) == 1
    assert cands[0].fully_qualified_name == "pathlib.Path.read_text"
    assert cands[0].role == "py:method"


def test_exact_short_name_multi_candidate_open(
    fake_chunks: list[Chunk], fake_symbols: list[SymbolEntry]
) -> None:
    """Plan §4 acceptance: short `open` should resolve to all 4 variants."""
    idx = SymbolIndex(fake_chunks, fake_symbols)
    cands = idx.exact("open")
    names = {c.fully_qualified_name for c in cands}
    assert names == {"open", "io.open", "os.open", "codecs.open"}


def test_exact_two_segment_short_form(
    fake_chunks: list[Chunk], fake_symbols: list[SymbolEntry]
) -> None:
    idx = SymbolIndex(fake_chunks, fake_symbols)
    cands = idx.exact("Path.read_text")
    assert {c.fully_qualified_name for c in cands} == {"pathlib.Path.read_text"}


def test_exact_unknown_returns_empty(
    fake_chunks: list[Chunk], fake_symbols: list[SymbolEntry]
) -> None:
    idx = SymbolIndex(fake_chunks, fake_symbols)
    assert idx.exact("zzznonexistent") == []


def test_exact_section_chunks_not_indexed(
    fake_chunks: list[Chunk], fake_symbols: list[SymbolEntry]
) -> None:
    """Only symbol_chunks populate the index; section_chunks are ignored."""
    idx = SymbolIndex(fake_chunks, fake_symbols)
    assert idx.exact("Examples") == []


def test_exact_candidate_fields_populated(
    fake_chunks: list[Chunk], fake_symbols: list[SymbolEntry]
) -> None:
    idx = SymbolIndex(fake_chunks, fake_symbols)
    cand = idx.exact("pathlib.Path.read_text")[0]
    assert cand.chunk_id == "symbol:pathlib.Path.read_text"
    assert cand.fully_qualified_name == "pathlib.Path.read_text"
    assert cand.role == "py:method"
    assert cand.parent_module == "pathlib"


# ------------------------------------------------------------------
# SymbolIndex.fuzzy
# ------------------------------------------------------------------


def test_fuzzy_case_difference_matches(
    fake_chunks: list[Chunk], fake_symbols: list[SymbolEntry]
) -> None:
    """`dict.fromKeys` should fuzzy-match `dict.fromkeys`."""
    idx = SymbolIndex(fake_chunks, fake_symbols)
    cands = idx.fuzzy("dict.fromKeys", threshold=85)
    assert "dict.fromkeys" in {c.fully_qualified_name for c in cands}


def test_fuzzy_typo_matches(fake_chunks: list[Chunk], fake_symbols: list[SymbolEntry]) -> None:
    """`pathlib.Path.read_txt` (missing 'e') should fuzzy-match read_text."""
    idx = SymbolIndex(fake_chunks, fake_symbols)
    cands = idx.fuzzy("pathlib.Path.read_txt", threshold=80)
    assert "pathlib.Path.read_text" in {c.fully_qualified_name for c in cands}


def test_fuzzy_threshold_filters_distant_matches(
    fake_chunks: list[Chunk], fake_symbols: list[SymbolEntry]
) -> None:
    """A query with no plausible match returns empty at high thresholds."""
    idx = SymbolIndex(fake_chunks, fake_symbols)
    assert idx.fuzzy("zzznonexistent", threshold=95) == []


# ------------------------------------------------------------------
# SymbolIndex.lookup
# ------------------------------------------------------------------


def test_lookup_exact_preferred(fake_chunks: list[Chunk], fake_symbols: list[SymbolEntry]) -> None:
    """Exact match wins; fuzzy is not consulted when exact returns results."""
    idx = SymbolIndex(fake_chunks, fake_symbols)
    cands = idx.lookup("pathlib.Path.read_text")
    assert {c.fully_qualified_name for c in cands} == {"pathlib.Path.read_text"}


def test_lookup_falls_back_to_fuzzy(
    fake_chunks: list[Chunk], fake_symbols: list[SymbolEntry]
) -> None:
    """No exact match → fuzzy()."""
    idx = SymbolIndex(fake_chunks, fake_symbols)
    cands = idx.lookup("dict.fromKeys")
    assert "dict.fromkeys" in {c.fully_qualified_name for c in cands}


def test_lookup_unknown_returns_empty(
    fake_chunks: list[Chunk], fake_symbols: list[SymbolEntry]
) -> None:
    idx = SymbolIndex(fake_chunks, fake_symbols)
    assert idx.lookup("zzznonexistent") == []
