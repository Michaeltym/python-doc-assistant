"""Tests for python_doc_assistant.ingest.chunker.

Hermetic — fake HTML written under tmp_path; no real docs needed.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from python_doc_assistant.ingest.chunker import (
    CHUNK_TYPE_SECTION,
    CHUNK_TYPE_SYMBOL,
    Chunk,
    _compute_source_hash,
    _make_chunk_id,
    _path_and_anchor,
    _slug,
    build_chunks,
)
from python_doc_assistant.ingest.parse_objects_inv import SymbolEntry

# ------------------------------------------------------------------
# Fixture builders
# ------------------------------------------------------------------


PATHLIB_HTML = """<!DOCTYPE html>
<html>
<head><title>pathlib --- Path class</title></head>
<body>
  <section id="module-pathlib">
    <h1>pathlib --- pure paths</h1>
    <p>Module-level intro paragraph.</p>

    <dl class="py class">
      <dt id="pathlib.Path">
        <em class="property">class </em><code>pathlib.Path</code>
      </dt>
      <dd>Path class docstring goes here.</dd>
    </dl>

    <section id="methods">
      <h2>Methods</h2>
      <dl class="py method">
        <dt id="pathlib.Path.read_text">
          <code>Path.read_text(encoding=None)</code>
        </dt>
        <dd>Read file content as text.</dd>
      </dl>
    </section>

    <section id="examples">
      <h2>Examples</h2>
      <p>Some example code without a symbol binding.</p>
    </section>
  </section>
</body>
</html>"""


@pytest.fixture
def fake_docs_dir(tmp_path: Path) -> Path:
    docs = tmp_path / "docs"
    library = docs / "library"
    library.mkdir(parents=True)
    (library / "pathlib.html").write_text(PATHLIB_HTML, encoding="utf-8")
    return docs


@pytest.fixture
def fake_symbols() -> list[SymbolEntry]:
    return [
        SymbolEntry(
            name="pathlib.Path",
            role="py:class",
            uri="library/pathlib.html#pathlib.Path",
            module="pathlib",
        ),
        SymbolEntry(
            name="pathlib.Path.read_text",
            role="py:method",
            uri="library/pathlib.html#pathlib.Path.read_text",
            module="pathlib",
        ),
    ]


# ------------------------------------------------------------------
# _path_and_anchor
# ------------------------------------------------------------------


def test_path_and_anchor_with_anchor() -> None:
    assert _path_and_anchor("library/pathlib.html#pathlib.Path.read_text") == (
        "library/pathlib.html",
        "pathlib.Path.read_text",
    )


def test_path_and_anchor_without_anchor() -> None:
    assert _path_and_anchor("library/pathlib.html") == ("library/pathlib.html", None)


def test_path_and_anchor_empty_anchor() -> None:
    assert _path_and_anchor("library/pathlib.html#") == ("library/pathlib.html", "")


def test_path_and_anchor_keeps_extra_hash_in_anchor() -> None:
    # Splits on the first '#' only.
    assert _path_and_anchor("library/foo.html#a#b") == ("library/foo.html", "a#b")


# ------------------------------------------------------------------
# _compute_source_hash
# ------------------------------------------------------------------


def test_compute_source_hash_known_input() -> None:
    expected = f"sha256:{hashlib.sha256(b'hello').hexdigest()}"
    assert _compute_source_hash(b"hello") == expected


def test_compute_source_hash_deterministic_and_distinct() -> None:
    assert _compute_source_hash(b"x") == _compute_source_hash(b"x")
    assert _compute_source_hash(b"x") != _compute_source_hash(b"y")


# ------------------------------------------------------------------
# _make_chunk_id
# ------------------------------------------------------------------


def test_make_chunk_id_symbol() -> None:
    assert _make_chunk_id("symbol", "pathlib.Path.read_text") == "symbol:pathlib.Path.read_text"


def test_make_chunk_id_section() -> None:
    assert (
        _make_chunk_id("section", "tutorial/datastructures#tut-tuples")
        == "section:tutorial/datastructures#tut-tuples"
    )


# ------------------------------------------------------------------
# _slug
# ------------------------------------------------------------------


def test_slug_lowercase_simple() -> None:
    assert _slug("Examples") == "examples"


def test_slug_replaces_spaces_with_hyphen() -> None:
    assert _slug("Path Methods") == "path-methods"


def test_slug_strips_special_chars() -> None:
    assert _slug("os.path & comparison") == "os-path-comparison"


def test_slug_collapses_consecutive_hyphens() -> None:
    assert _slug("hello---world") == "hello-world"


# ------------------------------------------------------------------
# build_chunks (end-to-end)
# ------------------------------------------------------------------


def test_build_chunks_emits_symbol_chunks_for_known_anchors(
    fake_docs_dir: Path, fake_symbols: list[SymbolEntry]
) -> None:
    chunks = build_chunks(fake_docs_dir, "3.12", fake_symbols)
    sym_chunks = [c for c in chunks if c.chunk_type == CHUNK_TYPE_SYMBOL]
    names = {s for c in sym_chunks for s in c.symbols}
    assert "pathlib.Path" in names
    assert "pathlib.Path.read_text" in names


def test_build_chunks_emits_section_chunk_for_unbound_section(
    fake_docs_dir: Path, fake_symbols: list[SymbolEntry]
) -> None:
    chunks = build_chunks(fake_docs_dir, "3.12", fake_symbols)
    sec_chunks = [c for c in chunks if c.chunk_type == CHUNK_TYPE_SECTION]
    titles = [c.title for c in sec_chunks]
    # The "Examples" <section> has no symbol binding, so it must surface as a section_chunk.
    assert any("Example" in t for t in titles)


def test_build_chunks_chunk_ids_globally_unique(
    fake_docs_dir: Path, fake_symbols: list[SymbolEntry]
) -> None:
    chunks = build_chunks(fake_docs_dir, "3.12", fake_symbols)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), "chunk_id must be globally unique"


def test_build_chunks_all_required_fields_populated(
    fake_docs_dir: Path, fake_symbols: list[SymbolEntry]
) -> None:
    chunks = build_chunks(fake_docs_dir, "3.12", fake_symbols)
    assert chunks
    for c in chunks:
        assert c.chunk_id
        assert c.chunk_type in (CHUNK_TYPE_SYMBOL, CHUNK_TYPE_SECTION)
        assert c.docs_version == "3.12"
        assert c.title
        assert c.text
        assert c.canonical_url
        assert c.source_path == "library/pathlib.html"
        assert c.source_hash.startswith("sha256:")


def test_build_chunks_symbol_chunk_text_includes_docstring(
    fake_docs_dir: Path, fake_symbols: list[SymbolEntry]
) -> None:
    chunks = build_chunks(fake_docs_dir, "3.12", fake_symbols)
    by_id = {c.chunk_id: c for c in chunks}
    target = by_id.get("symbol:pathlib.Path.read_text")
    assert target is not None
    assert "Read file content as text" in target.text


def test_build_chunks_section_chunk_excludes_absorbed_dl_text(
    fake_docs_dir: Path, fake_symbols: list[SymbolEntry]
) -> None:
    """The method's docstring lives in symbol_chunk; it must NOT also appear in any
    section_chunk (otherwise BM25 double-counts and downstream ranking is skewed)."""
    chunks = build_chunks(fake_docs_dir, "3.12", fake_symbols)
    for c in chunks:
        if c.chunk_type == CHUNK_TYPE_SECTION:
            assert "Read file content as text" not in c.text


# ------------------------------------------------------------------
# Chunk dataclass shape
# ------------------------------------------------------------------


def test_chunk_is_frozen() -> None:
    c = Chunk(
        chunk_id="symbol:foo",
        chunk_type="symbol",
        docs_version="3.12",
        title="foo",
        text="body",
        symbols=("foo",),
        canonical_url="library/foo.html#foo",
        anchor="foo",
        parent_module="foo",
        source_path="library/foo.html",
        source_hash="sha256:abc",
    )
    with pytest.raises(Exception):
        c.title = "bar"  # type: ignore[misc]
