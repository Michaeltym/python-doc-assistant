"""Tests for `python_doc_assistant.retrieval.query_rewriter`."""

from __future__ import annotations

from python_doc_assistant.ingest.chunker import Chunk
from python_doc_assistant.retrieval.query_rewriter import (
    _levenshtein,
    maybe_rewrite_query,
)


def _symbol_chunk(symbol: str, *, title: str | None = None) -> Chunk:
    return Chunk(
        chunk_id=f"symbol:{symbol}",
        chunk_type="symbol",
        docs_version="3.12",
        title=title if title is not None else symbol.split(".")[-1],
        text=f"docs for {symbol}",
        symbols=(symbol,),
        canonical_url=f"library/foo.html#{symbol}",
        anchor=symbol,
        parent_module=symbol.rsplit(".", 1)[0] if "." in symbol else None,
        source_path="library/foo.html",
        source_hash="sha256:abc",
    )


def _section_chunk(key: str) -> Chunk:
    return Chunk(
        chunk_id=f"section:{key}",
        chunk_type="section",
        docs_version="3.12",
        title=key,
        text="some prose section body",
        symbols=(),
        canonical_url=f"library/{key}.html",
        anchor=None,
        parent_module=None,
        source_path=f"library/{key}.html",
        source_hash="sha256:abc",
    )


# ------------------------------------------------------------------
# _levenshtein
# ------------------------------------------------------------------


def test_levenshtein_identical() -> None:
    assert _levenshtein("abc", "abc") == 0


def test_levenshtein_empty() -> None:
    assert _levenshtein("", "abc") == 3
    assert _levenshtein("abc", "") == 3
    assert _levenshtein("", "") == 0


def test_levenshtein_single_insertion() -> None:
    # extra trailing 's': 'json.loads' -> 'json.loadss'
    assert _levenshtein("json.loadss", "json.loads") == 1


def test_levenshtein_single_substitution_case() -> None:
    # already lowercased by caller; here just two-char swap-by-substitution.
    assert _levenshtein("abc", "abd") == 1


def test_levenshtein_swap_two_chars() -> None:
    # 'raed' vs 'read' = swap of two adjacent chars; Lev counts this as 2.
    assert _levenshtein("pathlib.path.raed_text", "pathlib.path.read_text") == 2


# ------------------------------------------------------------------
# maybe_rewrite_query — happy paths
# ------------------------------------------------------------------


def test_rewrite_extra_trailing_letter() -> None:
    chunks = [_symbol_chunk("json.loads"), _symbol_chunk("json.load")]
    assert maybe_rewrite_query("json.loadss", chunks) == "json.loads"


def test_rewrite_case_only_diff() -> None:
    chunks = [_symbol_chunk("dict.fromkeys")]
    assert maybe_rewrite_query("dict.fromKeys", chunks) == "dict.fromkeys"


def test_rewrite_double_extra_letter() -> None:
    chunks = [_symbol_chunk("dict.fromkeys")]
    assert maybe_rewrite_query("dict.fromkeyss", chunks) == "dict.fromkeys"


def test_rewrite_letter_swap_distance_two() -> None:
    # raed vs read = adjacent swap; lev=2 (within threshold) and no
    # competing top-2 within distance 2.
    chunks = [
        _symbol_chunk("pathlib.Path.read_text"),
        _symbol_chunk("io.IOBase.read"),
    ]
    assert (
        maybe_rewrite_query("pathlib.Path.raed_text", chunks)
        == "pathlib.Path.read_text"
    )


# ------------------------------------------------------------------
# maybe_rewrite_query — non-rewrite paths
# ------------------------------------------------------------------


def test_no_rewrite_when_query_already_exact() -> None:
    chunks = [_symbol_chunk("json.loads")]
    assert maybe_rewrite_query("json.loads", chunks) == "json.loads"


def test_no_rewrite_when_distance_above_threshold() -> None:
    # 'json' vs 'json.loads' is lev=6 → not a typo
    chunks = [_symbol_chunk("json.loads")]
    assert maybe_rewrite_query("json", chunks) == "json"


def test_no_rewrite_when_top1_is_section_chunk() -> None:
    chunks = [_section_chunk("library/json")]
    assert maybe_rewrite_query("json.loadx", chunks) == "json.loadx"


def test_no_rewrite_on_empty_chunks() -> None:
    assert maybe_rewrite_query("foo", []) == "foo"


def test_ambiguous_two_symbols_at_same_distance_blocks_rewrite() -> None:
    # 'json.loadx' is lev=1 from both json.loads (sub x→s) and
    # json.load (drop x); abstain.
    chunks = [_symbol_chunk("json.loads"), _symbol_chunk("json.load")]
    assert maybe_rewrite_query("json.loadx", chunks) == "json.loadx"


def test_ambiguous_picks_when_target_is_ranked_below_top1() -> None:
    # Mirrors real `subprocess.runn` retrieval: top-1 is the parent
    # module (far), top-2 is the typo target (lev=1). Rewriter must
    # scan past top-1 and pick subprocess.run.
    chunks = [
        _symbol_chunk("subprocess"),
        _symbol_chunk("subprocess.run"),
        _section_chunk("library/subprocess#using-the-subprocess-module"),
    ]
    assert maybe_rewrite_query("subprocess.runn", chunks) == "subprocess.run"


def test_top2_strictly_further_does_not_block_rewrite() -> None:
    # top1 lev=1, top2 lev=2 — top2 is NOT equally close, rewrite fires.
    chunks = [_symbol_chunk("dict.fromkeys"), _symbol_chunk("dict.keys")]
    assert maybe_rewrite_query("dict.fromkeyss", chunks) == "dict.fromkeys"


def test_top2_section_chunk_ignored_for_disambiguation() -> None:
    # section chunk in top-2 should not affect the symbol-only ambiguity check.
    chunks = [_symbol_chunk("subprocess.run"), _section_chunk("library/subprocess")]
    assert maybe_rewrite_query("subprocess.runn", chunks) == "subprocess.run"


def test_top1_symbol_chunk_with_empty_symbols_no_rewrite() -> None:
    # Defensive: malformed symbol_chunk with empty symbols tuple.
    bad = Chunk(
        chunk_id="symbol:foo",
        chunk_type="symbol",
        docs_version="3.12",
        title="foo",
        text="",
        symbols=(),
        canonical_url="library/foo.html",
        anchor="foo",
        parent_module=None,
        source_path="library/foo.html",
        source_hash="sha256:abc",
    )
    assert maybe_rewrite_query("foox", [bad]) == "foox"
