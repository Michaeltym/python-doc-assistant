"""Tests for python_doc_assistant.prompts.grounded.

Hermetic — pure string manipulation; no LLM calls.
"""

from __future__ import annotations

import pytest

from python_doc_assistant.ingest.chunker import Chunk
from python_doc_assistant.prompts.grounded import (
    QUERY_TYPE_STRUCTURE,
    REFUSAL_MARKER,
    ParsedResponse,
    _extract_citations,
    _format_chunks,
    _is_refusal,
    _query_type_structure,
    build_grounded_prompt,
    parse_response,
)
from python_doc_assistant.retrieval.router import QueryType

# ------------------------------------------------------------------
# Fixture builders
# ------------------------------------------------------------------


def _chunk(chunk_id: str, title: str = "T", text: str = "BODY") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        chunk_type="symbol",
        docs_version="3.12",
        title=title,
        text=text,
        symbols=(chunk_id.split(":", 1)[-1],),
        canonical_url=f"library/foo.html#{chunk_id}",
        anchor=chunk_id,
        parent_module=None,
        source_path="library/foo.html",
        source_hash="sha256:abc",
    )


# ------------------------------------------------------------------
# _extract_citations
# ------------------------------------------------------------------


def test_extract_citations_basic() -> None:
    text = "See [#symbol:pathlib.Path.read_text] for details."
    assert _extract_citations(text) == ("symbol:pathlib.Path.read_text",)


def test_extract_citations_multiple_in_order() -> None:
    text = "Refer to [#a] then [#b] then [#a] again."
    # Preserve order, deduplicate
    assert _extract_citations(text) == ("a", "b")


def test_extract_citations_none() -> None:
    assert _extract_citations("plain answer with no citations") == ()


def test_extract_citations_ignores_malformed() -> None:
    """`[chunk_id]` without `#`, or unclosed brackets, are not citations."""
    assert _extract_citations("[no-hash] and [#") == ()


# ------------------------------------------------------------------
# _is_refusal
# ------------------------------------------------------------------


def test_is_refusal_marker_present() -> None:
    assert _is_refusal(f"Some text\n{REFUSAL_MARKER}") is True


def test_is_refusal_marker_absent() -> None:
    assert _is_refusal("A normal answer.") is False


# ------------------------------------------------------------------
# _query_type_structure
# ------------------------------------------------------------------


def test_query_type_structure_none_returns_empty() -> None:
    assert _query_type_structure(None) == ""


def test_query_type_structure_known_types() -> None:
    """Every QueryType in QUERY_TYPE_STRUCTURE produces a non-empty hint."""
    for qt in QUERY_TYPE_STRUCTURE:
        assert _query_type_structure(qt)  # non-empty string


# ------------------------------------------------------------------
# _format_chunks
# ------------------------------------------------------------------


def test_format_chunks_includes_chunk_id_marker() -> None:
    chunks = [_chunk("symbol:pathlib.Path", title="Path", text="Path body.")]
    block = _format_chunks(chunks)
    assert "[#symbol:pathlib.Path]" in block
    assert "Path" in block
    assert "Path body." in block


def test_format_chunks_separates_multiple() -> None:
    chunks = [_chunk("c1", text="first"), _chunk("c2", text="second")]
    block = _format_chunks(chunks)
    assert "first" in block
    assert "second" in block
    assert block.find("first") < block.find("second")


def test_format_chunks_empty_input() -> None:
    """No chunks → empty (or near-empty) string; caller handles 0-context case."""
    assert _format_chunks([]) == ""


def test_format_chunks_inserts_separator_between() -> None:
    """`---` separator between chunks must have newlines on both sides.

    Catches a bug where `---` glues directly to next chunk's `[#cid]` marker.
    """
    chunks = [_chunk("c1", text="first"), _chunk("c2", text="second")]
    block = _format_chunks(chunks)
    assert "\n---\n" in block
    # Separator must sit between the two chunk bodies, not inside one
    assert block.find("first") < block.find("\n---\n") < block.find("second")


def test_format_chunks_single_chunk_has_no_separator() -> None:
    """One chunk → no `---` separator (separator only joins multiple)."""
    block = _format_chunks([_chunk("c1", text="only")])
    assert "---" not in block


# ------------------------------------------------------------------
# build_grounded_prompt
# ------------------------------------------------------------------


def test_build_grounded_prompt_contains_query() -> None:
    chunks = [_chunk("c1")]
    prompt = build_grounded_prompt("How to read a file?", chunks)
    assert "How to read a file?" in prompt


def test_build_grounded_prompt_contains_chunks_with_markers() -> None:
    chunks = [_chunk("symbol:foo", text="foo body")]
    prompt = build_grounded_prompt("q", chunks)
    assert "[#symbol:foo]" in prompt
    assert "foo body" in prompt


def test_build_grounded_prompt_contains_citation_instruction() -> None:
    """Prompt must instruct the model to cite with [#chunk_id]."""
    prompt = build_grounded_prompt("q", [_chunk("c1")])
    assert "[#" in prompt  # citation format mentioned somewhere


def test_build_grounded_prompt_contains_refusal_marker_instruction() -> None:
    """Prompt must teach the model the refusal marker."""
    prompt = build_grounded_prompt("q", [_chunk("c1")])
    assert REFUSAL_MARKER in prompt


def test_build_grounded_prompt_uses_query_type_structure() -> None:
    chunks = [_chunk("c1")]
    p_id = build_grounded_prompt("q", chunks, query_type=QueryType.IDENTIFIER)
    p_how = build_grounded_prompt("q", chunks, query_type=QueryType.HOWTO)
    # Different query_types produce different structure hints
    assert p_id != p_how
    assert QUERY_TYPE_STRUCTURE[QueryType.IDENTIFIER] in p_id
    assert QUERY_TYPE_STRUCTURE[QueryType.HOWTO] in p_how


def test_build_grounded_prompt_omits_structure_when_query_type_none() -> None:
    """query_type=None still produces a valid prompt without a structure hint."""
    chunks = [_chunk("c1")]
    p_none = build_grounded_prompt("q", chunks, query_type=None)
    # All structure hints should be absent
    for hint in QUERY_TYPE_STRUCTURE.values():
        assert hint not in p_none


def test_build_grounded_prompt_separates_sections_with_blank_lines() -> None:
    """Sections (system / structure / context / question / answer) must be
    visually separated by blank lines so the chat template renders cleanly.

    Catches a bug where the structure hint sits flush against the system
    block (single `\\n` instead of blank line).
    """
    prompt = build_grounded_prompt(
        "q", [_chunk("c1")], query_type=QueryType.IDENTIFIER
    )
    # At least: SYSTEM_GROUNDING / SYSTEM_CITATIONS / SYSTEM_REFUSAL +
    # structure hint + CONTEXT + QUESTION + ANSWER → expect 4+ blank lines
    assert prompt.count("\n\n") >= 4
    # Structure hint is NOT immediately glued after SYSTEM_REFUSAL
    structure_hint = QUERY_TYPE_STRUCTURE[QueryType.IDENTIFIER]
    assert f"\n\n{structure_hint}" in prompt


def test_build_grounded_prompt_no_chunks_still_well_formed() -> None:
    """Empty retrieval must still produce a parseable prompt (no CONTEXT block)."""
    prompt = build_grounded_prompt("q", [])
    assert "QUESTION: q" in prompt
    assert "ANSWER:" in prompt


# ------------------------------------------------------------------
# parse_response
# ------------------------------------------------------------------


def test_parse_response_plain_answer() -> None:
    pr = parse_response("This is a plain answer.")
    assert pr == ParsedResponse(
        text="This is a plain answer.",
        cited_chunk_ids=(),
        refused=False,
    )


def test_parse_response_with_citations() -> None:
    pr = parse_response("Per [#a] and [#b], the answer is foo.")
    assert pr.cited_chunk_ids == ("a", "b")
    assert pr.refused is False
    assert "Per" in pr.text


def test_parse_response_refusal() -> None:
    pr = parse_response(f"{REFUSAL_MARKER}")
    assert pr.refused is True
    # text is cleaned: marker stripped
    assert REFUSAL_MARKER not in pr.text


def test_parse_response_refusal_with_explanation() -> None:
    """Models often add a brief reason next to the marker; both should parse."""
    pr = parse_response(f"This is not covered by the docs.\n{REFUSAL_MARKER}")
    assert pr.refused is True
    assert REFUSAL_MARKER not in pr.text
    assert "not covered" in pr.text


def test_parse_response_returns_dataclass() -> None:
    """Sanity: result is the documented frozen dataclass."""
    pr = parse_response("hi")
    assert isinstance(pr, ParsedResponse)
    with pytest.raises(Exception):
        pr.text = "x"  # type: ignore[misc]


def test_parse_response_preserves_citation_markers_in_text() -> None:
    """Citations stay inline in the answer text (not stripped).

    Plan §2 contract: `cited_chunk_ids` is an *extraction* — the markers
    themselves remain in `text` so a downstream renderer can highlight them.
    """
    pr = parse_response("Per [#a] and [#b], the answer is foo.")
    assert "[#a]" in pr.text
    assert "[#b]" in pr.text
    assert pr.cited_chunk_ids == ("a", "b")
