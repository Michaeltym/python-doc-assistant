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
    text = "See [1] for details."
    assert _extract_citations(text) == (1,)


def test_extract_citations_multiple_in_order() -> None:
    text = "Refer to [1] then [2] then [1] again."
    # Preserve order, deduplicate
    assert _extract_citations(text) == (1, 2)


def test_extract_citations_none() -> None:
    assert _extract_citations("plain answer with no citations") == ()


def test_extract_citations_ignores_non_integer_brackets() -> None:
    """Non-integer brackets like [INSUFFICIENT-CONTEXT] or [text] are skipped."""
    assert _extract_citations("[INSUFFICIENT-CONTEXT] and [foo] and [text](url)") == ()


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


def test_format_chunks_includes_numeric_marker_and_title() -> None:
    chunks = [_chunk("symbol:pathlib.Path", title="Path", text="Path body.")]
    block = _format_chunks(chunks)
    assert "[1]" in block
    assert "Path" in block
    assert "Path body." in block


def test_format_chunks_numbers_chunks_one_indexed() -> None:
    """Multiple chunks are numbered [1], [2], [3], ..."""
    chunks = [_chunk("c1", text="first"), _chunk("c2", text="second"), _chunk("c3", text="third")]
    block = _format_chunks(chunks)
    assert "[1]" in block
    assert "[2]" in block
    assert "[3]" in block
    assert block.find("[1]") < block.find("[2]") < block.find("[3]")


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
    """`---` separator between chunks must have newlines on both sides."""
    chunks = [_chunk("c1", text="first"), _chunk("c2", text="second")]
    block = _format_chunks(chunks)
    assert "\n---\n" in block
    assert block.find("first") < block.find("\n---\n") < block.find("second")


def test_format_chunks_single_chunk_has_no_separator() -> None:
    """One chunk → no `---` separator (separator only joins multiple)."""
    block = _format_chunks([_chunk("c1", text="only")])
    assert "---" not in block


def test_format_chunks_does_not_leak_chunk_id() -> None:
    """chunk_id stays out of the prompt — only [N] number is shown."""
    block = _format_chunks([_chunk("symbol:pathlib.Path", title="Path", text="body")])
    assert "symbol:pathlib.Path" not in block


# ------------------------------------------------------------------
# build_grounded_prompt
# ------------------------------------------------------------------


def test_build_grounded_prompt_returns_messages_list() -> None:
    """Returns: system + real user = 2 messages."""
    msgs = build_grounded_prompt("q", [_chunk("c1")])
    assert isinstance(msgs, list)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"


def test_build_grounded_prompt_system_has_grounding_contracts() -> None:
    """System message holds grounding + citation + refusal instructions."""
    msgs = build_grounded_prompt("q", [_chunk("c1")])
    sys_content = msgs[0]["content"]
    assert "documentation chunks provided below" in sys_content  # SYSTEM_GROUNDING
    assert "[N]" in sys_content  # SYSTEM_CITATIONS — new numeric format
    assert REFUSAL_MARKER in sys_content  # SYSTEM_REFUSAL


def test_build_grounded_prompt_user_has_query() -> None:
    msgs = build_grounded_prompt("How to read a file?", [_chunk("c1")])
    assert "How to read a file?" in msgs[1]["content"]


def test_build_grounded_prompt_user_has_chunks_with_numeric_markers() -> None:
    msgs = build_grounded_prompt(
        "q", [_chunk("symbol:foo", text="foo body"), _chunk("symbol:bar", text="bar body")]
    )
    user_content = msgs[1]["content"]
    assert "[1]" in user_content
    assert "[2]" in user_content
    assert "foo body" in user_content
    assert "bar body" in user_content


def test_build_grounded_prompt_query_type_hint_in_user_not_system() -> None:
    """Structure hint goes into user message (not system)."""
    msgs = build_grounded_prompt("q", [_chunk("c1")], query_type=QueryType.HOWTO)
    hint = QUERY_TYPE_STRUCTURE[QueryType.HOWTO]
    assert hint in msgs[1]["content"]
    assert hint not in msgs[0]["content"]


def test_build_grounded_prompt_uses_query_type_structure() -> None:
    chunks = [_chunk("c1")]
    p_id = build_grounded_prompt("q", chunks, query_type=QueryType.IDENTIFIER)
    p_how = build_grounded_prompt("q", chunks, query_type=QueryType.HOWTO)
    assert p_id != p_how
    assert QUERY_TYPE_STRUCTURE[QueryType.IDENTIFIER] in p_id[1]["content"]
    assert QUERY_TYPE_STRUCTURE[QueryType.HOWTO] in p_how[1]["content"]


def test_build_grounded_prompt_omits_structure_when_query_type_none() -> None:
    """query_type=None → no structure hint anywhere in the messages."""
    msgs = build_grounded_prompt("q", [_chunk("c1")], query_type=None)
    joined = msgs[0]["content"] + msgs[1]["content"]
    for hint in QUERY_TYPE_STRUCTURE.values():
        assert hint not in joined


def test_build_grounded_prompt_no_chunks_still_well_formed() -> None:
    """Empty retrieval still produces a valid 2-message conversation."""
    msgs = build_grounded_prompt("q", [])
    assert len(msgs) == 2
    assert "QUESTION: q" in msgs[1]["content"]
    # CONTEXT block is omitted when no chunks
    assert "CONTEXT:" not in msgs[1]["content"]


# ------------------------------------------------------------------
# parse_response
# ------------------------------------------------------------------


def test_parse_response_plain_answer() -> None:
    pr = parse_response("This is a plain answer.")
    assert pr == ParsedResponse(
        text="This is a plain answer.",
        cited_indices=(),
        refused=False,
    )


def test_parse_response_with_citations() -> None:
    pr = parse_response("Per [1] and [2], the answer is foo.")
    assert pr.cited_indices == (1, 2)
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

    Plan §2 contract: `cited_indices` is an *extraction* — the [N] markers
    themselves remain in `text` so a downstream renderer can highlight them.
    """
    pr = parse_response("Per [1] and [2], the answer is foo.")
    assert "[1]" in pr.text
    assert "[2]" in pr.text
    assert pr.cited_indices == (1, 2)
