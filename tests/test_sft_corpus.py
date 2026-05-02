"""Tests for `python_doc_assistant.generation.tinydocs.sft_corpus`.

Pure-helper tests only; the orchestration in `scripts/build_sft_corpus.py`
is covered by an end-to-end smoke run (CLI `--smoke` flag), not by tests.
"""

from __future__ import annotations

from python_doc_assistant.generation.tinydocs.sft_corpus import (
    DEFAULT_BROKEN_PATTERNS,
    MIN_ANSWER_CHARS,
    build_question_generation_prompt,
    is_sft_rejected,
)
from python_doc_assistant.ingest.chunker import Chunk

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


def _symbol_chunk(chunk_id: str = "symbol:pathlib.Path.read_text") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        chunk_type="symbol",
        docs_version="3.12",
        title="Path.read_text",
        text="Open the file pointed to in text mode and return its decoded contents.",
        symbols=("pathlib.Path.read_text",),
        canonical_url="library/pathlib.html#pathlib.Path.read_text",
        anchor="pathlib.Path.read_text",
        parent_module="pathlib",
        source_path="library/pathlib.html",
        source_hash="sha256:deadbeef",
    )


def _section_chunk(chunk_id: str = "section:tutorial/datastructures#more-on-lists") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        chunk_type="section",
        docs_version="3.12",
        title="More on Lists",
        text="The list data type has some more methods. Here are all the list methods.",
        symbols=(),
        canonical_url="tutorial/datastructures.html#more-on-lists",
        anchor="more-on-lists",
        parent_module=None,
        source_path="tutorial/datastructures.html",
        source_hash="sha256:cafef00d",
    )


# ------------------------------------------------------------------
# is_sft_rejected
# ------------------------------------------------------------------


def test_accepts_normal_answer() -> None:
    long_text = "This is a perfectly normal answer with citation [1] over the minimum length."
    assert is_sft_rejected(long_text, refused=False) is None


def test_rejects_no_citation_when_required() -> None:
    """v3.1 §6 third pass: answers without `[N]` markers are dropped by default."""
    text = "This is a long enough answer but contains no citation marker at all."
    assert is_sft_rejected(text, refused=False) == "no_citation"


def test_no_citation_filter_can_be_disabled() -> None:
    """`require_citation=False` keeps the legacy behaviour."""
    text = "This is a long enough answer but contains no citation marker at all."
    assert is_sft_rejected(text, refused=False, require_citation=False) is None


def test_rejects_refused() -> None:
    assert is_sft_rejected("anything here even long enough text content", refused=True) == "refused"


def test_rejects_empty() -> None:
    assert is_sft_rejected("", refused=False) == "empty"


def test_rejects_whitespace_only() -> None:
    assert is_sft_rejected("   \n\t  ", refused=False) == "empty"


def test_rejects_too_short() -> None:
    short = "tiny"
    assert len(short) < MIN_ANSWER_CHARS
    assert is_sft_rejected(short, refused=False) == "too_short"


def test_rejects_broken_pattern_insufficient_context() -> None:
    text = "Here is some valid prose followed by [INSUFFICIENT-CONTEXT] which signals refusal"
    reason = is_sft_rejected(text, refused=False)
    assert reason is not None
    assert reason.startswith("matches:")
    assert "[INSUFFICIENT-CONTEXT]" in reason


def test_rejects_broken_pattern_i_cannot() -> None:
    text = "Sorry, I cannot answer this question without more information about the API."
    reason = is_sft_rejected(text, refused=False)
    assert reason is not None
    assert reason.startswith("matches:")


def test_refused_takes_priority_over_short() -> None:
    """If `refused=True`, that is the reason regardless of other defects."""
    assert is_sft_rejected("", refused=True) == "refused"


def test_custom_min_chars_threshold() -> None:
    """Caller-provided `min_chars` overrides default."""
    text = "exactly thirty-two characters[1]!"  # 33 chars including citation
    # Using stricter threshold rejects this
    assert is_sft_rejected(text, refused=False, min_chars=64) == "too_short"
    # Default accepts it
    assert is_sft_rejected(text, refused=False, min_chars=20) is None


def test_custom_broken_patterns_used() -> None:
    """Caller-provided `broken_patterns` overrides default list."""
    text = "This is a long enough answer with the FORBIDDEN-MARKER inside it."
    reason = is_sft_rejected(
        text,
        refused=False,
        broken_patterns=("FORBIDDEN-MARKER",),
    )
    assert reason is not None
    assert "FORBIDDEN-MARKER" in reason
    # The default patterns are *not* applied when caller overrides.
    # `[INSUFFICIENT-CONTEXT]` doubles as the `[N]` citation regex match
    # (it's `[<digits>]` shape — wait, no: contains "INSUFFICIENT-CONTEXT").
    # Need an explicit `[1]` to satisfy the citation requirement.
    assert (
        is_sft_rejected(
            "answer with [INSUFFICIENT-CONTEXT] [1] but caller passed empty broken_patterns",
            refused=False,
            broken_patterns=(),
        )
        is None
    )


# ------------------------------------------------------------------
# build_question_generation_prompt
# ------------------------------------------------------------------


def test_question_prompt_returns_two_messages() -> None:
    """Standard chat-template format: one system, one user."""
    chunk = _symbol_chunk()
    messages = build_question_generation_prompt(chunk)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


def test_question_prompt_includes_chunk_text_verbatim() -> None:
    """User message includes the chunk body so Qwen can see what to ask about."""
    chunk = _symbol_chunk()
    messages = build_question_generation_prompt(chunk)
    user_content = messages[1]["content"]
    # Verbatim text presence (not a summary or paraphrase)
    assert chunk.text in user_content


def test_question_prompt_includes_symbol_chunk_identifier() -> None:
    """Symbol chunks pass the API name (chunk_id or symbols) so the question can mention it."""
    chunk = _symbol_chunk()
    messages = build_question_generation_prompt(chunk)
    user_content = messages[1]["content"]
    assert (
        "pathlib.Path.read_text" in user_content
        or "Path.read_text" in user_content
        or chunk.chunk_id in user_content
    )


def test_question_prompt_works_for_section_chunk() -> None:
    """Section chunks have empty `symbols`; prompt must still build (use title / chunk_id)."""
    chunk = _section_chunk()
    messages = build_question_generation_prompt(chunk)
    assert len(messages) == 2
    user_content = messages[1]["content"]
    # Title or chunk_id should anchor the topic for Qwen
    assert chunk.title in user_content or chunk.chunk_id in user_content


def test_question_prompt_messages_are_strings() -> None:
    """Both `role` and `content` must be plain strings (chat-template convention)."""
    chunk = _symbol_chunk()
    messages = build_question_generation_prompt(chunk)
    for m in messages:
        assert isinstance(m["role"], str)
        assert isinstance(m["content"], str)
        assert m["content"].strip()  # non-empty


# ------------------------------------------------------------------
# Module constants
# ------------------------------------------------------------------


def test_default_constants() -> None:
    assert MIN_ANSWER_CHARS == 30
    assert "[INSUFFICIENT-CONTEXT]" in DEFAULT_BROKEN_PATTERNS
