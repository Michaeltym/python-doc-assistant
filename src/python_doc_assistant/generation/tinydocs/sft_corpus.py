"""Pure helpers for v3.1 §6 SFT corpus build (`scripts/build_sft_corpus.py`).

The orchestration (Qwen loading, retrieval routing, file IO) lives in the
script. The two pieces extracted here are unit-testable in isolation:

  - `is_sft_rejected`: post-generation quality filter
  - `build_question_generation_prompt`: chat messages asked of Qwen when
    we want it to generate a *user question* about a given chunk (used as
    the synthetic-query side of the SFT corpus)
"""

from __future__ import annotations

from typing import Final

from python_doc_assistant.ingest.chunker import Chunk

# ------------------------------------------------------------------
# Quality-filter constants
# ------------------------------------------------------------------

MIN_ANSWER_CHARS: Final[int] = 30

# Substrings indicating Qwen produced a refusal / unhelpful canned phrase
# even when no explicit refusal marker is present. Treat them as rejections
# so the SFT data does not learn to imitate them.
DEFAULT_BROKEN_PATTERNS: Final[tuple[str, ...]] = (
    "[INSUFFICIENT-CONTEXT]",
    "I am unable",
    "I cannot",
    "I do not have",
)


def is_sft_rejected(
    answer: str,
    refused: bool,
    *,
    min_chars: int = MIN_ANSWER_CHARS,
    broken_patterns: tuple[str, ...] = DEFAULT_BROKEN_PATTERNS,
) -> str | None:
    """Return a rejection reason string, or None if the answer is accepted.

    Reasons (in priority order):
      - "refused": the Generator emitted the refusal marker
      - "empty": answer is empty / whitespace-only
      - "too_short": len(answer) < min_chars
      - "matches:<pattern>": answer contains one of `broken_patterns`

    Used by the SFT corpus build to drop bad Qwen outputs *before* training
    so the student model does not learn to imitate them.
    """
    if refused:
        return "refused"
    stripped_answer = answer.strip()
    if not stripped_answer:
        return "empty"
    if len(stripped_answer) < min_chars:
        return "too_short"
    for p in broken_patterns:
        if p.lower() in answer.lower():
            return f"matches:<{p}>"
    return None


def build_question_generation_prompt(chunk: Chunk) -> list[dict[str, str]]:
    """Build chat messages asking Qwen to generate one user question about `chunk`.

    Returns a 2-message conversation (system + user). The intent is that
    Qwen, when fed these messages and called with greedy decoding, returns
    a single, concrete user question that a developer reading the Python
    docs might actually ask about this chunk.

    Required behaviour:
      - System message sets the role: "You are a Python developer; given a
        docs snippet, generate exactly one realistic user question..."
      - User message includes the chunk's `chunk_id`, `title`, and `text`
        (verbatim, no summarisation)
      - The output question should mention the API name when the chunk
        is a symbol_chunk; for section_chunks it should ask about the
        topic the section covers
      - Return type matches `build_grounded_prompt`'s `[{"role": ..., "content": ...}, ...]`

    Concrete prompt wording is up to the implementer; the test suite asserts
    only on shape (message count, role names, that chunk content appears
    in the user message verbatim).
    """
    system_content = (
        "You generate user questions for a Python documentation Q&A "
        "training set. Given a snippet from the Python docs, output "
        "exactly one realistic question that a Python developer might "
        "ask about it.\n\n"
        "Rules:\n"
        "- Output ONLY the question text — no preamble, numbering, "
        "quotes, or markdown.\n"
        '- One question, one line, ending with "?".\n'
        "- Must be answerable using only the snippet.\n"
        "- Prefer practical questions "
        '("how do I...", "when should I use...", "what does X return") '
        'over vague "what is..." questions.\n'
        "- If the snippet describes a specific API (function, method, "
        "class, module), include its exact name in the question."
    )

    api_line = f"API: {chunk.symbols[0]}\n" if chunk.symbols else ""
    user_content = (
        f"Snippet ID: {chunk.chunk_id}\nTitle: {chunk.title}\n{api_line}\nBody:\n{chunk.text}"
    )
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
