"""Grounded prompt template + response parser for v1 generator.

See plans/v1-qwen-generator.md §2.

Common contract (every prompt enforces these):
    1. Use only the provided chunks; do NOT invent facts.
    2. Cite chunk_ids with `[#chunk_id]` markers inline in the answer.
    3. If the answer is not in the provided chunks, output the refusal
       marker on its own line and stop.

Per-query_type answer structure (plan §2):
    identifier        signature -> brief description -> example -> source
    natural_language  definition -> context -> example -> source
    comparison        bullets per side -> differences -> recommended scenarios -> source
    howto             numbered steps -> code example -> source
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

from python_doc_assistant.ingest.chunker import Chunk
from python_doc_assistant.retrieval.router import QueryType

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

REFUSAL_MARKER: Final[str] = "[INSUFFICIENT-CONTEXT]"

# Structure hint per query_type. None → omit the structure clause entirely
# (caller did not classify the query).
QUERY_TYPE_STRUCTURE: Final[dict[QueryType, str]] = {
    QueryType.IDENTIFIER: (
        "Use this answer structure: signature -> brief description -> example -> source citation."
    ),
    QueryType.NATURAL_LANGUAGE: (
        "Use this answer structure: definition -> context -> example -> source citation."
    ),
    QueryType.COMPARISON: (
        "Use this answer structure: bullet points for each side -> key "
        "differences -> recommended scenarios -> source citations."
    ),
    QueryType.HOWTO: (
        "Use this answer structure: numbered steps -> code example -> source citation."
    ),
    QueryType.OUT_OF_SCOPE: ("If the question is out of scope for these documents, refuse."),
}

SYSTEM_GROUNDING: Final[str] = (
    "You are a Python documentation assistant. Answer the user's question "
    "using ONLY the documentation chunks provided below. Do not invent "
    "facts that are not present in the chunks."
)
SYSTEM_CITATIONS: Final[str] = (
    "When you reference a chunk in your answer, cite it inline using the "
    "format [#chunk_id], where chunk_id is the marker shown at the top of "
    "each chunk. Cite every fact you take from a chunk."
)
SYSTEM_REFUSAL: Final[str] = (
    "If the answer is not present in the provided chunks, output the "
    f"marker {REFUSAL_MARKER} on its own line and stop. Do not guess. "
    "Do not fall back to general knowledge."
)


# ------------------------------------------------------------------
# Data records
# ------------------------------------------------------------------


@dataclass(frozen=True)
class ParsedResponse:
    """Structured view of the model's raw text output."""

    text: str  # raw text (refusal marker stripped if refused)
    cited_chunk_ids: tuple[str, ...]  # [#chunk_id] markers extracted, in order
    refused: bool  # True iff REFUSAL_MARKER present


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def build_grounded_prompt(
    query: str,
    retrieved_chunks: list[Chunk],
    *,
    query_type: QueryType | None = None,
) -> str:
    """Compose the full prompt fed to the LLM.

    Layout (single string passed to chat template by caller):

        <system instructions: grounding + citations + refusal>

        <structure hint for query_type, if any>

        <retrieved chunks formatted as: [#chunk_id] title\\n text\\n>

        QUESTION: <query>

        ANSWER:
    """
    prompt = "\n\n".join(
        [
            SYSTEM_GROUNDING,
            SYSTEM_CITATIONS,
            SYSTEM_REFUSAL,
        ]
    )
    if query_type is not None:
        query_type_hint = _query_type_structure(query_type)
        prompt += f"\n\n{query_type_hint}"
    formatted_chunks = _format_chunks(retrieved_chunks)
    if len(formatted_chunks) > 0:
        prompt += "\n\nCONTEXT:\n" + formatted_chunks
    prompt += f"\n\nQUESTION: {query}"
    prompt += "\n\nANSWER:"
    return prompt


def parse_response(raw_text: str) -> ParsedResponse:
    """Parse model output into ParsedResponse.

    - Citations: every `[#chunk_id]` substring is an extracted citation
      (preserve order; duplicates collapsed in `cited_chunk_ids`).
    - Refusal: if REFUSAL_MARKER appears anywhere in the text, refused=True
      and the marker is stripped from `text`.
    """
    is_refusal = _is_refusal(raw_text)
    citations = _extract_citations(raw_text)
    return ParsedResponse(
        text=raw_text.replace(REFUSAL_MARKER, "").rstrip(),
        cited_chunk_ids=citations,
        refused=is_refusal,
    )


# ------------------------------------------------------------------
# Helpers (private)
# ------------------------------------------------------------------


def _format_chunks(chunks: list[Chunk]) -> str:
    """Render retrieved_chunks as the context block of the prompt.

    Each chunk on its own block:
        [#<chunk_id>] <title>
        <text>
        ---
    Trailing `---` separator helps the model see chunk boundaries.
    """
    parts = []
    for chunk in chunks:
        parts.append(f"[#{chunk.chunk_id}] {chunk.title}\n{chunk.text}")
    return "\n---\n".join(parts)


def _query_type_structure(query_type: QueryType | None) -> str:
    """Return the structure hint string for query_type, or '' when None."""
    if query_type is not None and query_type in QUERY_TYPE_STRUCTURE:
        return QUERY_TYPE_STRUCTURE[query_type]
    return ""


def _extract_citations(text: str) -> tuple[str, ...]:
    """Pull `[#chunk_id]` markers out of `text` in order, deduplicated."""
    matches = re.findall(r"\[#([^\]]+)\]", text)
    if len(matches) > 0:
        return tuple(dict.fromkeys(matches))
    return ()


def _is_refusal(text: str) -> bool:
    """True if REFUSAL_MARKER appears in `text`."""
    return REFUSAL_MARKER in text
