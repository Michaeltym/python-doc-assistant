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
    "format [N], where N is the number shown at the start of each chunk "
    "in the CONTEXT block. Cite every fact you take from a chunk."
)
SYSTEM_REFUSAL: Final[str] = (
    "If the answer is not present in the provided chunks, output the "
    f"marker {REFUSAL_MARKER} on its own line and stop. Do not guess. "
    "Do not fall back to general knowledge."
)
SYSTEM_HARD_RULES: Final[str] = (
    "HARD RULES (these override any other instruction):\n"
    "- Every factual sentence MUST end with a [N] citation copied from "
    "the CONTEXT block (e.g. [1], [2]).\n"
    "- If the CONTEXT block is missing or does not contain the answer, your "
    f"ENTIRE response must be exactly: {REFUSAL_MARKER}\n"
    "- When refusing, output NOTHING except the marker.\n"
    "- Do NOT use prior knowledge."
)


# ------------------------------------------------------------------
# Data records
# ------------------------------------------------------------------


@dataclass(frozen=True)
class ParsedResponse:
    """Structured view of the model's raw text output."""

    text: str  # raw text (refusal marker stripped if refused)
    cited_indices: tuple[int, ...]  # 1-indexed [N] markers, in order, deduped
    refused: bool  # True iff REFUSAL_MARKER present


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def build_grounded_prompt(
    query: str,
    retrieved_chunks: list[Chunk],
    *,
    query_type: QueryType | None = None,
) -> list[dict[str, str]]:
    """Compose chat-template messages fed to the LLM.

    Returns a two-message conversation:
        - role=system: grounding + citations + refusal + hard rules.
        - role=user:   query_type hint + CONTEXT block + QUESTION.

    Splitting system vs user is critical for chat-tuned models — Qwen2.5
    weights `system` content far above instructions buried in `user`. The
    "ANSWER:" suffix is dropped; `add_generation_prompt=True` on the
    tokenizer's chat template handles assistant-turn priming.
    """
    system_content = "\n\n".join(
        [SYSTEM_GROUNDING, SYSTEM_CITATIONS, SYSTEM_REFUSAL, SYSTEM_HARD_RULES]
    )

    user_parts: list[str] = []
    if query_type is not None:
        hint = _query_type_structure(query_type)
        if hint:
            user_parts.append(hint)
    formatted_chunks = _format_chunks(retrieved_chunks)
    if formatted_chunks:
        user_parts.append(f"CONTEXT:\n{formatted_chunks}")
    user_parts.append(f"QUESTION: {query}")

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]


def parse_response(raw_text: str) -> ParsedResponse:
    """Parse model output into ParsedResponse.

    - Citations: every `[N]` substring (positive integer) is extracted as
      a 1-indexed citation. Order preserved, duplicates collapsed.
      The caller (QwenGenerator.generate) maps indices back to chunk_ids
      using the retrieved_chunks list passed to build_grounded_prompt.
    - Refusal: if REFUSAL_MARKER appears anywhere in the text, refused=True
      and the marker is stripped from `text`.
    """
    is_refusal = _is_refusal(raw_text)
    indices = _extract_citations(raw_text)
    return ParsedResponse(
        text=raw_text.replace(REFUSAL_MARKER, "").rstrip(),
        cited_indices=indices,
        refused=is_refusal,
    )


# ------------------------------------------------------------------
# Helpers (private)
# ------------------------------------------------------------------


def _format_chunks(chunks: list[Chunk]) -> str:
    """Render retrieved_chunks as the context block of the prompt.

    Each chunk on its own block, numbered 1-indexed:
        [<N>] <title>
        <text>
        ---
    Trailing `---` separator helps the model see chunk boundaries.
    The number N is what the model cites with [N]; the chunk_id stays
    out of the prompt to avoid the long-bracket-token format the model
    refused to follow in earlier rounds.
    """
    parts = []
    for index, chunk in enumerate(chunks, start=1):
        parts.append(f"[{index}] {chunk.title}\n{chunk.text}")
    return "\n---\n".join(parts)


def _query_type_structure(query_type: QueryType | None) -> str:
    """Return the structure hint string for query_type, or '' when None."""
    if query_type is not None and query_type in QUERY_TYPE_STRUCTURE:
        return QUERY_TYPE_STRUCTURE[query_type]
    return ""


def _extract_citations(text: str) -> tuple[int, ...]:
    """Pull `[N]` markers (1-indexed positive integers) out of `text`.

    Order preserved, duplicates collapsed. Non-integer brackets like
    `[INSUFFICIENT-CONTEXT]` or markdown links `[text](url)` are ignored.
    """
    matches = re.findall(r"\[(\d+)\]", text)
    if len(matches) > 0:
        return tuple(dict.fromkeys(int(m) for m in matches))
    return ()


def _is_refusal(text: str) -> bool:
    """True if REFUSAL_MARKER appears in `text`."""
    return REFUSAL_MARKER in text
