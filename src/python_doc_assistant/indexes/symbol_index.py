"""Symbol-level index for exact + fuzzy lookup with multi-candidate support.

Designed for short-name collisions: e.g. `open` exists as builtin, `io.open`,
`os.open`, `codecs.open`. The index keeps every match and lets the caller
disambiguate.

See plans/v0-retrieval-eval.md §4 for the full contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from rapidfuzz import fuzz

from python_doc_assistant.ingest.chunker import Chunk
from python_doc_assistant.ingest.parse_objects_inv import SymbolEntry

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

DEFAULT_FUZZY_THRESHOLD: Final[int] = 85  # rapidfuzz score (0-100)

# Role priority for sorting multi-candidate exact results.
# Lower number = higher priority (surface first).
ROLE_PRIORITY: Final[dict[str, int]] = {
    "py:class": 0,
    "py:exception": 0,
    "py:module": 0,
    "py:method": 1,
    "py:classmethod": 1,
    "py:staticmethod": 1,
    "py:function": 2,
    "py:attribute": 3,
    "py:data": 4,
}
UNKNOWN_ROLE_PRIORITY: Final[int] = 999


# ------------------------------------------------------------------
# Data records
# ------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """One symbol_chunk match for a lookup query."""

    chunk_id: str  # "symbol:pathlib.Path.read_text"
    fully_qualified_name: str  # "pathlib.Path.read_text"
    role: str  # "py:method"
    parent_module: str | None  # "pathlib"


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


class SymbolIndex:
    """In-memory symbol → list[Candidate] index supporting exact + fuzzy lookup.

    Build once from a list[Chunk] (only chunk_type == "symbol" entries are indexed).
    Each symbol is registered under every short form (last 1, last 2 segments, full)
    so queries like `Path.read_text` and `read_text` both resolve.
    """

    def __init__(self, chunks: list[Chunk], symbols: list[SymbolEntry]) -> None:
        """Build the index from symbol_chunks. Other chunk types are ignored."""
        self.candidates: list[Candidate] = []
        for chunk in chunks:
            if chunk.chunk_type == "symbol":
                symbol_name = chunk.symbols[0]
                symbol = next((s for s in symbols if s.name == symbol_name), None)
                if symbol is not None:
                    self.candidates.append(
                        Candidate(
                            chunk_id=chunk.chunk_id,
                            fully_qualified_name=symbol.name,
                            role=symbol.role,
                            parent_module=symbol.module,
                        )
                    )
        self._exact_lookup: dict[str, list[Candidate]] = {}
        for candidate in self.candidates:
            for name in _short_names(candidate.fully_qualified_name):
                self._exact_lookup.setdefault(name, []).append(candidate)

    def exact(self, query: str) -> list[Candidate]:
        """Return candidates whose name (full or any short form) equals `query`.

        Multiple candidates are sorted by role priority then parent_module then name.
        Empty list when no match.
        """
        return _sort_candidates(self._exact_lookup.get(query, []))

    def fuzzy(self, query: str, *, threshold: int = DEFAULT_FUZZY_THRESHOLD) -> list[Candidate]:
        """Return candidates whose name matches `query` with rapidfuzz score >= threshold.

        Used for case-insensitive matches and small typos
        (e.g. `dict.fromKeys` -> `dict.fromkeys`).
        """
        return _sort_candidates(
            [
                candidate
                for candidate in self.candidates
                if fuzz.ratio(query, candidate.fully_qualified_name) >= threshold
            ]
        )

    def lookup(
        self, query: str, *, fuzzy_threshold: int = DEFAULT_FUZZY_THRESHOLD
    ) -> list[Candidate]:
        """Try exact() first; if empty, fall back to fuzzy()."""
        exact_results = self.exact(query)
        if len(exact_results) > 0:
            return exact_results
        fuzzy_results = self.fuzzy(query, threshold=fuzzy_threshold)
        return fuzzy_results


# ------------------------------------------------------------------
# Helpers (private)
# ------------------------------------------------------------------


def _short_names(fqn: str) -> list[str]:
    """Return distinct short-form aliases of `fqn` (full name included).

    Examples:
        "pathlib.Path.read_text" -> ["pathlib.Path.read_text", "Path.read_text", "read_text"]
        "dict.fromkeys"          -> ["dict.fromkeys", "fromkeys"]
        "open"                   -> ["open"]
    """
    parts = fqn.split(".")
    if fqn == "":
        return []
    short_names: list[str] = []
    for i in range(len(parts)):
        short_names.append(".".join(parts[i:]))
    return short_names


def _role_priority(role: str) -> int:
    """Lookup-order priority for `role`; unknown roles return UNKNOWN_ROLE_PRIORITY."""
    if role in ROLE_PRIORITY:
        return ROLE_PRIORITY[role]
    return UNKNOWN_ROLE_PRIORITY


def _sort_candidates(candidates: list[Candidate]) -> list[Candidate]:
    """Sort candidates by (role priority, parent_module, fully_qualified_name)."""
    return sorted(
        candidates,
        key=lambda c: (_role_priority(c.role), c.parent_module or "", c.fully_qualified_name),
    )
