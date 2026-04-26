"""BM25 index + analyzer for v0 retrieval.

See plans/v0-retrieval-eval.md §5 for the full contract.
"""

from __future__ import annotations

import pickle
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from rank_bm25 import BM25Okapi

from python_doc_assistant.ingest.chunker import Chunk

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

DEFAULT_K1: Final[float] = 1.5  # rank_bm25.BM25Okapi default
DEFAULT_B: Final[float] = 0.75  # rank_bm25.BM25Okapi default


# ------------------------------------------------------------------
# Data records
# ------------------------------------------------------------------


@dataclass(frozen=True)
class Hit:
    """One BM25 search result."""

    chunk_id: str
    score: float


# ------------------------------------------------------------------
# Public: analyzer (applied identically to indexed docs and queries)
# ------------------------------------------------------------------


def analyze(text: str) -> list[str]:
    """Tokenize per the 5-step pipeline (plan §5).

    Order matters — lowercase MUST run LAST so CamelCase boundaries survive.

    Pipeline:
        1. Split on '.'
        2. CamelCase split, keeping the merged form too
           (HTTPServer -> [HTTP, Server, HTTPServer]; BZ2File -> [BZ2, File, BZ2File])
        3. Underscore split, keeping the merged form too
           (read_text -> [read, text, read_text]; __init__ preserved as-is)
        4. Lowercase
        5. Drop empty and pure-punctuation tokens

    Examples:
        "pathlib.Path.read_text"      -> ["pathlib", "path", "read_text", "read", "text"]
        "dict.fromKeys"               -> ["dict", "fromkeys", "from", "keys"]
        "os.path.join"                -> ["os", "path", "join"]
        "how to iterate dict safely"  -> ["how", "to", "iterate", "dict", "safely"]
    """
    parts = text.split()
    dot_splited = _split_on_dots(parts)
    camel_case_splited = _split_camelcase(dot_splited)
    underscore_splited = _split_underscores(camel_case_splited)
    return _lowercase_and_drop_empty(underscore_splited)


# ------------------------------------------------------------------
# Public: BM25 index
# ------------------------------------------------------------------


class BM25Index:
    """Wraps rank_bm25.BM25Okapi with our analyzer + chunk_id mapping.

    Build once from a list[Chunk]; search() returns Hits ranked by BM25 score.
    Persist via save() / load() — the pickle format includes the chunk_id list
    and BM25Okapi state.
    """

    def __init__(
        self,
        chunks: list[Chunk],
        *,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
    ) -> None:
        """Tokenize each chunk's text + symbols field through `analyze`, build BM25Okapi.

        Suggested flow:
            1. self._chunk_ids: list[str] = [c.chunk_id for c in chunks]
            2. corpus: list[list[str]] = [
                   analyze(c.text + " " + " ".join(c.symbols)) for c in chunks
               ]
            3. self._bm25 = BM25Okapi(corpus, k1=k1, b=b)
        """
        self._chunk_ids = [chunk.chunk_id for chunk in chunks]
        corpus: list[list[str]] = []
        for chunk in chunks:
            text = chunk.text + "\n\n" + " ".join(chunk.symbols)
            tokens = analyze(text)
            corpus.append(tokens)
        self._bm25 = BM25Okapi(corpus, k1=k1, b=b)

    def search(self, query: str, *, k: int = 10) -> list[Hit]:
        """Return up to top-k Hits for `query`, sorted by score (highest first).

        Suggested flow:
            tokens = analyze(query)
            scores = self._bm25.get_scores(tokens)
            ranked = sorted(zip(self._chunk_ids, scores), key=lambda x: -x[1])
            return [Hit(cid, float(s)) for cid, s in ranked[:k] if s > 0]
        """
        tokens = analyze(query)
        scores = self._bm25.get_scores(tokens)
        sorted_chunks = sorted(zip(scores, self._chunk_ids), key=lambda v: v[0], reverse=True)
        return [
            Hit(score=float(score), chunk_id=chunk_id)
            for score, chunk_id in sorted_chunks[:k]
            if score > 0
        ]

    def save(self, path: Path) -> None:
        """Pickle the index to `path` (parents auto-created)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(
                {"chunk_ids": self._chunk_ids, "bm25": self._bm25},
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    @classmethod
    def load(cls, path: Path) -> BM25Index:
        if not path.exists():
            raise FileNotFoundError(f"BM25 index not found at {path}")
        if not path.is_file():
            raise IsADirectoryError(f"{path} exists but is not a regular file")
        with path.open("rb") as f:
            state = pickle.load(f)
            instance = cls.__new__(cls)
            instance._chunk_ids = state["chunk_ids"]
            instance._bm25 = state["bm25"]
            return instance


# ------------------------------------------------------------------
# Helpers (private analyzer stages)
# ------------------------------------------------------------------


def _split_on_dots(tokens: list[str]) -> list[str]:
    """For each token, split on '.'; preserve case."""
    splited: list[str] = []
    for token in tokens:
        splited.extend(token.split("."))
    return splited


def _split_camelcase(tokens: list[str]) -> list[str]:
    """For each token, split CamelCase boundaries AND keep the merged form.

    Boundary rules:
        - lowercase -> uppercase: `fromKeys` -> [from, Keys]
        - uppercase-abbrev -> Capitalized: `HTTPServer` -> [HTTP, Server]
        - digit -> uppercase: `BZ2File` -> [BZ2, File]

    Must run BEFORE lowercase (otherwise the case boundaries are gone).
    """
    splited: list[str] = []
    for token in tokens:
        parts = re.split(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z0-9])(?=[A-Z][a-z])", token)
        if len(parts) > 1:
            splited.append(token)
        splited.extend(parts)
    return splited


def _split_underscores(tokens: list[str]) -> list[str]:
    """For each token, split on '_' AND keep the merged form.

    Special case: dunder names (e.g. `__init__`) are preserved as-is in addition
    to the inner words (`init`).
    """
    splited: list[str] = []
    for token in tokens:
        parts = token.split("_")
        if len(parts) > 1:
            splited.append(token)
        splited.extend(parts)
    return splited


def _lowercase_and_drop_empty(tokens: list[str]) -> list[str]:
    """Final stage: lowercase every token; drop empty and pure-punctuation tokens."""
    return [
        token.lower()
        for token in tokens
        if token and not all(c in string.punctuation for c in token)
    ]
