"""Cross-encoder reranker for v2 §3.

Wraps `sentence_transformers.CrossEncoder` (`BAAI/bge-reranker-base`).
Scores `(query, chunk_text)` pairs in batches and returns the top-K
candidates sorted by cross-encoder score descending.

Decision (per plan §142): the original input score is NOT preserved.
`RerankedHit.score` is the cross-encoder output only. If downstream needs
to analyze rank movement vs. the input ranking, it can compare against
the per_query.jsonl `retrieved` field which still carries pre-rerank
scores.

NOTE: `sentence_transformers` is *lazily* imported inside the methods
that need it — importing this module without the `rerank` extra
installed must NOT raise. (Tests rely on this.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

from python_doc_assistant.ingest.chunker import Chunk

# ------------------------------------------------------------------
# Defaults (plan v2 §3)
# ------------------------------------------------------------------

DEFAULT_MODEL_ID: Final[str] = "BAAI/bge-reranker-base"
DEFAULT_TOP_K: Final[int] = 5
# Cross-encoder is much slower than embedding; batching the (query, doc)
# pairs through the GPU/MPS in chunks keeps latency manageable.
DEFAULT_BATCH_SIZE: Final[int] = 32


# ------------------------------------------------------------------
# Data records
# ------------------------------------------------------------------


@dataclass(frozen=True)
class RerankedHit:
    """One reranked candidate."""

    chunk_id: str
    score: float  # cross-encoder relevance score


# ------------------------------------------------------------------
# Public: cross-encoder reranker
# ------------------------------------------------------------------


class CrossEncoderReranker:
    """Reranks a candidate list with a cross-encoder model.

    Build once (model load is the slow part); call `rerank()` per query.

    Build path:
        reranker = CrossEncoderReranker()                      # default bge-reranker-base
        top5     = reranker.rerank(query, candidates, top_k=5)

    Test path (DI):
        reranker = CrossEncoderReranker(model=stub_cross_encoder)
    """

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_MODEL_ID,
        model: Any = None,
    ) -> None:
        """Load the cross-encoder model (or accept an injected one).

        Args:
            model_id: HuggingFace model id (used when `model` is None).
            model: pre-loaded `sentence_transformers.CrossEncoder` instance
                (or test stub exposing `.predict(pairs, batch_size=...,
                convert_to_numpy=True)`). When provided, `from_pretrained`
                is skipped entirely.

        Implementation outline:
            1. self._model_id = model_id
            2. if model is None:
                   from sentence_transformers import CrossEncoder
                   model = CrossEncoder(model_id)
               self._model = model
        """
        self._model_id = model_id
        if model is None:
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(model_id)
        self._model = model

    def rerank(
        self,
        query: str,
        chunks: list[Chunk],
        *,
        top_k: int = DEFAULT_TOP_K,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> list[RerankedHit]:
        """Score (query, chunk_text) for each candidate; return top-K.

        Args:
            query: the user query string.
            chunks: candidate chunks to rerank (typically top-20 from
                hybrid retrieval).
            top_k: number of reranked hits to return (default 5 per plan).
            batch_size: cross-encoder forward batch size.

        Returns:
            list[RerankedHit] sorted by cross-encoder score descending,
            capped at `top_k` entries. Empty `chunks` → [].

        Implementation outline:
            1. if not chunks: return []
            2. pairs = [(query, self._chunk_to_text(c)) for c in chunks]
            3. scores = self._model.predict(
                   pairs,
                   batch_size=batch_size,
                   convert_to_numpy=True,
                   show_progress_bar=False,
               )
            4. ranked = sorted(
                   zip(chunks, scores), key=lambda pair: -float(pair[1])
               )
            5. return [
                   RerankedHit(chunk_id=c.chunk_id, score=float(s))
                   for c, s in ranked[:top_k]
               ]
        """
        if len(chunks) == 0:
            return []
        pairs = [(query, self._chunk_to_text(c)) for c in chunks]
        scores = self._model.predict(
            pairs,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        sorted_chunk_scores = sorted(
            zip(chunks, scores),
            key=lambda t: -float(t[1]),
        )
        return [
            RerankedHit(chunk_id=chunk.chunk_id, score=float(score))
            for chunk, score in sorted_chunk_scores[:top_k]
        ]

    # ------------------------------------------------------------------
    # Helpers (private)
    # ------------------------------------------------------------------

    @staticmethod
    def _chunk_to_text(chunk: Chunk) -> str:
        """Compose the doc side of the (query, doc) pair fed to the cross-encoder.

        Default: `<title>\\n\\n<text>` (matches DenseIndex). Title
        disambiguates similar bodies (same `read_text` across modules).
        """
        return f"{chunk.title}\n\n{chunk.text}"
