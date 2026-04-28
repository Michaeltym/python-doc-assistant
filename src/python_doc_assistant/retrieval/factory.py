"""Retrieval factory for v2 §5 ablation.

Composes BM25 / symbol_index / dense / hybrid + optional cross-encoder
rerank into a single `(query, k) -> list[RetrievedChunk]` closure that
the eval pipeline (`evaluate` / `evaluate_with_generation`) consumes.

Supported retriever names:
    bm25            pure BM25 (no symbol routing)
    symbol+bm25     v0 router — exact symbol match → fallback to BM25
    dense           pure dense (cosine over normalized embeddings)
    hybrid-rrf      BM25 + dense fused via reciprocal rank fusion
    hybrid-linear   BM25 + dense blended via min-max + alpha weighting

Optional `reranker` wraps any of the above:
    inner.search(query, k=rerank_candidates)  →
    cross-encoder rerank → top-k
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final

from python_doc_assistant.evaluation.retrieval_metrics import RetrievedChunk
from python_doc_assistant.indexes.bm25_index import BM25Index
from python_doc_assistant.indexes.dense_index import DenseIndex
from python_doc_assistant.indexes.symbol_index import SymbolIndex
from python_doc_assistant.ingest.chunker import Chunk
from python_doc_assistant.retrieval.hybrid import linear_merge, rrf_merge
from python_doc_assistant.retrieval.rerank import CrossEncoderReranker
from python_doc_assistant.retrieval.router import route

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

VALID_RETRIEVERS: Final[frozenset[str]] = frozenset(
    {"bm25", "symbol+bm25", "dense", "hybrid-rrf", "hybrid-linear"}
)
DEFAULT_RERANK_CANDIDATES: Final[int] = 20  # top-N before rerank → top-K
DEFAULT_ALPHA: Final[float] = 0.5  # for hybrid-linear


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


RetrieveFn = Callable[[str, int], list[RetrievedChunk]]


def build_retrieve_fn(
    *,
    retriever: str,
    chunks_by_id: dict[str, Chunk],
    bm25_index: BM25Index | None = None,
    symbol_index: SymbolIndex | None = None,
    dense_index: DenseIndex | None = None,
    reranker: CrossEncoderReranker | None = None,
    alpha: float = DEFAULT_ALPHA,
    rerank_candidates: int = DEFAULT_RERANK_CANDIDATES,
) -> RetrieveFn:
    """Compose a retrieval closure for the eval pipeline.

    Args:
        retriever: one of VALID_RETRIEVERS.
        chunks_by_id: full chunk lookup; needed to fill RetrievedChunk
            fields (canonical_url + symbols) that downstream metrics need.
        bm25_index: required when retriever is bm25 / symbol+bm25 / hybrid-*.
        symbol_index: required when retriever is symbol+bm25.
        dense_index: required when retriever is dense / hybrid-*.
        reranker: if provided, fetch top-`rerank_candidates` from the
            inner retriever, then rerank to top-k via cross-encoder.
        alpha: blend weight for hybrid-linear (ignored otherwise).
        rerank_candidates: top-N before rerank (default 20). Plan §5
            ablation_constants pin this.

    Returns:
        retrieve_fn(query, k) -> list[RetrievedChunk]
            Each RetrievedChunk is rank-ordered (1-indexed),
            score-populated, and has canonical_url + symbols filled
            from `chunks_by_id`.

    Raises:
        ValueError on unknown retriever name or missing required index.

    Implementation outline:

        1. Validate retriever in VALID_RETRIEVERS.
        2. Validate required indexes per retriever (raise ValueError on miss).
        3. Define inner_retrieve(query, k_inner) -> list[(chunk_id, score)]:
             bm25         → bm25_index.search → (cid, score)
             symbol+bm25  → route() → enumerate to (cid, 1/(rank+1))
             dense        → dense_index.search → (cid, score)
             hybrid-rrf   → bm25.search + dense.search → rrf_merge → (cid, score)
             hybrid-linear→ bm25.search + dense.search → linear_merge → (cid, score)
        4. Define retrieve_fn(query, k) -> list[RetrievedChunk]:
             if reranker is None:
                 pairs = inner_retrieve(query, k)
             else:
                 candidates = inner_retrieve(query, rerank_candidates)
                 candidate_chunks = [chunks_by_id[cid] for cid, _ in candidates
                                      if cid in chunks_by_id]
                 reranked = reranker.rerank(query, candidate_chunks, top_k=k)
                 pairs = [(h.chunk_id, h.score) for h in reranked]
             return [
                 RetrievedChunk(
                     chunk_id=cid,
                     score=score,
                     rank=i,
                     canonical_url=chunks_by_id[cid].canonical_url,
                     symbols=chunks_by_id[cid].symbols,
                 )
                 for i, (cid, score) in enumerate(pairs, start=1)
                 if cid in chunks_by_id
             ]
        5. Return retrieve_fn (closure captures inner state).
    """
    if retriever not in VALID_RETRIEVERS:
        raise ValueError(f"Invalid retriever {retriever}")
    if bm25_index is None and retriever != "dense":
        raise ValueError("Missing bm25_index")
    if dense_index is None and (
        retriever == "dense" or retriever == "hybrid-rrf" or retriever == "hybrid-linear"
    ):
        raise ValueError("Missing dense_index")
    if symbol_index is None and retriever == "symbol+bm25":
        raise ValueError("Missing symbol_index")

    def retrieve_fn(query: str, k: int) -> list[RetrievedChunk]:

        if reranker is None:
            pairs = _inner_retrieve(
                query=query,
                k=k,
                retriever=retriever,
                bm25_index=bm25_index,
                symbol_index=symbol_index,
                dense_index=dense_index,
                alpha=alpha,
            )
        else:
            result = _inner_retrieve(
                query=query,
                k=rerank_candidates,
                retriever=retriever,
                bm25_index=bm25_index,
                symbol_index=symbol_index,
                dense_index=dense_index,
                alpha=alpha,
            )
            chunks = [chunks_by_id[chunk_id] for chunk_id, _ in result if chunk_id in chunks_by_id]
            pairs = [
                (r.chunk_id, r.score)
                for r in reranker.rerank(
                    query,
                    chunks,
                    top_k=k,
                )
            ]
        return [
            RetrievedChunk(
                chunk_id=chunk_id,
                score=score,
                rank=i,
                canonical_url=chunks_by_id[chunk_id].canonical_url,
                symbols=chunks_by_id[chunk_id].symbols,
            )
            for i, (chunk_id, score) in enumerate(pairs, start=1)
            if chunk_id in chunks_by_id
        ]

    return retrieve_fn


def _inner_retrieve(
    *,
    query: str,
    k: int,
    retriever: str,
    bm25_index: BM25Index | None = None,
    symbol_index: SymbolIndex | None = None,
    dense_index: DenseIndex | None = None,
    alpha: float = DEFAULT_ALPHA,
) -> list[tuple[str, float]]:
    if retriever == "bm25" and bm25_index is not None:
        bm_hits = bm25_index.search(query, k=k)
        return [(r.chunk_id, r.score) for r in bm_hits]
    if retriever == "symbol+bm25" and symbol_index is not None and bm25_index is not None:
        route_result = route(query, symbol_index=symbol_index, bm25_index=bm25_index, k=k)
        return [(chunk_id, 1 / i) for i, chunk_id in enumerate(route_result.chunk_ids, start=1)]
    if retriever == "dense" and dense_index is not None:
        dense_hits = dense_index.search(query, k=k)
        return [(r.chunk_id, r.score) for r in dense_hits]
    if (
        retriever in ("hybrid-rrf", "hybrid-linear")
        and bm25_index is not None
        and dense_index is not None
    ):
        bm25_result = bm25_index.search(query, k=k)
        dense_result = dense_index.search(query, k=k)
        rankings = [[r.chunk_id for r in bm25_result], [r.chunk_id for r in dense_result]]
        merged = (
            rrf_merge(rankings)
            if retriever == "hybrid-rrf"
            else linear_merge(
                [(r.chunk_id, r.score) for r in bm25_result],
                [(r.chunk_id, r.score) for r in dense_result],
                alpha=alpha,
            )
        )
        return [(r.chunk_id, r.score) for r in merged]
    return []
