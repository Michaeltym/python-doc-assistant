"""Eval pipeline that runs retrieval + generation per query.

See plans/v1-qwen-generator.md §4.

`evaluate_with_generation()` wraps the v0 retrieval-only `evaluate()` and
adds a generator pass: top-K chunks → Generator → Answer → store
generation fields on each `PerQueryResult`.

Two CLI invocations (one per model) produce the side-by-side runs that
feed `experiments/v1-qwen-grounded.md`:

    pdr eval --set <eval.jsonl> --tag v1-qwen \
        --model Qwen/Qwen2.5-1.5B-Instruct
    pdr eval --set <eval.jsonl> --tag v1-coder \
        --model Qwen/Qwen2.5-Coder-1.5B-Instruct
"""

from __future__ import annotations

from dataclasses import replace

from python_doc_assistant.evaluation.dataset import EvalQuery
from python_doc_assistant.evaluation.retrieval_metrics import (
    EvalRunResult,
    PerQueryResult,
    RetrieveFn,
    evaluate,
)
from python_doc_assistant.generation.interface import Generator
from python_doc_assistant.ingest.chunker import Chunk
from python_doc_assistant.retrieval.router import QueryType

# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def evaluate_with_generation(
    eval_queries: list[EvalQuery],
    retrieve_fn: RetrieveFn,
    generator: Generator,
    chunks_by_id: dict[str, Chunk],
    *,
    max_k: int = 10,
    k_for_generation: int = 5,
) -> EvalRunResult:
    """Run retrieval + generation per query.

    Steps:
        1. base = evaluate(eval_queries, retrieve_fn, max_k=max_k)
           — gives retrieval metrics + ranked retrieved chunks per query.
        2. For each PerQueryResult `pq` in base.queries:
              a. Take pq.retrieved[:k_for_generation] (top-K for the LLM).
              b. Map each RetrievedChunk → full Chunk via chunks_by_id;
                 silently drop ids that are missing from the dict.
              c. Resolve query_type → QueryType enum (skip if unknown so
                 the generator falls back to its generic template).
              d. answer = generator.generate(
                     pq.query, mapped_chunks, query_type=qt
                 )
              e. Build a new PerQueryResult with generation fields filled:
                 model_output_text          = answer.text
                 cited_chunk_ids            = answer.cited_chunk_ids
                 refused                    = answer.refused
                 generation_latency_seconds = answer.latency_seconds
                 (use dataclasses.replace; retrieval fields stay intact)
        3. Return a new EvalRunResult with the updated queries tuple;
           recall@5/recall@10/mrr/n_queries copied from `base`.

    Notes:
        - `chunks_by_id` typically comes from the CLI:
              `{c.chunk_id: c for c in _load_chunks(chunks_path)}`
        - The router's QueryType enum names align with EvalQuery.query_type
          strings ("identifier", "natural_language", "comparison", "howto",
          "out_of_scope"); use `QueryType(pq.query_type)` to resolve, and
          fall back to None on ValueError.
    """
    eval_result = evaluate(eval_queries, retrieve_fn, max_k=max_k)
    queries: list[PerQueryResult] = []
    for q in eval_result.queries:
        retrieved_chunks = _build_generation_chunks(q, chunks_by_id, k_for_generation)
        answer = generator.generate(
            q.query, retrieved_chunks, query_type=_resolve_query_type(q.query_type)
        )
        queries.append(
            _attach_answer(
                q, answer.text, answer.cited_chunk_ids, answer.refused, answer.latency_seconds
            )
        )
    return replace(eval_result, queries=tuple(queries))


# ------------------------------------------------------------------
# Helpers (private)
# ------------------------------------------------------------------


def _resolve_query_type(query_type_str: str) -> QueryType | None:
    """Map an eval-set query_type string to QueryType enum, or None on miss.

    Used by `evaluate_with_generation` so generator gets the per-type
    structure hint when the eval row classifies the query, and falls
    back to a generic template otherwise.
    """
    try:
        query_type = QueryType(query_type_str)
    except ValueError:
        query_type = None
    return query_type


def _build_generation_chunks(
    pq: PerQueryResult,
    chunks_by_id: dict[str, Chunk],
    k_for_generation: int,
) -> list[Chunk]:
    """Take pq.retrieved[:k] and map each RetrievedChunk → full Chunk.

    Silently drops chunk_ids that are missing from `chunks_by_id`
    (defensive: chunks file may have been re-ingested between retrieval
    indexing and this call).
    """
    retrieved_chunks: list[Chunk] = []
    for c in pq.retrieved[:k_for_generation]:
        chunk = chunks_by_id.get(c.chunk_id)
        if chunk is not None:
            retrieved_chunks.append(chunk)
    return retrieved_chunks


def _attach_answer(
    pq: PerQueryResult,
    answer_text: str,
    cited_chunk_ids: tuple[str, ...],
    refused: bool,
    latency_seconds: float,
) -> PerQueryResult:
    """Return a copy of pq with the four generation fields populated.

    Uses dataclasses.replace to keep retrieval fields untouched.
    """
    return replace(
        pq,
        model_output_text=answer_text,
        cited_chunk_ids=cited_chunk_ids,
        refused=refused,
        generation_latency_seconds=latency_seconds,
    )
