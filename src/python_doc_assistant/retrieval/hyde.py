"""HyDE-augmented retrieval for v4 sub-task 2.

Background:
    Short / vague natural-language queries embed to vectors that are
    far from the long, technical chunks they should match. The v4
    Week 2 hallucination bucket includes 9 such queries — `how to
    read a file in python`, `how to merge two dicts`, `Path vs
    os.path`, etc. — where the right chunk exists in the corpus but
    the dense retriever does not surface it because the query and the
    chunk live in different regions of embedding space.

    HyDE (Gao et al., 2022) addresses this by asking an LLM to write a
    hypothetical answer in document form, then embedding THAT
    hypothetical instead of the raw query. The hypothetical reads like
    documentation, so its embedding lands near actual documentation
    chunks, even when the LLM hallucinates non-existent APIs (the
    *topic* is right; the *facts* matter only at generation time, not
    retrieval time).

Pipeline (sub-task 2 design call, 2026-05-06):
    1. Classify the query. If `query_type in skip_query_types`
       (default: IDENTIFIER), skip HyDE and use the original query for
       dense — identifier queries are already in document form.
    2. Otherwise, ask the `HypotheticalGenerator` for a hypothetical
       documentation passage that would answer the query.
    3. Pass the hypothetical to `dense_index.search(...)` to fetch top-N
       candidates.
    4. If a reranker is provided, rerank with the ORIGINAL query (not
       the hypothetical) so the cross-encoder scores user intent, not
       the LLM's invention.
    5. Map results to `RetrievedChunk` (rank-ordered, score-populated,
       URL+symbols filled from `chunks_by_id`).

Why rerank with original query: the cross-encoder is trained to score
"does this passage answer this question?". Feeding it the hypothetical
would score "does this passage match this other passage?" — different
task, less reliable signal for our intent. The hypothetical's job is
done after dense retrieval.

This module never calls llama-cpp directly; the LLM dependency comes in
through `HypotheticalGenerator`. Tests inject a stub that returns
canned hypotheticals; production wires `QwenHypotheticalGenerator`
around the same Llama instance held by `QwenGGUFGenerator`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Final, Protocol

from python_doc_assistant.evaluation.retrieval_metrics import RetrievedChunk, RetrieveFn
from python_doc_assistant.indexes.dense_index import DenseIndex
from python_doc_assistant.ingest.chunker import Chunk
from python_doc_assistant.retrieval.rerank import CrossEncoderReranker
from python_doc_assistant.retrieval.router import QueryType, classify

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

DEFAULT_HYDE_PROMPT_TEMPLATE: Final[str] = (
    "Write a brief Python standard library documentation passage "
    '(3-5 sentences) that would directly answer this question: "{query}". '
    "Use canonical Python stdlib module/class/function names. If unsure "
    "of the exact API, describe the topic abstractly without inventing "
    "non-existent names. Do not refuse; do not add caveats."
)
DEFAULT_HYDE_MAX_TOKENS: Final[int] = 200
DEFAULT_HYDE_TEMPERATURE: Final[float] = 0.0
DEFAULT_RERANK_CANDIDATES: Final[int] = 20
DEFAULT_SKIP_QUERY_TYPES: Final[tuple[QueryType, ...]] = (QueryType.IDENTIFIER,)


# ------------------------------------------------------------------
# Hypothetical generator
# ------------------------------------------------------------------


class HypotheticalGenerator(Protocol):
    """Generates a hypothetical documentation passage for a query."""

    def generate(self, query: str) -> str:
        """Return a short documentation-shaped passage answering `query`."""
        ...


class QwenHypotheticalGenerator:
    """`HypotheticalGenerator` backed by a llama-cpp `Llama` instance.

    Reuse pattern: pass the SAME `Llama` instance held by
    `QwenGGUFGenerator` so the 4.7 GB model loads only once per process.

    Example:
        gen = QwenGGUFGenerator(model_path="...")
        hyp = QwenHypotheticalGenerator(llm=gen.llm)
    """

    def __init__(
        self,
        llm: Any,
        *,
        prompt_template: str = DEFAULT_HYDE_PROMPT_TEMPLATE,
        max_tokens: int = DEFAULT_HYDE_MAX_TOKENS,
        temperature: float = DEFAULT_HYDE_TEMPERATURE,
    ) -> None:
        """Store the LLM handle and decoding params.

        Args:
            llm: a llama-cpp `Llama` instance (or any duck-typed stub
                exposing `create_chat_completion(messages, max_tokens,
                temperature) -> dict`).
            prompt_template: format string with a single `{query}`
                placeholder. The default is `DEFAULT_HYDE_PROMPT_TEMPLATE`.
            max_tokens: max tokens for the hypothetical (3-5 sentences
                fit comfortably in 200).
            temperature: 0.0 = greedy. Hypotheticals are best
                deterministic so the same query always retrieves the
                same chunks.

        Stores: self.llm / prompt_template / max_tokens / temperature.

        Raises:
            ValueError if `prompt_template` does not contain `{query}`.
        """
        self.llm = llm
        if "{query}" not in prompt_template:
            raise ValueError("prompt_template must contain '{query}' placeholder")
        self.prompt_template = prompt_template
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(self, query: str) -> str:
        """Format the prompt, call the LLM, return the hypothetical text.

        Implementation outline:
            1. content = self.prompt_template.format(query=query)
            2. messages = [{"role": "user", "content": content}]
            3. result = self.llm.create_chat_completion(
                   messages=messages,
                   max_tokens=self.max_tokens,
                   temperature=self.temperature,
               )
            4. text = result["choices"][0]["message"]["content"]
            5. Return text.strip()  (strip leading/trailing whitespace
               from the model's output to keep the embedded text clean).
        """
        prompt = self.prompt_template.format(query=query)
        messages = [{"role": "user", "content": prompt}]
        result = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        text: str = result["choices"][0]["message"]["content"]
        return text.strip()


# ------------------------------------------------------------------
# HyDE retrieve_fn factory
# ------------------------------------------------------------------


def make_hyde_retrieve_fn(
    *,
    dense_index: DenseIndex,
    chunks_by_id: dict[str, Chunk],
    hypothetical_generator: HypotheticalGenerator,
    reranker: CrossEncoderReranker | None = None,
    rerank_candidates: int = DEFAULT_RERANK_CANDIDATES,
    skip_query_types: tuple[QueryType, ...] = DEFAULT_SKIP_QUERY_TYPES,
    classify_fn: Callable[[str], QueryType] = classify,
) -> RetrieveFn:
    """Build a `(query, k) -> list[RetrievedChunk]` closure that uses HyDE.

    Args:
        dense_index: the dense index used for embedding-based search.
            HyDE swaps the query argument; this index handles its own
            embedding step internally.
        chunks_by_id: full chunk lookup. Needed to populate
            `RetrievedChunk.canonical_url` + `.symbols`, and to convert
            `chunk_id`s to `Chunk` objects for the reranker.
        hypothetical_generator: produces the hypothetical text per query.
        reranker: optional cross-encoder reranker. When provided, dense
            fetches top-`rerank_candidates`, then rerank trims to top-k
            using the ORIGINAL query.
        rerank_candidates: top-N before rerank (only used when
            `reranker` is not None). Default matches the v2 ablation
            constant.
        skip_query_types: query types that bypass HyDE entirely (the
            original query is used for dense). Default: identifier
            queries (already in document form).
        classify_fn: query-type classifier. Override for tests.

    Returns:
        retrieve_fn(query, k) -> list[RetrievedChunk], rank-ordered.

    Implementation outline:

        def retrieve_fn(query: str, k: int) -> list[RetrievedChunk]:
            1. qt = classify_fn(query)
            2. if qt in skip_query_types:
                   search_query = query
               else:
                   search_query = hypothetical_generator.generate(query)
            3. n_dense = rerank_candidates if reranker is not None else k
               dense_hits = dense_index.search(search_query, k=n_dense)
            4. if reranker is None:
                   pairs = [(h.chunk_id, h.score) for h in dense_hits]
               else:
                   chunks = [chunks_by_id[h.chunk_id]
                             for h in dense_hits
                             if h.chunk_id in chunks_by_id]
                   reranked = reranker.rerank(query, chunks, top_k=k)
                   #  ^ rerank uses ORIGINAL query, not search_query.
                   pairs = [(r.chunk_id, r.score) for r in reranked]
            5. Build the RetrievedChunk list (1-indexed rank), drop ids
               missing from chunks_by_id (defensive), cap at k:
                   results: list[RetrievedChunk] = []
                   for i, (cid, score) in enumerate(pairs[:k], start=1):
                       if cid not in chunks_by_id:
                           continue
                       chunk = chunks_by_id[cid]
                       results.append(RetrievedChunk(
                           chunk_id=cid,
                           score=score,
                           rank=i,
                           canonical_url=chunk.canonical_url,
                           symbols=chunk.symbols,
                       ))
                   return results
        return retrieve_fn
    """

    def retrieve_fn(query: str, k: int) -> list[RetrievedChunk]:
        query_type = classify_fn(query)
        if query_type in skip_query_types:
            search_query = query
        else:
            search_query = hypothetical_generator.generate(query)
        n_dense = rerank_candidates if reranker is not None else k
        dense_hits = dense_index.search(search_query, k=n_dense)
        if reranker is None:
            pairs = [(h.chunk_id, h.score) for h in dense_hits]
        else:
            chunks = [chunks_by_id[h.chunk_id] for h in dense_hits if h.chunk_id in chunks_by_id]
            reranked = reranker.rerank(query, chunks, top_k=k)
            pairs = [(r.chunk_id, r.score) for r in reranked]

        results: list[RetrievedChunk] = []
        for i, (cid, score) in enumerate(pairs[:k], start=1):
            if cid not in chunks_by_id:
                continue
            chunk = chunks_by_id[cid]
            results.append(
                RetrievedChunk(
                    chunk_id=cid,
                    score=score,
                    rank=i,
                    canonical_url=chunk.canonical_url,
                    symbols=chunk.symbols,
                )
            )
        return results

    return retrieve_fn
