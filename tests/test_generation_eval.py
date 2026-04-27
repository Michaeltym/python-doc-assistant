"""Tests for python_doc_assistant.evaluation.generation_eval.

Hermetic — no real Qwen model loaded. Uses a stub Generator that returns
canned Answer objects so the retrieval-then-generation pipeline can be
exercised end-to-end without transformers / torch.
"""

from __future__ import annotations

from python_doc_assistant.evaluation.dataset import EvalQuery
from python_doc_assistant.evaluation.generation_eval import (
    _build_generation_chunks,
    _resolve_query_type,
    evaluate_with_generation,
)
from python_doc_assistant.evaluation.retrieval_metrics import (
    PerQueryResult,
    RetrievedChunk,
)
from python_doc_assistant.generation.interface import Answer, Generator
from python_doc_assistant.ingest.chunker import Chunk
from python_doc_assistant.retrieval.router import QueryType

# ------------------------------------------------------------------
# Fixture builders
# ------------------------------------------------------------------


def _eval_query(
    query: str = "Path.read_text",
    *,
    query_type: str = "identifier",
    expected_symbols: tuple[str, ...] = ("pathlib.Path.read_text",),
    expected_urls: tuple[str, ...] = ("library/pathlib.html",),
) -> EvalQuery:
    return EvalQuery(
        query=query,
        query_type=query_type,
        expected_symbols=expected_symbols,
        expected_urls=expected_urls,
        match_policy="any",
        url_match="strip_anchor",
        notes=None,
    )


def _chunk(
    chunk_id: str = "symbol:pathlib.Path.read_text",
    *,
    title: str = "Path.read_text",
    text: str = "Return the file contents as a string.",
    canonical_url: str = "library/pathlib.html#pathlib.Path.read_text",
    symbols: tuple[str, ...] = ("pathlib.Path.read_text",),
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        chunk_type="symbol",
        docs_version="3.12",
        title=title,
        text=text,
        symbols=symbols,
        canonical_url=canonical_url,
        anchor=chunk_id,
        parent_module="pathlib",
        source_path="library/pathlib.html",
        source_hash="sha256:abc",
    )


def _retrieved(chunk_id: str, rank: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        score=1.0 / rank,
        rank=rank,
        canonical_url=f"library/foo.html#{chunk_id}",
        symbols=(chunk_id.split(":", 1)[-1],),
    )


class _StubGenerator(Generator):
    """Returns canned Answer + records what was passed."""

    def __init__(
        self,
        *,
        text: str = "stub answer",
        cited_chunk_ids: tuple[str, ...] = (),
        refused: bool = False,
        latency_seconds: float = 0.5,
    ) -> None:
        self._answer_text = text
        self._cited_chunk_ids = cited_chunk_ids
        self._refused = refused
        self._latency = latency_seconds
        self.calls: list[dict[str, object]] = []

    def generate(
        self,
        query: str,
        retrieved_chunks: list[Chunk],
        *,
        query_type: QueryType | None = None,
        stream: bool = False,
    ) -> Answer:
        self.calls.append(
            {
                "query": query,
                "retrieved_chunks": retrieved_chunks,
                "query_type": query_type,
                "stream": stream,
            }
        )
        return Answer(
            text=self._answer_text,
            cited_chunk_ids=self._cited_chunk_ids,
            refused=self._refused,
            latency_seconds=self._latency,
        )


def _retrieve_fn_returning(
    chunk_ids: list[str],
) -> object:
    """Build a closure returning canned RetrievedChunk list, deterministic."""

    def retrieve_fn(query: str, k: int) -> list[RetrievedChunk]:
        return [_retrieved(cid, rank=i + 1) for i, cid in enumerate(chunk_ids[:k])]

    return retrieve_fn


# ------------------------------------------------------------------
# _resolve_query_type
# ------------------------------------------------------------------


def test_resolve_query_type_known_string_returns_enum() -> None:
    assert _resolve_query_type("identifier") == QueryType.IDENTIFIER
    assert _resolve_query_type("howto") == QueryType.HOWTO
    assert _resolve_query_type("comparison") == QueryType.COMPARISON


def test_resolve_query_type_unknown_string_returns_none() -> None:
    """Unknown strings (e.g. typo / future type) → None for generic template."""
    assert _resolve_query_type("not_a_real_type") is None


# ------------------------------------------------------------------
# _build_generation_chunks
# ------------------------------------------------------------------


def test_build_generation_chunks_takes_top_k() -> None:
    pq = PerQueryResult(
        query="q",
        query_type="identifier",
        match_policy="any",
        url_match="strip_anchor",
        expected_symbols=(),
        expected_urls=(),
        retrieved=tuple(_retrieved(f"c{i}", rank=i) for i in range(1, 6)),
        hit_at_5=False,
        hit_at_10=False,
        rank_for_mrr=None,
    )
    chunks_by_id = {f"c{i}": _chunk(f"c{i}") for i in range(1, 6)}
    out = _build_generation_chunks(pq, chunks_by_id, k_for_generation=3)
    assert len(out) == 3
    assert [c.chunk_id for c in out] == ["c1", "c2", "c3"]


def test_build_generation_chunks_drops_missing_ids() -> None:
    """Defensive: chunks_by_id may be a stale dict; missing ids are skipped."""
    pq = PerQueryResult(
        query="q",
        query_type="identifier",
        match_policy="any",
        url_match="strip_anchor",
        expected_symbols=(),
        expected_urls=(),
        retrieved=(_retrieved("present", rank=1), _retrieved("missing", rank=2)),
        hit_at_5=False,
        hit_at_10=False,
        rank_for_mrr=None,
    )
    chunks_by_id = {"present": _chunk("present")}
    out = _build_generation_chunks(pq, chunks_by_id, k_for_generation=10)
    assert len(out) == 1
    assert out[0].chunk_id == "present"


# ------------------------------------------------------------------
# evaluate_with_generation
# ------------------------------------------------------------------


def test_evaluate_with_generation_populates_per_query_fields() -> None:
    """Every PerQueryResult gets the four generation fields filled."""
    eval_queries = [_eval_query()]
    retrieve_fn = _retrieve_fn_returning(["symbol:pathlib.Path.read_text"])
    chunks_by_id = {"symbol:pathlib.Path.read_text": _chunk()}
    gen = _StubGenerator(
        text="Use Path.read_text() [1].",
        cited_chunk_ids=("symbol:pathlib.Path.read_text",),
        refused=False,
        latency_seconds=2.5,
    )

    result = evaluate_with_generation(
        eval_queries, retrieve_fn, gen, chunks_by_id, max_k=10, k_for_generation=5
    )

    assert len(result.queries) == 1
    pq = result.queries[0]
    assert pq.model_output_text == "Use Path.read_text() [1]."
    assert pq.cited_chunk_ids == ("symbol:pathlib.Path.read_text",)
    assert pq.refused is False
    assert pq.generation_latency_seconds == 2.5


def test_evaluate_with_generation_preserves_retrieval_metrics() -> None:
    """Aggregate retrieval metrics (recall@5/10/mrr) come straight from evaluate()."""
    eval_queries = [_eval_query()]
    retrieve_fn = _retrieve_fn_returning(["symbol:pathlib.Path.read_text"])
    chunks_by_id = {"symbol:pathlib.Path.read_text": _chunk()}
    gen = _StubGenerator()

    result = evaluate_with_generation(
        eval_queries, retrieve_fn, gen, chunks_by_id
    )

    assert result.n_queries == 1
    assert result.recall_at_5 == 1.0  # retrieved chunk matches expected_symbols
    assert result.recall_at_10 == 1.0
    assert result.mrr == 1.0


def test_evaluate_with_generation_passes_query_type_to_generator() -> None:
    """Generator receives the QueryType enum derived from EvalQuery.query_type."""
    eval_queries = [_eval_query(query_type="howto")]
    retrieve_fn = _retrieve_fn_returning(["c1"])
    chunks_by_id = {"c1": _chunk("c1")}
    gen = _StubGenerator()

    evaluate_with_generation(eval_queries, retrieve_fn, gen, chunks_by_id)

    assert len(gen.calls) == 1
    assert gen.calls[0]["query_type"] == QueryType.HOWTO


def test_evaluate_with_generation_unknown_query_type_passes_none() -> None:
    """Unknown query_type strings → generator gets None (generic template)."""
    eval_queries = [_eval_query(query_type="totally_made_up")]
    retrieve_fn = _retrieve_fn_returning(["c1"])
    chunks_by_id = {"c1": _chunk("c1")}
    gen = _StubGenerator()

    evaluate_with_generation(eval_queries, retrieve_fn, gen, chunks_by_id)

    assert gen.calls[0]["query_type"] is None


def test_evaluate_with_generation_uses_k_for_generation_not_max_k() -> None:
    """Generator sees only top-k_for_generation chunks, even when max_k is larger."""
    eval_queries = [_eval_query()]
    retrieve_fn = _retrieve_fn_returning([f"c{i}" for i in range(1, 11)])
    chunks_by_id = {f"c{i}": _chunk(f"c{i}") for i in range(1, 11)}
    gen = _StubGenerator()

    evaluate_with_generation(
        eval_queries, retrieve_fn, gen, chunks_by_id, max_k=10, k_for_generation=3
    )

    chunks_passed = gen.calls[0]["retrieved_chunks"]
    assert isinstance(chunks_passed, list)
    assert len(chunks_passed) == 3
    assert [c.chunk_id for c in chunks_passed] == ["c1", "c2", "c3"]


def test_evaluate_with_generation_runs_one_call_per_query() -> None:
    eval_queries = [_eval_query("q1"), _eval_query("q2"), _eval_query("q3")]
    retrieve_fn = _retrieve_fn_returning(["c1"])
    chunks_by_id = {"c1": _chunk("c1")}
    gen = _StubGenerator()

    evaluate_with_generation(eval_queries, retrieve_fn, gen, chunks_by_id)

    assert len(gen.calls) == 3
    assert [call["query"] for call in gen.calls] == ["q1", "q2", "q3"]


def test_evaluate_with_generation_refused_answer_propagates() -> None:
    """A refused Answer surfaces as refused=True and empty text in PerQueryResult."""
    eval_queries = [_eval_query()]
    retrieve_fn = _retrieve_fn_returning(["c1"])
    chunks_by_id = {"c1": _chunk("c1")}
    gen = _StubGenerator(text="", cited_chunk_ids=(), refused=True, latency_seconds=0.1)

    result = evaluate_with_generation(eval_queries, retrieve_fn, gen, chunks_by_id)

    pq = result.queries[0]
    assert pq.refused is True
    assert pq.model_output_text == ""
    assert pq.cited_chunk_ids == ()
