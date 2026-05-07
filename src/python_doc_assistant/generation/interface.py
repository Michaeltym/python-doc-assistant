"""Generator abstract base class + Answer dataclass.

See plans/v1-qwen-generator.md §1.

Why ABC at v1 (not v0): v1 wires Qwen2.5; v3 (optional) plugs in a hand-written
TinyDocs backend. Both share the same call shape, so the CLI and eval pipeline
do not need to know which backend is active.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from python_doc_assistant.ingest.chunker import Chunk
from python_doc_assistant.retrieval.router import QueryType

# ------------------------------------------------------------------
# Data records
# ------------------------------------------------------------------


@dataclass(frozen=True)
class Answer:
    """Output of Generator.generate().

    Fields:
        text: generated answer text (empty string when refused=True).
        cited_chunk_ids: chunk_ids the model explicitly referenced
            (parsed out of the prompt's citation marker, e.g. [#chunk_id]).
        refused: True iff the model emitted the refusal marker — i.e. the
            answer was not present in the retrieved chunks. Used to compute
            v1 refusal rate.
        latency_seconds: wall-clock seconds spent in .generate(); enables
            per-query latency tracking for v1 / v2 narrative analysis.
    """

    text: str
    cited_chunk_ids: tuple[str, ...]
    refused: bool
    latency_seconds: float


@dataclass(frozen=True)
class RawCompletion:
    """Output of Generator.generate_raw() — ungrounded text completion.

    Fields:
        text: continuation text the model produced for the prompt.
        latency_seconds: wall-clock seconds spent generating.
    """

    text: str
    latency_seconds: float


# ------------------------------------------------------------------
# Public ABC
# ------------------------------------------------------------------


class Generator(ABC):
    """Pluggable generator backend (Qwen / SmolLM / TinyDocs / ...).

    Subclasses must override `generate()`. The CLI and eval code depend
    only on this interface, so swapping backends is a one-line change.

    Instance attributes (subclasses set these in `__init__` so the eval
    pipeline can record decoding params per run):
        temperature, top_p, max_new_tokens.
    """

    temperature: float
    top_p: float
    max_new_tokens: int

    @abstractmethod
    def generate(
        self,
        query: str,
        retrieved_chunks: list[Chunk],
        *,
        query_type: QueryType | None = None,
        stream: bool = False,
    ) -> Answer:
        """Generate an Answer grounded in `retrieved_chunks`.

        Args:
            query: raw user query.
            retrieved_chunks: top-K chunks from the retrieval layer
                (router → SymbolIndex / BM25 / future hybrid).
            query_type: optional classifier hint from the eval schema or
                router.classify(); prompt template selection depends on
                this. None means the backend may fall back to a generic
                template.
            stream: if True, the backend may stream tokens (CLI v1 wires
                this to print partial output). Tests and eval keep False.

        Returns:
            Answer dataclass with text + citations + refusal flag + latency.
        """

    def generate_raw(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> RawCompletion:
        """Continue ``prompt`` as plain text — no retrieval, no grounding.

        Used by the v4 web-UI playground tab to show off a model's raw
        text-generation behaviour (e.g. an SFT-light TinyDocs that
        cannot follow instructions still has interesting LM continuations).

        Default raises NotImplementedError. Subclasses opt in by
        overriding; the eval pipeline only ever calls `generate()`.

        Args:
            prompt: raw text to continue from. No system message, no
                chat template wrapping (each backend decides whether
                its underlying API expects a chat or completion call).
            max_tokens: decode budget.
            temperature: 0.0 = greedy. > 0 enables sampling.

        Returns:
            ``RawCompletion(text, latency_seconds)``. ``text`` is the
            model's continuation only — the prompt is NOT echoed back.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement generate_raw()")
