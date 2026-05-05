"""Tests for python_doc_assistant.generation.qwen_gguf_backend.

Hermetic — no real GGUF model loaded. Strategy:
  - DI path: pass a mock `llm` object so __init__ skips Llama() construction.
  - Pipeline: subclass QwenGGUFGenerator and override `_call_model` to
    return canned text.
  - `_call_model` smoke: stub `llm.create_chat_completion` to return the
    OpenAI-shaped chat-completion dict; assert routing + return.
"""

from __future__ import annotations

import pytest

from python_doc_assistant.generation.interface import Answer, Generator
from python_doc_assistant.generation.qwen_gguf_backend import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_N_CTX,
    DEFAULT_N_GPU_LAYERS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    QwenGGUFGenerator,
)
from python_doc_assistant.ingest.chunker import Chunk
from python_doc_assistant.prompts.grounded import REFUSAL_MARKER

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


def _chunk(chunk_id: str = "symbol:foo") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        chunk_type="symbol",
        docs_version="3.12",
        title="foo",
        text="body",
        symbols=("foo",),
        canonical_url="library/foo.html#foo",
        anchor="foo",
        parent_module=None,
        source_path="library/foo.html",
        source_hash="sha256:abc",
    )


class _StubLlm:
    """Plain object captured as `llm` to skip Llama() construction in __init__.

    `create_chat_completion` is monkeypatched per-test when needed.
    """


class _StubGen(QwenGGUFGenerator):
    """Subclass that bypasses Llama() construction and stubs _call_model."""

    def __init__(self, canned: str = "stub answer") -> None:
        self.model_path = None
        self.max_new_tokens = DEFAULT_MAX_NEW_TOKENS
        self.temperature = DEFAULT_TEMPERATURE
        self.top_p = DEFAULT_TOP_P
        self.n_ctx = DEFAULT_N_CTX
        self.n_gpu_layers = DEFAULT_N_GPU_LAYERS
        self.llm = _StubLlm()
        self._canned = canned
        self.last_prompt: list[dict[str, str]] | None = None

    def _call_model(self, prompt: list[dict[str, str]]) -> str:  # type: ignore[override]
        self.last_prompt = prompt
        return self._canned


# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------


def test_default_constants() -> None:
    assert DEFAULT_MAX_NEW_TOKENS == 512
    assert DEFAULT_TEMPERATURE == 0.0
    assert DEFAULT_TOP_P == 1.0
    assert DEFAULT_N_CTX == 4096
    assert DEFAULT_N_GPU_LAYERS == -1


# ------------------------------------------------------------------
# __init__ argument checks
# ------------------------------------------------------------------


def test_init_requires_model_path_or_llm() -> None:
    with pytest.raises(ValueError, match="model_path or llm"):
        QwenGGUFGenerator()


def test_init_missing_model_path_raises_filenotfound(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        QwenGGUFGenerator(model_path=tmp_path / "does-not-exist.gguf")


def test_init_di_skips_llama_construction() -> None:
    """Providing llm bypasses Llama() — should not import llama_cpp at all."""
    gen = QwenGGUFGenerator(llm=_StubLlm())
    assert gen.llm is not None
    assert gen.model_path is None
    assert gen.n_ctx == DEFAULT_N_CTX
    assert gen.n_gpu_layers == DEFAULT_N_GPU_LAYERS


def test_init_is_a_generator_subclass() -> None:
    gen = QwenGGUFGenerator(llm=_StubLlm())
    assert isinstance(gen, Generator)


# ------------------------------------------------------------------
# generate() — pipeline assertions via _StubGen
# ------------------------------------------------------------------


def test_generate_returns_answer_dataclass() -> None:
    gen = _StubGen(canned="The answer is foo [1].")
    ans = gen.generate("question?", [_chunk("symbol:foo")])
    assert isinstance(ans, Answer)
    assert ans.text == "The answer is foo [1]."
    assert ans.refused is False
    assert ans.cited_chunk_ids == ("symbol:foo",)
    assert ans.latency_seconds >= 0.0


def test_generate_passes_chunks_into_prompt() -> None:
    gen = _StubGen(canned="ok [1]")
    gen.generate("question?", [_chunk("symbol:bar")])
    assert gen.last_prompt is not None
    # Two-message conversation: system + user.
    assert {m["role"] for m in gen.last_prompt} == {"system", "user"}
    user_msg = next(m for m in gen.last_prompt if m["role"] == "user")
    assert "question?" in user_msg["content"]


def test_generate_refusal_marker_yields_refused_true() -> None:
    gen = _StubGen(canned=REFUSAL_MARKER)
    ans = gen.generate("q?", [_chunk("symbol:x")])
    assert ans.refused is True
    assert ans.text == ""
    assert ans.cited_chunk_ids == ()


def test_generate_drops_out_of_range_citations() -> None:
    gen = _StubGen(canned="foo [1] bar [99]")
    chunks = [_chunk("symbol:a"), _chunk("symbol:b")]
    ans = gen.generate("q?", chunks)
    # [1] maps to symbol:a; [99] is out of range, dropped.
    assert ans.cited_chunk_ids == ("symbol:a",)


def test_generate_dedupes_citation_chunk_ids() -> None:
    gen = _StubGen(canned="foo [1] bar [2] baz [1]")
    chunks = [_chunk("symbol:a"), _chunk("symbol:b")]
    ans = gen.generate("q?", chunks)
    # parse_response collapses duplicate indices in order.
    assert ans.cited_chunk_ids == ("symbol:a", "symbol:b")


def test_generate_stream_true_raises() -> None:
    gen = _StubGen()
    with pytest.raises(NotImplementedError):
        gen.generate("q?", [_chunk()], stream=True)


# ------------------------------------------------------------------
# _call_model — stubbed Llama.create_chat_completion
# ------------------------------------------------------------------


def test_call_model_routes_to_create_chat_completion() -> None:
    """_call_model should forward messages to llm.create_chat_completion
    and return the assistant message content unchanged.
    """
    captured: dict[str, object] = {}

    class _LlmCapturing:
        def create_chat_completion(self, *, messages, max_tokens, temperature, top_p):
            captured["messages"] = messages
            captured["max_tokens"] = max_tokens
            captured["temperature"] = temperature
            captured["top_p"] = top_p
            return {"choices": [{"message": {"content": "hello world"}}]}

    gen = QwenGGUFGenerator(llm=_LlmCapturing())
    out = gen._call_model(
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ]
    )
    assert out == "hello world"
    assert captured["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]
    assert captured["max_tokens"] == DEFAULT_MAX_NEW_TOKENS
    assert captured["temperature"] == DEFAULT_TEMPERATURE
    assert captured["top_p"] == DEFAULT_TOP_P
