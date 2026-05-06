"""Tests for python_doc_assistant.generation.qwen_backend.

Hermetic — no real Qwen model loaded. Strategy:
  - For pipeline tests: subclass QwenGenerator with `_StubQwenGenerator`,
    bypass the parent __init__, override `_call_model` to return canned
    text. Exercises build-prompt -> call-model -> parse -> Answer end-to-end
    without touching transformers / torch.
  - For `_detect_device` tests: monkeypatch torch availability flags;
    skipped with `pytest.importorskip` when torch is not installed.
  - For DI test: pass plain stub objects so __init__ skips from_pretrained.
"""

from __future__ import annotations

import pytest

from python_doc_assistant.generation.interface import Answer, Generator
from python_doc_assistant.generation.qwen_backend import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MODEL_ID,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    QwenGenerator,
)
from python_doc_assistant.ingest.chunker import Chunk
from python_doc_assistant.prompts.grounded import (
    QUERY_TYPE_STRUCTURE,
    REFUSAL_MARKER,
)
from python_doc_assistant.retrieval.router import QueryType

# ------------------------------------------------------------------
# Fixture builders
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


class _StubQwenGenerator(QwenGenerator):
    """Test double — bypasses transformers loading, returns canned text.

    Captures the last messages seen by `_call_model` so tests can assert on it.
    """

    def __init__(self, canned_response: str = "stub answer") -> None:
        # Deliberately skip QwenGenerator.__init__ — no model load.
        self.model_id = DEFAULT_MODEL_ID
        self.max_new_tokens = DEFAULT_MAX_NEW_TOKENS
        self.temperature = DEFAULT_TEMPERATURE
        self.top_p = DEFAULT_TOP_P
        self.device = "cpu"
        self.tokenizer = None
        self.model = None
        self._canned = canned_response
        self.last_prompt: list[dict[str, str]] | None = None

    def _call_model(self, prompt: list[dict[str, str]]) -> str:  # type: ignore[override]
        self.last_prompt = prompt
        return self._canned


class _StubTokenizer:
    """Used only for the DI smoke test — never actually called."""


class _StubModel:
    """Used only for the DI smoke test — never actually called."""

    def to(self, device: str) -> "_StubModel":
        return self


# ------------------------------------------------------------------
# Module constants
# ------------------------------------------------------------------


def test_default_constants_match_plan() -> None:
    """Plan §3 + greedy decoding decision."""
    assert DEFAULT_MODEL_ID == "Qwen/Qwen2.5-1.5B-Instruct"
    assert DEFAULT_MAX_NEW_TOKENS == 512
    assert DEFAULT_TEMPERATURE == 0.0
    assert DEFAULT_TOP_P == 1.0


# ------------------------------------------------------------------
# Inheritance
# ------------------------------------------------------------------


def test_qwen_is_a_generator_subclass() -> None:
    assert issubclass(QwenGenerator, Generator)


# ------------------------------------------------------------------
# generate() pipeline (via stub)
# ------------------------------------------------------------------


def test_generate_returns_answer_with_text_and_citations() -> None:
    """[N] numeric citations are mapped back to chunk_ids using retrieved order."""
    gen = _StubQwenGenerator(canned_response="Use [1] then [2].")
    answer = gen.generate("how to foo?", [_chunk("symbol:foo"), _chunk("symbol:bar")])
    assert isinstance(answer, Answer)
    assert "[1]" in answer.text
    assert answer.cited_chunk_ids == ("symbol:foo", "symbol:bar")
    assert answer.refused is False


def test_generate_drops_out_of_range_citations() -> None:
    """Model may emit [99] when only 2 chunks exist — silently drop those."""
    gen = _StubQwenGenerator(canned_response="See [1] and [99].")
    answer = gen.generate("q", [_chunk("symbol:foo"), _chunk("symbol:bar")])
    assert answer.cited_chunk_ids == ("symbol:foo",)


def test_generate_no_citations_yields_empty_chunk_ids() -> None:
    """Model emits prose with no [N] markers — cited_chunk_ids is empty."""
    gen = _StubQwenGenerator(canned_response="A plain answer with no citations.")
    answer = gen.generate("q", [_chunk("symbol:foo")])
    assert answer.cited_chunk_ids == ()


def test_generate_refusal_yields_empty_text() -> None:
    """REFUSAL_MARKER in raw → refused=True, text='' per Answer contract."""
    gen = _StubQwenGenerator(canned_response=f"Not in docs.\n{REFUSAL_MARKER}")
    answer = gen.generate("q", [_chunk("c1")])
    assert answer.refused is True
    assert answer.text == ""
    assert REFUSAL_MARKER not in answer.text


def test_generate_refusal_clears_cited_chunk_ids() -> None:
    """Refused answer must have empty cited_chunk_ids even when model leaks
    a [N] marker alongside the refusal (e.g. '[1] [INSUFFICIENT-CONTEXT]').

    Catches a Codex review finding from the v2 dense+rerank+qwen run
    where two refused rows still carried non-empty cited_chunk_ids.
    """
    gen = _StubQwenGenerator(canned_response=f"[1] {REFUSAL_MARKER}")
    answer = gen.generate("q", [_chunk("c1"), _chunk("c2")])
    assert answer.refused is True
    assert answer.text == ""
    assert answer.cited_chunk_ids == ()


def test_generate_records_nonneg_latency() -> None:
    gen = _StubQwenGenerator(canned_response="ok")
    answer = gen.generate("q", [_chunk("c1")])
    assert answer.latency_seconds >= 0.0


def test_generate_passes_query_type_to_prompt() -> None:
    """query_type must reach build_grounded_prompt — verify via captured prompt."""
    gen = _StubQwenGenerator(canned_response="ok")
    gen.generate("q", [_chunk("c1")], query_type=QueryType.IDENTIFIER)
    assert gen.last_prompt is not None
    assert QUERY_TYPE_STRUCTURE[QueryType.IDENTIFIER] in gen.last_prompt[1]["content"]


def test_generate_includes_query_in_prompt() -> None:
    gen = _StubQwenGenerator(canned_response="ok")
    gen.generate("how to read a file?", [_chunk("c1")])
    assert gen.last_prompt is not None
    assert "how to read a file?" in gen.last_prompt[1]["content"]


def test_generate_includes_chunk_markers_in_prompt() -> None:
    """Numeric [N] markers in CONTEXT, not chunk_ids."""
    gen = _StubQwenGenerator(canned_response="ok")
    gen.generate("q", [_chunk("symbol:foo"), _chunk("symbol:bar")])
    assert gen.last_prompt is not None
    user_content = gen.last_prompt[1]["content"]
    assert "[1]" in user_content
    assert "[2]" in user_content


def test_generate_handles_empty_chunks() -> None:
    """Empty retrieval still produces a valid prompt + Answer (model usually refuses)."""
    gen = _StubQwenGenerator(canned_response=REFUSAL_MARKER)
    answer = gen.generate("q", [])
    assert answer.refused is True
    assert answer.text == ""


def test_generate_stream_true_raises_notimplemented() -> None:
    """v1 §3 only implements stream=False; CLI §5 will revisit streaming."""
    gen = _StubQwenGenerator()
    with pytest.raises(NotImplementedError):
        gen.generate("q", [_chunk("c1")], stream=True)


# ------------------------------------------------------------------
# _detect_device (needs torch, gracefully skipped if absent)
# ------------------------------------------------------------------


def test_detect_device_returns_known_value() -> None:
    pytest.importorskip("torch")
    device = QwenGenerator._detect_device()
    assert device in {"cuda", "mps", "cpu"}


def test_detect_device_prefers_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("torch")
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert QwenGenerator._detect_device() == "cuda"


def test_detect_device_falls_back_to_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("torch")
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert QwenGenerator._detect_device() == "mps"


def test_detect_device_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("torch")
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert QwenGenerator._detect_device() == "cpu"


# ------------------------------------------------------------------
# __init__ DI smoke (no transformers required)
# ------------------------------------------------------------------


def test_init_with_di_skips_from_pretrained() -> None:
    """Both tokenizer + model passed → no from_pretrained / network call."""
    tok = _StubTokenizer()
    mdl = _StubModel()
    gen = QwenGenerator(tokenizer=tok, model=mdl, device="cpu")
    assert gen.tokenizer is tok
    assert gen.model is mdl
    assert gen.device == "cpu"
    assert gen.model_id == DEFAULT_MODEL_ID
    assert gen.max_new_tokens == DEFAULT_MAX_NEW_TOKENS
    assert gen.temperature == DEFAULT_TEMPERATURE
    assert gen.top_p == DEFAULT_TOP_P


def test_init_overrides_decoding_params() -> None:
    gen = QwenGenerator(
        tokenizer=_StubTokenizer(),
        model=_StubModel(),
        device="cpu",
        max_new_tokens=128,
        temperature=0.0,
        top_p=1.0,
    )
    assert gen.max_new_tokens == 128
    assert gen.temperature == 0.0
    assert gen.top_p == 1.0
