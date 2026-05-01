"""Tests for python_doc_assistant.generation.tinydocs_backend.

Hermetic — no real TinyDocs model loaded. Strategy:
  - For pipeline tests: subclass TinyDocsGenerator with `_StubTinyDocsGenerator`,
    bypass the parent __init__, override `_call_model` to return canned
    text. Exercises build-prompt → flatten → call-model → parse → Answer
    end-to-end without touching torch.
  - For `_detect_device` tests: monkeypatch torch availability flags;
    skipped with `pytest.importorskip` when torch is not installed.
  - For DI test: pass plain stub objects so __init__ skips file loading.
"""

from __future__ import annotations

import pytest

from python_doc_assistant.generation.interface import Answer, Generator
from python_doc_assistant.generation.tinydocs_backend import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
    TinyDocsGenerator,
)
from python_doc_assistant.ingest.chunker import Chunk
from python_doc_assistant.prompts.grounded import REFUSAL_MARKER

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


class _StubTinyDocsGenerator(TinyDocsGenerator):
    """Test double — bypasses checkpoint loading, returns canned text.

    Captures the last prompt_text seen by `_call_model` so tests can
    assert on it.
    """

    def __init__(self, canned_response: str = "stub answer") -> None:
        # Deliberately skip TinyDocsGenerator.__init__ — no model load.
        self.max_new_tokens = DEFAULT_MAX_NEW_TOKENS
        self.temperature = DEFAULT_TEMPERATURE
        self.device = "cpu"
        self.tokenizer = None
        self.model = None
        self.model_max_seq_len = 256
        self._canned = canned_response
        self.last_prompt_text: str | None = None

    def _call_model(self, prompt_text: str) -> str:  # type: ignore[override]
        self.last_prompt_text = prompt_text
        return self._canned


class _StubTokenizer:
    """Used only for the DI smoke test — never actually called."""


class _StubModel:
    """Used only for the DI smoke test — never actually called."""


# ------------------------------------------------------------------
# Module constants
# ------------------------------------------------------------------


def test_default_constants_match_plan() -> None:
    """Plan §6 — greedy MVP, conservative max_new_tokens budget."""
    assert DEFAULT_MAX_NEW_TOKENS == 64
    assert DEFAULT_TEMPERATURE == 0.0


# ------------------------------------------------------------------
# ABC conformance
# ------------------------------------------------------------------


def test_is_generator_subclass() -> None:
    """TinyDocsGenerator must be a Generator (so CLI / eval can swap)."""
    assert issubclass(TinyDocsGenerator, Generator)


# ------------------------------------------------------------------
# generate() — pipeline glue (using stub _call_model)
# ------------------------------------------------------------------


def test_generate_returns_answer_with_text() -> None:
    """Happy path: canned text → Answer.text echoed; not refused."""
    gen = _StubTinyDocsGenerator(canned_response="hello world")
    answer = gen.generate("q", [_chunk("symbol:a")])
    assert isinstance(answer, Answer)
    assert answer.text == "hello world"
    assert answer.refused is False
    assert answer.cited_chunk_ids == ()
    assert answer.latency_seconds >= 0.0


def test_generate_records_latency() -> None:
    """latency_seconds must be a finite non-negative float."""
    gen = _StubTinyDocsGenerator(canned_response="x")
    answer = gen.generate("q", [_chunk()])
    assert answer.latency_seconds >= 0.0


def test_generate_passes_flattened_text_to_call_model() -> None:
    """generate() must flatten messages and hand a string to _call_model."""
    gen = _StubTinyDocsGenerator(canned_response="ok")
    gen.generate("how to use Path?", [_chunk("symbol:pathlib.Path")])
    assert gen.last_prompt_text is not None
    # Flattened concat — must not still be a list[dict].
    assert isinstance(gen.last_prompt_text, str)
    # Question made it into the flattened prompt.
    assert "how to use Path?" in gen.last_prompt_text


def test_generate_maps_citation_to_chunk_id() -> None:
    """`[1]` in raw → mapped to retrieved_chunks[0].chunk_id."""
    gen = _StubTinyDocsGenerator(canned_response="see [1]")
    chunks = [_chunk("symbol:a"), _chunk("symbol:b")]
    answer = gen.generate("q", chunks)
    assert answer.cited_chunk_ids == ("symbol:a",)
    assert answer.refused is False


def test_generate_drops_out_of_range_citations() -> None:
    """`[99]` with only 2 chunks must be dropped silently."""
    gen = _StubTinyDocsGenerator(canned_response="see [99]")
    answer = gen.generate("q", [_chunk("symbol:a"), _chunk("symbol:b")])
    assert answer.cited_chunk_ids == ()


def test_generate_refusal_returns_empty_text_and_no_citations() -> None:
    """REFUSAL_MARKER → refused=True, text='', cited=() per Answer contract."""
    gen = _StubTinyDocsGenerator(canned_response=f"Not in docs.\n{REFUSAL_MARKER}")
    answer = gen.generate("q", [_chunk()])
    assert answer.refused is True
    assert answer.text == ""
    assert answer.cited_chunk_ids == ()


def test_generate_refusal_with_citation_drops_citation() -> None:
    """Even if the model emits `[1]` alongside refusal, contract zeros it."""
    gen = _StubTinyDocsGenerator(canned_response=f"[1] {REFUSAL_MARKER}")
    answer = gen.generate("q", [_chunk("symbol:a")])
    assert answer.refused is True
    assert answer.text == ""
    assert answer.cited_chunk_ids == ()


def test_generate_stream_true_raises() -> None:
    """stream=True is not implemented in §6 MVP."""
    gen = _StubTinyDocsGenerator()
    with pytest.raises(NotImplementedError):
        gen.generate("q", [_chunk()], stream=True)


# ------------------------------------------------------------------
# _flatten_messages
# ------------------------------------------------------------------


def test_flatten_messages_concatenates_contents_with_blank_line() -> None:
    """Messages collapsed into one string; both contents present in order."""
    flat = TinyDocsGenerator._flatten_messages(
        [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "USR"},
        ]
    )
    assert "SYS" in flat
    assert "USR" in flat
    # System content precedes user content in the flattened string.
    assert flat.index("SYS") < flat.index("USR")


def test_flatten_messages_drops_role_markers() -> None:
    """No `<system>` / `<user>` style markup in the flattened text."""
    flat = TinyDocsGenerator._flatten_messages(
        [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "USR"},
        ]
    )
    assert "system" not in flat.lower() or "system" in "SYS".lower()
    # Stricter: no role-tag markup characters from common templates.
    assert "<system>" not in flat
    assert "<|im_start|>" not in flat


# ------------------------------------------------------------------
# DI smoke test — __init__ skips file loading when both objects given
# ------------------------------------------------------------------


def test_init_with_di_skips_file_loading() -> None:
    """Both tokenizer and model provided → no checkpoint / tokenizer path needed."""
    tok = _StubTokenizer()
    model = _StubModel()
    gen = TinyDocsGenerator(tokenizer=tok, model=model, device="cpu")
    assert gen.tokenizer is tok
    assert gen.model is model
    assert gen.device == "cpu"


def test_init_requires_both_di_or_neither() -> None:
    """Asymmetric DI (only tokenizer or only model) is a usage error."""
    with pytest.raises(ValueError):
        TinyDocsGenerator(tokenizer=_StubTokenizer(), model=None, device="cpu")
    with pytest.raises(ValueError):
        TinyDocsGenerator(tokenizer=None, model=_StubModel(), device="cpu")


# ------------------------------------------------------------------
# _detect_device
# ------------------------------------------------------------------


def test_detect_device_prefers_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert TinyDocsGenerator._detect_device() == "cuda"


def test_detect_device_falls_back_to_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert TinyDocsGenerator._detect_device() == "mps"


def test_detect_device_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert TinyDocsGenerator._detect_device() == "cpu"
