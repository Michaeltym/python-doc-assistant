"""Qwen2.5-Instruct backend for the Generator ABC.

See plans/v1-qwen-generator.md §3.

Eager-loads tokenizer + model in __init__ unless `tokenizer` / `model` are
injected (testing path). Auto-detects device (cuda > mps > cpu) when
`device` is None.

NOTE: `transformers` and `torch` are *lazily* imported inside the methods
that need them — importing this module without those packages installed
must NOT raise. (Tests rely on this; CI runs them without the `generation`
extra.)
"""

from __future__ import annotations

import time
from typing import Any, Final

from python_doc_assistant.generation.interface import Answer, Generator
from python_doc_assistant.ingest.chunker import Chunk
from python_doc_assistant.prompts.grounded import build_grounded_prompt, parse_response
from python_doc_assistant.retrieval.router import QueryType

# ------------------------------------------------------------------
# Defaults (plan §3 — greedy decoding for v1 grounded RAG)
# ------------------------------------------------------------------

DEFAULT_MODEL_ID: Final[str] = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_MAX_NEW_TOKENS: Final[int] = 512
# Greedy decoding (temperature=0, do_sample=False) — small instruct models
# follow strict format rules far more reliably under greedy than sampled.
DEFAULT_TEMPERATURE: Final[float] = 0.0
DEFAULT_TOP_P: Final[float] = 1.0


# ------------------------------------------------------------------
# Concrete generator
# ------------------------------------------------------------------


class QwenGenerator(Generator):
    """Qwen2.5-Instruct backed Generator (v1 default model).

    Attributes (set by __init__):
        model_id: HuggingFace id used to load weights.
        max_new_tokens / temperature / top_p: decoding params.
        device: "cuda" / "mps" / "cpu" — used by `_call_model`.
        tokenizer: PreTrainedTokenizerBase (or DI stub).
        model: PreTrainedModel (or DI stub).
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        device: str | None = None,
        tokenizer: Any = None,
        model: Any = None,
    ) -> None:
        """Construct + eager-load model.

        Args:
            model_id: HuggingFace model identifier.
            max_new_tokens / temperature / top_p: decoding params.
            device: explicit device string. None → `_detect_device()`.
            tokenizer / model: dependency injection for tests. If BOTH are
                provided, skip `from_pretrained` entirely (treat them as
                already loaded; do NOT call `.to(device)` on them).
                Otherwise call `AutoTokenizer.from_pretrained(model_id)`
                and `AutoModelForCausalLM.from_pretrained(model_id)`, then
                move the model to `device`.

        Stores: self.model_id / max_new_tokens / temperature / top_p /
                device / tokenizer / model.
        """
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.device = device if device is not None else self._detect_device()
        if tokenizer is None and model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto").to(
                self.device  # type: ignore[arg-type]
            )
        elif tokenizer is not None and model is not None:
            self.tokenizer = tokenizer
            self.model = model
        else:
            raise ValueError("tokenizer and model must both be None or both provided")

    def generate(
        self,
        query: str,
        retrieved_chunks: list[Chunk],
        *,
        query_type: QueryType | None = None,
        stream: bool = False,
    ) -> Answer:
        """Run the grounded-generation pipeline.

        Steps:
            1. build_grounded_prompt(query, retrieved_chunks, query_type=...)
            2. raw = self._call_model(prompt)
            3. parsed = parse_response(raw)
            4. Map parsed.cited_indices ([N] numbers) back to chunk_ids using
               retrieved_chunks order; drop indices out of [1, len].
            5. Wrap in Answer:
                 - text = "" if parsed.refused else parsed.text
                   (interface.py contract: refused ⇒ empty text)
                 - cited_chunk_ids = mapped chunk_ids from step 4
                 - refused = parsed.refused
                 - latency_seconds = wall-clock seconds spent in steps 2 + 3

        `stream=True` is not implemented in v1 §3 → raise NotImplementedError
        (CLI §5 will revisit streaming).
        """
        if stream:
            raise NotImplementedError

        prompt = build_grounded_prompt(query, retrieved_chunks, query_type=query_type)
        start_time = time.perf_counter()
        raw_text = self._call_model(prompt)
        parsed_response = parse_response(raw_text)
        # Map 1-indexed [N] citations back to chunk_ids; drop out-of-range
        # numbers the model may emit (e.g. [99] when only 5 chunks exist).
        cited_chunk_ids = tuple(
            retrieved_chunks[i - 1].chunk_id
            for i in parsed_response.cited_indices
            if 1 <= i <= len(retrieved_chunks)
        )
        return Answer(
            text="" if parsed_response.refused else parsed_response.text,
            cited_chunk_ids=() if parsed_response.refused else cited_chunk_ids,
            refused=parsed_response.refused,
            latency_seconds=time.perf_counter() - start_time,
        )

    def _call_model(self, prompt: list[dict[str, str]]) -> str:
        """Apply chat template to messages, run model.generate, decode new tokens.

        `prompt` is the message list returned by `build_grounded_prompt`
        (system + user roles). Tests subclass QwenGenerator and override
        this method to return canned text (no transformers call).
        """
        text = self.tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(  # pyright: ignore[reportAttributeAccessIssue]
            **model_inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=self.temperature > 0,
        )
        input_len = model_inputs.input_ids.shape[1]
        new_tokens = generated_ids[0][input_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)  # type: ignore[return-value]

    @staticmethod
    def _detect_device() -> str:
        """Pick the best available device. Order: cuda > mps > cpu.

        Lazy-imports torch (so importing this module without torch works).
        """
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
