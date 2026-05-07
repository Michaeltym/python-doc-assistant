"""TinyDocs backend for the Generator ABC.

See plans/v3-tiny-llm.md §6.

Wraps a self-trained TinyDocs checkpoint behind the same Generator
interface used by QwenGenerator. The base LM is not chat-tuned and
uses a custom BPE tokenizer, so the chat-template messages from
build_grounded_prompt are flattened to plain text before encoding.
KV cache enables incremental decoding.

NOTE: torch is *lazily* imported inside the methods that need it —
importing this module without torch installed must NOT raise. (Tests
rely on this; CI runs them without the `tinydocs` extra.)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Final

from python_doc_assistant.generation.interface import Answer, Generator, RawCompletion
from python_doc_assistant.generation.tinydocs.config import TinyDocsConfig
from python_doc_assistant.ingest.chunker import Chunk
from python_doc_assistant.prompts.grounded import build_grounded_prompt, parse_response
from python_doc_assistant.retrieval.router import QueryType

# ------------------------------------------------------------------
# Defaults (plan §6 — greedy MVP, prompt+answer fits in trained max_seq_len)
# ------------------------------------------------------------------

DEFAULT_MAX_NEW_TOKENS: Final[int] = 64
# Greedy only — v3 §6 MVP: pipeline-runs criterion. Sampling deferred.
DEFAULT_TEMPERATURE: Final[float] = 0.0


# ------------------------------------------------------------------
# Concrete generator
# ------------------------------------------------------------------


class TinyDocsGenerator(Generator):
    """TinyDocs (self-trained) backed Generator.

    The base LM is not chat-tuned and uses a custom BPE tokenizer, so
    the prompt is flattened to plain text and encoded with
    TinyDocsTokenizer. Generation is greedy with KV cache.

    Attributes (set by __init__):
        max_new_tokens / temperature: decoding params.
        device: "cuda" / "mps" / "cpu" — used by `_call_model`.
        tokenizer: TinyDocsTokenizer (or DI stub).
        model: TinyDocsModel (or DI stub).
        model_max_seq_len: int — the trained model's max sequence length;
            prompt is truncated so len(prompt) + max_new_tokens fits.
    """

    def __init__(
        self,
        checkpoint_path: Path | str | None = None,
        tokenizer_path: Path | str | None = None,
        *,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        device: str | None = None,
        tokenizer: Any = None,
        model: Any = None,
    ) -> None:
        """Construct + load checkpoint and tokenizer.

        Args:
            checkpoint_path: Path to a step_<N>.pt produced by
                `train_tinydocs`. The dict must contain `model_state` and
                `model_config` (asdict TinyDocsConfig).
            tokenizer_path: Path to a saved TinyDocsTokenizer JSON.
            max_new_tokens: decode budget. Constrained by the trained
                `max_seq_len`; prompt is truncated so that
                len(prompt) + max_new_tokens <= model_max_seq_len.
            temperature: reserved; v3 §6 MVP does greedy only (must be 0.0).
            device: explicit torch device. None → `_detect_device()`.
            tokenizer / model: dependency injection for tests. If BOTH are
                provided, skip checkpoint / tokenizer file loading entirely
                (treat them as already loaded; do NOT call .to(device) on
                them). Otherwise BOTH paths must be provided and are
                loaded eagerly:
                    - TinyDocsTokenizer.load(tokenizer_path)
                    - torch.load(checkpoint_path) → dict; rebuild
                      TinyDocsModel(TinyDocsConfig(**ckpt["model_config"])),
                      load_state_dict(ckpt["model_state"]), .to(device),
                      .eval()

        Stores: self.max_new_tokens / temperature / device / tokenizer /
                model / model_max_seq_len.
        """
        if (tokenizer is None) != (model is None):
            raise ValueError("tokenizer and model must both be None or both provided")
        if (checkpoint_path is None) != (tokenizer_path is None):
            raise ValueError(
                "checkpoint_path and tokenizer_path must both be None or both provided"
            )

        has_di = tokenizer is not None and model is not None
        has_paths = checkpoint_path is not None and tokenizer_path is not None
        if has_di == has_paths:
            raise ValueError("Provide EITHER DI OR paths, not both / neither")

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = device or self._detect_device()
        if tokenizer is not None and model is not None:  # ← Pyright narrows tokenizer/model here
            self.tokenizer = tokenizer
            self.model = model
            default_config = TinyDocsConfig()
            self.model_max_seq_len = default_config.max_seq_len  # not needed for DI stubs
        elif checkpoint_path is not None and tokenizer_path is not None:  # ← narrows paths
            import torch

            from python_doc_assistant.generation.tinydocs.model import TinyDocsModel
            from python_doc_assistant.generation.tinydocs.tokenizer import TinyDocsTokenizer

            self.tokenizer = TinyDocsTokenizer.load(Path(tokenizer_path))
            checkpoint = torch.load(Path(checkpoint_path), weights_only=False, map_location="cpu")
            model_config = checkpoint["model_config"]
            self.model_max_seq_len = model_config["max_seq_len"]
            self.model = TinyDocsModel(TinyDocsConfig(**model_config))
            self.model.load_state_dict(checkpoint["model_state"])
            self.model.to(self.device)
            self.model.eval()

    def generate(
        self,
        query: str,
        retrieved_chunks: list[Chunk],
        *,
        query_type: QueryType | None = None,
        stream: bool = False,
    ) -> Answer:
        """Run grounded generation through the TinyDocs base LM.

        Steps:
            1. build_grounded_prompt(query, retrieved_chunks, query_type=...)
            2. text = self._flatten_messages(messages)
            3. raw = self._call_model(text)  — wall-clock timed
            4. parsed = parse_response(raw)
            5. Map parsed.cited_indices to chunk_ids using
               retrieved_chunks order; drop indices outside [1, len].
            6. Wrap in Answer:
                 - text = "" if parsed.refused else parsed.text
                   (interface.py contract: refused ⇒ empty text)
                 - cited_chunk_ids = mapped chunk_ids from step 5
                 - refused = parsed.refused
                 - latency_seconds = wall-clock seconds spent in steps 3 + 4

        `stream=True` is not implemented → raise NotImplementedError.

        Note: the base LM is not instruction-tuned; expect parsed.refused
        almost always False and parsed.cited_indices almost always empty
        because the model will not naturally emit `[N]` or REFUSAL_MARKER.
        That is fine for MVP — §6 only requires the pipeline to run.
        """
        if stream:
            raise NotImplementedError("Streaming generation not implemented for TinyDocsGenerator.")
        prompt = build_grounded_prompt(query, retrieved_chunks, query_type=query_type)
        text = self._flatten_messages(prompt)
        start_time = time.perf_counter()
        raw = self._call_model(text)
        parsed = parse_response(raw)
        cited_indices = parsed.cited_indices
        cited_chunk_ids = [
            retrieved_chunks[idx - 1].chunk_id
            for idx in cited_indices
            if 1 <= idx <= len(retrieved_chunks)
        ]
        return Answer(
            text="" if parsed.refused else parsed.text,
            cited_chunk_ids=tuple(cited_chunk_ids) if not parsed.refused else (),
            refused=parsed.refused,
            latency_seconds=time.perf_counter() - start_time,
        )

    def _call_model(self, prompt_text: str) -> str:
        """Encode prompt, run greedy decode loop, decode new tokens.

        Encoding rules:
            - tokenizer.encode(prompt_text, add_bos=True, add_eos=False)
            - if len(encoded) > self.model_max_seq_len - self.max_new_tokens,
              keep the LAST (model_max_seq_len - max_new_tokens) tokens
              (tail-biased truncation: question sits at the end of the
              flattened prompt, so the tail is the most informative slice).

        Decode loop is delegated to `_decode_loop`. Decoding the new
        token id list back to a string uses tokenizer.decode (which
        skips bos/eos/pad per TinyDocsTokenizer contract).
        """
        encoded = self.tokenizer.encode(prompt_text, add_bos=True, add_eos=False)
        keep = self.model_max_seq_len - self.max_new_tokens
        if len(encoded) > keep:
            encoded = encoded[-keep:]
        result: str = self.tokenizer.decode(self._decode_loop(encoded))
        return result

    def _decode_loop(self, input_ids: Any) -> list[int]:
        """Greedy autoregressive decode with KV cache.

        Args:
            input_ids: torch.LongTensor of shape (1, T_in).

        Steps:
            1. Prefill: model(input_ids, caches=None, position=0) →
               (logits, caches). logits shape (1, T_in, vocab).
            2. next_token = logits[:, -1, :].argmax(dim=-1)  # (1,)
               If next_token == eos_id, return [].
            3. Loop until eos_id emitted OR len(generated) == max_new_tokens:
                 model(next_token[:, None], caches=caches,
                       position=T_in + len(generated))
                 → logits (1, 1, vocab); argmax for next.
                 Append (excluding the trailing eos) to `generated`.
            4. Return list[int] of generated token ids (prompt excluded;
               trailing eos excluded if emitted).

        Runs under torch.no_grad().
        """
        import torch

        with torch.no_grad():
            input_tensor = torch.tensor(input_ids, dtype=torch.long, device=self.device).unsqueeze(
                0
            )
            logits, caches = self.model(input_tensor)
            next_token = logits[:, -1, :].argmax(dim=-1)
            if next_token.item() == self.tokenizer.eos_id:
                return []
            generated = [next_token.item()]
            for position in range(self.max_new_tokens - 1):
                logits, caches = self.model(
                    next_token[:, None], caches=caches, position=position + len(input_ids)
                )
                next_token = logits[:, -1, :].argmax(dim=-1)
                if next_token.item() == self.tokenizer.eos_id:
                    break
                generated.append(next_token.item())
            return generated

    def generate_raw(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> RawCompletion:
        """Plain-text continuation: tokenizer.encode → _decode_loop → decode.

        Bypasses the grounded prompt entirely so the v4 web-UI playground
        can show off raw LM continuations. Useful precisely because the
        v3.1 TinyDocs checkpoint is base-LM-quality and not yet
        instruction-tuned — the playground exposes the underlying
        language modelling without the grounded RAG mask.

        Implementation outline:
            1. start = time.perf_counter()
            2. encoded = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
            3. budget = self.model_max_seq_len - max_tokens
               if len(encoded) > budget:
                   encoded = encoded[-budget:]
            4. previous_max = self.max_new_tokens
               previous_temp = self.temperature
               self.max_new_tokens = max_tokens
               self.temperature = temperature
               try:
                   ids = self._decode_loop(encoded)
               finally:
                   self.max_new_tokens = previous_max
                   self.temperature = previous_temp
            5. text = self.tokenizer.decode(ids)
            6. return RawCompletion(text=text, latency_seconds=time.perf_counter() - start)

        Note: temperature > 0 is currently a no-op because `_decode_loop`
        is greedy (argmax). A follow-up wires sampling through; the
        signature accepts the parameter so the playground UI can already
        send it.
        """
        start = time.perf_counter()
        encoded = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        budget = self.model_max_seq_len - max_tokens
        if len(encoded) > budget:
            encoded = encoded[-budget:]
        previous_max = self.max_new_tokens
        previous_temp = self.temperature
        self.max_new_tokens = max_tokens
        self.temperature = temperature
        try:
            ids = self._decode_loop(encoded)
        finally:
            self.max_new_tokens = previous_max
            self.temperature = previous_temp
        text = self.tokenizer.decode(ids)
        return RawCompletion(text=text, latency_seconds=time.perf_counter() - start)

    @staticmethod
    def _flatten_messages(messages: list[dict[str, str]]) -> str:
        """Flatten chat-template messages into plain text.

        Concat policy (MVP §6 Option A — simplest):
            - Concatenate every message's `content` joined with "\n\n"
            - Roles dropped (no `<system>` markup)
            - Trailing newline appended

        Why no markup: the base LM is not chat-tuned, so role tokens
        carry no learned signal. Plain concat is the lightest scaffold;
        future SFT could revisit.
        """
        return "\n\n".join(m["content"] for m in messages) + "\n"

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
