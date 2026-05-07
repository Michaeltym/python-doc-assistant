"""Qwen GGUF backend (llama.cpp) for the Generator ABC.

See `plans/v4-prod-ready.md` (Revision 2026-05-05 — Qwen-only path),
sub-task 3' "Qwen 7B GGUF backend".

Wraps `llama-cpp-python`'s `Llama` around the same Generator ABC the
v1 Qwen-transformers backend uses. Reuses `build_grounded_prompt`
and `parse_response`. The ABC contract (`Answer.text`,
`cited_chunk_ids`, `refused`, `latency_seconds`) is identical so the
v0/v1/v2 eval pipeline can swap backends transparently.

NOTE: `llama_cpp` is *lazily* imported inside the methods that need
it — importing this module without the package installed must NOT
raise. (Tests rely on this; CI runs them without the `llama-cpp`
extra.)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Final

from python_doc_assistant.generation.interface import Answer, Generator, RawCompletion
from python_doc_assistant.ingest.chunker import Chunk
from python_doc_assistant.prompts.grounded import build_grounded_prompt, parse_response
from python_doc_assistant.retrieval.router import QueryType

# ------------------------------------------------------------------
# Defaults (tuned for M1 Pro 16 GB on Qwen2.5-7B-Instruct Q4_K_M)
# ------------------------------------------------------------------

DEFAULT_MAX_NEW_TOKENS: Final[int] = 512
DEFAULT_TEMPERATURE: Final[float] = 0.0
DEFAULT_TOP_P: Final[float] = 1.0
DEFAULT_N_CTX: Final[int] = 8192
DEFAULT_N_GPU_LAYERS: Final[int] = -1  # -1 = offload all layers to Metal


class QwenGGUFGenerator(Generator):
    """Qwen GGUF (llama.cpp) backed Generator.

    Attributes (set by __init__):
        model_path: Path to the GGUF model file (first shard if multi-shard;
            llama.cpp resolves the rest by suffix).
        max_new_tokens / temperature / top_p: decoding params.
        n_ctx: context window in tokens.
        n_gpu_layers: layers to offload to GPU (Metal on M1). -1 = all.
        llm: `llama_cpp.Llama` instance (or DI stub for tests).
    """

    def __init__(
        self,
        model_path: Path | str | None = None,
        *,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        n_ctx: int = DEFAULT_N_CTX,
        n_gpu_layers: int = DEFAULT_N_GPU_LAYERS,
        llm: Any = None,
    ) -> None:
        """Construct + eager-load llama.cpp model.

        Args:
            model_path: Path to the GGUF file. Required when `llm` is
                None; ignored when `llm` is provided (DI path).
            max_new_tokens: decode budget.
            temperature: 0.0 = greedy. > 0 enables sampling.
            top_p: nucleus sampling threshold (only used when
                temperature > 0).
            n_ctx: total context window the underlying llama.cpp model
                is allocated. Prompt + answer must fit here.
            n_gpu_layers: forwarded to `Llama(n_gpu_layers=...)`. -1
                offloads every layer to Metal on M1.
            llm: dependency injection for tests. If provided, skip
                `Llama(...)` construction and use it directly.

        Stores: self.model_path / max_new_tokens / temperature / top_p /
                n_ctx / n_gpu_layers / llm.

        Raises:
            ValueError: if neither model_path nor llm is provided.
            FileNotFoundError: if model_path is provided but missing.
        """
        if llm is None and model_path is None:
            raise ValueError("provide either model_path or llm")
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        if llm is not None:
            self.llm = llm
            self.model_path = Path(model_path) if model_path else None
        else:
            self.model_path = Path(model_path)  # type: ignore[arg-type]
            if not self.model_path.exists():
                raise FileNotFoundError(f"GGUF model not found: {self.model_path}")
            from llama_cpp import Llama

            self.llm = Llama(
                model_path=str(self.model_path),
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                verbose=False,
            )

    def generate(
        self,
        query: str,
        retrieved_chunks: list[Chunk],
        *,
        query_type: QueryType | None = None,
        stream: bool = False,
    ) -> Answer:
        """Run the grounded-generation pipeline.

        Steps (mirrors `QwenGenerator.generate`):
            1. build_grounded_prompt(query, retrieved_chunks, query_type=...)
            2. raw = self._call_model(prompt)
            3. parsed = parse_response(raw)
            4. Map parsed.cited_indices ([N]) back to chunk_ids in
               retrieved_chunks order; drop indices outside [1, len].
            5. Wrap in Answer:
                 - text = "" if parsed.refused else parsed.text
                 - cited_chunk_ids = mapped chunk_ids
                 - refused = parsed.refused
                 - latency_seconds = wall-clock seconds in steps 2 + 3

        `stream=True` is not implemented → raise NotImplementedError.
        """
        if stream:
            raise NotImplementedError("Streaming generation not implemented for QwenGGUFGenerator.")
        prompt = build_grounded_prompt(query, retrieved_chunks, query_type=query_type)
        start_time = time.perf_counter()
        raw_text = self._call_model(prompt)
        parsed = parse_response(raw_text)
        cited_chunk_ids = tuple(
            retrieved_chunks[i - 1].chunk_id
            for i in parsed.cited_indices
            if 1 <= i <= len(retrieved_chunks)
        )
        return Answer(
            text="" if parsed.refused else parsed.text,
            cited_chunk_ids=() if parsed.refused else cited_chunk_ids,
            refused=parsed.refused,
            latency_seconds=time.perf_counter() - start_time,
        )

    def _call_model(self, prompt: list[dict[str, str]]) -> str:
        """Run the chat-completion call against the underlying Llama.

        `prompt` is the (system, user) message list from
        `build_grounded_prompt`. Forward it directly to
        `Llama.create_chat_completion`:

            out = self.llm.create_chat_completion(
                messages=prompt,
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            return out["choices"][0]["message"]["content"]

        Returns the raw assistant message text (no role wrapper, no
        leading/trailing whitespace stripping beyond what llama.cpp
        does itself).
        """
        res = self.llm.create_chat_completion(
            messages=prompt,
            max_tokens=self.max_new_tokens,
            top_p=self.top_p,
            temperature=self.temperature,
        )
        text: str = res["choices"][0]["message"]["content"]
        return text

    def generate_raw(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> RawCompletion:
        """Plain-text continuation via llama.cpp's `create_completion`.

        This bypasses the chat template and the grounded prompt, so the
        model behaves like a base / completion model. For instruction-
        tuned Qwen the output still tends to be coherent text, but
        without a chat-template wrapping the model is no longer in
        "assistant turn" mode — useful for the playground side-by-side
        with TinyDocs (which has no chat template at all).

        Implementation outline:
            1. import time
            2. start = time.perf_counter()
            3. res = self.llm.create_completion(
                   prompt=prompt,
                   max_tokens=max_tokens,
                   top_p=self.top_p,
                   temperature=temperature,
               )
            4. text: str = res["choices"][0]["text"]
            5. return RawCompletion(text=text, latency_seconds=time.perf_counter() - start)
        """
        import time

        start = time.perf_counter()
        res = self.llm.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            top_p=self.top_p,
            temperature=temperature,
        )
        text: str = res["choices"][0]["text"]
        return RawCompletion(text=text, latency_seconds=time.perf_counter() - start)
