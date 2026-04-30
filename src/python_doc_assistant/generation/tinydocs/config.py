"""Hyperparameter dataclass for TinyDocs model.

See plans/v3-tiny-llm.md §1 for architecture decisions and §2 for the
tokenizer / vocab story.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TinyDocsConfig:
    """Architecture hyperparameters."""

    vocab_size: int = 32000
    hidden_dim: int = 512
    n_layers: int = 12
    n_heads: int = 8
    n_kv_heads: int = 8
    head_dim: int = 64
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    ffn_mult: int = 4
    norm_eps: float = 1e-6
    tie_word_embeddings: bool = True
