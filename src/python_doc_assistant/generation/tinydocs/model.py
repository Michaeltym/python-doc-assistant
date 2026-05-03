"""TinyDocs decoder-only model — interface skeleton.

Six modules to implement (plans/v3-tiny-llm.md §1):
  - RMSNorm
  - RotaryEmbedding
  - Attention (with KV cache)
  - SwiGLUFFN
  - TransformerBlock
  - TinyDocsModel
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from python_doc_assistant.generation.tinydocs.config import TinyDocsConfig

# ------------------------------------------------------------------
# KV cache
# ------------------------------------------------------------------


@dataclass
class KVCache:
    """Per-layer KV cache for incremental decoding."""

    keys: Tensor
    values: Tensor


# ------------------------------------------------------------------
# Modules
# ------------------------------------------------------------------


class RMSNorm(nn.Module):
    """Root mean square layer norm."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return (x / torch.sqrt(torch.mean(x**2, keepdim=True, dim=-1) + self.eps)) * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary positional embedding."""

    cos_cache: Tensor
    sin_cache: Tensor

    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10000.0) -> None:
        super().__init__()
        half_head_dim = head_dim // 2
        thetas = 1 / theta ** (torch.arange(0, half_head_dim) * 2 / head_dim)
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        angles = positions.unsqueeze(1) @ thetas.unsqueeze(0)
        self.register_buffer("cos_cache", angles.cos())
        self.register_buffer("sin_cache", angles.sin())
        self.max_seq_len = max_seq_len
        self.half_head_dim = half_head_dim

    # q, k shape [B, num_heads, T(seq_len), head_dim]
    def forward(self, q: Tensor, k: Tensor, *, position: int = 0) -> tuple[Tensor, Tensor]:
        seq_len = q.size(2)
        if seq_len + position > self.max_seq_len:
            raise ValueError(f"Input too long, current max_seq_len={self.max_seq_len}")
        return self._rotate(q, seq_len, position), self._rotate(k, seq_len, position)

    def _rotate(self, t: Tensor, seq_len: int, position: int) -> Tensor:

        first_half = t[..., : self.half_head_dim]
        second_half = t[..., self.half_head_dim :]
        cos = self.cos_cache[position : position + seq_len]
        sin = self.sin_cache[position : position + seq_len]
        rotated_first_half = first_half * cos - second_half * sin
        rotated_second_half = first_half * sin + second_half * cos
        rotated = torch.cat([rotated_first_half, rotated_second_half], dim=-1)
        return rotated


class Attention(nn.Module):
    """Multi-head causal self-attention with optional KV cache.

    Two attention implementations gated by `config.attention_impl`:
      - "manual": hand-written `q @ k.T / sqrt(d) → softmax → @ v` (default,
        matches v3 §1–§8 narrative).
      - "sdpa":   `nn.functional.scaled_dot_product_attention`. Prefill uses
        `is_causal=True` (Metal Flash Attention path on MPS); decode uses an
        explicit `attn_mask` derived from `~causal_mask[position:..., :T_kv]`
        (SDPA convention is `True = attend`).

    Numerically equivalent to within fp32 rounding; ~17 % MPS speedup
    measured on the v3.1 probe.
    """

    causal_mask: Tensor

    def __init__(self, config: TinyDocsConfig) -> None:
        super().__init__()
        hidden_dim = config.hidden_dim
        max_seq_len = config.max_seq_len
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.attention_impl = config.attention_impl
        mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)

    def forward(
        self,
        x: Tensor,
        rope: RotaryEmbedding,
        *,
        cache: KVCache | None = None,
        position: int = 0,
    ) -> tuple[Tensor, KVCache | None]:
        B, T, _ = x.shape
        q: Tensor = self.q_proj(x)
        k_new: Tensor = self.k_proj(x)
        v_new: Tensor = self.v_proj(x)
        # after _reshape, shape is [B, n_heads, T, head_dim]
        q, k_new = rope(self._reshape(q), self._reshape(k_new), position=position)
        v_new = self._reshape(v_new)
        # prefill mode
        if cache is None:
            full_k, full_v = k_new, v_new
        # decode mode
        else:
            full_k = torch.cat([cache.keys, k_new], dim=-2)
            full_v = torch.cat([cache.values, v_new], dim=-2)

        if self.attention_impl == "sdpa":
            attended = self._sdpa_attention(q, full_k, full_v, T, position)
        else:
            attended = self._manual_attention(q, full_k, full_v, T, position)
        attended = attended.transpose(1, 2).reshape(B, T, self.head_dim * self.n_heads)
        return self.o_proj(attended), KVCache(keys=full_k, values=full_v)

    def _manual_attention(
        self, q: Tensor, full_k: Tensor, full_v: Tensor, T: int, position: int
    ) -> Tensor:
        """Hand-written attention (v3 §1 baseline): scores → masked softmax → values."""
        T_kv = full_k.shape[-2]
        mask_slice = self.causal_mask[position : position + T, :T_kv]
        scores = q @ full_k.transpose(-2, -1) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(mask_slice, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        return weights @ full_v

    def _sdpa_attention(
        self, q: Tensor, full_k: Tensor, full_v: Tensor, T: int, position: int
    ) -> Tensor:
        """SDPA path (v3.1 §4) — routes prefill through Metal Flash on MPS.

        Required behaviour (caller passes q/k/v already shaped [B, H, T*, D]):
          - Prefill (cache was None upstream → T == T_kv == prefill length):
            call `nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)`.
            `is_causal=True` lets PyTorch pick the fastest backend on MPS.
          - Decode (cache provided upstream → T_kv > T): build an explicit
            `attn_mask` from `~self.causal_mask[position:position+T, :T_kv]`
            (SDPA convention: True = attend, opposite of our stored mask)
            and call `scaled_dot_product_attention(q, k, v, attn_mask=mask)`.

        Returns the same `(B, n_heads, T, head_dim)` tensor as `_manual_attention`.
        """
        T_kv = full_k.shape[-2]
        if T == T_kv:
            return nn.functional.scaled_dot_product_attention(q, full_k, full_v, is_causal=True)
        attn_mask = ~self.causal_mask[position : position + T, :T_kv]
        return nn.functional.scaled_dot_product_attention(q, full_k, full_v, attn_mask=attn_mask)

    def _reshape(self, t: Tensor) -> Tensor:
        return t.reshape(*t.shape[:-1], self.n_heads, self.head_dim).transpose(1, 2)


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward block."""

    def __init__(self, hidden_dim: int, ffn_mult: int) -> None:
        super().__init__()
        ffn_inner_dim = int(hidden_dim * ffn_mult * 2 / 3)
        ffn_inner_dim = 256 * ((ffn_inner_dim + 255) // 256)
        self.gate_proj = nn.Linear(hidden_dim, ffn_inner_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, ffn_inner_dim, bias=False)
        self.down_proj = nn.Linear(ffn_inner_dim, hidden_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gate: Tensor = self.gate_proj(x)
        up: Tensor = self.up_proj(x)
        hidden: Tensor = nn.functional.silu(gate) * up
        out: Tensor = self.down_proj(hidden)
        return out


class TransformerBlock(nn.Module):
    """One transformer decoder block (norm → attn → norm → ffn)."""

    def __init__(self, config: TinyDocsConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.hidden_dim, config.norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_dim, config.norm_eps)
        self.attention = Attention(config)
        self.swiglu = SwiGLUFFN(config.hidden_dim, config.ffn_mult)

    def forward(
        self,
        x: Tensor,
        rope: RotaryEmbedding,
        *,
        cache: KVCache | None = None,
        position: int = 0,
    ) -> tuple[Tensor, KVCache | None]:
        attention_norm = self.attention_norm(x)
        attention_output, new_cache = self.attention(
            attention_norm, rope, cache=cache, position=position
        )
        x = x + attention_output
        ffn_norm = self.ffn_norm(x)
        ffn_out = self.swiglu(ffn_norm)
        return x + ffn_out, new_cache


class TinyDocsModel(nn.Module):
    """Full decoder-only LM."""

    def __init__(self, config: TinyDocsConfig) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.rope = RotaryEmbedding(config.head_dim, config.max_seq_len, config.rope_theta)
        self.final_norm = RMSNorm(config.hidden_dim, config.norm_eps)
        self.vocab_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        if config.tie_word_embeddings:
            self.vocab_head.weight = self.embeddings.weight

        def _init_weights(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

        self.apply(_init_weights)

    def forward(
        self,
        token_ids: Tensor,  # shape is [B, T]
        *,
        caches: list[KVCache] | None = None,
        position: int = 0,
    ) -> tuple[Tensor, list[KVCache] | None]:
        """Returns (logits, updated_caches). Logits shape: (batch, seq, vocab)."""
        hidden = self.embeddings(token_ids)
        new_caches: list[KVCache] = []
        for i, block in enumerate(self.blocks):
            hidden, cache = block(
                hidden,
                self.rope,
                cache=caches[i] if caches is not None else None,
                position=position,
            )
            new_caches.append(cache)
        hidden = self.final_norm(hidden)
        logits = self.vocab_head(hidden)
        return logits, new_caches
