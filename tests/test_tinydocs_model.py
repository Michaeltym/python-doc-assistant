"""Shape / contract tests for v3 TinyDocs model.

All tests are skipped until v3 §1 implementation lands. Each test
encodes a single behavior contract; the implementer chooses how to
satisfy it.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from python_doc_assistant.generation.tinydocs.config import TinyDocsConfig  # noqa: E402
from python_doc_assistant.generation.tinydocs.model import (  # noqa: E402
    Attention,
    RMSNorm,
    RotaryEmbedding,
    SwiGLUFFN,
    TinyDocsModel,
    TransformerBlock,
)

# ------------------------------------------------------------------
# RMSNorm
# ------------------------------------------------------------------


def test_rmsnorm_preserves_shape() -> None:
    norm = RMSNorm(dim=512)
    x = torch.randn(2, 16, 512)
    y = norm(x)
    assert y.shape == x.shape


def test_rmsnorm_unit_variance_on_unit_input() -> None:
    norm = RMSNorm(dim=512)
    x = torch.ones(1, 1, 512)
    y = norm(x)
    assert torch.allclose(y, x, atol=1e-3)


def test_rmsnorm_actually_normalizes_random_input() -> None:
    """With weight=1 (default init), output RMS along the feature dim must be ~1."""
    norm = RMSNorm(dim=512)
    x = torch.randn(2, 16, 512) * 5.0  # large random input
    y = norm(x)
    rms_y = y.float().pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms_y, torch.ones_like(rms_y), atol=1e-3)


# ------------------------------------------------------------------
# RotaryEmbedding
# ------------------------------------------------------------------


def test_rope_preserves_shape() -> None:
    rope = RotaryEmbedding(head_dim=64, max_seq_len=2048)
    q = torch.randn(2, 8, 16, 64)
    k = torch.randn(2, 8, 16, 64)
    q_rot, k_rot = rope(q, k)
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape


def test_rope_position_zero_is_identity() -> None:
    rope = RotaryEmbedding(head_dim=64, max_seq_len=2048)
    q = torch.randn(1, 1, 1, 64)
    k = torch.randn(1, 1, 1, 64)
    q_rot, k_rot = rope(q, k, position=0)
    assert torch.allclose(q_rot[..., 0, :], q[..., 0, :], atol=1e-5)


def test_rope_relative_distance_invariance() -> None:
    """Q·K after RoPE depends only on relative distance, not absolute position.

    This catches: wrong freq formula, wrong pair convention, dim mix-up that
    happens to preserve shape but breaks the rotation semantics.
    """
    head_dim = 64
    rope = RotaryEmbedding(head_dim=head_dim, max_seq_len=2048)
    q = torch.randn(1, 1, 1, head_dim)
    k = torch.randn(1, 1, 1, head_dim)

    delta = 5
    scores = []
    for m in (0, 50, 100, 500):
        n = m + delta
        q_rot, _ = rope(q, q, position=m)  # q rotated at pos m
        _, k_rot = rope(k, k, position=n)  # k rotated at pos n
        score = (q_rot * k_rot).sum().item()
        scores.append(score)

    for s in scores[1:]:
        assert abs(s - scores[0]) < 1e-3, (
            f"score={s} differs from scores[0]={scores[0]} — "
            f"RoPE failed to preserve Q·K invariance under same relative delta"
        )


def test_rope_kv_cache_position_offset() -> None:
    """Rotating the n-th token alone at position=n must match the n-th
    token of a full-sequence rotation at position=0. Verifies that
    `position` slicing of the cache is correct for KV-cache decoding."""
    head_dim = 64
    rope = RotaryEmbedding(head_dim=head_dim, max_seq_len=2048)

    full = torch.randn(1, 1, 5, head_dim)  # 5 tokens at positions 0..4
    full_rot, _ = rope(full, full, position=0)

    single = full[:, :, 2:3, :]  # the 3rd token (index 2)
    single_rot, _ = rope(single, single, position=2)

    assert torch.allclose(full_rot[:, :, 2:3, :], single_rot, atol=1e-5)


# ------------------------------------------------------------------
# Attention
# ------------------------------------------------------------------


def test_attention_output_shape() -> None:
    cfg = TinyDocsConfig(hidden_dim=512, n_heads=8, n_kv_heads=8, head_dim=64)
    attn = Attention(cfg)
    rope = RotaryEmbedding(head_dim=64, max_seq_len=2048)
    x = torch.randn(2, 16, 512)
    y, _ = attn(x, rope)
    assert y.shape == x.shape


def test_attention_kv_cache_round_trip() -> None:
    cfg = TinyDocsConfig(hidden_dim=512, n_heads=8, n_kv_heads=8, head_dim=64)
    attn = Attention(cfg)
    rope = RotaryEmbedding(head_dim=64, max_seq_len=2048)
    x1 = torch.randn(1, 4, 512)
    x2 = torch.randn(1, 1, 512)
    _, cache1 = attn(x1, rope, cache=None, position=0)
    assert cache1 is not None
    _, cache2 = attn(x2, rope, cache=cache1, position=4)
    assert cache2 is not None
    assert cache2.keys.shape[-2] == 5


def test_attention_causal_mask_isolates_future() -> None:
    """Output at position i must depend only on positions 0..i (causal).

    Verify by running attention on the full sequence and on only the
    first i+1 tokens; output at position i must match.
    Catches: missing mask, mask pointed wrong direction, broadcasting bugs.
    """
    cfg = TinyDocsConfig(hidden_dim=64, n_heads=4, n_kv_heads=4, head_dim=16)
    attn = Attention(cfg)
    rope = RotaryEmbedding(head_dim=16, max_seq_len=64)

    x_full = torch.randn(1, 5, 64)
    out_full, _ = attn(x_full, rope, cache=None, position=0)

    x_partial = x_full[:, :3, :]  # first 3 tokens
    out_partial, _ = attn(x_partial, rope, cache=None, position=0)

    assert torch.allclose(out_full[:, 2:3, :], out_partial[:, 2:3, :], atol=1e-5)


def test_attention_kv_cache_matches_prefill() -> None:
    """Incremental decoding via KV cache must give the same final output
    as prefill of the full sequence.
    Catches: wrong RoPE position with cache, K/V cache concat axis bug,
    missing/extra position offset.
    """
    cfg = TinyDocsConfig(hidden_dim=64, n_heads=4, n_kv_heads=4, head_dim=16)
    attn = Attention(cfg)
    rope = RotaryEmbedding(head_dim=16, max_seq_len=64)

    x = torch.randn(1, 5, 64)

    # Path A: prefill all 5 at once
    out_prefill, _ = attn(x, rope, cache=None, position=0)

    # Path B: prefill first 4, then feed token 5 incrementally with cache
    out_first4, cache = attn(x[:, :4, :], rope, cache=None, position=0)
    out_last, _ = attn(x[:, 4:5, :], rope, cache=cache, position=4)

    # Token 5's output must match across the two paths
    assert torch.allclose(out_prefill[:, 4:5, :], out_last, atol=1e-5)


# ------------------------------------------------------------------
# Attention SDPA path (v3.1 §4)
# ------------------------------------------------------------------


def _equivalence_config() -> TinyDocsConfig:
    """Small architecture for SDPA / manual equivalence tests."""
    return TinyDocsConfig(
        vocab_size=64,
        hidden_dim=64,
        n_layers=2,
        n_heads=4,
        head_dim=16,
        max_seq_len=32,
    )


def test_sdpa_prefill_matches_manual() -> None:
    """SDPA prefill (cache=None, is_causal=True) must equal hand-written attention."""
    cfg_manual = _equivalence_config()  # attention_impl="manual" default
    cfg_sdpa = TinyDocsConfig(**{**cfg_manual.__dict__, "attention_impl": "sdpa"})

    torch.manual_seed(0)
    rope = RotaryEmbedding(head_dim=cfg_manual.head_dim, max_seq_len=cfg_manual.max_seq_len)

    attn_manual = Attention(cfg_manual)
    attn_sdpa = Attention(cfg_sdpa)
    # Sync weights — only difference should be the forward path
    attn_sdpa.load_state_dict(attn_manual.state_dict())

    x = torch.randn(2, 16, cfg_manual.hidden_dim)
    out_manual, cache_manual = attn_manual(x, rope, cache=None, position=0)
    out_sdpa, cache_sdpa = attn_sdpa(x, rope, cache=None, position=0)

    assert torch.allclose(out_manual, out_sdpa, atol=1e-5), "SDPA prefill diverges from manual"
    assert torch.allclose(cache_manual.keys, cache_sdpa.keys)
    assert torch.allclose(cache_manual.values, cache_sdpa.values)


def test_sdpa_decode_matches_manual() -> None:
    """SDPA decode (cache provided, T=1) must equal hand-written attention.

    Tests the explicit attn_mask path (vs prefill's is_causal=True).
    """
    cfg_manual = _equivalence_config()
    cfg_sdpa = TinyDocsConfig(**{**cfg_manual.__dict__, "attention_impl": "sdpa"})

    torch.manual_seed(0)
    rope = RotaryEmbedding(head_dim=cfg_manual.head_dim, max_seq_len=cfg_manual.max_seq_len)

    attn_manual = Attention(cfg_manual)
    attn_sdpa = Attention(cfg_sdpa)
    attn_sdpa.load_state_dict(attn_manual.state_dict())

    # Prefill 8 tokens to build a cache.
    x_prefill = torch.randn(1, 8, cfg_manual.hidden_dim)
    _, cache_manual = attn_manual(x_prefill, rope, cache=None, position=0)
    _, cache_sdpa = attn_sdpa(x_prefill, rope, cache=None, position=0)

    # Decode one new token at position=8.
    x_new = torch.randn(1, 1, cfg_manual.hidden_dim)
    out_manual, _ = attn_manual(x_new, rope, cache=cache_manual, position=8)
    out_sdpa, _ = attn_sdpa(x_new, rope, cache=cache_sdpa, position=8)

    assert torch.allclose(out_manual, out_sdpa, atol=1e-5), "SDPA decode diverges from manual"


def test_sdpa_full_model_forward_matches_manual() -> None:
    """End-to-end model forward under both impls must produce equal logits."""
    cfg_manual = _equivalence_config()
    cfg_sdpa = TinyDocsConfig(**{**cfg_manual.__dict__, "attention_impl": "sdpa"})

    torch.manual_seed(42)
    model_manual = TinyDocsModel(cfg_manual)
    model_sdpa = TinyDocsModel(cfg_sdpa)
    model_sdpa.load_state_dict(model_manual.state_dict())

    input_ids = torch.randint(0, cfg_manual.vocab_size, (2, 12))
    logits_manual, _ = model_manual(input_ids)
    logits_sdpa, _ = model_sdpa(input_ids)

    assert torch.allclose(logits_manual, logits_sdpa, atol=1e-4), (
        "Full-model SDPA forward diverges from manual"
    )


# ------------------------------------------------------------------
# SwiGLUFFN
# ------------------------------------------------------------------


def test_swiglu_preserves_shape() -> None:
    ffn = SwiGLUFFN(hidden_dim=512, ffn_mult=4)
    x = torch.randn(2, 16, 512)
    y = ffn(x)
    assert y.shape == x.shape


def test_swiglu_param_count_follows_2_3_rule() -> None:
    """SwiGLU has 3 projections (gate, up, down). To keep total params close
    to a classic 2-projection FFN, inner dim is shrunk by 2/3.

    Total params should be ≈ 3 × hidden_dim × (2/3 × ffn_mult × hidden_dim)
    = 2 × ffn_mult × hidden_dim². Allow ±15% for alignment / round-up
    strategies (e.g. Llama rounds inner to a multiple of 256).
    """
    hidden_dim = 512
    ffn_mult = 4
    ffn = SwiGLUFFN(hidden_dim=hidden_dim, ffn_mult=ffn_mult)
    n = sum(p.numel() for p in ffn.parameters())
    expected = 2 * ffn_mult * hidden_dim * hidden_dim  # 2 × 4 × 512² = 2_097_152
    assert abs(n - expected) / expected < 0.15, (
        f"SwiGLU param count {n} deviates from expected ≈{expected} by more than 15%; "
        f"check ffn_inner_dim formula"
    )


# ------------------------------------------------------------------
# TransformerBlock
# ------------------------------------------------------------------


def test_block_preserves_shape() -> None:
    cfg = TinyDocsConfig(hidden_dim=512, n_heads=8, n_kv_heads=8, head_dim=64)
    block = TransformerBlock(cfg)
    rope = RotaryEmbedding(head_dim=64, max_seq_len=2048)
    x = torch.randn(2, 16, 512)
    y, _ = block(x, rope)
    assert y.shape == x.shape


def test_block_has_independent_norms() -> None:
    """The two RMSNorm sub-layers (pre-attn and pre-ffn) must be independent
    instances with separate `weight` parameters.

    Catches: accidentally reusing one RMSNorm instance for both pre-norms
    (which silently halves expressive capacity since the same weights apply
    before attention and before FFN).
    """
    from python_doc_assistant.generation.tinydocs.model import RMSNorm as RMSNormCls

    cfg = TinyDocsConfig(hidden_dim=64, n_heads=4, n_kv_heads=4, head_dim=16)
    block = TransformerBlock(cfg)

    norms = [m for m in block.modules() if isinstance(m, RMSNormCls)]
    assert len(norms) == 2, f"expected 2 RMSNorm modules in a block, got {len(norms)}"
    assert norms[0] is not norms[1], "the two RMSNorm instances must be distinct objects"
    assert norms[0].weight is not norms[1].weight, (
        "the two RMSNorm instances must have separate `weight` parameters"
    )


def test_block_residual_is_non_destructive() -> None:
    """The forward pass must not mutate the input tensor in place.

    Catches: `x += sublayer(...)` instead of `x = x + sublayer(...)`. In-place
    ops can break autograd in deeper computation graphs and silently corrupt
    upstream tensors.
    """
    cfg = TinyDocsConfig(hidden_dim=64, n_heads=4, n_kv_heads=4, head_dim=16)
    block = TransformerBlock(cfg)
    rope = RotaryEmbedding(head_dim=16, max_seq_len=64)
    x = torch.randn(1, 4, 64)
    x_clone = x.clone()
    _ = block(x, rope)
    assert torch.equal(x, x_clone), "forward must not mutate the input tensor"


# ------------------------------------------------------------------
# TinyDocsModel
# ------------------------------------------------------------------


def test_model_forward_logits_shape() -> None:
    cfg = TinyDocsConfig(
        vocab_size=32000, hidden_dim=512, n_layers=4, n_heads=8, n_kv_heads=8, head_dim=64
    )
    model = TinyDocsModel(cfg)
    token_ids = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, _ = model(token_ids)
    assert logits.shape == (2, 16, cfg.vocab_size)


def test_model_kv_cache_incremental_decoding() -> None:
    cfg = TinyDocsConfig(
        vocab_size=32000, hidden_dim=512, n_layers=4, n_heads=8, n_kv_heads=8, head_dim=64
    )
    model = TinyDocsModel(cfg)
    prompt = torch.randint(0, cfg.vocab_size, (1, 8))
    next_tok = torch.randint(0, cfg.vocab_size, (1, 1))
    _, caches = model(prompt)
    assert caches is not None and len(caches) == cfg.n_layers
    logits, caches2 = model(next_tok, caches=caches, position=8)
    assert logits.shape == (1, 1, cfg.vocab_size)
    assert caches2 is not None and caches2[0].keys.shape[-2] == 9


def test_model_one_batch_overfit() -> None:
    """Smoke test: 1-batch overfit verifies forward+backward correctness."""
    cfg = TinyDocsConfig(
        vocab_size=128, hidden_dim=64, n_layers=2, n_heads=4, n_kv_heads=4, head_dim=16
    )
    model = TinyDocsModel(cfg)
    token_ids = torch.randint(0, cfg.vocab_size, (1, 8))
    targets = torch.randint(0, cfg.vocab_size, (1, 8))

    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    loss_first = None
    for step in range(50):
        logits, _ = model(token_ids)
        loss = loss_fn(logits.view(-1, cfg.vocab_size), targets.view(-1))
        if step == 0:
            loss_first = loss.item()
        optim.zero_grad()
        loss.backward()
        optim.step()
    assert loss_first is not None
    assert loss.item() < loss_first * 0.5
