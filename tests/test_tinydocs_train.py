"""Tests for v3 §4b training loop (LR schedule + checkpoint + mini train run)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from torch.optim import Optimizer  # noqa: E402

from python_doc_assistant.generation.tinydocs.config import TinyDocsConfig  # noqa: E402
from python_doc_assistant.generation.tinydocs.dataset import TinyDocsDataset  # noqa: E402
from python_doc_assistant.generation.tinydocs.model import TinyDocsModel  # noqa: E402
from python_doc_assistant.generation.tinydocs.train import (  # noqa: E402
    TrainConfig,
    get_lr,
    load_checkpoint,
    save_checkpoint,
    train_tinydocs,
)

# ------------------------------------------------------------------
# get_lr
# ------------------------------------------------------------------


def test_get_lr_starts_at_zero() -> None:
    """At step 0 lr should be 0 (warmup just beginning)."""
    lr = get_lr(0, base_lr=3e-4, warmup_steps=100, total_steps=5000)
    assert lr == pytest.approx(0.0)


def test_get_lr_linear_during_warmup() -> None:
    """During warmup lr scales linearly with step."""
    base_lr = 1e-3
    warmup = 100
    lr_at_50 = get_lr(50, base_lr=base_lr, warmup_steps=warmup, total_steps=5000)
    lr_at_25 = get_lr(25, base_lr=base_lr, warmup_steps=warmup, total_steps=5000)
    assert lr_at_50 == pytest.approx(base_lr * 0.5, rel=1e-6)
    assert lr_at_25 == pytest.approx(base_lr * 0.25, rel=1e-6)


def test_get_lr_peaks_at_warmup_end() -> None:
    """At step == warmup_steps lr equals base_lr."""
    lr = get_lr(100, base_lr=3e-4, warmup_steps=100, total_steps=5000)
    assert lr == pytest.approx(3e-4, rel=1e-6)


def test_get_lr_cosine_decays_after_warmup() -> None:
    """Lr should monotonically decrease from warmup_end onward."""
    base_lr = 3e-4
    samples = [
        get_lr(step, base_lr=base_lr, warmup_steps=100, total_steps=1000)
        for step in (100, 300, 600, 900)
    ]
    for a, b in zip(samples, samples[1:]):
        assert a > b, f"LR not decreasing: {a} -> {b}"


def test_get_lr_near_zero_at_total_steps() -> None:
    """At step == total_steps the cosine decay reaches 0."""
    lr = get_lr(1000, base_lr=3e-4, warmup_steps=100, total_steps=1000)
    assert lr == pytest.approx(0.0, abs=1e-8)


# ------------------------------------------------------------------
# save_checkpoint / load_checkpoint
# ------------------------------------------------------------------


def _tiny_model_and_optim() -> tuple[TinyDocsModel, Optimizer]:
    cfg = TinyDocsConfig(
        vocab_size=64, hidden_dim=32, n_layers=2, n_heads=4, n_kv_heads=4, head_dim=8
    )
    model = TinyDocsModel(cfg)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return model, optim


def test_checkpoint_save_load_round_trip(tmp_path: Path) -> None:
    """Save → load reproduces model + optimizer state and the saved step."""
    model_a, optim_a = _tiny_model_and_optim()
    # Take one step so optimizer has non-default state
    x = torch.randint(0, 64, (1, 4))
    logits, _ = model_a(x)
    loss = logits.sum()
    loss.backward()
    optim_a.step()

    path = tmp_path / "ckpt.pt"
    save_checkpoint(model_a, optim_a, step=42, path=path)

    model_b, optim_b = _tiny_model_and_optim()
    restored_step = load_checkpoint(path, model_b, optim_b)

    assert restored_step == 42
    # Compare a parameter tensor between the two models
    for (n_a, p_a), (n_b, p_b) in zip(
        model_a.named_parameters(), model_b.named_parameters(), strict=True
    ):
        assert n_a == n_b
        assert torch.equal(p_a, p_b), f"param {n_a} mismatch after restore"


# ------------------------------------------------------------------
# train_tinydocs (integration)
# ------------------------------------------------------------------


def _tiny_dataset() -> TinyDocsDataset:
    """A small fake dataset of 8 segments, each (seq_len + 1) = 17 tokens."""
    segments = torch.randint(0, 64, (8, 17), dtype=torch.long)
    return TinyDocsDataset(segments)


def test_train_writes_log_and_decreases_loss(tmp_path: Path) -> None:
    """50-step mini run on a tiny dataset: loss should decrease meaningfully
    and the JSONL log should record per-step (step, loss, lr) entries."""
    model_cfg = TinyDocsConfig(
        vocab_size=64, hidden_dim=32, n_layers=2, n_heads=4, n_kv_heads=4, head_dim=8
    )
    train_cfg = TrainConfig(
        base_lr=1e-3,
        warmup_steps=10,
        total_steps=50,
        batch_size=2,
        checkpoint_every=25,
        log_every=1,
        device="cpu",
    )
    ds = _tiny_dataset()
    log_path = tmp_path / "loss.jsonl"
    ckpt_dir = tmp_path / "ckpts"

    train_tinydocs(model_cfg, train_cfg, ds, ckpt_dir=ckpt_dir, log_path=log_path)

    # Log written
    assert log_path.exists()
    lines = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    assert len(lines) >= 10
    for entry in lines:
        assert "step" in entry
        assert "loss" in entry
        assert "lr" in entry

    # Loss decreases (early loss > late loss)
    early_loss = sum(e["loss"] for e in lines[:5]) / 5
    late_loss = sum(e["loss"] for e in lines[-5:]) / 5
    assert late_loss < early_loss * 0.9, (
        f"Expected late_loss ({late_loss:.4f}) < 0.9 × early_loss ({early_loss:.4f})"
    )


def test_train_writes_checkpoints(tmp_path: Path) -> None:
    """Checkpoints should be written every `checkpoint_every` steps."""
    model_cfg = TinyDocsConfig(
        vocab_size=64, hidden_dim=32, n_layers=2, n_heads=4, n_kv_heads=4, head_dim=8
    )
    train_cfg = TrainConfig(
        base_lr=1e-3,
        warmup_steps=5,
        total_steps=30,
        batch_size=2,
        checkpoint_every=10,
        log_every=5,
        device="cpu",
    )
    ds = _tiny_dataset()
    log_path = tmp_path / "loss.jsonl"
    ckpt_dir = tmp_path / "ckpts"

    train_tinydocs(model_cfg, train_cfg, ds, ckpt_dir=ckpt_dir, log_path=log_path)

    # At least 2 checkpoints (at step 10, 20, possibly final at 30)
    ckpts = sorted(ckpt_dir.glob("step_*.pt"))
    assert len(ckpts) >= 2
