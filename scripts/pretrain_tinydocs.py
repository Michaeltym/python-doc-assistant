"""CLI entry point: pretrain TinyDocs on pre-encoded segments.

Usage:
    uv run python scripts/encode_corpus.py \\
        --corpus data/pretrain/corpus.jsonl \\
        --tokenizer data/tokenizer/tokenizer.json \\
        --seq-len 2048 \\
        --out data/pretrain/segments.pt

    uv run python scripts/pretrain_tinydocs.py \\
        --segments data/pretrain/segments.pt \\
        --ckpt-dir data/checkpoints/run-1 \\
        --log data/checkpoints/run-1/loss.jsonl \\
        --total-steps 1000 \\
        --batch-size 4 \\
        --device mps
"""

from __future__ import annotations

from pathlib import Path

import click
import torch

from python_doc_assistant.generation.tinydocs.config import TinyDocsConfig
from python_doc_assistant.generation.tinydocs.dataset import TinyDocsDataset
from python_doc_assistant.generation.tinydocs.train import (
    TrainConfig,
    train_tinydocs,
)


@click.command()
@click.option(
    "--segments",
    required=True,
    type=click.Path(exists=True),
    help="Path to segments.pt from `scripts/encode_corpus.py`.",
)
@click.option(
    "--ckpt-dir",
    required=True,
    type=click.Path(),
    help="Directory to write step_<N>.pt checkpoints.",
)
@click.option(
    "--log",
    required=True,
    type=click.Path(),
    help="Output JSONL log path (one line per logging step).",
)
@click.option("--total-steps", default=1000, type=int, help="Total optimizer steps.")
@click.option("--warmup-steps", default=100, type=int, help="LR warmup steps.")
@click.option("--base-lr", default=3e-4, type=float, help="Peak learning rate.")
@click.option("--batch-size", default=4, type=int, help="Per-step batch size.")
@click.option("--checkpoint-every", default=200, type=int, help="Save ckpt every N steps.")
@click.option("--log-every", default=10, type=int, help="Log every N steps.")
@click.option(
    "--device",
    default="mps",
    type=click.Choice(["mps", "cuda", "cpu"]),
    help="Torch device for training.",
)
@click.option("--seed", default=42, type=int, help="Random seed.")
def main(
    segments: str,
    ckpt_dir: str,
    log: str,
    total_steps: int,
    warmup_steps: int,
    base_lr: float,
    batch_size: int,
    checkpoint_every: int,
    log_every: int,
    device: str,
    seed: int,
) -> None:
    """Pretrain TinyDocs on a pre-encoded segments tensor."""
    payload = torch.load(Path(segments), weights_only=False)
    segments_tensor = payload["segments"]
    seq_len = payload["seq_len"]
    vocab_size = payload["vocab_size"]

    click.echo(
        f"loaded segments: shape={tuple(segments_tensor.shape)} seq_len={seq_len} "
        f"vocab_size={vocab_size}"
    )

    model_config = TinyDocsConfig(
        vocab_size=vocab_size,
        max_seq_len=seq_len,
    )
    dataset = TinyDocsDataset(segments_tensor)

    train_config = TrainConfig(
        base_lr=base_lr,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        batch_size=batch_size,
        checkpoint_every=checkpoint_every,
        log_every=log_every,
        device=device,
        seed=seed,
    )
    train_tinydocs(
        model_config,
        train_config,
        dataset,
        ckpt_dir=Path(ckpt_dir),
        log_path=Path(log),
    )
    click.echo(f"training complete; checkpoints in {ckpt_dir}")


if __name__ == "__main__":
    main()
