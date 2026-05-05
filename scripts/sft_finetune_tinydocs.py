"""CLI entry point: SFT fine-tune a TinyDocs ckpt on Qwen-distilled triples.

Usage:
    uv run python scripts/sft_finetune_tinydocs.py \\
        --init-ckpt data/checkpoints/run-v31/step_final.pt \\
        --tokenizer data/tokenizer/tokenizer-mix.json \\
        --sft-data data/sft/sft_corpus.jsonl \\
        --chunks data/chunks/3.12/a5c1a35a5a02/chunks.jsonl \\
        --ckpt-dir data/checkpoints/run-sft-v31 \\
        --log data/checkpoints/run-sft-v31/loss.jsonl \\
        --epochs 8 \\
        --batch-size 4 \\
        --lr 5e-6 \\
        --device mps
"""

from __future__ import annotations

from pathlib import Path

import click
import torch

from python_doc_assistant.generation.tinydocs.config import TinyDocsConfig
from python_doc_assistant.generation.tinydocs.sft_train import (
    SFTConfig,
    SFTDataset,
    build_sft_records,
    train_sft,
)
from python_doc_assistant.generation.tinydocs.tokenizer import TinyDocsTokenizer


@click.command()
@click.option(
    "--init-ckpt",
    required=True,
    type=click.Path(exists=True),
    help="Path to pretrained step_final.pt to fine-tune.",
)
@click.option(
    "--tokenizer",
    required=True,
    type=click.Path(exists=True),
    help="Path to tokenizer JSON (must match the one used in pretrain).",
)
@click.option(
    "--sft-data",
    required=True,
    type=click.Path(exists=True),
    help="Path to sft_corpus.jsonl (rows from build_sft_corpus).",
)
@click.option(
    "--chunks",
    required=True,
    type=click.Path(exists=True),
    help="Path to chunks.jsonl for chunk_id → text lookup.",
)
@click.option("--ckpt-dir", required=True, type=click.Path())
@click.option("--log", required=True, type=click.Path())
@click.option("--epochs", default=8, type=int)
@click.option("--batch-size", default=4, type=int)
@click.option("--lr", default=5e-6, type=float)
@click.option("--warmup-steps", default=20, type=int)
@click.option("--checkpoint-every-epochs", default=4, type=int)
@click.option("--log-every", default=10, type=int)
@click.option(
    "--device",
    default="mps",
    type=click.Choice(["mps", "cuda", "cpu"]),
)
@click.option("--seed", default=42, type=int)
def main(
    init_ckpt: str,
    tokenizer: str,
    sft_data: str,
    chunks: str,
    ckpt_dir: str,
    log: str,
    epochs: int,
    batch_size: int,
    lr: float,
    warmup_steps: int,
    checkpoint_every_epochs: int,
    log_every: int,
    device: str,
    seed: int,
) -> None:
    """SFT fine-tune a TinyDocs ckpt on grounded RAG triples."""
    tok = TinyDocsTokenizer.load(Path(tokenizer))
    init_payload = torch.load(Path(init_ckpt), weights_only=False, map_location="cpu")
    model_config = TinyDocsConfig(**init_payload["model_config"])

    click.echo(f"loaded init ckpt step={init_payload['step']}")
    click.echo(f"building SFT records from {sft_data}")
    records = build_sft_records(
        Path(sft_data),
        Path(chunks),
        tok,
        max_seq_len=model_config.max_seq_len,
    )
    click.echo(f"records ready: {len(records)}")
    dataset = SFTDataset(records)

    sft_config = SFTConfig(
        base_lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        warmup_steps=warmup_steps,
        checkpoint_every_epochs=checkpoint_every_epochs,
        log_every=log_every,
        device=device,
        seed=seed,
    )
    train_sft(
        model_config,
        sft_config,
        dataset,
        init_ckpt_path=Path(init_ckpt),
        ckpt_dir=Path(ckpt_dir),
        log_path=Path(log),
        pad_token_id=tok.pad_id,
    )
    click.echo(f"SFT complete; ckpts in {ckpt_dir}")


if __name__ == "__main__":
    main()
