"""CLI entry point: encode corpus.jsonl into pre-tokenized segments.

Outputs a .pt file with {segments, seq_len, vocab_size, tokenizer_path,
corpus_path, build_timestamp}. Slow step (BPE encoding) runs once;
training scripts then load the cached tensor in seconds.

Usage:
    uv run python scripts/encode_corpus.py \\
        --corpus data/pretrain/corpus.jsonl \\
        --tokenizer data/tokenizer/tokenizer.json \\
        --seq-len 2048 \\
        --out data/pretrain/segments.pt
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import click
import torch

from python_doc_assistant.generation.tinydocs.dataset import build_segments
from python_doc_assistant.generation.tinydocs.tokenizer import TinyDocsTokenizer


@click.command()
@click.option(
    "--corpus",
    required=True,
    type=click.Path(exists=True),
    help="Path to corpus.jsonl from `scripts/build_pretrain_corpus.py`.",
)
@click.option(
    "--tokenizer",
    required=True,
    type=click.Path(exists=True),
    help="Path to tokenizer.json from `scripts/train_tokenizer.py`.",
)
@click.option("--seq-len", required=True, type=int, help="Per-segment input length.")
@click.option(
    "--out",
    required=True,
    type=click.Path(),
    help="Output segments.pt path (dict with segments + metadata).",
)
def main(corpus: str, tokenizer: str, seq_len: int, out: str) -> None:
    """Encode a corpus to pre-tokenized segments.pt."""
    corpus_path = Path(corpus)
    tokenizer_path = Path(tokenizer)
    out_path = Path(out)

    click.echo(f"loading tokenizer: {tokenizer_path}")
    tok = TinyDocsTokenizer.load(tokenizer_path)

    click.echo(f"encoding corpus: {corpus_path} (this is the slow step)")
    segments = build_segments(corpus_path, tok, seq_len=seq_len, show_progress=True)

    payload = {
        "segments": segments,
        "seq_len": seq_len,
        "vocab_size": tok.vocab_size,
        "tokenizer_path": str(tokenizer_path),
        "corpus_path": str(corpus_path),
        "build_timestamp": datetime.now(timezone.utc).isoformat(),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)

    click.echo(f"segments written to {out_path}")
    click.echo(f"  shape: {tuple(segments.shape)}")
    click.echo(f"  seq_len: {seq_len}")
    click.echo(f"  vocab_size: {tok.vocab_size}")


if __name__ == "__main__":
    main()
