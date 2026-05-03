"""CLI entry point: train BPE tokenizer on a corpus.

Usage:
    uv run python scripts/train_tokenizer.py \\
        --corpus data/pretrain/corpus.jsonl \\
        --vocab-size 32000 \\
        --out data/tokenizer/tokenizer.json \\
        --incremental
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import click

from python_doc_assistant.generation.tinydocs.tokenizer import TinyDocsTokenizer
from python_doc_assistant.generation.tinydocs.tokenizer_train import (
    train_bpe,
    train_bpe_incremental,
)


@click.command()
@click.option(
    "--corpus",
    required=True,
    type=click.Path(exists=True),
    help='Path to corpus.jsonl (each line: {"text": ..., ...}).',
)
@click.option("--vocab-size", required=True, type=int, help="Target vocabulary size.")
@click.option("--out", required=True, type=click.Path(), help="Output tokenizer.json path.")
@click.option(
    "--incremental",
    is_flag=True,
    help="Use incremental BPE training (10–30× faster on large corpora; v3.1 §1.2).",
)
def main(corpus: str, vocab_size: int, out: str, incremental: bool) -> None:
    """Train a BPE tokenizer from a pretrain corpus."""
    corpus_path = Path(corpus)
    texts: list[str] = []
    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            texts.append(data["text"])
    click.echo(f"loaded {len(texts)} texts from {corpus_path}")

    special_tokens = ("<pad>", "<bos>", "<eos>", "<unk>", "<sp>")
    train_fn = train_bpe_incremental if incremental else train_bpe
    click.echo(
        f"training BPE (vocab_size={vocab_size}, "
        f"algo={'incremental' if incremental else 'naive'})..."
    )
    t0 = time.time()
    vocab, merges = train_fn(texts, vocab_size=vocab_size, special_tokens=special_tokens)
    elapsed = time.time() - t0
    click.echo(f"  done in {elapsed:.1f}s")

    tokenizer = TinyDocsTokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)
    tokenizer.save(Path(out))
    click.echo(f"tokenizer written to {out}")
    click.echo(f"  vocab_size: {len(vocab)}")
    click.echo(f"  merges: {len(merges)}")
    click.echo(f"  source: {len(texts)} texts from {corpus_path}")


if __name__ == "__main__":
    main()
