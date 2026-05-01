"""CLI entry point: byte-target Bernoulli sample of a pretrain corpus.

Used by v3.1 §1.1 to extract a ~100 MB representative sample from the
mix corpus (FineWeb + docs) for BPE tokenizer training. See
`src/.../tinydocs/sample.py` for the algorithm.

Usage:
    uv run python scripts/sample_corpus.py \\
        --corpus data/pretrain/mix_corpus.jsonl \\
        --out data/pretrain/mix_sample_100mb.jsonl \\
        --target-bytes 100_000_000 \\
        --seed 42
"""

from __future__ import annotations

from pathlib import Path

import click

from python_doc_assistant.generation.tinydocs.sample import (
    count_bytes,
    sample_lines_to_byte_target,
)


@click.command()
@click.option(
    "--corpus",
    required=True,
    type=click.Path(exists=True),
    help="Input jsonl corpus path.",
)
@click.option(
    "--out",
    required=True,
    type=click.Path(),
    help="Output sample jsonl path.",
)
@click.option(
    "--target-bytes",
    required=True,
    type=int,
    help="Target total bytes of sampled lines (e.g. 100_000_000 for ~100MB).",
)
@click.option("--seed", default=42, type=int, help="Random seed.")
def main(corpus: str, out: str, target_bytes: int, seed: int) -> None:
    """Two-pass Bernoulli sample of `corpus` to ~`target_bytes` of lines."""
    in_path = Path(corpus)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    click.echo(f"counting input bytes: {in_path}")
    total = count_bytes(in_path)
    click.echo(f"  total: {total / 1e6:.1f} MB")
    click.echo(f"  target: {target_bytes / 1e6:.1f} MB ({100 * target_bytes / total:.1f} %)")

    click.echo(f"sampling → {out_path}")
    written = 0
    n_lines = 0
    with out_path.open("w", encoding="utf-8") as f:
        for line in sample_lines_to_byte_target(in_path, target_bytes, seed=seed):
            f.write(line)
            written += len(line.encode("utf-8"))
            n_lines += 1

    click.echo(f"\nwritten {n_lines:,} lines, {written / 1e6:.1f} MB")
    click.echo(f"  ratio actual: {100 * written / total:.2f} %")


if __name__ == "__main__":
    main()
