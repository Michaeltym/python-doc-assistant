"""CLI entry point: build pretrain corpus from Python docs chunks.

Usage:
    uv run python scripts/build_pretrain_corpus.py \\
        --chunks data/chunks/3.12/a5c1a35a5a02/chunks.jsonl \\
        --out data/pretrain/corpus.jsonl \\
        --seed 42
"""

from __future__ import annotations

from pathlib import Path

import click

from python_doc_assistant.generation.tinydocs.data_mix import build_corpus_from_chunks


@click.command()
@click.option(
    "--chunks",
    required=True,
    type=click.Path(exists=True),
    help="Path to chunks.jsonl from `pdr build-index`.",
)
@click.option("--out", required=True, type=click.Path(), help="Output corpus.jsonl path.")
@click.option("--seed", default=42, type=int, help="Shuffle seed for reproducibility.")
@click.option(
    "--manifest", default=None, type=click.Path(), help="Optional path to write manifest.json."
)
def main(chunks: str, out: str, seed: int, manifest: str | None = None) -> None:
    """Build pretrain corpus from Python docs chunks."""
    manifest_dict = build_corpus_from_chunks(
        Path(chunks), Path(out), seed=seed, manifest_path=Path(manifest) if manifest else None
    )
    click.echo(f"corpus written to {out}")
    click.echo(f"manifest: {manifest_dict}")


if __name__ == "__main__":
    main()
