"""CLI entry point: build mixed pretrain corpus (FineWeb-Edu + Python docs).

v3.1 §2 deliverable. Streams `HuggingFaceFW/fineweb-edu` from HuggingFace
until a target byte size is reached, then concatenates with the v0 docs
corpus tiled to ~5 % of total bytes (so the BPE / pretrain see Python-docs
vocabulary even though FineWeb dominates).

Output:
    data/pretrain/mix_corpus.jsonl  — one JSON per line, schema {"text": "..."}
    data/pretrain/mix_manifest.json — reproducibility (byte counts, seeds,
                                       fineweb subset name, build timestamp)

Usage:
    uv run python scripts/build_fineweb_corpus.py \\
        --out data/pretrain/mix_corpus.jsonl \\
        --target-bytes 2_400_000_000 \\
        --docs-corpus data/pretrain/corpus.jsonl

Notes:
  - 2.4 GB ≈ 430 M GPT-2 tokens, ≈ 380–400 M tokens with the docs-tuned
    32k BPE we'll train in §1.4.
  - HF datasets streaming caches per-row downloads but doesn't write the
    full 10 BT subset to disk; only the slice we read.
  - `HF_TOKEN` env var enables higher rate limits — recommended for the
    full 2.4 GB pull.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import IO

import click

DEFAULT_DOCS_CORPUS = Path("data/pretrain/corpus.jsonl")
DOCS_RATIO = 0.05  # 5 % of total bytes (docs gets tiled to reach this)


def _stream_fineweb_to_byte_target(
    out_handle: IO[str],
    target_bytes: int,
    sample_name: str,
) -> tuple[int, int, str | None]:
    """Stream FineWeb-Edu rows; return (bytes_written, rows_written, revision).

    Implementation note: uses `huggingface_hub.hf_hub_download` to fetch each
    parquet file to the local HF cache, then pyarrow's row-batch iterator to
    stream rows without loading the whole table into RAM.

    The earlier `datasets.load_dataset(..., streaming=True)` code path failed
    on this machine with "Cannot send a request, as the client has been
    closed." inside fsspec/httpx — a known compatibility issue. This direct
    download path uses `huggingface_hub`'s requests-based client and works
    reliably.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    repo_id = "HuggingFaceFW/fineweb-edu"
    # sample-10BT → sample/10BT (dataset config name → file prefix)
    if sample_name.startswith("sample-"):
        file_prefix = "sample/" + sample_name.removeprefix("sample-") + "/"
    else:
        file_prefix = sample_name + "/"

    click.echo(
        f"resolving HuggingFaceFW/fineweb-edu parquet files (subset={sample_name})..."
    )
    api = HfApi()
    all_files = api.list_repo_files(repo_id, repo_type="dataset")
    parquet_files = sorted(
        f for f in all_files if f.startswith(file_prefix) and f.endswith(".parquet")
    )
    if not parquet_files:
        raise click.UsageError(
            f"no parquet files found under {file_prefix} in {repo_id}"
        )
    click.echo(f"  found {len(parquet_files)} parquet files; will read in order")

    revision: str | None = None
    written = 0
    rows = 0
    for fname in parquet_files:
        if written >= target_bytes:
            break
        click.echo(f"  downloading {fname} (cached after first time)...")
        local_path = hf_hub_download(repo_id=repo_id, filename=fname, repo_type="dataset")
        # Capture revision (parent of snapshots/<rev>/)
        if revision is None:
            parts = Path(local_path).resolve().parts
            if "snapshots" in parts:
                idx = parts.index("snapshots")
                if idx + 1 < len(parts):
                    revision = parts[idx + 1]

        click.echo(f"  streaming rows from {fname}...")
        pf = pq.ParquetFile(local_path)
        for batch in pf.iter_batches(batch_size=1000, columns=["text"]):
            for text_scalar in batch.column("text"):
                text = text_scalar.as_py()
                if not text or not text.strip():
                    continue
                record = json.dumps({"text": text}, ensure_ascii=False) + "\n"
                out_handle.write(record)
                written += len(record.encode("utf-8"))
                rows += 1
                if rows % 1000 == 0:
                    pct = 100 * written / target_bytes
                    click.echo(
                        f"  fineweb rows={rows:>7}  written={written / 1e6:>6.1f} MB "
                        f"({pct:5.1f} %)"
                    )
                if written >= target_bytes:
                    break
            if written >= target_bytes:
                break
    click.echo(f"  done. fineweb rows={rows}, bytes={written / 1e6:.1f} MB")
    return written, rows, revision


def _append_docs_tiled(
    out_handle: IO[str],
    docs_path: Path,
    target_bytes: int,
) -> tuple[int, int]:
    """Tile docs corpus until `target_bytes` reached. Each tile re-emits the file in order."""
    if not docs_path.exists():
        raise FileNotFoundError(f"docs corpus not found: {docs_path}")

    with docs_path.open() as f:
        docs_lines = [line for line in f if line.strip()]
    click.echo(f"docs corpus: {len(docs_lines)} non-empty lines")

    written = 0
    tile = 0
    while written < target_bytes:
        tile += 1
        for line in docs_lines:
            out_handle.write(line)
            written += len(line.encode("utf-8"))
            if written >= target_bytes:
                break
        if written < target_bytes:
            click.echo(
                f"  tile {tile} complete; {written / 1e6:.1f} MB; tiling again..."
            )
    click.echo(f"  done. docs tiles={tile}, bytes={written / 1e6:.1f} MB")
    return written, tile


@click.command()
@click.option("--out", required=True, type=click.Path())
@click.option(
    "--target-bytes",
    default=2_400_000_000,  # ~430M GPT-2 tokens ≈ 380M custom-BPE tokens
    type=int,
    help="Total target bytes (FineWeb + docs combined).",
)
@click.option(
    "--fineweb-subset",
    default="sample-10BT",
    help="HF FineWeb-Edu subset name. sample-10BT = 10B-token slice.",
)
@click.option(
    "--docs-corpus",
    default=str(DEFAULT_DOCS_CORPUS),
    type=click.Path(exists=True),
)
@click.option(
    "--manifest",
    default=None,
    type=click.Path(),
    help="Manifest output path. Defaults to mix_manifest.json next to --out.",
)
@click.option(
    "--docs-ratio",
    default=DOCS_RATIO,
    type=float,
    help="Fraction of total bytes that should be docs (tiled). Default 0.05 (5 %).",
)
def main(
    out: str,
    target_bytes: int,
    fineweb_subset: str,
    docs_corpus: str,
    manifest: str | None,
    docs_ratio: float,
) -> None:
    """Build a mixed FineWeb-Edu + docs corpus for v3.1 pretraining."""
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = (
        Path(manifest) if manifest else out_path.with_name("mix_manifest.json")
    )

    docs_target = int(target_bytes * docs_ratio)
    fw_target = target_bytes - docs_target

    click.echo(f"target total: {target_bytes / 1e9:.2f} GB")
    click.echo(f"  fineweb: {fw_target / 1e9:.2f} GB ({100 * (1 - docs_ratio):.0f} %)")
    click.echo(f"  docs:    {docs_target / 1e6:.0f} MB ({100 * docs_ratio:.0f} %, tiled)")

    started_at = datetime.now(timezone.utc)
    with out_path.open("w", encoding="utf-8") as f:
        fw_bytes, fw_rows, fw_rev = _stream_fineweb_to_byte_target(
            f, fw_target, fineweb_subset
        )
        docs_bytes, docs_tiles = _append_docs_tiled(f, Path(docs_corpus), docs_target)
    finished_at = datetime.now(timezone.utc)

    total = fw_bytes + docs_bytes
    click.echo(f"\nfinal: {total / 1e9:.2f} GB written → {out_path}")

    manifest_data = {
        "out_path": str(out_path),
        "target_bytes": target_bytes,
        "fineweb": {
            "subset": fineweb_subset,
            "revision": fw_rev,
            "rows": fw_rows,
            "bytes": fw_bytes,
        },
        "docs": {
            "source": str(Path(docs_corpus).resolve()),
            "tiles": docs_tiles,
            "bytes": docs_bytes,
            "ratio_target": docs_ratio,
        },
        "total_bytes": total,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "wall_clock_seconds": (finished_at - started_at).total_seconds(),
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    click.echo(f"manifest → {manifest_path}")


if __name__ == "__main__":
    main()
