"""CLI entry point for python-doc-assistant.

Subcommands (plan §7):
    ingest        Download docs archive (calls fetch_docs.ingest_docs).
    build-index   Parse objects.inv + chunk HTML + persist chunks.jsonl + bm25.pkl.
    search        Load indexes + route query + print top-k.

`eval` is added in §9 once retrieval_metrics + run_writer exist.
"""

from __future__ import annotations

import json
import tomllib
from dataclasses import asdict
from pathlib import Path

import click

from python_doc_assistant.indexes.bm25_index import BM25Index, analyze
from python_doc_assistant.indexes.symbol_index import SymbolIndex
from python_doc_assistant.ingest.chunker import Chunk, build_chunks
from python_doc_assistant.ingest.fetch_docs import ingest_docs
from python_doc_assistant.ingest.parse_objects_inv import SymbolEntry, parse_objects_inv
from python_doc_assistant.retrieval.router import route

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

DEFAULT_CONFIG_PATH: Path = Path("config.toml")
DEFAULT_DATA_ROOT: Path = Path("data")


# ------------------------------------------------------------------
# Click group
# ------------------------------------------------------------------


@click.group()
def main() -> None:
    """python-doc-assistant CLI."""


# ------------------------------------------------------------------
# Subcommand: ingest
# ------------------------------------------------------------------


@main.command()
@click.option("--version", required=True, help="Major.minor docs branch (e.g. 3.12).")
@click.option("--force-switch", is_flag=True, help="Allow switching the active sha.")
def ingest(version: str, force_switch: bool) -> None:
    """Download the Python docs archive for --version.

    Suggested flow:
        result = ingest_docs(version, force_switch=force_switch)
        click.echo(f"docs_sha_short={result.sha_short}  skipped={result.skipped}")
        click.echo(f"docs_dir={result.docs_dir}")
    """
    result = ingest_docs(version, force_switch=force_switch)
    click.echo(f"docs_sha_short={result.sha_short}  skipped={result.skipped}")
    click.echo(f"docs_dir={result.docs_dir}")


# ------------------------------------------------------------------
# Subcommand: build-index
# ------------------------------------------------------------------


@main.command(name="build-index")
@click.option("--version", default=None, help="Override config.toml DOCS_VERSION.")
@click.option("--docs-sha", default=None, help="Use a specific sha_short (else current.txt).")
def build_index(version: str | None, docs_sha: str | None) -> None:
    """Parse symbols + chunk HTML + persist chunks.jsonl + bm25.pkl.

    Suggested flow:
        eff_version = _resolve_docs_version(version)
        eff_sha     = _resolve_docs_sha(eff_version, docs_sha)
        docs_dir    = DEFAULT_DATA_ROOT / "docs"   / eff_version / eff_sha
        chunks_path = DEFAULT_DATA_ROOT / "chunks" / eff_version / eff_sha / "chunks.jsonl"
        bm25_path   = DEFAULT_DATA_ROOT / "indexes"/ eff_version / eff_sha / "bm25.pkl"

        symbols = parse_objects_inv(docs_dir)
        chunks  = build_chunks(docs_dir, eff_version, symbols)
        _save_chunks(chunks, chunks_path)

        idx = BM25Index(chunks)
        idx.save(bm25_path)

        click.echo(f"chunks={len(chunks)}  -> {chunks_path}")
        click.echo(f"bm25  -> {bm25_path}")
    """
    docs_version = _resolve_docs_version(version)
    docs_sha = _resolve_docs_sha(docs_version, docs_sha)
    docs_dir = DEFAULT_DATA_ROOT / "docs" / docs_version / docs_sha
    chunks_path = DEFAULT_DATA_ROOT / "chunks" / docs_version / docs_sha / "chunks.jsonl"
    bm25_path = DEFAULT_DATA_ROOT / "indexes" / docs_version / docs_sha / "bm25.pkl"
    symbols = parse_objects_inv(docs_dir)
    chunks = build_chunks(docs_dir, docs_version, symbols)
    _save_chunks(chunks, chunks_path)
    bm25_index = BM25Index(chunks)
    bm25_index.save(bm25_path)
    click.echo(f"chunks={len(chunks)}  -> {chunks_path}")
    click.echo(f"bm25  -> {bm25_path}")


# ------------------------------------------------------------------
# Subcommand: search
# ------------------------------------------------------------------


@main.command()
@click.argument("query")
@click.option("--k", "k", type=int, default=5, help="Top-k results.")
@click.option("--version", default=None)
@click.option("--docs-sha", default=None)
@click.option("--debug", is_flag=True, help="Print scores / routing decision / tokens.")
def search(query: str, k: int, version: str | None, docs_sha: str | None, debug: bool) -> None:
    """Search the persisted indexes for QUERY.

    Suggested flow:
        eff_version = _resolve_docs_version(version)
        eff_sha     = _resolve_docs_sha(eff_version, docs_sha)
        chunks      = _load_chunks(... chunks.jsonl path ...)
        bm25        = BM25Index.load(... bm25.pkl path ...)
        symbols     = ... rebuild SymbolEntry list, e.g. parse_objects_inv on docs_dir
        sym_idx     = SymbolIndex(chunks, symbols)

        result = route(query, symbol_index=sym_idx, bm25_index=bm25, k=k)
        for cid in result.chunk_ids:
            click.echo(cid)

        if debug:
            click.echo(f"query_type={result.query_type.value}  used={result.used}")
            click.echo(f"docs_sha_short={eff_sha}")
            click.echo(f"tokens={analyze(query)}")
    """
    docs_version = _resolve_docs_version(version)
    docs_sha = _resolve_docs_sha(docs_version, docs_sha)
    docs_dir = DEFAULT_DATA_ROOT / "docs" / docs_version / docs_sha
    chunks_path = DEFAULT_DATA_ROOT / "chunks" / docs_version / docs_sha / "chunks.jsonl"
    bm25_path = DEFAULT_DATA_ROOT / "indexes" / docs_version / docs_sha / "bm25.pkl"
    chunks = _load_chunks(chunks_path)
    symbols = parse_objects_inv(docs_dir)
    symbol_index = SymbolIndex(chunks, symbols)
    bm25_index = BM25Index.load(bm25_path)
    result = route(query, symbol_index=symbol_index, bm25_index=bm25_index, k=k)
    for cid in result.chunk_ids:
        click.echo(cid)

    if debug:
        click.echo(f"query_type={result.query_type.value}  used={result.used}")
        click.echo(f"docs_sha_short={docs_sha}")
        click.echo(f"tokens={analyze(query)}")


# ------------------------------------------------------------------
# Helpers (private)
# ------------------------------------------------------------------


def _resolve_docs_version(
    override: str | None,
    *,
    config_path: Path | None = None,
) -> str:
    """Effective DOCS_VERSION: --version override beats config.toml.

    Raises click.ClickException if neither source provides a value.
    """
    if override:
        return override
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise click.ClickException(f"{config_path} is not found")
    with config_path.open("rb") as f:
        data = tomllib.load(f)
    version = data.get("DOCS_VERSION")
    if not isinstance(version, str):
        raise click.ClickException("DOCS_VERSION missing in config")
    return version.strip()


def _resolve_docs_sha(
    version: str,
    override: str | None,
    *,
    data_root: Path | None = None,
) -> str:
    """Effective sha_short: --docs-sha override beats data/docs/<version>/current.txt.

    Prints `resolved docs-sha=<sha>` to stdout when falling back to current.txt
    (plan §7). Raises click.ClickException if no source resolves.
    """
    if override:
        return override
    if data_root is None:
        data_root = DEFAULT_DATA_ROOT
    path = data_root / "docs" / version / "current.txt"
    if not path.exists():
        raise click.ClickException(f"{path} is not found")
    with path.open(encoding="utf-8") as f:
        sha_short = f.read()
        if sha_short:
            click.echo(f"resolved docs-sha={sha_short}")
            return sha_short.strip()

    raise click.ClickException("No docs sha found")


def _save_chunks(chunks: list[Chunk], path: Path) -> None:
    """Write chunks.jsonl (one JSON object per line; symbols tuple -> list)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk)) + "\n")


def _load_chunks(path: Path) -> list[Chunk]:
    """Read chunks.jsonl back into list[Chunk] (symbols list -> tuple)."""
    if not path.exists():
        raise click.ClickException(f"{path} is not found")
    if not path.is_file():
        raise click.ClickException(f"{path} is not a regular file")
    chunks: list[Chunk] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            data["symbols"] = tuple(data["symbols"])
            chunks.append(Chunk(**data))
    return chunks


# ------------------------------------------------------------------
# Imports referenced by subcommands when implemented
# ------------------------------------------------------------------

# These are listed here so the type checker sees the dependency graph; the
# subcommands above import + call them at implementation time.
__all__ = [
    "BM25Index",
    "Chunk",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_DATA_ROOT",
    "SymbolEntry",
    "SymbolIndex",
    "analyze",
    "build_chunks",
    "build_index",
    "ingest",
    "ingest_docs",
    "main",
    "parse_objects_inv",
    "route",
    "search",
]
