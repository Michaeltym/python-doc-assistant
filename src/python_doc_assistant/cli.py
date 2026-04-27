"""CLI entry point for python-doc-assistant.

Subcommands (plan §7 + §9):
    ingest        Download docs archive (calls fetch_docs.ingest_docs).
    build-index   Parse objects.inv + chunk HTML + persist chunks.jsonl + bm25.pkl.
    search        Load indexes + route query + print top-k.
    eval          Run eval set; write results.json + per_query.jsonl (plan §9).
"""

from __future__ import annotations

import json
import sys
import tomllib
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import click

from python_doc_assistant.evaluation.dataset import load_eval_set
from python_doc_assistant.evaluation.retrieval_metrics import (
    EvalRunResult,
    RetrievedChunk,
    evaluate,
)
from python_doc_assistant.evaluation.run_writer import RunMetadata, make_run_dir, write_run
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
# Subcommand: eval
# ------------------------------------------------------------------


@main.command(name="eval")
@click.option(
    "--set",
    "set_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to JSONL eval set.",
)
@click.option("--tag", required=True, help="Run tag (e.g. 'v0-bm25').")
@click.option("--version", default=None, help="Override config.toml DOCS_VERSION.")
@click.option("--docs-sha", default=None, help="Use a specific sha_short (else current.txt).")
@click.option("--k", "k", type=int, default=10, help="Top-k retrieved per query.")
@click.option(
    "--model",
    "model_id",
    default=None,
    help=(
        "HuggingFace model_id to run grounded generation per query "
        "(v1 §4). Omit for retrieval-only eval (v0). Examples: "
        "Qwen/Qwen2.5-1.5B-Instruct, Qwen/Qwen2.5-Coder-1.5B-Instruct."
    ),
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Force overwrite if the run directory already exists.",
)
def eval_cmd(
    set_path: Path,
    tag: str,
    version: str | None,
    docs_sha: str | None,
    k: int,
    model_id: str | None,
    overwrite: bool,
) -> None:
    """Run eval set against persisted indexes; write results.json + per_query.jsonl.

    Suggested flow (plan §9):
        eff_version = _resolve_docs_version(version, config_path=DEFAULT_CONFIG_PATH)
        eff_sha     = _resolve_docs_sha(eff_version, docs_sha, data_root=DEFAULT_DATA_ROOT)

        chunks   = _load_chunks(<chunks.jsonl path>)
        bm25     = BM25Index.load(<bm25.pkl path>)
        symbols  = parse_objects_inv(<docs_dir>)
        sym_idx  = SymbolIndex(chunks, symbols)

        eval_queries = load_eval_set(set_path)

        retrieve_fn = _make_retrieve_fn(sym_idx, bm25, chunks_by_id)
        run_result  = evaluate(eval_queries, retrieve_fn, max_k=k)

        manifest = _load_ingest_manifest(<docs_dir>)
        metadata = RunMetadata(
            docs_version=eff_version,
            docs_served_version=manifest["docs_served_version"],
            docs_sha_short=eff_sha,
            ingest_manifest=manifest,
            config={"retrieval_mode": "bm25+symbol", "k": k, "eval_set": str(set_path)},
            tag=tag,
            command=" ".join(sys.argv),
        )

        out_dir = make_run_dir(tag)
        write_run(out_dir, run_result, metadata, overwrite=overwrite)

        click.echo(f"recall@5={run_result.recall_at_5:.3f}  "
                   f"recall@10={run_result.recall_at_10:.3f}  "
                   f"mrr={run_result.mrr:.3f}")
        click.echo(f"run_dir={out_dir}")
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
    eval_queries = load_eval_set(set_path)

    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    retrieve_fn = _make_retrieve_fn(symbol_index, bm25_index, chunks_by_id)
    run_result, model_field, decoding_params = _run_eval_with_optional_generation(
        eval_queries=eval_queries,
        retrieve_fn=retrieve_fn,
        chunks_by_id=chunks_by_id,
        max_k=k,
        model_id=model_id,
    )

    manifest = _load_ingest_manifest(docs_dir)
    metadata = RunMetadata(
        docs_version=docs_version,
        docs_served_version=manifest["docs_served_version"],
        docs_sha_short=docs_sha,
        ingest_manifest=manifest,
        config={"retrieval_mode": "bm25+symbol", "k": k, "eval_set": str(set_path)},
        tag=tag,
        command=" ".join(sys.argv),
        model=model_field,
        decoding_params=decoding_params,
    )
    out_dir = make_run_dir(tag)
    write_run(out_dir, run_result, metadata, overwrite=overwrite)
    click.echo(
        f"recall@5={run_result.recall_at_5:.3f}  "
        f"recall@10={run_result.recall_at_10:.3f}  "
        f"mrr={run_result.mrr:.3f}"
    )
    click.echo(f"run_dir={out_dir}")


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


def _load_ingest_manifest(docs_dir: Path) -> dict[str, Any]:
    """Read docs_dir/ingest_manifest.json into a dict for RunMetadata.ingest_manifest.

    Raises click.ClickException if the manifest is missing or malformed.
    """
    path = docs_dir / "ingest_manifest.json"
    if not path.exists():
        raise click.ClickException(f"{path} is not found")
    if not path.is_file():
        raise click.ClickException(f"{path} is not a regular file")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
        if not isinstance(data, dict):
            raise click.ClickException(f"manifest at {path} is not a JSON object")
    return data


def _make_retrieve_fn(
    symbol_index: SymbolIndex,
    bm25_index: BM25Index,
    chunks_by_id: dict[str, Chunk],
) -> Callable[[str, int], list[RetrievedChunk]]:
    """Closure factory: returns a retrieve_fn for evaluate().

    The returned fn calls route() and converts result.chunk_ids into
    RetrievedChunk records (chunk_id + score + 1-indexed rank + url + symbols).

    Score note: route() does not surface BM25/symbol-index scores in v0.
    Use rank-derived placeholder (e.g. 1.0 / rank or 0.0) and document this
    in v0 narrative; v1 may extend route() to return scored hits.
    """

    def retrieve_fn(query: str, k: int) -> list[RetrievedChunk]:
        result = route(query, symbol_index=symbol_index, bm25_index=bm25_index, k=k)
        chunks: list[RetrievedChunk] = []
        for index, chunk_id in enumerate(result.chunk_ids, start=1):
            chunk = chunks_by_id.get(chunk_id)
            if chunk:
                chunks.append(
                    RetrievedChunk(
                        chunk_id=chunk_id,
                        score=1.0 / index,
                        rank=index,
                        canonical_url=chunk.canonical_url,
                        symbols=chunk.symbols,
                    )
                )

        return chunks

    return retrieve_fn


def _run_eval_with_optional_generation(
    *,
    eval_queries: list[Any],
    retrieve_fn: Callable[[str, int], list[RetrievedChunk]],
    chunks_by_id: dict[str, Chunk],
    max_k: int,
    model_id: str | None,
) -> tuple[EvalRunResult, str | None, dict[str, Any] | None]:
    """Dispatch retrieval-only vs retrieval+generation eval.

    Returns:
        (run_result, model_field, decoding_params)

        - model_id is None → retrieval-only (v0 path); model + decoding
          stay None.
        - model_id is set → retrieval + generation (v1 §4); model returns
          the id verbatim and decoding_params surfaces the generator's
          decoding config so results.json is reproducible.

    Splitting this out keeps `eval_cmd` body small and lets tests cover
    the dispatch without spinning up a real model.
    """
    if model_id is None:
        run_result = evaluate(eval_queries, retrieve_fn, max_k=max_k)
        return run_result, None, None

    from python_doc_assistant.evaluation.generation_eval import evaluate_with_generation
    from python_doc_assistant.generation.qwen_backend import QwenGenerator

    generator = QwenGenerator(model_id)
    run_result = evaluate_with_generation(
        eval_queries, retrieve_fn, generator, chunks_by_id, max_k=max_k
    )
    decoding_params = {
        "temperature": generator.temperature,
        "top_p": generator.top_p,
        "max_new_tokens": generator.max_new_tokens,
    }
    return run_result, model_id, decoding_params


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
    "EvalRunResult",
    "RetrievedChunk",
    "RunMetadata",
    "SymbolEntry",
    "SymbolIndex",
    "analyze",
    "build_chunks",
    "build_index",
    "eval_cmd",
    "evaluate",
    "ingest",
    "ingest_docs",
    "load_eval_set",
    "main",
    "make_run_dir",
    "parse_objects_inv",
    "route",
    "search",
    "write_run",
]
