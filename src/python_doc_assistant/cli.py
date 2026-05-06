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
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

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

if TYPE_CHECKING:
    from python_doc_assistant.generation.interface import Generator
    from python_doc_assistant.retrieval.hyde import HypotheticalGenerator

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
@click.option(
    "--with-dense",
    is_flag=True,
    help=(
        "Also build the dense embedding index (saves to dense.npy + dense.json "
        "next to bm25.pkl). Requires the `embedding` extra installed."
    ),
)
def build_index(
    version: str | None,
    docs_sha: str | None,
    with_dense: bool,
) -> None:
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
    if with_dense:
        from python_doc_assistant.indexes.dense_index import DenseIndex

        dense_path = DEFAULT_DATA_ROOT / "indexes" / docs_version / docs_sha / "dense.npy"
        dense_index = DenseIndex(chunks)
        dense_index.save(dense_path)
        click.echo(f"dense -> {dense_path}")


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
# Subcommand: ask (plan v1 §5)
# ------------------------------------------------------------------


@main.command(name="ask")
@click.argument("query")
@click.option("--k", "k", type=int, default=5, help="Top-k chunks to retrieve.")
@click.option(
    "--backend",
    type=click.Choice(["qwen", "qwen-gguf", "tinydocs"]),
    default="qwen",
    help=(
        "Generator backend. 'qwen' = HF transformers (v1). "
        "'qwen-gguf' = llama.cpp + GGUF (v4 sub-task 3'). "
        "'tinydocs' loads a self-trained checkpoint (v3 §6)."
    ),
)
@click.option(
    "--model",
    "model_id",
    default="Qwen/Qwen2.5-1.5B-Instruct",
    help="HuggingFace model_id (qwen backend only).",
)
@click.option(
    "--gguf-model",
    "gguf_model_path",
    default=None,
    type=click.Path(exists=True),
    help="Path to GGUF model file (required when --backend=qwen-gguf).",
)
@click.option(
    "--tinydocs-ckpt",
    default=None,
    type=click.Path(exists=True),
    help="Path to TinyDocs step_<N>.pt (required when --backend=tinydocs).",
)
@click.option(
    "--tinydocs-tok",
    default=None,
    type=click.Path(exists=True),
    help="Path to TinyDocs tokenizer.json (required when --backend=tinydocs).",
)
@click.option("--version", default=None, help="Override config.toml DOCS_VERSION.")
@click.option("--docs-sha", default=None, help="Use a specific sha_short (else current.txt).")
@click.option(
    "--debug",
    is_flag=True,
    help="Also print retrieved chunks (id + score), final prompt, citation match.",
)
def ask(
    query: str,
    k: int,
    backend: str,
    model_id: str,
    gguf_model_path: str | None,
    tinydocs_ckpt: str | None,
    tinydocs_tok: str | None,
    version: str | None,
    docs_sha: str | None,
    debug: bool,
) -> None:
    """Answer QUERY using grounded retrieval + Qwen generation.

    Default output:  the model's answer text on stdout (one block).
    With --debug:    additionally prints the four blocks plan §5 requires:
        1. retrieved chunks (rank + score + chunk_id)
        2. final prompt — full chat-template messages fed to the LLM
        3. (the answer is already printed)
        4. citation validation — for each chunk_id the model cited,
           whether it was in the top-K retrieved set

    Suggested flow:
        eff_version = _resolve_docs_version(version)
        eff_sha     = _resolve_docs_sha(eff_version, docs_sha)
        docs_dir    = DEFAULT_DATA_ROOT / "docs"   / eff_version / eff_sha
        chunks_path = DEFAULT_DATA_ROOT / "chunks" / eff_version / eff_sha / "chunks.jsonl"
        bm25_path   = DEFAULT_DATA_ROOT / "indexes"/ eff_version / eff_sha / "bm25.pkl"

        chunks       = _load_chunks(chunks_path)
        symbols      = parse_objects_inv(docs_dir)
        symbol_index = SymbolIndex(chunks, symbols)
        bm25_index   = BM25Index.load(bm25_path)
        chunks_by_id = {c.chunk_id: c for c in chunks}

        retrieve_fn = _make_retrieve_fn(symbol_index, bm25_index, chunks_by_id)
        retrieved   = retrieve_fn(query, k)            # list[RetrievedChunk]
        gen_chunks  = [chunks_by_id[r.chunk_id] for r in retrieved if r.chunk_id in chunks_by_id]

        from python_doc_assistant.generation.qwen_backend import QwenGenerator
        from python_doc_assistant.prompts.grounded import build_grounded_prompt
        from python_doc_assistant.retrieval.router import classify

        generator = QwenGenerator(model_id)
        qt        = classify(query)
        answer    = generator.generate(query, gen_chunks, query_type=qt)

        click.echo(answer.text or "[INSUFFICIENT-CONTEXT]")

        if debug:
            click.echo("")
            click.echo("[debug] retrieved:")
            for r in retrieved:
                click.echo(f"  rank={r.rank}  score={r.score:.3f}  id={r.chunk_id}")
            click.echo("")
            click.echo("[debug] prompt (chat messages):")
            messages = build_grounded_prompt(query, gen_chunks, query_type=qt)
            for m in messages:
                click.echo(f"--- {m['role']} ---")
                click.echo(m["content"])
            click.echo("")
            click.echo("[debug] citations:")
            retrieved_ids = {r.chunk_id for r in retrieved}
            if not answer.cited_chunk_ids:
                click.echo("  (none)")
            else:
                for cid in answer.cited_chunk_ids:
                    in_set = "yes" if cid in retrieved_ids else "no (not in top-K)"
                    click.echo(f"  cited={cid}  in_retrieved={in_set}")
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

    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    retrieve_fn = _make_retrieve_fn(symbol_index, bm25_index, chunks_by_id)
    retrieved = retrieve_fn(query, k)
    gen_chunks = [chunks_by_id[r.chunk_id] for r in retrieved if r.chunk_id in chunks_by_id]
    from python_doc_assistant.prompts.grounded import build_grounded_prompt
    from python_doc_assistant.retrieval.query_rewriter import maybe_rewrite_query
    from python_doc_assistant.retrieval.router import classify

    generator: Generator
    if backend == "tinydocs":
        if tinydocs_ckpt is None or tinydocs_tok is None:
            raise click.UsageError("--backend=tinydocs requires --tinydocs-ckpt and --tinydocs-tok")
        from python_doc_assistant.generation.tinydocs_backend import TinyDocsGenerator

        generator = TinyDocsGenerator(
            checkpoint_path=Path(tinydocs_ckpt),
            tokenizer_path=Path(tinydocs_tok),
        )
    elif backend == "qwen-gguf":
        if gguf_model_path is None:
            raise click.UsageError("--backend=qwen-gguf requires --gguf-model")
        from python_doc_assistant.generation.qwen_gguf_backend import QwenGGUFGenerator

        generator = QwenGGUFGenerator(model_path=Path(gguf_model_path))
    else:
        from python_doc_assistant.generation.qwen_backend import QwenGenerator

        generator = QwenGenerator(model_id)
    qt = classify(query)
    generator_query = maybe_rewrite_query(query, gen_chunks)
    answer = generator.generate(generator_query, gen_chunks, query_type=qt)
    click.echo(answer.text or "[INSUFFICIENT-CONTEXT]")

    if debug:
        click.echo("")
        if generator_query != query:
            click.echo(f"[debug] query rewritten: {query!r} -> {generator_query!r}")
            click.echo("")
        click.echo("[debug] retrieved:")
        for r in retrieved:
            click.echo(f"  rank={r.rank}  score={r.score:.3f}  id={r.chunk_id}")
        click.echo("")
        click.echo("[debug] prompt (chat messages):")
        messages = build_grounded_prompt(generator_query, gen_chunks, query_type=qt)
        for m in messages:
            click.echo(f"--- {m['role']} ---")
            click.echo(m["content"])
        click.echo("")
        click.echo("[debug] citations:")
        retrieved_ids = {r.chunk_id for r in retrieved}
        if not answer.cited_chunk_ids:
            click.echo("  (none)")
        else:
            for cid in answer.cited_chunk_ids:
                in_set = "yes" if cid in retrieved_ids else "no (not in top-K)"
                click.echo(f"  cited={cid}  in_retrieved={in_set}")


# ------------------------------------------------------------------
# Subcommand: judge (plan v2 §6)
# ------------------------------------------------------------------


@main.command(name="judge")
@click.option(
    "--run-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Path to a run dir (must contain per_query.jsonl + results.json).",
)
@click.option(
    "--judge-model",
    default="claude-haiku-4-5-20251001",
    help="Anthropic model id used for scoring.",
)
@click.option("--temperature", type=float, default=0.0)
@click.option("--max-tokens", type=int, default=200)
@click.option(
    "--max-rows",
    type=int,
    default=None,
    help="Truncate to first N rows (smoke / cost cap).",
)
@click.option(
    "--rerun-existing",
    is_flag=True,
    help="Re-judge even if `<run-dir>/judge_scores.jsonl` exists.",
)
def judge_cmd(
    run_dir: Path,
    judge_model: str,
    temperature: float,
    max_tokens: int,
    max_rows: int | None,
    rerun_existing: bool,
) -> None:
    """Score every per_query.jsonl row with an LLM judge (plan v2 §6).

    Workflow:
        1. Loads `run_dir/per_query.jsonl` (one row per evaluated query).
        2. Loads `data/chunks/<docs_version>/<docs_sha>/chunks.jsonl` so
           the judge prompt can include retrieved chunk TEXT (not just
           ids). docs_version + docs_sha are read from
           `run_dir/results.json`.
        3. For each row, calls `evaluation.judge.judge_one(...)` →
           Anthropic API → parsed `JudgeRecord`.
        4. Writes `run_dir/judge_scores.jsonl` (full JudgeRecord per
           row; reproducibility metadata included).
        5. Updates `run_dir/results.json` with two new top-level keys:
              "judge"           — model_id / judge_prompt_hash /
                                  temperature / max_tokens / n_records /
                                  n_errors / timestamp_started /
                                  timestamp_completed
              "judge_aggregate" — output of `evaluation.human_scoring.aggregate()`
                                  on the parsed tiers.

    Skips silently when `judge_scores.jsonl` already exists, unless
    `--rerun-existing` is passed.

    Suggested implementation flow (your code goes in here):

        # Skip-existing gate
        out_path = run_dir / "judge_scores.jsonl"
        if out_path.exists() and not rerun_existing:
            click.echo(f"SKIP — {out_path.name} exists; pass --rerun-existing")
            return

        # Lazy-import so v0 / no-judge-extra paths still import this module.
        import anthropic
        from python_doc_assistant.evaluation.judge import (
            JudgeError, judge_one, judge_prompt_hash,
            judge_records_to_human_scores, write_judge_records,
        )
        from python_doc_assistant.evaluation.human_scoring import aggregate

        # Load run metadata to find the chunks.jsonl
        results_path = run_dir / "results.json"
        results = json.loads(results_path.read_text(encoding="utf-8"))
        chunks_path = (
            DEFAULT_DATA_ROOT / "chunks" / results["docs_version"]
            / results["docs_sha_short"] / "chunks.jsonl"
        )
        chunks_by_id = {c.chunk_id: c for c in _load_chunks(chunks_path)}

        # Read per_query.jsonl rows (truncate if --max-rows)
        rows = ... read run_dir / "per_query.jsonl" line-by-line ...
        if max_rows:
            rows = rows[:max_rows]

        # Loop with API client + collect records
        client = anthropic.Anthropic()
        records, errors = [], 0
        from datetime import datetime, timezone
        started = datetime.now(timezone.utc).isoformat()
        for i, row in enumerate(rows, start=1):
            retrieved_ids = [r["chunk_id"] for r in row.get("retrieved", [])][:5]
            retrieved_chunks = [
                chunks_by_id[cid] for cid in retrieved_ids if cid in chunks_by_id
            ]
            try:
                rec = judge_one(
                    query=row["query"],
                    expected_symbols=tuple(row.get("expected_symbols") or ()),
                    retrieved_chunks=retrieved_chunks,
                    cited_chunk_ids=tuple(row.get("cited_chunk_ids") or ()),
                    refused=bool(row.get("refused")),
                    model_output_text=row.get("model_output_text") or "",
                    client=client,
                    model_id=judge_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                records.append(rec)
            except JudgeError as exc:
                errors += 1
                click.echo(f"  parse-fail row {i}: {exc}")

        completed = datetime.now(timezone.utc).isoformat()

        # Write JudgeRecord JSONL + update results.json
        write_judge_records(records, out_path)
        agg = aggregate(judge_records_to_human_scores(records))
        results["judge"] = {
            "judge_model": judge_model,
            "judge_prompt_hash": judge_prompt_hash(),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "n_records": len(records),
            "n_errors": errors,
            "timestamp_started": started,
            "timestamp_completed": completed,
        }
        results["judge_aggregate"] = agg
        results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

        click.echo(
            f"hallucination_rate={agg['hallucination_rate']:.3f}  "
            f"correct_rate={agg['correct_rate']:.3f}  "
            f"n_records={len(records)}  n_errors={errors}"
        )
        click.echo(f"out: {out_path}")
    """
    judge_scores_path = run_dir / "judge_scores.jsonl"
    per_query_path = run_dir / "per_query.jsonl"
    results_path = run_dir / "results.json"
    if not rerun_existing and judge_scores_path.exists():
        click.echo(f"SKIP: {judge_scores_path} exists")
        return
    if not results_path.exists():
        raise FileNotFoundError(f"{results_path} does not exist")
    with results_path.open("r", encoding="utf-8") as f:
        results = json.load(f)
    docs_version = results["docs_version"]
    docs_sha_short = results["docs_sha_short"]
    chunks_path = DEFAULT_DATA_ROOT / "chunks" / docs_version / docs_sha_short / "chunks.jsonl"
    chunks = _load_chunks(chunks_path)
    import anthropic

    from python_doc_assistant.evaluation.human_scoring import aggregate
    from python_doc_assistant.evaluation.judge import (
        JudgeError,
        JudgeRecord,
        judge_one,
        judge_prompt_hash,
        judge_records_to_human_scores,
        write_judge_records,
    )

    records: list[JudgeRecord] = []
    errors = 0
    client = anthropic.Anthropic()
    chunks_by_id = {c.chunk_id: c for c in chunks}
    if not per_query_path.exists():
        raise FileNotFoundError(f"{per_query_path} does not exist")
    started = datetime.now(timezone.utc).isoformat()
    with per_query_path.open("r", encoding="utf-8") as f:
        total = sum(1 for line in f if line.strip())
    if max_rows is not None:
        total = min(total, max_rows)
    click.echo(f"judge {run_dir.name} — {total} rows", nl=True)
    import time as _time

    last_t = _time.monotonic()
    with per_query_path.open("r", encoding="utf-8") as f:
        for index, line in enumerate(f, start=1):
            if max_rows is not None and index > max_rows:
                break
            stripped_line = line.strip()
            if not stripped_line:
                continue
            obj = json.loads(stripped_line)
            if not isinstance(obj, dict):
                raise ValueError(f"Line {index}: invalid json object")
            query = obj["query"]
            expected_symbols = obj["expected_symbols"]
            retrieved_chunks = [
                chunks_by_id[r["chunk_id"]]
                for r in obj.get("retrieved", [])
                if r["chunk_id"] in chunks_by_id
            ]
            cited_chunk_ids = obj["cited_chunk_ids"]
            refused = obj["refused"]
            model_output_text = obj["model_output_text"]
            try:
                record = judge_one(
                    query,
                    expected_symbols,
                    retrieved_chunks,
                    cited_chunk_ids,
                    refused,
                    model_output_text,
                    client=client,
                    model_id=judge_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                records.append(record)
                tier = record.tier
            except JudgeError as e:
                errors += 1
                tier = f"PARSE-FAIL: {e}"
            now = _time.monotonic()
            click.echo(
                f"[{index:3d}/{total}] dt={now - last_t:5.2f}s tier={tier}",
                nl=True,
            )
            sys.stdout.flush()
            last_t = now

    write_judge_records(records, judge_scores_path)
    judge_agg = aggregate(judge_records_to_human_scores(records))
    results["judge"] = {
        "judge_model": judge_model,
        "judge_prompt_hash": judge_prompt_hash(),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "n_records": len(records),
        "n_errors": errors,
        "timestamp_started": started,
        "timestamp_completed": datetime.now(timezone.utc).isoformat(),
    }
    results["judge_aggregate"] = judge_agg
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    click.echo(
        f"hallucination_rate={judge_agg['hallucination_rate']:.3f}  "
        f"correct_rate={judge_agg['correct_rate']:.3f}  "
        f"n_records={len(records)}  n_errors={errors}"
    )
    click.echo(f"out: {results_path}")


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
    "--retriever",
    type=click.Choice(
        ["bm25", "symbol+bm25", "dense", "hybrid-rrf", "hybrid-linear"],
        case_sensitive=True,
    ),
    default="symbol+bm25",
    help=(
        "Retrieval pipeline (v2 §5 ablation). 'symbol+bm25' is the v0 baseline "
        "router. dense / hybrid-* require a dense index built via "
        "`pdr build-index --with-dense`."
    ),
)
@click.option(
    "--alpha",
    type=float,
    default=0.5,
    help="Linear-merge alpha (only used when --retriever=hybrid-linear).",
)
@click.option(
    "--rerank",
    is_flag=True,
    help=(
        "Wrap the chosen retriever with a cross-encoder reranker: "
        "fetch top-N candidates → rerank → top-k. Requires the `rerank` extra."
    ),
)
@click.option(
    "--rerank-candidates",
    type=int,
    default=20,
    help="N candidates fetched from the inner retriever before rerank (default 20).",
)
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
    "--backend",
    type=click.Choice(["qwen", "qwen-gguf"]),
    default="qwen",
    help=(
        "Generation backend. 'qwen' = HF transformers (v1 default). "
        "'qwen-gguf' = llama.cpp + GGUF (v4 sub-task 3')."
    ),
)
@click.option(
    "--gguf-model",
    "gguf_model_path",
    default=None,
    type=click.Path(exists=True),
    help="Path to GGUF model file (required when --backend=qwen-gguf).",
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Force overwrite if the run directory already exists.",
)
@click.option(
    "--hyde",
    is_flag=True,
    default=False,
    help="Enable HyDE preprocessing (requires --backend=qwen-gguf + --retriever=dense).",
)
def eval_cmd(
    set_path: Path,
    tag: str,
    version: str | None,
    docs_sha: str | None,
    k: int,
    retriever: str,
    alpha: float,
    rerank: bool,
    rerank_candidates: int,
    model_id: str | None,
    backend: str,
    gguf_model_path: str | None,
    overwrite: bool,
    hyde: bool,
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

    if backend == "qwen-gguf" and gguf_model_path is None:
        raise click.UsageError("--backend=qwen-gguf requires --gguf-model")
    if hyde and backend != "qwen-gguf":
        raise click.UsageError("--hyde requires --backend=qwen-gguf")
    if hyde and not rerank:
        click.echo(
            "warning: --hyde without --rerank loses the original-query rerank safety net",
            err=True,
        )

    generator: Generator | None = None
    hypothetical_generator: HypotheticalGenerator | None = None
    if hyde and backend == "qwen-gguf" and gguf_model_path is not None:
        from python_doc_assistant.generation.qwen_gguf_backend import QwenGGUFGenerator
        from python_doc_assistant.retrieval.hyde import QwenHypotheticalGenerator

        generator = QwenGGUFGenerator(model_path=Path(gguf_model_path))
        hypothetical_generator = QwenHypotheticalGenerator(generator.llm)

    retrieve_fn = _build_eval_retrieve_fn(
        retriever=retriever,
        chunks=chunks,
        chunks_by_id=chunks_by_id,
        docs_version=docs_version,
        docs_sha=docs_sha,
        bm25_index=bm25_index,
        symbol_index=symbol_index,
        alpha=alpha,
        rerank=rerank,
        rerank_candidates=rerank_candidates,
        hyde=hyde,
        hypothetical_generator=hypothetical_generator,
    )
    run_result, model_field, decoding_params = _run_eval_with_optional_generation(
        eval_queries=eval_queries,
        retrieve_fn=retrieve_fn,
        chunks_by_id=chunks_by_id,
        max_k=k,
        model_id=model_id,
        backend=backend,
        gguf_model_path=gguf_model_path,
        generator=generator,
    )

    manifest = _load_ingest_manifest(docs_dir)
    metadata = RunMetadata(
        docs_version=docs_version,
        docs_served_version=manifest["docs_served_version"],
        docs_sha_short=docs_sha,
        ingest_manifest=manifest,
        config={
            "retriever": retriever,
            "k": k,
            "eval_set": str(set_path),
            "alpha": alpha if retriever == "hybrid-linear" else None,
            "rerank": rerank,
            "rerank_candidates": rerank_candidates if rerank else None,
        },
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
# Subcommand: serve (v4 sub-tasks 7 / 9 — HTTP server + web UI)
# ------------------------------------------------------------------


@main.command(name="serve")
@click.option(
    "--gguf-model",
    "gguf_model_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to GGUF model file (Qwen2.5-7B-Instruct Q4_K_M GGUF first shard).",
)
@click.option("--version", default=None, help="Override config.toml DOCS_VERSION.")
@click.option("--docs-sha", default=None, help="Use a specific sha_short (else current.txt).")
@click.option(
    "--k", type=int, default=5, help="Default top-k chunks per request (client may override)."
)
@click.option(
    "--retriever",
    type=click.Choice(
        ["bm25", "symbol+bm25", "dense", "hybrid-rrf", "hybrid-linear"],
        case_sensitive=True,
    ),
    default="dense",
    help=(
        "Retrieval pipeline. v4 production stack uses 'dense' + --rerank + --hyde. "
        "Other retrievers work but the eval numbers in the v4 narrative assume dense+rerank."
    ),
)
@click.option(
    "--alpha",
    type=float,
    default=0.5,
    help="Linear-merge alpha (only used when --retriever=hybrid-linear).",
)
@click.option(
    "--rerank/--no-rerank",
    default=True,
    help="Wrap the chosen retriever with a cross-encoder reranker (default on).",
)
@click.option(
    "--rerank-candidates",
    type=int,
    default=20,
    help="N candidates fetched from the inner retriever before rerank (default 20).",
)
@click.option(
    "--hyde/--no-hyde",
    default=True,
    help=(
        "Enable HyDE preprocessing for non-identifier queries (default on; requires "
        "--retriever=dense)."
    ),
)
@click.option("--host", default="127.0.0.1", help="HTTP bind host (default 127.0.0.1).")
@click.option("--port", type=int, default=8000, help="HTTP bind port (default 8000).")
@click.option(
    "--frontend-dist",
    default="frontend/dist",
    type=click.Path(file_okay=False, path_type=Path),
    help=(
        "Path to the built React frontend. When the directory exists, FastAPI mounts it at "
        "'/'. Run `cd frontend && npm run build` first, or use `npm run dev` (Vite at :5173 "
        "with /api proxy) instead."
    ),
)
def serve_cmd(
    gguf_model_path: Path,
    version: str | None,
    docs_sha: str | None,
    k: int,
    retriever: str,
    alpha: float,
    rerank: bool,
    rerank_candidates: int,
    hyde: bool,
    host: str,
    port: int,
    frontend_dist: Path,
) -> None:
    """Start an HTTP server exposing /api/ask + /health + the web UI.

    The server holds a single QwenGGUFGenerator instance + a single
    retrieve_fn closure for the lifetime of the process. Requests
    serialise behind an asyncio.Lock — concurrent clients queue. See
    `src/python_doc_assistant/service/app.py` for the FastAPI wiring.

    Implementation outline:

        1. Resolve docs_version + docs_sha exactly like `eval_cmd` so
           the same chunks/indexes the v4 numbers were measured
           against are loaded.

        2. Load the persisted artefacts:
              chunks      = _load_chunks(chunks_path)
              symbols     = parse_objects_inv(docs_dir)
              symbol_idx  = SymbolIndex(chunks, symbols)
              bm25_index  = BM25Index.load(bm25_path)
              chunks_by_id = {c.chunk_id: c for c in chunks}

        3. Validate flag combos (mirror eval_cmd):
              - hyde requires backend=qwen-gguf (here always true) AND
                retriever == "dense" (raise click.UsageError otherwise).

        4. Build the QwenGGUFGenerator first (so the HyDE generator
           can share its Llama instance — no double-loading the 4.7 GB
           model):
              from python_doc_assistant.generation.qwen_gguf_backend import QwenGGUFGenerator
              generator = QwenGGUFGenerator(model_path=gguf_model_path)

        5. If hyde, build the QwenHypotheticalGenerator on the same
           Llama:
              from python_doc_assistant.retrieval.hyde import QwenHypotheticalGenerator
              hypothetical_generator = QwenHypotheticalGenerator(generator.llm)

        6. Build retrieve_fn via the existing helper:
              retrieve_fn = _build_eval_retrieve_fn(
                  retriever=retriever,
                  chunks=chunks,
                  chunks_by_id=chunks_by_id,
                  docs_version=docs_version,
                  docs_sha=docs_sha,
                  bm25_index=bm25_index,
                  symbol_index=symbol_idx,
                  alpha=alpha,
                  rerank=rerank,
                  rerank_candidates=rerank_candidates,
                  hyde=hyde,
                  hypothetical_generator=hypothetical_generator,
              )

        7. Build the AskState. The asyncio.Lock can be a fresh one;
           uvicorn binds the same event loop the FastAPI handlers
           run on, so the lock is shared correctly:
              import asyncio
              from python_doc_assistant.service.app import AskState, build_app

              static_root = frontend_dist if frontend_dist.is_dir() else None
              state = AskState(
                  generator=generator,
                  retrieve_fn=retrieve_fn,
                  chunks_by_id=chunks_by_id,
                  lock=asyncio.Lock(),
                  static_root=static_root,
              )

        8. Build the FastAPI app + run it via uvicorn:
              import uvicorn
              app = build_app(state)
              click.echo(f"serving on http://{host}:{port}")
              if static_root:
                  click.echo(f"  ui: mounted from {static_root}")
              else:
                  click.echo("  ui: not built (run `cd frontend && npm run build`)")
              uvicorn.run(app, host=host, port=port, log_level="info")

    All FastAPI / uvicorn / generator imports MUST stay inside this
    function so the v0 / no-extras install path keeps `python -m
    python_doc_assistant` importable.
    """
    import asyncio

    import uvicorn

    from python_doc_assistant.generation.qwen_gguf_backend import QwenGGUFGenerator
    from python_doc_assistant.retrieval.hyde import QwenHypotheticalGenerator
    from python_doc_assistant.service.app import AskState, build_app

    if hyde and retriever != "dense":
        raise click.UsageError("--hyde requires --retriever=dense")

    eff_version = _resolve_docs_version(version)
    eff_sha = _resolve_docs_sha(eff_version, docs_sha)
    docs_dir = DEFAULT_DATA_ROOT / "docs" / eff_version / eff_sha
    chunks_path = DEFAULT_DATA_ROOT / "chunks" / eff_version / eff_sha / "chunks.jsonl"
    bm25_path = DEFAULT_DATA_ROOT / "indexes" / eff_version / eff_sha / "bm25.pkl"

    click.echo(f"loading chunks + indexes from {docs_dir}")
    chunks = _load_chunks(chunks_path)
    symbols = parse_objects_inv(docs_dir)
    symbol_idx = SymbolIndex(chunks, symbols)
    bm25_index = BM25Index.load(bm25_path)
    chunks_by_id = {c.chunk_id: c for c in chunks}

    click.echo(f"loading qwen-gguf generator: {gguf_model_path.name}")
    generator = QwenGGUFGenerator(model_path=gguf_model_path)

    hypothetical_generator: HypotheticalGenerator | None = None
    if hyde:
        hypothetical_generator = QwenHypotheticalGenerator(generator.llm)

    retrieve_fn = _build_eval_retrieve_fn(
        retriever=retriever,
        chunks=chunks,
        chunks_by_id=chunks_by_id,
        docs_version=eff_version,
        docs_sha=eff_sha,
        bm25_index=bm25_index,
        symbol_index=symbol_idx,
        alpha=alpha,
        rerank=rerank,
        rerank_candidates=rerank_candidates,
        hyde=hyde,
        hypothetical_generator=hypothetical_generator,
    )

    # Default top-k. The frontend's AskRequest may override this per-call;
    # the eval pipeline reuses k from the CLI flag. Stash it on state for
    # observability — the actual k flowing into retrieve_fn comes from
    # AskRequest.k inside _ask_stream.
    _ = k

    static_root = frontend_dist if frontend_dist.is_dir() else None
    state = AskState(
        generator=generator,
        retrieve_fn=retrieve_fn,
        chunks_by_id=chunks_by_id,
        lock=asyncio.Lock(),
        static_root=static_root,
    )

    app = build_app(state)
    click.echo(f"serving on http://{host}:{port}")
    click.echo(f"  retriever={retriever} rerank={rerank} hyde={hyde}")
    if static_root:
        click.echo(f"  ui mounted: {static_root}")
    else:
        click.echo("  ui not built (run `cd frontend && npm run build` or `npm run dev`)")
    uvicorn.run(app, host=host, port=port, log_level="info")


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


def _build_eval_retrieve_fn(
    *,
    retriever: str,
    chunks: list[Chunk],
    chunks_by_id: dict[str, Chunk],
    docs_version: str,
    docs_sha: str,
    bm25_index: BM25Index,
    symbol_index: SymbolIndex,
    alpha: float,
    rerank: bool,
    rerank_candidates: int,
    hyde: bool,
    hypothetical_generator: HypotheticalGenerator | None,
) -> Callable[[str, int], list[RetrievedChunk]]:
    """Wire the retrieval factory for `eval_cmd` (v2 §5 prereq).

    Loads the dense index and/or cross-encoder reranker only when the
    chosen retriever / `--rerank` flag needs them. `bm25_index` and
    `symbol_index` are passed in already-built by `eval_cmd`.

    Implementation outline:

        1. Decide what the retriever needs:
              needs_dense   = retriever in {"dense", "hybrid-rrf", "hybrid-linear"}
              needs_symbol  = retriever == "symbol+bm25"

        2. Lazy-load DenseIndex when needed:
              from python_doc_assistant.indexes.dense_index import DenseIndex
              dense_path = (
                  DEFAULT_DATA_ROOT / "indexes" / docs_version / docs_sha
                  / "dense.npy"
              )
              dense_index = DenseIndex.load(dense_path)
           Otherwise dense_index = None.

        3. Lazy-load CrossEncoderReranker when rerank=True:
              from python_doc_assistant.retrieval.rerank import CrossEncoderReranker
              reranker = CrossEncoderReranker()
           Otherwise reranker = None.

        4. Lazy-import the factory and return its closure:
              from python_doc_assistant.retrieval.factory import build_retrieve_fn
              return build_retrieve_fn(
                  retriever=retriever,
                  chunks_by_id=chunks_by_id,
                  bm25_index=bm25_index,
                  symbol_index=symbol_index if needs_symbol else None,
                  dense_index=dense_index,
                  reranker=reranker,
                  alpha=alpha,
                  rerank_candidates=rerank_candidates,
              )

    All three of DenseIndex / CrossEncoderReranker / factory.build_retrieve_fn
    must be imported INSIDE this function (not at module top level) so
    the v0 / no-extras install path keeps `python -m python_doc_assistant`
    working.
    """
    needs_dense = retriever in {"dense", "hybrid-rrf", "hybrid-linear"}
    needs_symbol = retriever == "symbol+bm25"
    dense_index = None
    reranker = None

    if needs_dense:
        from python_doc_assistant.indexes.dense_index import DenseIndex

        dense_path = DEFAULT_DATA_ROOT / "indexes" / docs_version / docs_sha / "dense.npy"
        dense_index = DenseIndex.load(dense_path)
    if rerank:
        from python_doc_assistant.retrieval.rerank import CrossEncoderReranker

        reranker = CrossEncoderReranker()
    from python_doc_assistant.retrieval.factory import build_retrieve_fn

    if hyde:
        from python_doc_assistant.retrieval.hyde import make_hyde_retrieve_fn

        if retriever != "dense":
            raise click.UsageError("--hyde requires --retriever=dense")
        if hypothetical_generator is None:
            raise click.UsageError("--hyde requires hypothetical_generator")
        assert dense_index is not None  # retriever=="dense" → loaded above

        return make_hyde_retrieve_fn(
            dense_index=dense_index,
            chunks_by_id=chunks_by_id,
            hypothetical_generator=hypothetical_generator,
            reranker=reranker,
            rerank_candidates=rerank_candidates,
        )
    return build_retrieve_fn(
        retriever=retriever,
        chunks_by_id=chunks_by_id,
        bm25_index=bm25_index,
        symbol_index=symbol_index if needs_symbol else None,
        dense_index=dense_index,
        reranker=reranker,
        alpha=alpha,
        rerank_candidates=rerank_candidates,
    )


def _run_eval_with_optional_generation(
    *,
    eval_queries: list[Any],
    retrieve_fn: Callable[[str, int], list[RetrievedChunk]],
    chunks_by_id: dict[str, Chunk],
    max_k: int,
    model_id: str | None,
    backend: str = "qwen",
    gguf_model_path: str | None = None,
    generator: Generator | None = None,
) -> tuple[EvalRunResult, str | None, dict[str, Any] | None]:
    """Dispatch retrieval-only vs retrieval+generation eval.

    Returns:
        (run_result, model_field, decoding_params)

        - retrieval-only (v0): selected when backend == "qwen" and
          model_id is None. Model + decoding stay None.
        - retrieval + qwen transformers (v1): backend == "qwen" and
          model_id is set.
        - retrieval + qwen GGUF (v4 sub-task 3'): backend == "qwen-gguf"
          and gguf_model_path is set; model_field returns the GGUF
          filename for reproducibility.

    `generator` argument: when the caller has already built a Generator
    (e.g. the HyDE path needs to share the same Llama instance with
    `QwenHypotheticalGenerator`), pass it in to avoid double-loading the
    4.7 GB Qwen GGUF. When None, the function builds one based on
    `backend` + `model_id` / `gguf_model_path`.

    Splitting this out keeps `eval_cmd` body small and lets tests cover
    the dispatch without spinning up a real model.
    """
    if backend == "qwen" and model_id is None and generator is None:
        run_result = evaluate(eval_queries, retrieve_fn, max_k=max_k)
        return run_result, None, None

    from python_doc_assistant.evaluation.generation_eval import evaluate_with_generation

    model_field: str
    if generator is not None:
        if backend == "qwen-gguf":
            assert gguf_model_path is not None
            model_field = Path(gguf_model_path).name
        elif model_id is not None:
            model_field = model_id
        else:
            model_field = ""
    elif backend == "qwen-gguf":
        if gguf_model_path is None:
            raise ValueError("backend='qwen-gguf' requires gguf_model_path")
        from python_doc_assistant.generation.qwen_gguf_backend import QwenGGUFGenerator

        generator = QwenGGUFGenerator(model_path=Path(gguf_model_path))
        model_field = Path(gguf_model_path).name
    else:
        from python_doc_assistant.generation.qwen_backend import QwenGenerator

        assert model_id is not None
        generator = QwenGenerator(model_id)
        model_field = model_id

    run_result = evaluate_with_generation(
        eval_queries, retrieve_fn, generator, chunks_by_id, max_k=max_k
    )
    decoding_params = {
        "temperature": generator.temperature,
        "top_p": generator.top_p,
        "max_new_tokens": generator.max_new_tokens,
    }
    return run_result, model_field, decoding_params


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
