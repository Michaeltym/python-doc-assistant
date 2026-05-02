"""CLI entry point: build SFT corpus by distilling Qwen2.5-1.5B answers.

v3.1 §6 (data-generation half). Two source streams produce
`(query, retrieved_chunks, qwen_answer)` triples for fine-tuning the
v3.1 base:

  1. **Eval-set queries** — every query in `eval_sets/{v0,v1,v2}.jsonl`
     deduped, run through retrieval + Qwen.
  2. **Synthetic queries** — random-sampled chunks → Qwen-generated user
     questions → retrieval + Qwen-generated answers.

Outputs:
  - `data/sft/sft_corpus.jsonl`: accepted (query, chunks, answer) records
  - `data/sft/sft_rejected.jsonl`: rejected records with reason
  - `data/sft/manifest.json`: reproducibility (counts, seed, timestamps,
    docs sha, decoding params)

Usage:
    uv run python scripts/build_sft_corpus.py \\
        --out data/sft/sft_corpus.jsonl \\
        --rejected data/sft/sft_rejected.jsonl \\
        --manifest data/sft/manifest.json \\
        --n-synthetic 1000 \\
        --seed 42

Recommended overnight wrapper to prevent macOS sleep:
    caffeinate -di uv run python scripts/build_sft_corpus.py [...]
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click

from python_doc_assistant.evaluation.dataset import load_eval_set
from python_doc_assistant.generation.qwen_backend import QwenGenerator
from python_doc_assistant.generation.tinydocs.sft_corpus import (
    build_question_generation_prompt,
    is_sft_rejected,
)
from python_doc_assistant.indexes.bm25_index import BM25Index
from python_doc_assistant.indexes.symbol_index import SymbolIndex
from python_doc_assistant.ingest.chunker import Chunk
from python_doc_assistant.ingest.parse_objects_inv import parse_objects_inv
from python_doc_assistant.retrieval.router import classify, route

# ------------------------------------------------------------------
# Defaults
# ------------------------------------------------------------------

DOCS_VERSION = "3.12"
DEFAULT_DATA_ROOT = Path("data")
DEFAULT_EVAL_SETS = (
    Path("eval_sets/v2_full.jsonl"),
    Path("eval_sets/v0_core.jsonl"),
    Path("eval_sets/v1_out_of_scope_20.jsonl"),
)
ANSWER_MAX_NEW_TOKENS = 256  # 128 cuts off mid-example; [N] sits at end
QUESTION_MAX_NEW_TOKENS = 64
TOP_K = 3


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _resolve_docs_sha(docs_version: str) -> str:
    """Read `data/docs/<version>/current.txt` for the active sha_short."""
    current = DEFAULT_DATA_ROOT / "docs" / docs_version / "current.txt"
    if not current.exists():
        raise click.UsageError(f"missing {current}; run `pdr ingest` first")
    return current.read_text().strip()


def _load_chunks(path: Path) -> list[Chunk]:
    chunks = []
    with path.open() as f:
        for line in f:
            obj = json.loads(line)
            chunks.append(
                Chunk(
                    chunk_id=obj["chunk_id"],
                    chunk_type=obj["chunk_type"],
                    docs_version=obj["docs_version"],
                    title=obj["title"],
                    text=obj["text"],
                    symbols=tuple(obj["symbols"]),
                    canonical_url=obj["canonical_url"],
                    anchor=obj.get("anchor"),
                    parent_module=obj.get("parent_module"),
                    source_path=obj["source_path"],
                    source_hash=obj["source_hash"],
                )
            )
    return chunks


def _load_unique_eval_queries(paths: tuple[Path, ...]) -> list[Any]:
    """Load eval queries from N jsonl files, deduped by query string."""
    seen: set[str] = set()
    out = []
    for path in paths:
        if not path.exists():
            click.echo(f"  skipping (missing): {path}")
            continue
        for eq in load_eval_set(path):
            if eq.query in seen:
                continue
            seen.add(eq.query)
            out.append(eq)
    return out


def gen_question(generator: QwenGenerator, chunk: Chunk) -> str:
    """Ask Qwen to produce one user question about `chunk`.

    Calls `_call_model` directly (bypassing build_grounded_prompt) since
    the question-generation prompt is custom. Restores max_new_tokens
    after the call so subsequent answer generation uses the longer budget.
    """
    messages = build_question_generation_prompt(chunk)
    saved = generator.max_new_tokens
    generator.max_new_tokens = QUESTION_MAX_NEW_TOKENS
    try:
        raw = generator._call_model(messages)
    finally:
        generator.max_new_tokens = saved
    return raw.strip().split("\n")[0]


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


@click.command()
@click.option("--out", required=True, type=click.Path())
@click.option("--rejected", default=None, type=click.Path())
@click.option("--manifest", default=None, type=click.Path())
@click.option("--docs-version", default=DOCS_VERSION)
@click.option(
    "--n-synthetic",
    default=1000,
    type=int,
    help="Number of synthetic chunk-derived queries to generate (0 disables).",
)
@click.option(
    "--n-eval",
    default=None,
    type=int,
    help="Cap eval-set queries (None = use all unique).",
)
@click.option("--seed", default=42, type=int)
@click.option(
    "--smoke",
    is_flag=True,
    help="Tiny run: 5 eval queries + 10 synthetic, for end-to-end validation only.",
)
def main(
    out: str,
    rejected: str | None,
    manifest: str | None,
    docs_version: str,
    n_synthetic: int,
    n_eval: int | None,
    seed: int,
    smoke: bool,
) -> None:
    """Build the v3.1 SFT corpus."""
    # ---- Paths
    docs_sha = _resolve_docs_sha(docs_version)
    docs_path = DEFAULT_DATA_ROOT / "docs" / docs_version / docs_sha
    chunks_jsonl = (
        DEFAULT_DATA_ROOT / "chunks" / docs_version / docs_sha / "chunks.jsonl"
    )
    bm25_path = DEFAULT_DATA_ROOT / "indexes" / docs_version / docs_sha / "bm25.pkl"

    out_path = Path(out)
    rejected_path = Path(rejected) if rejected else out_path.with_name("sft_rejected.jsonl")
    manifest_path = Path(manifest) if manifest else out_path.with_name("manifest.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rejected_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- Load corpus + indexes + Qwen
    click.echo(f"loading chunks: {chunks_jsonl}")
    chunks = _load_chunks(chunks_jsonl)
    chunks_by_id = {c.chunk_id: c for c in chunks}
    click.echo(f"  {len(chunks)} chunks")

    click.echo("loading indexes (symbol + bm25)...")
    eval_symbols = parse_objects_inv(docs_path)
    symbol_index = SymbolIndex(chunks, eval_symbols)
    bm25_index = BM25Index.load(bm25_path)

    click.echo("loading Qwen2.5-1.5B-Instruct (slow first time)...")
    generator = QwenGenerator(max_new_tokens=ANSWER_MAX_NEW_TOKENS)

    # ---- Path A: eval-set queries
    click.echo("loading eval-set queries...")
    eval_queries = _load_unique_eval_queries(DEFAULT_EVAL_SETS)
    click.echo(f"  unique: {len(eval_queries)}")
    if smoke:
        eval_queries = eval_queries[:5]
    elif n_eval is not None:
        eval_queries = eval_queries[:n_eval]

    # ---- Path B: synthetic chunk-derived queries
    rng = random.Random(seed)
    n_syn_effective = 10 if smoke else n_synthetic
    n_syn_effective = min(n_syn_effective, len(chunks))
    synthetic_chunks = rng.sample(chunks, n_syn_effective) if n_syn_effective > 0 else []
    click.echo(f"synthetic chunks sampled: {len(synthetic_chunks)}")

    # ---- Stats
    started_at = datetime.now(timezone.utc)
    n_accepted = 0
    n_rejected = 0
    n_skipped_bad_question = 0
    rejection_reasons: dict[str, int] = {}

    total = len(eval_queries) + len(synthetic_chunks)
    click.echo(f"\nbuilding SFT corpus: {total} queries total")
    click.echo(f"  → accepted: {out_path}")
    click.echo(f"  → rejected: {rejected_path}\n")

    with out_path.open("w", encoding="utf-8") as out_f, rejected_path.open(
        "w", encoding="utf-8"
    ) as rej_f:
        # ---- Path A: eval-set queries
        for i, eq in enumerate(eval_queries, 1):
            n_accepted, n_rejected = _run_one(
                query=eq.query,
                source="eval",
                source_chunk_id=None,
                generator=generator,
                symbol_index=symbol_index,
                bm25_index=bm25_index,
                chunks_by_id=chunks_by_id,
                out_f=out_f,
                rej_f=rej_f,
                n_accepted=n_accepted,
                n_rejected=n_rejected,
                rejection_reasons=rejection_reasons,
            )
            click.echo(
                f"[A {i:3d}/{len(eval_queries)}] "
                f"acc={n_accepted} rej={n_rejected}  q={eq.query[:50]}"
            )

        # ---- Path B: synthetic queries (Qwen-generated questions)
        for i, chunk in enumerate(synthetic_chunks, 1):
            question = gen_question(generator, chunk)
            if not question or "?" not in question:
                n_skipped_bad_question += 1
                click.echo(
                    f"[B {i:3d}/{len(synthetic_chunks)}] "
                    f"SKIP_BAD_Q  chunk={chunk.chunk_id}"
                )
                continue
            n_accepted, n_rejected = _run_one(
                query=question,
                source="synthetic",
                source_chunk_id=chunk.chunk_id,
                generator=generator,
                symbol_index=symbol_index,
                bm25_index=bm25_index,
                chunks_by_id=chunks_by_id,
                out_f=out_f,
                rej_f=rej_f,
                n_accepted=n_accepted,
                n_rejected=n_rejected,
                rejection_reasons=rejection_reasons,
            )
            click.echo(
                f"[B {i:3d}/{len(synthetic_chunks)}] "
                f"acc={n_accepted} rej={n_rejected}  q={question[:50]}"
            )

    # ---- Manifest
    finished_at = datetime.now(timezone.utc)
    manifest_data = {
        "docs_version": docs_version,
        "docs_sha_short": docs_sha,
        "n_eval_queries": len(eval_queries),
        "n_synthetic_chunks": len(synthetic_chunks),
        "n_skipped_bad_question": n_skipped_bad_question,
        "n_accepted": n_accepted,
        "n_rejected": n_rejected,
        "rejection_reasons": rejection_reasons,
        "qwen_model_id": generator.model_id,
        "decoding": {
            "answer_max_new_tokens": ANSWER_MAX_NEW_TOKENS,
            "question_max_new_tokens": QUESTION_MAX_NEW_TOKENS,
            "temperature": generator.temperature,
            "top_p": generator.top_p,
            "top_k": TOP_K,
        },
        "seed": seed,
        "smoke": smoke,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "wall_clock_seconds": (finished_at - started_at).total_seconds(),
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    click.echo(f"\n{'=' * 60}")
    click.echo(f"DONE: accepted={n_accepted}, rejected={n_rejected}, "
               f"skipped_bad_q={n_skipped_bad_question}")
    click.echo(f"  out:      {out_path}")
    click.echo(f"  rejected: {rejected_path}")
    click.echo(f"  manifest: {manifest_path}")
    click.echo(f"  wall-clock: {manifest_data['wall_clock_seconds']:.0f}s")


def _run_one(
    *,
    query: str,
    source: str,
    source_chunk_id: str | None,
    generator: QwenGenerator,
    symbol_index: SymbolIndex,
    bm25_index: BM25Index,
    chunks_by_id: dict[str, Chunk],
    out_f: Any,
    rej_f: Any,
    n_accepted: int,
    n_rejected: int,
    rejection_reasons: dict[str, int],
) -> tuple[int, int]:
    """Process one query: retrieve top-K → Qwen answer → filter → write.

    Returns updated (n_accepted, n_rejected).
    """
    qt = classify(query)
    result = route(query, symbol_index=symbol_index, bm25_index=bm25_index, k=TOP_K)

    # Retrieval-quality filter for synthetic path: when we already know the
    # source chunk, require it to appear in retrieval. Skips ~30 % of synthetic
    # samples but cuts the "fabricated API" failure mode (audit found 3/20
    # fabrications were on retrieval misses).
    if source_chunk_id is not None and source_chunk_id not in result.chunk_ids:
        record_miss: dict[str, Any] = {
            "query": query,
            "query_type": qt.value,
            "source": source,
            "source_chunk_id": source_chunk_id,
            "retrieved_chunk_ids": list(result.chunk_ids),
            "rejection_reason": "retrieval_miss",
        }
        rej_f.write(json.dumps(record_miss) + "\n")
        rej_f.flush()
        n_rejected += 1
        rejection_reasons["retrieval_miss"] = (
            rejection_reasons.get("retrieval_miss", 0) + 1
        )
        return n_accepted, n_rejected

    retrieved = [
        chunks_by_id[cid] for cid in result.chunk_ids if cid in chunks_by_id
    ]
    answer = generator.generate(query, retrieved, query_type=qt)

    record: dict[str, Any] = {
        "query": query,
        "query_type": qt.value,
        "source": source,
        "retrieved_chunk_ids": [c.chunk_id for c in retrieved],
        "qwen_answer": answer.text,
        "qwen_refused": answer.refused,
        "qwen_latency_seconds": answer.latency_seconds,
    }
    if source_chunk_id is not None:
        record["source_chunk_id"] = source_chunk_id

    reason = is_sft_rejected(answer.text, answer.refused)
    if reason is None:
        out_f.write(json.dumps(record) + "\n")
        out_f.flush()
        n_accepted += 1
    else:
        record["rejection_reason"] = reason
        rej_f.write(json.dumps(record) + "\n")
        rej_f.flush()
        n_rejected += 1
        rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
    return n_accepted, n_rejected


if __name__ == "__main__":
    main()
