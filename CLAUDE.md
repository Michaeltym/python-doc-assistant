# CLAUDE.md

Claude-specific guidance for this repository. `AGENTS.md` is the canonical instruction file; read it first and follow it unless the user gives newer instructions.

## First Steps

Before coding, read:

1. `AGENTS.md`
2. `PLAN.md`
3. The relevant file under `plans/`

This project is intentionally staged. Keep work aligned with the active stage.

## Project Summary

Build a Python documentation RAG assistant with:

- pinned Python docs corpus
- reproducible ingest, chunking, indexing, retrieval, and eval
- symbol search plus BM25 in v0
- grounded Qwen generation in v1
- dense, hybrid, and rerank ablations in v2
- optional tiny decoder-only learning backend in v3

Do not introduce LangChain, LlamaIndex, managed vector databases, or unrelated frameworks.

## Working Style

- Make small, focused edits.
- Preserve user changes and untracked files.
- Prefer direct implementation over broad refactors.
- Keep generated artifacts out of commits unless the user asks to include them.
- Use the repository's documented decisions before inventing new patterns.

## Commands

Use `uv` only. Do not add `requirements.txt`.

Common validation:

```bash
uv sync --extra dev
uv run ruff format .
uv run ruff check .
uv run mypy src
uv run pytest
```

v0 validation:

```bash
uv sync --extra dev --extra ingest --extra retrieval
uv run pdr ingest --version 3.12
uv run pdr build-index
uv run pdr search "Path.read_text" --k 5 --debug
uv run pdr eval --set eval_sets/v0_core.jsonl --tag v0-bm25
```

Do not install torch, transformers, sentence-transformers, or reranker dependencies during v0 unless the task explicitly requires them.

## Implementation Notes

- Prefer `src/python_doc_assistant/` for package code.
- Add typed dataclasses for structured records such as symbols, chunks, retrieval hits, answers, and eval rows.
- Keep parser, index, retrieval, generation, and eval logic independently testable.
- Treat evaluation data as a product artifact. Add it before optimizing implementation behavior.
- Record reproducibility fields in every eval run output: `docs_version` (major.minor), `docs_served_version` (actual patch), `docs_sha_short` (which sha-keyed corpus was used), and the full `ingest_manifest` snapshot. Chunk metadata carries `docs_version` + `source_hash`. See AGENTS.md §Evaluation Rules and PLAN.md §8 for the full contract.
- Normalize symbols, URLs, and anchors consistently before matching.
- Avoid one-to-one short-name symbol maps because Python docs symbols can collide.

## Commit And PR Rules

Commit messages:

- 30 words or fewer
- short, clean, clear

PR titles:

```text
[type] short description
```

PR descriptions must include:

```markdown
### What

### Why

### How

### Changes

### Testing
```

When replying to a PR comment, keep the message short, and make clear the reply was authored by Claude (analogous to the Codex attribution rule in `AGENTS.md`).
