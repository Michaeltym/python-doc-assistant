# AGENTS.md

Guidance for coding agents working in this repository.

## Project Context

This repository is a from-scratch Python documentation RAG assistant. Read these files before making architectural or implementation changes:

- `PLAN.md`
- `plans/v0-retrieval-eval.md`
- `plans/v1-qwen-generator.md`
- `plans/v2-ablation.md`
- `plans/v3-tiny-llm.md`

The project priorities are reproducibility, measurable retrieval quality, grounded answers, and small, reviewable changes.

## Core Engineering Rules

- Follow the staged roadmap. Do not implement v1/v2/v3 features while working on v0 unless explicitly requested.
- Keep the project framework-light. Do not add LangChain, LlamaIndex, hosted vector databases, or broad orchestration frameworks.
- Use `uv` for dependency and environment management. Do not add `requirements.txt`.
- Keep dependencies split by stage using `pyproject.toml` optional dependencies.
- Pin documentation versions through `DOCS_VERSION` in config and record that version in chunks and eval runs.
- Prefer simple, typed Python modules over premature abstractions.
- Keep generated data, downloaded docs, indexes, model checkpoints, and large artifacts out of git unless explicitly requested.

## Repository Layout

Use a standard Python `src` layout when adding code:

```text
src/python_doc_assistant/
```

Prefer these top-level areas:

- `src/python_doc_assistant/ingest/` for downloading, parsing, and chunking docs
- `src/python_doc_assistant/indexes/` for symbol, BM25, dense, and persisted indexes (plural to avoid collision with Python's `list.index()`)
- `src/python_doc_assistant/retrieval/` for routing, hybrid merge, and reranking
- `src/python_doc_assistant/generation/` for generator interfaces and backends
- `src/python_doc_assistant/evaluation/` for datasets, metrics, scoring, and run writers (not `eval/` — avoids shadowing built-in `eval()`)
- `eval_sets/` for committed evaluation datasets
- `experiments/` for human-readable experiment reports and selected machine-readable run outputs
- `tests/` for unit and integration tests

## Development Commands

Use the lightest command that validates the change.

```bash
uv sync --extra dev
uv run ruff format .
uv run ruff check .
uv run mypy src
uv run pytest
```

For v0 work:

```bash
uv sync --extra dev --extra ingest --extra retrieval
uv run pdr ingest --version 3.12
uv run pdr build-index
uv run pdr search "Path.read_text" --k 5 --debug
uv run pdr eval --set eval_sets/v0_core_30.jsonl --tag v0-bm25
```

Do not install generation, embedding, rerank, or training dependencies during v0 unless the task explicitly requires it.

## Code Standards

- Python version: 3.11 or newer.
- Use type hints for public functions, dataclasses, and interfaces.
- Prefer dataclasses or typed containers for structured records such as chunks, symbols, retrieval hits, answers, and eval results.
- Keep IO boundaries explicit. Parsing, indexing, retrieval, generation, and evaluation should be testable independently.
- Use structured parsers for HTML, TOML, JSON, and JSONL. Avoid brittle string parsing when a parser is available.
- Write small tests for scoring, matching, routing, serialization, and parser edge cases.
- Do not hide failures. Surface malformed input, missing docs versions, and invalid eval rows with clear errors.
- Tests must not depend on real network. Use fixtures under `tests/fixtures/` (small HTML samples, pre-parsed inventory JSON). Mock HTTP in any test that exercises `fetch_docs.py`.

## Evaluation Rules

- Evaluation sets come before implementation changes that optimize retrieval.
- Do not tune the evaluation set to match current output.
- Evaluation supports `match_policy: any | all` per query and `url_match: exact | strip_anchor | prefix`. See PLAN.md §8 for hit semantics.
- Every eval run should record:
  - `docs_version` (major.minor) and `docs_served_version` (actual patch parsed from HTML)
  - `docs_sha_short` (the sha-keyed corpus directory the run actually used; resolved from `--docs-sha` or `current.txt`)
  - full `ingest_manifest` snapshot (docs URL, archive sha256, ingest timestamp)
  - config
  - command or entry point
  - retrieval mode
  - model name, prompt hash, and decoding params when generation is involved
  - aggregate metrics
  - per-query top-k results and scores
- For LLM-as-judge runs, additionally record `judge_model`, `judge_prompt_hash`, temperature, and each sample's raw (unparsed) judge output.
- Retrieval metrics must include Recall@5, Recall@10, and MRR.
- Generation evaluation must keep retrieval config, prompt, model, decoding params, and top-k fixed when comparing retrieval strategies.

## RAG-Specific Guidance

- Keep `objects.inv` as a symbol-to-URI source, not as full structured documentation.
- The canonical chunk schema is defined in PLAN.md §8. All modules downstream of ingest (eval, debug, citation, rerank) depend on its field names.
- Symbol lookup must handle short-name collisions. Do not model it as one key to one chunk unless uniqueness is guaranteed.
- BM25 analyzer must split tokens on `.`, CamelCase, and `_` to handle symbol-style queries (e.g. `pathlib.Path.read_text`, `dict.fromKeys`). See plans/v0-retrieval-eval.md §5.
- Normalize URLs and anchors before eval matching.
- Keep `symbol_chunk` and `section_chunk` extraction rules explicit to avoid accidental duplicate content.
- CLI debug output should expose routing decisions, scores, chunk IDs, URLs, and citations.

## Git Hygiene

- Check `git status --short` before editing.
- Do not revert or delete user changes unless explicitly asked.
- Keep commits focused on one purpose.
- Commit messages must be short, clean, clear, and 30 words or fewer.

## Pull Requests

PR title format:

```text
[type] short description
```

Example:

```text
[fix] handle null response in user API
```

PR descriptions must include:

```markdown
### What

Brief summary of what changed.

### Why

Explain the problem this PR solves.

### How

Describe the implementation approach at a high level.

### Changes

- List key changes.

### Testing

Explain how this was tested, including commands when applicable.
```

Keep PRs small and focused. One purpose per PR.

## Replying To Pull Request Comments

Replies should be short and clear, and must make clear the reply was authored by Codex.

Example:

```text
addressed this by normalizing chunk URLs before eval matching. — Codex
```
