# python-doc-assistant

A Python documentation RAG assistant built from scratch, with reproducible
ingest → chunk → index → retrieve → evaluate pipeline. Pinned to a specific
docs version + sha so every eval run is replayable.

**Status:** v0 (retrieval + evaluation) complete. v1 (Qwen generator) next.

| Metric | v0 BM25-only baseline |
| ------ | --------------------- |
| Recall@5 | 0.824 |
| Recall@10 | 0.853 |
| MRR | 0.674 |

Run details: [`experiments/v0-bm25-only.md`](experiments/v0-bm25-only.md).

---

## Quick start

```bash
# Install deps (no torch in v0)
uv sync --extra dev --extra ingest --extra retrieval

# Download Python 3.12 docs archive (~50 MB, sha-keyed cache)
uv run pdr ingest --version 3.12

# Parse symbols + chunk HTML + persist chunks.jsonl + bm25.pkl
uv run pdr build-index

# Search
uv run pdr search "Path.read_text" --k 5
uv run pdr search "how to iterate dict safely" --k 5 --debug

# Run the eval set (writes experiments/runs/<timestamp>-<tag>/)
uv run pdr eval --set eval_sets/v0_core_30.jsonl --tag v0-bm25
```

---

## Architecture

```
                       ┌──────────────────────┐
   user query ────────▶│  Query router        │
                       │  (identifier vs NL)  │
                       └─────────┬────────────┘
                                 │
                ┌────────────────┴────────────────┐
                │                                 │
                ▼                                 ▼
     ┌────────────────────┐           ┌──────────────────────┐
     │  SymbolIndex       │           │  BM25Index           │
     │  exact + fuzzy     │  miss     │  (rank_bm25)         │
     │  multi-candidate   │ ────────▶ │  analyzer:           │
     │                    │  fallback │   . / Camel / _ /    │
     │                    │           │   lowercase last     │
     └─────────┬──────────┘           └──────────┬───────────┘
               │                                 │
               └────────────┬────────────────────┘
                            ▼
                   ┌────────────────┐
                   │  Top-K chunks  │
                   └────────┬───────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
    ┌──────────────────┐        ┌──────────────────┐
    │  CLI search:     │        │  CLI eval:       │
    │  print chunk_ids │        │  evaluate() →    │
    │  + debug info    │        │  results.json +  │
    │                  │        │  per_query.jsonl │
    └──────────────────┘        └──────────────────┘


             Build-time pipeline (pdr ingest + pdr build-index)

   docs.python.org
   archive.tar.bz2
        │
        ▼
   ┌────────────────┐    ┌────────────────────┐    ┌──────────────────┐
   │  fetch_docs    │───▶│  parse_objects_inv │───▶│  build_chunks    │
   │  + sha-keyed   │    │  (sphobjinv)       │    │  symbol+section  │
   │  cache         │    │  → SymbolEntry[]   │    │  → Chunk[]       │
   └────────────────┘    └────────────────────┘    └─────────┬────────┘
                                                              │
                            ┌─────────────────────────────────┤
                            ▼                                 ▼
                   ┌──────────────────┐             ┌──────────────────┐
                   │  chunks.jsonl    │             │  bm25.pkl        │
                   │  (per-chunk      │             │  (BM25Okapi      │
                   │   metadata)      │             │   pickled)       │
                   └──────────────────┘             └──────────────────┘
```

Every artifact (docs / chunks / indexes) is keyed by docs major.minor +
archive sha256 short, so old eval runs always replay against the corpus
they were measured on.

---

## Tools and libraries

| Layer | Choice | Why |
| ----- | ------ | --- |
| Dependency manager | [`uv`](https://github.com/astral-sh/uv) + `pyproject.toml` + `uv.lock` | Reproducible environments without `requirements.txt`. |
| Build backend | `hatchling` | PEP 621 compliant, src-layout friendly. |
| CLI | `click` | Standard, composable subcommand pattern. |
| HTML parsing | `beautifulsoup4` + `lxml` | Proven, fast, predictable. |
| Sphinx inventory | `sphobjinv` | Reads `objects.inv` symbol → URI maps cleanly. |
| HTTP | `requests` | Streaming download with retry. |
| BM25 | `rank_bm25` | Lightweight, no Elasticsearch dependency. |
| Fuzzy matching | `rapidfuzz` | C-backed `fuzz.ratio` for SymbolIndex.fuzzy. |
| Lint + format | `ruff` (E / F / I rules) | Replaces black + isort + flake8. |
| Type checking | `mypy --strict` | Catches API drift early; `py.typed` marker for downstream consumers. |
| Test runner | `pytest` (+ `CliRunner` for CLI tests) | 218 hermetic tests; no real network. |
| Future stages | `transformers`, `sentence-transformers`, `torch` | Wired in via `pyproject.toml` extras (`generation`, `embedding`, `rerank`). Not installed in v0. |

---

## Stage roadmap

| Stage | Deliverable | Plan | Status |
| ----- | ----------- | ---- | ------ |
| **v0** | Retrieval + evaluation: ingest, chunker, BM25 + symbol index, router, CLI, eval set, metrics, run writer | [`plans/v0-retrieval-eval.md`](plans/v0-retrieval-eval.md) | ✅ Complete |
| **v1** | Wire `Qwen2.5-1.5B-Instruct` as a grounded generator with citations + refusal | [`plans/v1-qwen-generator.md`](plans/v1-qwen-generator.md) | Planned |
| **v2** | Ablation: dense embeddings + hybrid (RRF / linear) + cross-encoder rerank, eval set scaled to 100–200 queries | [`plans/v2-ablation.md`](plans/v2-ablation.md) | Planned |
| **v3** | (Optional, side track) Hand-written decoder-only LLM (RoPE / RMSNorm / SwiGLU / KV cache) plugged into the same RAG pipeline as a comparison backend | [`plans/v3-tiny-llm.md`](plans/v3-tiny-llm.md) | Optional |

Top-level project plan: [`PLAN.md`](PLAN.md). Per-stage plans are the
authoritative source for sub-task ordering, acceptance criteria, and
deliverables.

---

## Repository layout

```
python-doc-assistant/
├── PLAN.md                         # Top-level project plan
├── AGENTS.md                       # Cross-agent rules (Codex / Claude)
├── CLAUDE.md                       # Claude-specific guidance
├── README.md                       # this file
├── pyproject.toml                  # deps + tool config
├── uv.lock                         # pinned versions
├── config.toml                     # DOCS_VERSION = "3.12"
├── plans/                          # per-stage plans
├── eval_sets/
│   └── v0_core_30.jsonl            # 34 hand-written queries
├── experiments/
│   ├── v0-bm25-only.md             # narrative
│   └── runs/<ts>-<tag>/            # machine-readable run snapshots
├── data/                           # gitignored: docs / chunks / indexes
└── src/python_doc_assistant/
    ├── ingest/
    │   ├── fetch_docs.py           # download + sha-key + manifest
    │   ├── parse_objects_inv.py    # SymbolEntry list
    │   └── chunker.py              # symbol_chunk + section_chunk
    ├── indexes/
    │   ├── symbol_index.py         # exact + fuzzy multi-candidate
    │   └── bm25_index.py           # analyzer + BM25Okapi + persistence
    ├── retrieval/
    │   └── router.py               # identifier vs NL dispatch
    ├── evaluation/
    │   ├── dataset.py              # eval set schema + JSONL loader
    │   ├── retrieval_metrics.py    # is_hit + Recall@K + MRR
    │   └── run_writer.py           # results.json + per_query.jsonl
    └── cli.py                      # pdr ingest / build-index / search / eval
```

---

## Development

```bash
uv sync --extra dev --extra ingest --extra retrieval

# Lint + format
uv run ruff check .
uv run ruff format .

# Type-check
uv run mypy src tests

# Run all tests (218 hermetic; no network)
uv run pytest
```

`tests/` mirrors `src/` with one test module per source file. Tests
mock HTTP via `monkeypatch`, build small in-memory tarballs / HTML
fixtures, and use `tmp_path` for filesystem isolation. None of them
read real docs or real `objects.inv`.

---

## Reproducibility

Every eval run is written to `experiments/runs/<ISO-timestamp>-<tag>/`
with two files:

| File | Contents |
| ---- | -------- |
| `results.json` | aggregate metrics (Recall@5 / Recall@10 / MRR / n_queries) + 7 reproducibility fields (`docs_version`, `docs_served_version`, `docs_sha_short`, full `ingest_manifest` snapshot, `config`, `tag`, `command`) |
| `per_query.jsonl` | one line per EvalQuery: retrieved chunk_ids + scores + ranks + hit flags + rank_for_mrr |

The `<docs_sha_short>` directory under `data/docs/<version>/` is never
overwritten by re-ingest — `pdr ingest` errors out on sha mismatch
unless `--force-switch` is passed and creates a new sibling
directory. Old runs always resolve back to the corpus they were
measured against.

---

## Constraints (not goals)

- **Framework-light.** No LangChain, LlamaIndex, hosted vector DBs, or
  general orchestration frameworks. Direct stdlib + targeted libraries
  only.
- **Evaluation-first.** Eval set design (`eval_sets/v0_core_30.jsonl`)
  precedes retrieval optimization. Failing queries are data signals
  for the next stage, not bugs to "fix" by adjusting expected values.
- **Stage-isolated dependencies.** v0 deliberately does not install
  `torch` / `transformers` / `sentence-transformers`. They land in v1+
  via `pyproject.toml` extras (`generation`, `embedding`, `rerank`).
- **Reproducible per-run.** Docs version pinned via `DOCS_VERSION`;
  archive sha-keyed; manifest snapshotted into every eval result. No
  silent corpus drift between runs.

---

## License

Not yet declared. Add a `LICENSE` file before publishing.
