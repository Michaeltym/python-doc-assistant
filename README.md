# python-doc-assistant

A Python documentation RAG assistant built from scratch, with reproducible
ingest → chunk → index → retrieve → evaluate pipeline. Pinned to a specific
docs version + sha so every eval run is replayable.

**Status:** v0 (retrieval + eval) and v1 (Qwen generator + grounded prompt) complete. v2 (dense / hybrid / rerank ablation + LLM-as-judge) — retrieval ablation + judge module complete; generation eval (6 configs × n=111 with Haiku 4.5 judge) running.

Best retrieval configuration (v2, on `eval_sets/v2_full.jsonl`, n=111):

| Configuration | Recall@5 | Recall@10 | MRR |
| ------------- | -------: | --------: | --: |
| v0 BM25 baseline (`bm25`)   | 0.712 | 0.766 | 0.567 |
| v0 router (`symbol+bm25`)   | 0.730 | 0.775 | 0.625 |
| v2 best (`dense + rerank`)  | **0.838** | 0.883 | 0.705 |

Generation eval (`Qwen2.5-1.5B-Instruct` + LLM-as-judge with Haiku 4.5)
on the same six retrieval configs is the v2 §6 deliverable; numbers are
filled in [`experiments/v2-ablation.md`](experiments/v2-ablation.md) as
each judge run completes.

Run details: [`experiments/v0-bm25-only.md`](experiments/v0-bm25-only.md),
[`experiments/v1-qwen-grounded.md`](experiments/v1-qwen-grounded.md),
[`experiments/v2-ablation.md`](experiments/v2-ablation.md).

---

## Quick start

```bash
# v0 install: ingest + retrieval, no torch
uv sync --extra dev --extra ingest --extra retrieval

# Download Python 3.12 docs archive (~50 MB, sha-keyed cache)
uv run pdr ingest --version 3.12

# Parse symbols + chunk HTML + persist chunks.jsonl + bm25.pkl
uv run pdr build-index

# Search (v0 retrieval)
uv run pdr search "Path.read_text" --k 5
uv run pdr search "how to iterate dict safely" --k 5 --debug

# v0 retrieval-only eval
uv run pdr eval --set eval_sets/v0_core.jsonl --tag v0-bm25
```

For v1 (grounded generation) and v2 (dense / hybrid / rerank):

```bash
# v1+ install: add generation, embedding, rerank, judge extras
uv sync --extra dev --extra ingest --extra retrieval \
  --extra generation --extra embedding --extra rerank --extra judge

# Build dense embedding index (sentence-transformers, BAAI/bge-small-en-v1.5)
uv run pdr build-index --with-dense

# v2 retrieval ablation (12 configs against v2_full)
uv run pdr eval --set eval_sets/v2_full.jsonl --retriever symbol+bm25 --tag v2-bm25
uv run pdr eval --set eval_sets/v2_full.jsonl --retriever dense --tag v2-dense
uv run pdr eval --set eval_sets/v2_full.jsonl --retriever hybrid-rrf --tag v2-hybrid-rrf
uv run pdr eval --set eval_sets/v2_full.jsonl --retriever hybrid-linear --alpha 0.3 --tag v2-a03
uv run pdr eval --set eval_sets/v2_full.jsonl --retriever dense --rerank --tag v2-dense-rerank

# v1/v2 generation: layer Qwen2.5-1.5B-Instruct grounded prompt on top of any retriever
uv run pdr eval --set eval_sets/v2_full.jsonl --retriever dense --rerank \
  --model Qwen/Qwen2.5-1.5B-Instruct --tag v2-dense-rerank-qwen

# v2 §6 LLM-as-judge (requires ANTHROPIC_API_KEY)
uv run pdr judge --run-dir experiments/runs/<timestamp>-v2-dense-rerank-qwen
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
| Dense embedding | `sentence-transformers` (`BAAI/bge-small-en-v1.5`, 384-dim) | v2 §1; L2-normalized, cosine == inner product. |
| Cross-encoder reranker | `sentence-transformers.CrossEncoder` (`BAAI/bge-reranker-base`) | v2 §3; top-20 → top-5 rerank. |
| Generation backend | `transformers` (`Qwen/Qwen2.5-1.5B-Instruct`) on Mac MPS | v1 §2; greedy decoding, max_new_tokens=512. |
| LLM-as-judge | `anthropic` (`claude-haiku-4-5-20251001`) | v2 §6; 4-tier rubric + Cohen's kappa. |
| Lint + format | `ruff` (E / F / I rules) | Replaces black + isort + flake8. |
| Type checking | `mypy --strict` | Catches API drift early; `py.typed` marker for downstream consumers. |
| Test runner | `pytest` (+ `CliRunner` for CLI tests) | Hermetic; no real network. |
| Optional extras | `pyproject.toml` extras: `ingest`, `retrieval`, `generation`, `embedding`, `rerank` | v0 installs only `ingest` + `retrieval`; later stages opt-in. |

---

## Stage roadmap

| Stage | Deliverable | Plan | Status |
| ----- | ----------- | ---- | ------ |
| **v0** | Retrieval + evaluation: ingest, chunker, BM25 + symbol index, router, CLI, eval set, metrics, run writer | [`plans/v0-retrieval-eval.md`](plans/v0-retrieval-eval.md) | ✅ Complete |
| **v1** | `Qwen2.5-1.5B-Instruct` as a grounded generator with citations + refusal; out-of-scope eval set | [`plans/v1-qwen-generator.md`](plans/v1-qwen-generator.md) | ✅ Complete |
| **v2** | Ablation: dense embeddings + hybrid (RRF / linear) + cross-encoder rerank, eval set scaled to 111 queries, LLM-as-judge with kappa-calibrated rubric | [`plans/v2-ablation.md`](plans/v2-ablation.md) | ✅ Retrieval + judge module complete; generation eval in progress |
| **v3** | (Side track, learning) Hand-written decoder-only LLM (RoPE / RMSNorm / SwiGLU / KV cache) plugged into the same RAG pipeline as a comparison backend | [`plans/v3-tiny-llm.md`](plans/v3-tiny-llm.md) | Research side track |

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
│   ├── v0_core.jsonl               # 34 hand-written queries (v0)
│   ├── v1_out_of_scope_20.jsonl    # 20 OOS queries (v1)
│   └── v2_full.jsonl               # 111 queries (v0_core + 77 new for v2)
├── experiments/
│   ├── v0-bm25-only.md             # v0 narrative
│   ├── v1-qwen-grounded.md         # v1 narrative
│   ├── v2-ablation.md              # v2 narrative
│   └── runs/<ts>-<tag>/            # machine-readable run snapshots
├── data/                           # gitignored: docs / chunks / indexes
└── src/python_doc_assistant/
    ├── ingest/
    │   ├── fetch_docs.py           # download + sha-key + manifest
    │   ├── parse_objects_inv.py    # SymbolEntry list
    │   └── chunker.py              # symbol_chunk + section_chunk
    ├── indexes/
    │   ├── symbol_index.py         # exact + fuzzy multi-candidate
    │   ├── bm25_index.py           # analyzer + BM25Okapi + persistence
    │   └── dense_index.py          # v2 §1: bge-small embeddings + numpy
    ├── retrieval/
    │   ├── router.py               # identifier vs NL dispatch
    │   ├── hybrid.py               # v2 §2: RRF + linear merge
    │   ├── rerank.py               # v2 §3: cross-encoder reranker
    │   └── factory.py              # build retriever from CLI flags
    ├── generation/
    │   ├── interface.py            # v1 §2: Generator ABC + grounded prompt + citation parser
    │   └── qwen_backend.py         # v1 §2: Qwen2.5-1.5B-Instruct backend
    ├── evaluation/
    │   ├── dataset.py              # eval set schema + JSONL loader
    │   ├── retrieval_metrics.py    # is_hit + Recall@K + MRR
    │   ├── run_writer.py           # results.json + per_query.jsonl
    │   ├── generation_eval.py      # v1 §4: per-query generation pipeline
    │   ├── human_scoring.py        # v1 §6: 4-tier scoring schema + aggregate
    │   └── judge.py                # v2 §6: LLM-as-judge (Anthropic Haiku 4.5)
    └── cli.py                      # pdr ingest / build-index / search / eval / judge
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
| `results.json` | aggregate retrieval metrics (Recall@5 / Recall@10 / MRR / n_queries) + 7 reproducibility fields (`docs_version`, `docs_served_version`, `docs_sha_short`, full `ingest_manifest` snapshot, `config`, `tag`, `command`); for generation runs adds `model` + `decoding_params`; for judged runs adds `judge` (model + prompt hash + timing) and `judge_aggregate` (tier counts + hallucination_rate + correct_rate). |
| `per_query.jsonl` | one line per EvalQuery: retrieved chunk_ids + scores + ranks + hit flags + rank_for_mrr; generation runs also include `model_output_text`, `cited_chunk_ids`, `refused`. |
| `judge_scores.jsonl` | (v2 §6 only) one `JudgeRecord` per query: tier (correct / partial / wrong / hallucination / refused), notes, raw judge output, judge_model, judge_prompt_hash, timestamp. |

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
- **Evaluation-first.** Eval set design (`eval_sets/v0_core.jsonl`)
  precedes retrieval optimization. Failing queries are data signals
  for the next stage, not bugs to "fix" by adjusting expected values.
- **Stage-isolated dependencies.** v0 deliberately does not install
  `torch` / `transformers` / `sentence-transformers` / `anthropic`. They
  land in v1+ via `pyproject.toml` extras (`generation`, `embedding`,
  `rerank`, `judge`).
- **Reproducible per-run.** Docs version pinned via `DOCS_VERSION`;
  archive sha-keyed; manifest snapshotted into every eval result. No
  silent corpus drift between runs.

---

## License

Not yet declared. Add a `LICENSE` file before publishing.
