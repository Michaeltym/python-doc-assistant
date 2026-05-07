# Python Docs RAG — Project Plan

## 1. Project Positioning

A Python API docs RAG assistant with **complete retrieval evaluation**, supporting **multiple pluggable generator backends**.
Qwen2.5 serves as the default baseline; a homegrown tiny LLM is an optional learning-oriented comparison backend (non-blocking side track).

**Core constraint: standalone new repository, built from scratch.**

## 2. Goals

- Full RAG-stack engineering: ingest / chunking / indexing / retrieval / reranking / grounding / evaluation
- At least one non-trivial optimization: embedding fine-tuning or hybrid retrieval ablation
- Every key design decision has a documented rationale under `experiments/`
- Quantifiable evaluation data: Recall@k, MRR, generation quality scores, ablation tables
- **Reproducibility**: docs version pinned; every eval run produces machine-readable JSON/JSONL
- Observability: CLI exposes the retrieval process (scores, chunks, citations) for debugging and regression

## 3. Non-Goals (Explicitly Out of Scope)

- The homegrown LLM is not the primary generator
- Not aiming to surpass Qwen on answer quality
- No LangChain / LlamaIndex / other heavyweight frameworks
- No vector-database operations; numpy or FAISS is sufficient
- v0–v2 ship no frontend; CLI is enough

## 4. Repository and Environment

**Suggested path:** `/Users/michaeltan/Desktop/training/python-doc-assistant/`

**Python and dependency management (industry-standard)**

- Python 3.11+, with `.python-version` locking the version
- **`uv`** as the project and dependency management tool (replacing pip + venv + pip-tools)
- `pyproject.toml` declares dependencies and tool configuration (PEP 621)
- `uv.lock` locks exact versions and is committed alongside the code
- **Do not use `requirements.txt`**
- Dependencies split into `optional-dependencies` by stage (see §10); v0 does not install torch/transformers

**Docs corpus version locking + artifact provenance**

- The `DOCS_VERSION` config is **major.minor** (e.g. `"3.12"`; docs.python.org only publishes archives per branch); written into `config.toml` / CLI arguments
- Each ingest extracts to a **sha-keyed** subdirectory `data/docs/<DOCS_VERSION>/<sha_short>/` (sha_short = first 12 chars of the archive sha256); `current.txt` tracks the current active sha; on sha conflict, the default is to error out unless `--force-switch` is passed
- Each ingest produces `data/docs/<DOCS_VERSION>/<sha_short>/ingest_manifest.json`, recording five fields:
  - `docs_version` (major.minor, e.g. `"3.12"`)
  - `docs_served_version` (actual patch, parsed from HTML title, e.g. `"3.12.13"`)
  - `docs_url` (full archive download URL)
  - `docs_archive_sha256` (sha256 of the full archive)
  - `ingest_timestamp` (ISO 8601)
- chunk metadata carries `docs_version` + `source_hash` (see §8 chunk schema)
- Each eval run snapshots the manifest into `results.json`
- Switching versions / switching sha = re-ingest + rebuild indexes; **eval results across different `DOCS_VERSION` or different `docs_archive_sha256` are not directly comparable** (sha drift within the same DOCS_VERSION also implies content changes)

**Code quality tools (all configured in `pyproject.toml`)**

- `ruff` — linting + formatting (replaces black + isort + flake8)
- `mypy` — static type checking
- `pytest` — testing
- Optional: `pre-commit` to run all of the above automatically before commits

A standalone git repository, with an isolated virtual environment automatically created by `uv venv` and reproducibility guaranteed via `uv.lock`.

## 5. System Architecture

```text
query
  ↓
query router  (identifier vs natural language)
  ↓
retrieval
  ├─ symbol index (exact + fuzzy)
  ├─ BM25
  └─ dense embedding        (v2)
  ↓
score merge / rerank        (v2)
  ↓
top-K chunks
  ↓
Generator backend (pluggable)
  ├─ qwen       (default baseline, v1; quantised GGUF in v4)
  └─ tinydocs   (homegrown, v3)
  ↓
grounded answer + citations
```

## 6. Technology Stack

| Layer       | Choice                                | Rationale                                                           |
| ----------- | ------------------------------------- | ------------------------------------------------------------------- |
| Corpus source | Python official HTML (pinned to a major.minor branch, e.g. `3.12`) | docs.python.org only publishes archives per branch; `docs_served_version` + sha256 record the actual patch |
| Symbol mapping | `objects.inv` (via `sphobjinv`)    | Sphinx-generated **symbol → URI mapping**; not equivalent to a complete structured doc source |
| HTML parsing | BeautifulSoup / lxml                  | Extract real content from `section` / `dl` / `dt` / `dd`            |
| BM25        | `rank_bm25` or hand-rolled            | A few thousand chunks does not need Elasticsearch                   |
| Embedding   | `BAAI/bge-small-en-v1.5`              | Small, strong, open source, standard baseline                       |
| Vector index | numpy cosine (v2) → FAISS (as needed) | numpy suffices for a few thousand chunks (v0 does not build a dense index) |
| Rerank      | `BAAI/bge-reranker-base` (v2)         | Standard cross-encoder                                              |
| Generator   | `Qwen2.5-1.5B-Instruct` (default)     | Apache 2.0, runs on MPS; the v1 stage compares against `Qwen2.5-Coder-1.5B-Instruct` (potentially better suited for code-adjacent content) |
| Inference backend | `transformers` → `llama.cpp` (as needed) | Get it working first, then swap                                |
| Evaluation  | Hand-written scripts + optional RAGAS | Hand-written first                                                  |

## 7. Five-Stage Roadmap

> Each stage's detailed task breakdown, acceptance criteria, and decision points are in [`plans/`](plans/):
>
> - [`plans/v0-retrieval-eval.md`](plans/v0-retrieval-eval.md)
> - [`plans/v1-qwen-generator.md`](plans/v1-qwen-generator.md)
> - [`plans/v2-ablation.md`](plans/v2-ablation.md)
> - [`plans/v3-tiny-llm.md`](plans/v3-tiny-llm.md)
> - [`plans/v4-prod-ready.md`](plans/v4-prod-ready.md)

### v0 — Retrieval-only + eval set bootstrap (about 1 week)

Deliverables:

- `objects.inv` parsing (using `sphobjinv`) to extract symbol → URI mappings
- HTML section chunking + **two coexisting chunk types**:
  - `symbol_chunk`: chunked by API symbol, carrying fully qualified name / signature / docstring / parent module
  - `section_chunk`: chunked by tutorial / HOWTO / concept section (covers queries like `how to use Path` and `how to use the with statement`)
- Each chunk carries `DOCS_VERSION` metadata
- BM25 index
- Symbol exact + fuzzy matching (`dict.fromKeys` → `dict.fromkeys`)
- CLI: input a query, output the Top-K raw chunks
- **eval set v0: 30–50 entries**, multi-answer + `match_policy` schema (see §8), three categories mixed:
  - identifier (about 40%): `dict.fromkeys`, `Path.read_text`
  - natural language (about 50%): `how to use Path`, `iterate dict safely`
  - comparison / cross-symbol (about 10%): `Path vs os.path`

Metrics: Recall@5, Recall@10, MRR

Completion criteria:

- 30–50 evaluation-first queries committed
- `ingest / index / search / eval` reproducible with a single command
- Recall@5 has a baseline number (**target ≥ 0.8, not a blocking gate**)
- Each query has per-query JSON output of top-k, scores, and matched chunks
- **Eval set written first, code written second**

### v1 — Wire up the Qwen generator (about 1 week)

Deliverables:

- `Generator` abstract interface (`generate(query, retrieved_chunks) -> answer`)
- `QwenGenerator` implementation, defaulting to `Qwen2.5-1.5B-Instruct`
- Run `Qwen2.5-Coder-1.5B-Instruct` in parallel for comparison (Python docs lean code-adjacent)
- Grounded prompt: enforce citations, refuse out-of-doc questions
- Answer format conditioned on `query_type`: `identifier` → signature + brief description + example + citation; `comparison` → key points on each side + differences + scenarios + citation; `howto` → steps + code example + citation; `natural_language` / concept → definition + background + example + citation
- **Manual generation-quality scoring: covers the entire current eval query set** (the 30–50 from v0)
- CLI gains `--debug` to display retrieved chunks and scores

Metrics: 4-tier generation quality (correct / partially correct / wrong / hallucination); refusal rate

Completion criteria: generation quality clearly beats v0's raw-text-only display; hallucination rate < 10%.

### v2 — Ablation and optimization (about 1 week)

Deliverables:

- Add dense embedding retrieval
- Hybrid merging (RRF or linear weighting)
- Cross-encoder rerank (Top-20 → Top-5)
- Expand the eval set to 100–200 entries
- **Ablation table**:

| Configuration           | Recall@5 | MRR | Answer quality | Hallucination rate |
| ----------------------- | -------- | --- | -------------- | ------------------ |
| BM25-only               |          |     |                |                    |
| Dense-only              |          |     |                |                    |
| Hybrid-RRF              |          |     |                |                    |
| Hybrid-linear (α sweep) |          |     |                |                    |
| Hybrid-RRF + rerank     |          |     |                |                    |
| Hybrid-best + rerank    |          |     |                |                    |

- LLM-as-judge (Claude API / Qwen API / local `Qwen2.5-72B-Instruct`) assists scoring; each run records `judge_model / judge_prompt_hash / temperature / raw_judge_output` (details in plans/v2)
- `experiments/v2-ablation.md` records decisions; `experiments/runs/<YYYY-MM-DDTHH-MM-SS>-v2-<config>/` stores machine-readable results (each config gets its own timestamped directory)

Completion criteria: ablation data can directly answer "how much did rerank contribute" and "on which queries does dense beat BM25".

### v3 (optional) — Homegrown tiny LLM comparison backend (+2 weeks)

**MVP deliverables (under the realistic constraint of 2 weeks + local MPS):**

- Write a modern decoder-only model from scratch (RoPE + RMSNorm + SwiGLU + weight tying + KV cache)
- Reuse the `Qwen2.5` tokenizer
- Get the architecture working (1-batch overfit verifies forward / backward correctness)
- `TinyDocsGenerator(Generator)` adapter plugs into the same RAG pipeline (what it generates does not matter, just having the pipeline wired up end-to-end is enough)

**Stretch goals (do them only if time allows; **not** pass criteria):**

- Small-scale pretraining (general text + Python docs mixture)
- SFT: `(query, retrieved_chunks) → answer` (distill from Qwen)
- Compare against Qwen on the same eval set

Positioning: a standalone learning objective; not chasing quality.

Completion criteria: all MVP deliverables wired up end-to-end; **generation quality is not a pass criterion**; stretch goals count for whatever gets done. Details in [`plans/v3-tiny-llm.md`](plans/v3-tiny-llm.md).

### v4 — Production-track accuracy (4–6 weeks)

Building on the v2 §9 cross-generator follow-up, which established that the local Qwen 1.5B is the dominant constraint on hallucination at the v2 retrieval ceiling, v4 lifts answer accuracy on Python documentation queries from `0.685` (Qwen 1.5B baseline) / `0.757` (capacity-class generator on the same retrieval) to `≥ 0.90` with `hallucination_rate ≤ 0.03` on a scaled eval set of `n ≥ 300`.

Deliverables (sub-task ordering driven by v2 §9 refusal triage):

- **Refusal calibration** — soften the `[INSUFFICIENT-CONTEXT]` trigger and add in-prompt calibration examples; targets the 8/21 false-refuse rows on retrieval-hit queries observed in v2 §9
- **Retrieval-miss recovery** — query rewrite (HyDE) for natural-language queries, comparison decomposition for `match_policy=all` rows, chunker re-cut to favor finer section/symbol boundaries
- **Generator backend integration** — `claude_backend.py` exposing the Anthropic Messages API as a `Generator` implementation; `--backend {qwen, claude}` CLI flag
- **Self-verification loop** — verify-and-revise pass after the initial generation
- **Eval set expansion** — `eval_sets/v4_prod.jsonl` with `n ≥ 300` in-scope plus 30 out-of-scope queries
- **Diagnostic metrics** — per-query-type accuracy / hallucination breakdown in `results.json`; refusal precision / recall / F1; failure-mode triage tooling
- **Interactive `pdr ask` subcommand** with streaming, citation rendering, pre-flight checks
- **Optional**: agentic / tool-use generator (only if accuracy after the above is < 0.88); HTTP API + web UI; MCP server

Completion criteria: `accuracy ≥ 0.90` on `eval_sets/v4_prod.jsonl`, `hallucination_rate ≤ 0.03`, `latency_p50 ≤ 3 s`, 95% CIs reported on every aggregate accuracy claim. Details in [`plans/v4-prod-ready.md`](plans/v4-prod-ready.md).

## 8. Evaluation Strategy (Spans All Stages)

**Core principle: the eval set is built starting from v0, not patched in at the end.**

**Chunk schema (locked in at v0; eval / debug / citation / rerank all depend on it)**

```json
{
  "chunk_id": "symbol:pathlib.Path.read_text",
  "chunk_type": "symbol",
  "docs_version": "3.12",
  "title": "Path.read_text",
  "text": "body text",
  "symbols": ["pathlib.Path.read_text"],
  "canonical_url": "library/pathlib.html#pathlib.Path.read_text",
  "anchor": "pathlib.Path.read_text",
  "parent_module": "pathlib",
  "source_path": "library/pathlib.html",
  "source_hash": "sha256:..."
}
```

`symbols` is an array (a chunk may contain multiple related symbols); `symbols` may be empty for a `section_chunk`. Full field definitions are in [`plans/v0-retrieval-eval.md`](plans/v0-retrieval-eval.md) §3.

Eval set construction:

1. Hand-pick the 30 most frequently asked symbols in the Python standard library to start (`dict`, `list`, `pathlib.Path`, `os`, `json`, `collections`, `functools`, `itertools`, etc.)
2. Annotate each query with the **multi-answer** schema: `expected_symbols` (array) + `expected_urls` (array) + `query_type`
3. File format: `eval_sets/v0_core.jsonl`, one JSON per line

Schema example:

```json
{
  "query_id": "v0-001",
  "query": "Path vs os.path",
  "query_type": "comparison",
  "expected_symbols": ["pathlib.Path", "os.path"],
  "expected_urls": ["library/pathlib.html", "library/os.path.html"],
  "match_policy": "all",
  "url_match": "strip_anchor",
  "notes": "multi-hop / comparison"
}
```

`query_id` is **required and globally unique** within the eval set (format `v0-NNN`). v1 human scoring and v2 LLM-as-judge join back on this id; query text alone is not a stable key (collides on punctuation / casing).

`query_type` enum: `identifier` / `natural_language` / `comparison` / `howto` / `out_of_scope`

`match_policy` enum: `"any"` (default) / `"all"` (commonly used for comparison)

The hit definition forks on `match_policy`:

- `"any"`: any chunk in top-k whose `symbols` intersect `expected_symbols`, **or** whose `canonical_url` matches any item in `expected_urls` per the URL matching rules
- `"all"`: **every item** in `expected_symbols` and `expected_urls` is covered by some chunk in top-k (used for comparison-type queries to avoid "top-k only hit one side" being judged as a pass)

**URL matching rules** (controlled by the optional `url_match` field on an eval entry; default `"strip_anchor"`):

- `"exact"`: full URLs strictly equal
- `"strip_anchor"` (default): drop everything after `#` on both sides, then compare for equality — `library/pathlib.html#Path.read_text` matches `library/pathlib.html`
- `"prefix"`: **path-prefix** match. Expected URL must end in `.html` or `/`; chunk URL counts only if it equals the expected URL or extends past a `#` or `/`. Avoids `library/pathlib` accidentally matching a future `library/pathlibx.html`.

Metric layers:

- Retrieval layer: Recall@{5,10}, MRR — reported both **aggregate** and **per `query_type`** (`identifier` / `natural_language` / `comparison` / `howto` / `out_of_scope`); aggregate hides the per-type spread that the v1 generator needs to act on
- Generation layer: 4-tier manual scoring (from v1 onward) → LLM-as-judge (from v2 onward)
- Refusal layer: refusal rate on out-of-scope queries

**Machine-readable experiment results**

Run directory naming: `experiments/runs/<YYYY-MM-DDTHH-MM-SS>-<tag>/` (ISO 8601 timestamp down to the second to prevent multiple runs on the same day from overwriting each other; `run_writer` defaults to refusing to write into an existing directory; `--overwrite` overrides explicitly).

Each eval run simultaneously produces:

- `experiments/<stage>.md` — narrates decisions and the story (human-readable)
- `<run_dir>/results.json` — aggregated metrics + config + `DOCS_VERSION` + `docs_sha_short` + `docs_served_version` + `ingest_manifest` snapshot
- `<run_dir>/per_query.jsonl` — top-k, scores, matched chunks, and hit details for each query

Markdown narrates the decisions; JSON proves it can actually be reproduced.

## 9. Optional Side Track: embedding fine-tuning (consider after v2)

Does not block the main project; the goal is to verify on the Python docs domain whether "fine-tuning the embedding gives a measurable recall lift over off-the-shelf `bge-small`":

- Use a large model (Claude / GPT-4) to generate synthetic `(query, doc chunk)` pairs
- Fine-tune `bge-small-en` with contrastive learning
- A/B compare Recall@k improvements
- Pluggable as a standalone module into the dense retrieval layer

## 10. Repository Structure (Initial Skeleton)

```text
python-doc-assistant/
├── README.md
├── PLAN.md
├── pyproject.toml              # dependencies (split into extras by stage) + tool configuration
├── uv.lock                     # locked versions, committed to git
├── .python-version             # read by uv
├── .pre-commit-config.yaml     # optional
├── config.toml                 # DOCS_VERSION and other config
├── src/
│   └── python_doc_assistant/       # standard src layout
│       ├── __init__.py
│       ├── ingest/
│       │   ├── fetch_docs.py           # download a specific version + record manifest sha256
│       │   ├── parse_objects_inv.py    # sphobjinv parses symbol → URI
│       │   └── chunker.py              # symbol_chunk + section_chunk
│       ├── indexes/                    # plural, to avoid colliding with Python's `index()`
│       │   ├── symbol_index.py         # exact + fuzzy
│       │   ├── bm25_index.py           # includes analyzer (`.` / CamelCase / `_` tokenization)
│       │   └── dense_index.py          # v2
│       ├── retrieval/
│       │   ├── router.py               # identifier vs NL
│       │   ├── hybrid.py               # v2
│       │   └── rerank.py               # v2
│       ├── generation/
│       │   ├── interface.py            # Generator ABC
│       │   ├── qwen_backend.py         # v1 (transformers + MPS)
│       │   ├── qwen_gguf_backend.py    # v4 (llama-cpp-python, quantised default)
│       │   └── tinydocs_backend.py     # v3
│       ├── prompts/
│       │   └── grounded.py
│       ├── evaluation/                 # not eval/, to avoid colliding with the builtin eval()
│       │   ├── dataset.py              # load the eval set
│       │   ├── retrieval_metrics.py    # Recall@k, MRR, supports match_policy
│       │   ├── answer_metrics.py       # manual scoring + LLM judge
│       │   └── run_writer.py           # writes results.json + per_query.jsonl
│       └── cli.py                      # unified entry point
├── data/                           # all gitignored
│   ├── docs/
│   │   └── <DOCS_VERSION>/         # by major.minor branch (e.g. `3.12/`)
│   │       ├── current.txt                # current active sha_short
│   │       └── <sha_short>/               # multiple shas may coexist under the same DOCS_VERSION
│   │           ├── ingest_manifest.json
│   │           └── library/...            # raw HTML extracted from the archive
│   ├── chunks/
│   │   └── <DOCS_VERSION>/
│   │       └── <sha_short>/
│   │           └── chunks.jsonl
│   └── indexes/
│       └── <DOCS_VERSION>/
│           └── <sha_short>/
│               ├── bm25.pkl
│               └── dense.npy              # v2
├── eval_sets/
│   ├── v0_core.jsonl            # v0 initial version (multi-answer schema)
│   └── v2_full_200.jsonl           # v2 expansion
├── experiments/
│   ├── v0-bm25-only.md
│   ├── v1-qwen-grounded.md
│   ├── v2-ablation.md
│   ├── v3-tinydocs-vs-qwen.md
│   └── runs/                       # machine-readable results — committed (see policy below)
│       └── <YYYY-MM-DDTHH-MM-SS>-<tag>/   # second-level timestamp prevents same-day overwrite
│           ├── results.json
│           └── per_query.jsonl
└── tests/
```

**Commit policy for `experiments/runs/`:**

- `results.json` and `per_query.jsonl` (and v1+ `human_scores.jsonl`) are **committed** into git. Each run directory is an immutable experiment snapshot — being able to `git log` back to a numerical baseline is worth the small repo bloat.
- Anything that grows fast or is hard to interpret without external tooling — training logs, tensorboard event files, model-generated raw responses larger than ~1 MB — stays out of the run directory or is gitignored on a case-by-case basis.
- `data/` is gitignored in full; do not commit corpora, indexes, checkpoints. `experiments/` stays committed because the artifacts there are the project's reproducibility evidence.

**`pyproject.toml` dependency structure (split into extras by stage)**:

```toml
[project]
dependencies = ["click"]  # core minimal dependencies (`tomllib` is already in the 3.11+ standard library)

[project.scripts]
python-doc-assistant = "python_doc_assistant.cli:main"
pdr = "python_doc_assistant.cli:main"  # development alias

[project.optional-dependencies]
dev = ["pytest", "ruff", "mypy"]
ingest = ["beautifulsoup4", "lxml", "sphobjinv", "requests"]
retrieval = ["rank-bm25", "numpy", "rapidfuzz"]
embedding = ["sentence-transformers", "numpy"]        # v2
generation = ["transformers", "torch", "accelerate"]  # v1
rerank = ["sentence-transformers"]                    # v2
```

v0 environment: `uv sync --extra dev --extra ingest --extra retrieval` (**does not install torch**; lightweight environment).

## 11. Risks and Mitigations

| Risk                           | Mitigation                                         |
| ------------------------------ | -------------------------------------------------- |
| Eval set serves the current implementation (self-deception) | Write the queries first, then look at the output; evaluation-first |
| Docs version drift makes results irreproducible | Pin `DOCS_VERSION`; record it in chunk metadata + every eval run |
| Qwen latency too high on MPS  | Drop to 0.5B, or switch to a `llama.cpp` quantized build |
| `objects.inv` and HTML do not line up | Get one of the two paths working first, then consider merging |
| v3 training time blows up     | v3 itself is optional; if necessary, do only the architecture and skip SFT |
| Over-designing interfaces too early | Hand-roll in v0; only abstract the `Generator` ABC starting in v1 |

## 12. Time Budget

| Stage                | Estimate    | Deliverable form                                |
| -------------------- | ----------- | ----------------------------------------------- |
| v0                   | 1 week      | retrieval CLI + eval set + first metric         |
| v1                   | 1 week      | Qwen generation + refusal + citations           |
| v2                   | 1 week      | hybrid + rerank + ablation table + expanded eval set |
| v3 (optional)        | 2 weeks     | homegrown tiny LLM backend + comparison experiment |
| Side track (embedding fine-tuning) | as-needed after v2 | standalone module A/B                |

**Minimum usable version (MVP): v0 + v1 + v2, about 3 weeks.**
