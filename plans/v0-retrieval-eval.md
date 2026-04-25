# v0 — Retrieval-only + Eval Set Bootstrap

**Parent doc:** [../PLAN.md](../PLAN.md) §7 v0 / §8 Evaluation Strategy

**Estimated duration:** ~1 week

**Core goal:** Get the end-to-end loop "fetch docs → chunk → build index → retrieve → evaluate" working. **No LLM yet** — just return raw chunks.

**Core principle:** Write the eval set first, then the code.

---

## Prerequisites

- [ ] Create new git repo `python-doc-assistant`
- [ ] `uv init` and write `pyproject.toml`
- [ ] **Package uses standard src layout: `src/python_doc_assistant/`**
- [ ] Add `[project.scripts]` to `pyproject.toml`: `python-doc-assistant = "python_doc_assistant.cli:main"` (also wire up the dev alias `pdr`)
- [ ] v0 extras: `uv sync --extra dev --extra ingest --extra retrieval` (**do not install torch**)
- [ ] `.python-version` pinned to 3.11
- [ ] `config.toml` sets `DOCS_VERSION` (**major.minor**, e.g. `"3.12"`; docs.python.org only serves archives by branch — see §1)
- [ ] Get `ruff check` + `mypy` + `pytest` running (empty project, just to verify the toolchain)
- [ ] Create `tests/fixtures/`: small HTML + `objects.inv` fixtures; **no unit test depends on real network**, the `fetch_docs.py` test mocks out HTTP

## Subtasks (in order)

### 1. Ingest — download docs + record manifest

File: `src/python_doc_assistant/ingest/fetch_docs.py`

**DOCS_VERSION semantics:** `docs.python.org` only serves archives by **major.minor** (e.g. `3.12`); the URL `https://docs.python.org/<major.minor>/archives/python-<major.minor>-docs-html.tar.bz2` serves the **latest patch** docs for that branch (currently `3.12` is actually `3.12.13`). Our pinning strategy:

- `DOCS_VERSION` is configured as **major.minor** (e.g. `"3.12"`)
- The actual patch version is parsed from the unpacked HTML `<title>` (something like `"3.12.13 Documentation"`) and recorded in the manifest's `docs_served_version`
- The archive's sha256 **is used as the storage directory key** (`data/docs/<DOCS_VERSION>/<sha_short>/`); multiple shas can coexist under the same `DOCS_VERSION`, so old evals can always be replayed against the original corpus and won't be overwritten by new ingests

**Flow:**

- Download from `https://docs.python.org/<DOCS_VERSION>/archives/python-<DOCS_VERSION>-docs-html.tar.bz2` (e.g. `https://docs.python.org/3.12/archives/python-3.12-docs-html.tar.bz2`)
- Compute the archive sha256 and take the first 12 chars as `<sha_short>`
- Unpack into `data/docs/<DOCS_VERSION>/<sha_short>/` (e.g. `data/docs/3.12/a1b2c3d4e5f6/`) — **corpora with different shas do not overwrite each other**, old subdirectories are preserved, old evals can always be replayed against the original corpus
- `data/docs/<DOCS_VERSION>/current.txt` records the currently active sha
- **Idempotent + overwrite-protection rules:**
  - Same archive (matching sha) re-ingested → skip, exit 0
  - Sha conflict (newly downloaded sha differs from `current.txt`) → **error out by default**; switching versions requires `--force-switch` (creates a new `<sha_short>` subdirectory and updates `current.txt`)
- Write manifest (see below)

**Downstream paths in sync (chunks / indexes are layered under the same `<DOCS_VERSION>/<sha_short>/`):**

- `data/chunks/<DOCS_VERSION>/<sha_short>/chunks.jsonl`
- `data/indexes/<DOCS_VERSION>/<sha_short>/bm25.pkl` / `dense.npy` etc.

`data/docs/<DOCS_VERSION>/<sha_short>/ingest_manifest.json`:

```json
{
  "docs_version": "3.12",
  "docs_served_version": "3.12.13",
  "docs_url": "https://docs.python.org/3.12/archives/python-3.12-docs-html.tar.bz2",
  "docs_archive_sha256": "a1b2c3...",
  "ingest_timestamp": "2026-04-24T10:30:00Z"
}
```

At eval run time, snapshot this manifest into `results.json` to ensure traceability at the artifact level.

**Archive cross-machine reproducibility (`data/` is gitignored, the sha-keyed directory only solves the local-machine problem):**

- Also drop the downloaded `.tar.bz2` archive into a **content-addressed cache**: `~/.cache/python-doc-assistant/archives/<docs_archive_sha256>.tar.bz2` (full sha256 as the filename)
- If the same sha is already cached → skip download
- To reproduce an old eval on another machine: pack the archive from the cache directory, ship it over, and unpack
- v0 ships with a local cache only; **for future team collaboration, document an external artifact store (S3 / GCS / etc.), addressed by full `docs_archive_sha256`** (not implemented in v0, but the extension point is reserved)

**Acceptance criteria:**

- `data/docs/3.12/<sha_short>/library/pathlib.html` exists and is readable
- `data/docs/3.12/current.txt` points at the currently active sha
- `ingest_manifest.json` has all five fields, and `docs_served_version` is correctly parsed from HTML
- Re-ingesting the same sha → idempotent exit 0; a different sha without `--force-switch` → errors out
- Unit tests use fixtures to verify manifest generation + sha-conflict logic, with no network access

### 2. Parse objects.inv

File: `src/python_doc_assistant/ingest/parse_objects_inv.py`

- Use the `sphobjinv` library to read `objects.inv`
- Produce `list[SymbolEntry]`, where each entry contains:
  - `name` (e.g. `pathlib.Path.read_text`)
  - `role` (`py:method` / `py:class` / `py:function` / ...)
  - `uri` (e.g. `library/pathlib.html#pathlib.Path.read_text`)
  - `module` (e.g. `pathlib`)

**Acceptance criteria:** Lists 6000+ symbols, including `pathlib.Path.read_text`; unit tests use a fixture inventory and don't depend on a real download

### 3. Chunker — two chunk types + fixed schema

File: `src/python_doc_assistant/ingest/chunker.py`

Two chunk types:

- **`symbol_chunk`**: For each `objects.inv` symbol, jump to the matching HTML anchor and grab the `<dl class="py method">` / `<dl class="py class">` block, including signature + docstring
- **`section_chunk`**: HTML `<section>` elements not bound to a symbol (tutorial / HOWTO / concept), split by h2/h3

**Deduplication rule (so the same text doesn't get fed to BM25 twice):**

First run the `symbol_chunk` pass, recording the absorbed HTML nodes (`<dl class="py ...">` and their subtrees). Then run the `section_chunk` pass; during DFS, skip those nodes so they don't end up in the section text. Otherwise `pathlib.Path.read_text` would appear in both the symbol_chunk **and** the section_chunk that contains it, distorting BM25 scores and any downstream rerank.

**Unified chunk schema (locked in for v0; downstream eval / debug / citation / rerank all depend on it):**

```json
{
  "chunk_id": "symbol:pathlib.Path.read_text",
  "chunk_type": "symbol",
  "docs_version": "3.12",
  "title": "Path.read_text",
  "text": "body text...",
  "symbols": ["pathlib.Path.read_text"],
  "canonical_url": "library/pathlib.html#pathlib.Path.read_text",
  "anchor": "pathlib.Path.read_text",
  "parent_module": "pathlib",
  "source_path": "library/pathlib.html",
  "source_hash": "sha256:abc123..."
}
```

Field notes:

- `chunk_id`: globally unique, format `<chunk_type>:<stable-key>`
- `chunk_type`: `"symbol"` or `"section"`
- `symbols`: array — a chunk may contain multiple related symbols; `section_chunk` may have an empty array
- `canonical_url`: complete URL (relative to docs root, including anchor)
- `anchor`: the fragment after `#`, for direct linking
- `parent_module`: top-level module (`pathlib` / `os` / `json` / ...); `section_chunk` may be `null`
- `source_path`: path to the HTML file, relative to `data/docs/<DOCS_VERSION>/<sha_short>/`
- `source_hash`: sha256 of this chunk's source text, used for dedup + incremental refresh detection

All chunks are written to `data/chunks/<DOCS_VERSION>/<sha_short>/chunks.jsonl` (JSON Lines; shares the same `<sha_short>` as docs).

**Acceptance criteria:**

- `chunks.jsonl` has 5000+ entries
- Reasonable symbol/section ratio (symbols expected to dominate)
- All required fields are present
- `chunk_id` is globally unique (verified by enumeration in unit tests)

### 4. Symbol index — exact + fuzzy (multi-candidate)

File: `src/python_doc_assistant/indexes/symbol_index.py`

Short names collide — `open` exists in builtin / `io.open` / `os.open` / `codecs.open`; many method names also appear across multiple classes. The index must be **multi-candidate**:

```python
exact_index: dict[str, list[Candidate]]
```

`Candidate` contains at least `chunk_id / fully_qualified_name / role / parent_module`.

Query behavior:

- `pathlib.Path.read_text` (fully qualified) → typically 1 candidate
- `Path.read_text` (short name) → multiple candidates, sorted by `parent_module` + `role` priority (`py:class` / `py:method` first), or return the whole list to the upstream router/ranker
- Fuzzy matching uses `rapidfuzz` to handle case and minor typos (`dict.fromKeys` → `dict.fromkeys`), also returning `list[Candidate]`

**Acceptance criteria:**

- Hand-written tests (using fixture chunks) cover 10 spelling variants; the candidate set for each query contains the expected chunk
- At least 1 **short-name multi-candidate** test case (e.g. short name `open` hits builtin `open` + `io.open` + `os.open` + `codecs.open` and others)

### 5. BM25 index + analyzer

File: `src/python_doc_assistant/indexes/bm25_index.py`

**Analyzer rules (applied uniformly to both query and doc indexing; **order matters**):**

1. Split on `.`: `pathlib.Path` → `pathlib`, `Path` (preserve original case)
2. Split CamelCase: `fromKeys` → `from`, `Keys`; also keep the merged form `fromKeys`
3. Split underscores: `read_text` → `read`, `text`; also keep `read_text`
4. **Finally** lowercase everything
5. Drop empty tokens and pure punctuation

> **Why lowercase must come after CamelCase splitting:** If you lowercase first, `fromKeys` becomes `fromkeys` and you can no longer recover the `from` / `Keys` boundary.

**Examples:**

- `pathlib.Path.read_text` → `[pathlib, path, read_text, read, text]`
- `dict.fromKeys` → `[dict, fromkeys, from, keys]`
- `os.path.join` → `[os, path, join]`
- `how to iterate dict safely` → `[how, to, iterate, dict, safely]`

**Why split this way?** Many Python docs queries are symbol-shaped (`Path.read_text`, `dict.fromKeys`). With plain English whitespace tokenization, BM25 can't match `read` or `keys` against the index at all, and recall tanks.

Doc side: run the analyzer over the chunk's `text` + `symbols` (expanded and concatenated) together.

**Index engine:** `rank_bm25.BM25Okapi`

**Persistence:** Serialize to `data/indexes/<DOCS_VERSION>/<sha_short>/bm25.pkl` and support reload.

**Acceptance criteria:**

- Analyzer unit tests: token output for the 4 examples above matches exactly
- 10 symbol-shaped queries (`Path.read_text` / `dict.fromKeys` / ...) must hit the corresponding chunk in Top-5

### 6. Query router

File: `src/python_doc_assistant/retrieval/router.py`

- Heuristically decide whether the query is an identifier (contains `.`, no spaces, looks like code) or natural language
- Identifier → try the symbol index first, fall back to BM25 on miss
- Natural language → straight to BM25

v0 uses rules, no ML.

**Acceptance criteria:** 10 manually labeled queries route correctly

### 7. CLI

File: `src/python_doc_assistant/cli.py`

Exposed via `pyproject.toml`'s `[project.scripts]`:

```text
python-doc-assistant ingest --version 3.12
python-doc-assistant build-index  [--docs-sha <sha_short>]
python-doc-assistant search "Path.read_text" --k 5  [--docs-sha <sha_short>]
python-doc-assistant search "how to iterate dict safely" --k 5 --debug
python-doc-assistant eval --set eval_sets/v0_core_30.jsonl --tag v0-bm25  [--docs-sha <sha_short>]
```

(The `pdr` alias is available during development)

**Sha resolution rules (shared by `build-index` / `search` / `eval`):**

- `--docs-sha <sha_short>` explicitly specified → use that subdirectory directly (recommended for eval runs to guarantee reproducibility)
- `--docs-sha` not passed → default to reading `data/docs/<DOCS_VERSION>/current.txt`, and print `resolved docs-sha=<sha_short>` to stdout as a notice
- **Eval runs must record the actually-used `docs_sha_short` in `results.json`** (whether it came from `--docs-sha` or `current.txt`); this prevents an old eval from becoming irreproducible if `current.txt` is later changed

`--debug` shows each chunk's score + routing decision + analyzer-emitted tokens + the `docs_sha_short` in use.

**Acceptance criteria:** All 5 commands run end-to-end from a clean state; `eval` output's `results.json` contains the `docs_sha_short` field

### 8. Eval set v0 (**finished before the code**)

File: `eval_sets/v0_core_30.jsonl`

- 30–50 queries, **hand-written**
- Multi-answer + match_policy schema:

```json
{
  "query": "Path vs os.path",
  "query_type": "comparison",
  "expected_symbols": ["pathlib.Path", "os.path"],
  "expected_urls": ["library/pathlib.html", "library/os.path.html"],
  "match_policy": "all",
  "notes": "multi-hop / comparison"
}
```

- `query_type` enum: `identifier` / `natural_language` / `comparison` / `howto`
- `match_policy` enum:
  - `"any"` (default): a hit is recorded if any chunk in top-k matches any expected item
  - `"all"` (default for comparison): top-k must cover every expected item
- `url_match` field is optional, defaults to `"strip_anchor"`; enum `"exact"` / `"strip_anchor"` / `"prefix"` (see §9 URL matching rules)
- Mix: 40% identifier / 50% natural language / 10% comparison

**Acceptance criteria:** Valid JSON Lines, passes the validator script (schema-compliant + valid `match_policy` value)

### 9. Eval metrics + run writer

Files:

- `src/python_doc_assistant/evaluation/retrieval_metrics.py`
- `src/python_doc_assistant/evaluation/run_writer.py`

**Hit logic:**

For each query, take Top-K and decide based on `match_policy`:

- `match_policy: "any"`: any chunk in top-k whose `symbols` intersect `expected_symbols`, **or** whose `canonical_url` matches any of `expected_urls` per the URL matching rules
- `match_policy: "all"`: **every item** in `expected_symbols` and `expected_urls` is covered by some chunk in top-k

**URL matching rules** (controlled by the optional `url_match` field on each eval entry, defaults to `"strip_anchor"`):

- `"exact"`: full URL strict equality
- `"strip_anchor"` (default): strip everything after `#` on both sides and compare for equality (`library/pathlib.html#pathlib.Path.read_text` matches `library/pathlib.html`)
- `"prefix"`: chunk URL starts with the expected URL counts as a match (for permissive scenarios like "the entire pathlib module counts as correct")

**Metric definitions (Recall@K and MRR both branch on `match_policy`):**

- **Recall@K:** Decide whether the query is a "hit" (0 / 1) per match_policy, then mean over queries
- **MRR:**
  - `match_policy: "any"` → use the rank of the **first** matching item in top-K; RR = 0 if nothing hits (standard MRR)
  - `match_policy: "all"` → use the rank at which **all expected items are covered** (i.e. the rank where the last expected item appears); RR = 0 if any expected item is missing from top-K

Compute Recall@5, Recall@10, MRR.

**Output:**

Run directory naming: `experiments/runs/<YYYY-MM-DDTHH-MM-SS>-<tag>/` (ISO 8601 timestamp to seconds + custom tag, e.g. `2026-04-25T14-30-00-v0-bm25`).

- `results.json`: aggregate metrics + config + `DOCS_VERSION` + `docs_served_version` + `docs_sha_short` + `ingest_manifest` snapshot
- `per_query.jsonl`: top-k chunks, scores, match_policy, and hit details for each query

**Overwrite-protection:** `run_writer` refuses by default to write to an existing directory; overwriting requires explicit `--overwrite`. Timestamps go down to the second, so collisions are unlikely under normal circumstances; the edge case of two runs in the same second is covered by the `--overwrite` fallback.

CLI:

```text
python-doc-assistant eval --set eval_sets/v0_core_30.jsonl --tag v0-bm25
```

**Acceptance criteria:**

- Running the command produces both files
- Both `any` and `all` policies are covered by unit tests
- `results.json` round-trips cleanly back to a Python object

### 10. Experiment narrative doc

File: `experiments/v0-bm25-only.md`

Half a page to a page, answering:

- What the current configuration is
- The Recall@5 / Recall@10 / MRR numbers
- Which query categories failed (with 3–5 concrete examples)
- What v1 should change

## Completion criteria

- [ ] `eval_sets/v0_core_30.jsonl` has 30–50 queries, multi-answer + `match_policy` schema
- [ ] A single command runs end-to-end from a clean state: `uv sync && python-doc-assistant ingest && python-doc-assistant build-index && python-doc-assistant eval`
- [ ] `ingest_manifest.json` has all five fields (`docs_version` / `docs_served_version` / `docs_url` / `docs_archive_sha256` / `ingest_timestamp`)
- [ ] Chunk schema has all 11 fields (`chunk_id` / `chunk_type` / `docs_version` / `title` / `text` / `symbols` / `canonical_url` / `anchor` / `parent_module` / `source_path` / `source_hash`), with `chunk_id` globally unique
- [ ] BM25 analyzer rules have unit tests
- [ ] Recall@5 has a baseline number (**target ≥ 0.8, non-blocking**)
- [ ] `experiments/runs/<YYYY-MM-DDTHH-MM-SS>-v0-bm25/` contains `results.json` + `per_query.jsonl`
- [ ] `experiments/v0-bm25-only.md` is written
- [ ] All unit tests use fixtures, no network access

## Decisions to make during execution (I'll ask you)

- Which docs branch to pin `DOCS_VERSION` to (`3.12` / `3.13` / ...)
- HTML parsing details: how to split nested `<section>` elements on each module page
- Maximum length for `section_chunk` (to prevent a whole page becoming one chunk)
- Fuzzy matching threshold
- Specific conditions for the query router heuristics
- Which 30 starter symbols to include in the eval set
- Whether CamelCase splitting handles all-caps acronyms specially (`HTTPServer` → `http`, `server` or letter by letter)
