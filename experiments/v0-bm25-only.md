# v0 BM25-only baseline

## What is the current configuration

Retrieval pipeline (no LLM, no dense embeddings, no rerank):

| Layer | Implementation |
| ----- | -------------- |
| Corpus | Python 3.12 docs (`docs.python.org/3.12` archive, sha `a5c1a35a5a02`, served patch `3.12.13`) |
| Chunker | `symbol_chunk` (per `objects.inv` entry) + `section_chunk` (h2/h3 split, 1500 char cap), `<dl class="py ...">` dedup. 11581 chunks total (9943 symbol + 1638 section). |
| Symbol index | Multi-key dict; full + last-2 + last-1 segment short forms; `rapidfuzz.fuzz.ratio` ≥ 85 fallback. |
| BM25 | `rank_bm25.BM25Okapi` with k1=1.5, b=0.75. Analyzer (plan §5): split on `.` → CamelCase split (uppercase-abbrev + digit boundaries; merged form preserved) → `_` split (dunder preserved) → lowercase → drop empty/punctuation. |
| Router | Heuristic (no ML). `^[A-Za-z0-9._]+$` and no whitespace → IDENTIFIER → SymbolIndex.lookup; else NATURAL_LANGUAGE → BM25. Identifier miss → fallback BM25 (`used=("bm25",)`). |
| Eval set | `eval_sets/v0_core_30.jsonl` — 34 queries (14 identifier / 17 natural_language / 3 comparison). Multi-answer schema with `match_policy` (any / all) and `url_match` (default strip_anchor). |

## Aggregate metrics

Run: `experiments/runs/2026-04-27T07-17-09-v0-bm25/`

| Metric | Value |
| ------ | ----- |
| Recall@5 | **0.824** |
| Recall@10 | 0.853 |
| MRR | 0.674 |
| n_queries | 34 |

Plan §0 / §8 acceptance threshold (Recall@5 ≥ 0.8) — passed.

## Per-query-type breakdown

| query_type | n | Recall@5 | Recall@10 | MRR |
| ---------- | -- | -------- | --------- | --- |
| identifier | 14 | 1.00 | 1.00 | 0.95 |
| natural_language | 17 | 0.82 | 0.88 | 0.57 |
| comparison | 3 | **0.00** | **0.00** | 0.00 |

The aggregate hides a sharp gradient: identifier queries are essentially solved by SymbolIndex, natural-language queries are at BM25's typical strength for short technical text, and comparison queries collapse entirely.

## Where it fails (6 queries at Recall@5 = 0)

### 1. `how to read a file in python` (natural_language)

Expected `pathlib.Path.read_text` / `io.open` / `open` (or their pages). BM25 is dominated by `read` / `file` token frequency across thousands of unrelated chunks (FileHandler, readline, IOBase, etc.). The canonical answers do not surface in top-5.

### 2. `how to memoize a function` (natural_language)

Expected `functools.lru_cache` / `functools.cache`. The string `memoize` is rare in Python docs (functools uses "cached results" phrasing) so BM25 has no tf-idf signal connecting the query to the right chunk.

### 3. `how to run a shell command from python` (natural_language)

Expected `subprocess.run` / `subprocess.Popen`. Tokens `shell` / `command` / `run` are scattered (os.system, os.popen, runpy, runner classes), and the `subprocess` module's introductory text does not concentrate them.

### 4. `Path vs os.path` (comparison, match_policy=all)

Top-5 surfaces pathlib chunks but does not also cover `os.path` chunks within the first 5 results, so the all-policy evaluation reports a miss even though one side is correctly retrieved.

### 5. `list vs tuple` (comparison, match_policy=all)

Top-5 is dominated by individual list / tuple methods (`list.sort`, `list.append`, `tuple.count`, `tuple.index`) rather than the parent classes. BM25 has no notion that `list` and `tuple` themselves are the answers.

### 6. `json vs pickle` (comparison, match_policy=all)

Top-1 is `library/pickle.html#comparison-with-json` — a section chunk that literally compares the two — but `expected_symbols=["json", "pickle"]` and `match_policy=all` requires both module-level symbols to appear, which they do not.

## What v1 should change

Priorities, ordered by expected impact:

1. **Cross-encoder rerank** (`bge-reranker-base`, plan §v2 §3): rerank Top-20 → Top-5 should pull canonical answers (`pathlib.Path.read_text`, `subprocess.run`) above token-frequency noise. Targets failures #1 and #3.

2. **Dense embeddings** (`bge-small-en-v1.5`, plan §v2 §1): semantic retrieval handles synonyms (`memoize` ↔ `cache`, `shell command` ↔ `subprocess`) that BM25 lexical matching misses. Targets failure #2.

3. **Comparison routing** (out of scope for v0): comparison queries need either (a) a comparison-section bias when both module names appear, or (b) a metrics relaxation that accepts a single comparison section_chunk as covering both sides. The current `match_policy=all` insists on per-module coverage, which BM25 alone cannot satisfy on a single page. Targets failures #4–#6.

4. **Score surfacing through `route()`**: v0 uses a `1.0 / rank` placeholder because `route()` does not expose BM25 / SymbolIndex scores upstream. Real scores are needed for v2 hybrid merge (RRF / linear weighting).

5. **Out-of-scope queries** (plan §v1): no row in `v0_core_30.jsonl` exercises refusal behavior because v0 has no generator. v1 should add `eval_sets/v1_out_of_scope_20.jsonl` to measure refusal rate after Qwen is wired in.

## Reproducibility

All runs are written to `experiments/runs/<timestamp>-<tag>/` with `results.json` + `per_query.jsonl`. The above run can be reproduced from this commit:

```
uv run pdr ingest --version 3.12
uv run pdr build-index
uv run pdr eval --set eval_sets/v0_core_30.jsonl --tag v0-bm25
```

`results.json` records `docs_version` / `docs_served_version` / `docs_sha_short` / full `ingest_manifest` snapshot / config / command, so the corpus and pipeline state are recoverable even if `current.txt` later moves to a newer sha.
