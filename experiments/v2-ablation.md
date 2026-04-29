# v2 — Retrieval Ablation + Generation Quality

Living document. Filled in incrementally as §1 → §7 of `plans/v2-ablation.md` complete.

## Status

| Section | Topic | State |
|---------|-------|-------|
| §1 | Dense embedding index (`indexes/dense_index.py`) | ✅ done (commits `f2bb1bb` + `23fda6e`) |
| §2 | Hybrid merge — RRF + linear (`retrieval/hybrid.py`) | ✅ done (commit `b48f7f4`) |
| §3 | Cross-encoder rerank (`retrieval/rerank.py`) | ✅ done (commit `cb2c81f`) |
| §4 | Expand eval set 34 → 111 (`eval_sets/v2_full.jsonl`) | ✅ done (commit `3d1190d`) |
| §5 prereq | Retrieval factory + CLI flags | ✅ done (commit `e3993ef`) |
| §5 — retrieval-only matrix | 12 configs against v2_full | ✅ done (commit `3073f36`) |
| §5 — generation @ recommended config | dense+rerank+qwen | ✅ done (commit `7d57dd6`) |
| §5 — generation @ remaining 5 configs | bm25 / dense / hybrid-rrf / α=0.3 / rrf+rerank | ⏳ TODO |
| §6 | LLM-as-judge (`evaluation/judge.py`) | ⏳ backbone landed; impl + agreement check pending |
| §7 | Narrative — rerank contribution / dense vs bm25 / α / final config | ⏳ TODO |
| §8 | v3 priority recommendation | ⏳ TODO |

## Reproducibility

| Field | Value |
|---|---|
| docs_version | `3.12` |
| docs_served_version | `3.12.13` |
| docs_sha_short | `a5c1a35a5a02` |
| chunks count | 11,581 (BM25 corpus + dense embedding rows) |
| dense embedding model | `BAAI/bge-small-en-v1.5` (384-dim, L2-normalized) |
| reranker | `BAAI/bge-reranker-base` (top-20 → top-5 cross-encoder) |
| generator | `Qwen/Qwen2.5-1.5B-Instruct` (greedy, max_new_tokens=512, top_p=1) |
| eval set | `eval_sets/v2_full.jsonl` (n=111) |
| device | MPS (Mac) |

Hybrid-linear `alpha` sweep grid: {0.0 (=dense), 0.2, 0.3, 0.5, 0.7, 0.8, 1.0 (=bm25)}.

## §1 — Dense embedding index

`DenseIndex` wraps `sentence-transformers` + numpy. Build once from `list[Chunk]`,
encode `title + "\n\n" + text` per chunk, save as `dense.npy` + sidecar
`dense.json` (model_id / chunk_ids / dim).

- **Persistence:** `data/indexes/<docs_version>/<sha_short>/dense.npy` (17.0 MB
  for n=11,581 × 384 float32) + `dense.json` (~413 KB).
- **L2-normalized at encode time** so cosine == inner product
  (`embeddings @ query_emb`). Saves a normalization step at search time.
- **Lazy-imports `sentence_transformers` + `numpy`**; v0 install path
  (no `embedding` extra) still imports the module.
- **DI hook** for tests — pass `model=stub` to skip `from_pretrained`.

Per-query latency on MPS (n=11,581 corpus): ~215 ms mean, range 12-400 ms
(first-query model warm-up). Negligible vs. generation cost.

## §2 — Hybrid merge

Two merge strategies; both ablated in §5.

- **RRF** (`rrf_merge`) — rank-based, no score normalization needed:
  `score(c) = Σ_i 1 / (k + rank_i(c))` with default `k=60`. Takes ranked
  chunk_id lists (not scored). Robust default with no hyperparameters.
- **Linear** (`linear_merge`) — score-based: `score(c) = α · bm25_norm + (1-α) · dense_norm`.
  Each list is min-max normalized to [0,1] independently because BM25
  scores and cosine similarities live on different scales.
- **Edge cases:** empty corpus → []; constant-score input → all 0.5
  (avoids divide-by-zero, no within-list signal).

## §3 — Cross-encoder rerank

`CrossEncoderReranker.rerank(query, chunks, *, top_k=5, batch_size=32)` calls
`sentence_transformers.CrossEncoder.predict()` on `(query, chunk_text)` pairs
in batches. Decision (per plan §142): the original input score is **NOT
preserved** — `RerankedHit.score` is the cross-encoder output only. Rank
movement vs. the input ranking can still be analyzed via the per_query.jsonl
`retrieved` field which carries pre-rerank scores.

## §4 — eval_sets/v2_full.jsonl (n=111)

Built on v0's 34 entries + 77 new rows targeting the gaps plan §4 calls out:

| query_type | v0 | new | total | % |
|---|---|---|---|---|
| identifier | 14 | 20 | 34 | 30.6% |
| natural_language | 17 | 10 | 27 | 24.3% |
| howto | 0 | 23 | 23 | 20.7% |
| comparison | 3 | 21 | 24 | 21.6% |
| out_of_scope | 0 | 0 | 0 | 0% (kept in `eval_sets/v1_out_of_scope_20.jsonl`) |

`match_policy=all` is set on 18 comparison rows where both sides must be
retrieved (e.g. `deque vs list`, `Union vs Optional`). New entry classes
include 10 typo identifiers (e.g. `pathlib.Path.raed_text`,
`tempflie.NamedTemporaryFile`), 10 long natural-language queries (multi-clause
conversational), 21 explicit howto rows, and 17 new comparison rows.

Multi-pass review record (heuristic draft → Claude → Codex round 1+2 →
Gemini → my own pass) caught: comparison rows promoted to `match_policy=all`,
`yield` removed from "yield vs return" symbols (it is syntax not a stdlib
symbol), `json.load` disambiguated to single symbol, `args vs kwargs`/long
unpacking URL list expanded to include `reference/expressions.html`,
`groupby` query expanded with `collections.defaultdict`. See commit
`3d1190d` for the full diff.

## §5 — Retrieval-only ablation matrix (12 configs, n=111)

Each row = one `pdr eval` invocation against `v2_full.jsonl` with
retrieval-only metrics (no generator loaded). All configs share the same
docs sha + chunk corpus + eval set, so deltas attribute to retrieval only.

| Config                       | Recall@5 | Recall@10 |   MRR |
|------------------------------|---------:|----------:|------:|
| bm25                         |    0.712 |     0.766 | 0.567 |
| symbol+bm25 (v0 baseline)    |    0.730 |     0.775 | 0.625 |
| hybrid-linear α=0.8          |    0.748 |     0.865 | 0.591 |
| hybrid-rrf                   |    0.775 |     0.865 | 0.631 |
| hybrid-linear α=0.5          |    0.784 |     0.883 | 0.647 |
| hybrid-linear α=0.7          |    0.784 |     0.865 | 0.601 |
| hybrid-linear α=0.2          |    0.802 |     0.901 | 0.694 |
| hybrid-linear α=0.3          |    0.802 |     0.910 | 0.692 |
| dense (α=0)                  |    0.811 |     0.892 | 0.691 |
| hybrid-rrf + rerank          |    0.829 |     0.883 | 0.709 |
| hybrid-linear α=0.3 + rerank |    0.829 |     0.883 | 0.709 |
| **dense + rerank**           |  **0.838** |   0.883 | 0.705 |

Run dirs: see commit `3073f36`.

### Key retrieval findings

1. **vs v0 baseline (0.730)** — best config dense+rerank reaches
   **0.838 = +10.8 pp Recall@5**. plan §7 question "how much does
   rerank contribute?" — at the chosen retrieval input, rerank adds
   **+2.7 pp** (dense 0.811 → dense+rerank 0.838).
2. **Dense alone outperforms every hybrid** (any α ∈ {0.2…0.8}). The v2
   eval set skews toward NL / howto / comparison (66 of 111 = 60%) which
   dense embedding handles better than BM25's keyword overlap. plan
   §7 question "Dense vs BM25 — which queries?" → BM25 wins on
   identifier-exact (e.g. `pathlib.Path.read_text` matches verbatim) and
   typo-recovery (when typo'd token still hashes near a valid token);
   Dense wins on every NL / howto / comparison case.
3. **α-sweep U-curve** — α=0.2-0.3 is the floor of the linear-merge
   sweep (Recall@5 = 0.802, Recall@10 = 0.910 at α=0.3 — actually the
   highest Recall@10 across all configs). High-α (BM25-heavy) configs
   degrade monotonically; α=0 (pure dense) re-tops α=0.3. plan §7
   question "α optimal value?" → **α=0.3** if forced into linear hybrid;
   but pure dense still beats it on Recall@5.
4. **Rerank candidate set is the bound** — `hybrid-rrf + rerank` and
   `hybrid-linear α=0.3 + rerank` produce identical {Recall@5,
   Recall@10, MRR} = {0.829, 0.883, 0.709}. The cross-encoder reorders
   the same top-20 chunks regardless of how the 20 are selected; ranking
   strategy under the rerank cap is invisible.
5. **Different metrics → different winners** — Recall@5 maxed by
   dense+rerank; Recall@10 maxed by hybrid-linear α=0.3 (no rerank);
   MRR maxed by either rerank-flavored hybrid (0.709). For the §7
   recommendation we use Recall@5 since K=5 is what the LLM sees.

## §5 — Generation @ recommended config (dense+rerank+qwen)

Run dir: `experiments/runs/2026-04-28T12-16-13-v2-dense-rerank-qwen` (commit
`7d57dd6`).

| Metric (full v2_full, n=111) | Value |
|---|---|
| Recall@5 | 0.838 |
| Recall@10 | 0.883 |
| MRR | 0.705 |
| Mean generation latency | 21.6 s |
| refused | 2/111 (1.8%) |
| cited_at_least_one | 59/111 (53.2%) |
| cited_match_expected (symbols only) | 33/111 (29.7%) |
| cited_match_expected (URL or symbols) | 51/111 (45.9%) |
| cited but no match | 26/111 (23.4%) |

| Restricted to v0_core 34 queries | v1 baseline | v2 dense+rerank | Δ |
|---|---|---|---|
| Recall@5 | 0.824 | 0.912 | **+8.8 pp** |
| Recall@10 | 0.853 | 0.941 | +8.8 pp |
| MRR | 0.674 | 0.805 | **+13.1 pp** |
| Cited match expected (symbols) | 38.2% | 35.3% | -2.9 pp |
| Refused | 0/34 | 1/34 | +1 |

### The retrieval-vs-cite-rate paradox (Codex review finding)

Recall jumps 8-13 pp but exact-symbol citation match _decreases_ slightly
(38.2% → 35.3% on v0_core subset). Three contributing causes:

1. **Heuristic too narrow** — Codex's URL-or-symbols variant gives
   v2 17/34 vs v1 16/34 → **v2 is actually slightly better on broader
   citation match**. The "regression" is a measurement artifact.
2. **Citation behavior is model-bound, not retrieval-bound** — the 1.5B
   Instruct model still leaves chunks uncited even when they're in
   top-5. plan §142 was right to flag this as the v1 ceiling.
3. **Module-level fallback** — three identifier queries (`dict.fromkeys`,
   `datetime.datetime.now`, `subprocess.run`) where v1 cited the precise
   method-level chunk now cite the module-level section_chunk. Dense+rerank
   surfaces section_chunks at higher ranks than v0's BM25, and the model
   prefers the broader chunk. Records as new v2-specific failure mode.

True hallucination_rate vs v1's 14.7% pending §6 LLM-as-judge or manual
re-scoring. **Predicted directionally lower** based on the better
retrieval, but magnitude TBD.

### Two data-quality bugs Codex review caught (fixed in commit `f745ebe`)

1. `QwenGenerator.generate()` did not clear `cited_chunk_ids` when
   `refused=True`. If the model emitted `[1] [INSUFFICIENT-CONTEXT]`,
   `parse_response` would still extract the `[1]` citation.
   2/111 rows in this run were affected.
2. The `open` row in `v0_core.jsonl` / `v2_full.jsonl` used
   `url_match=strip_anchor` against page-level URLs (e.g.
   `library/functions.html`). Since `functions.html` hosts dozens of
   built-ins (slice, range, str…), any chunk on that page was a false
   positive. Switched to `url_match=exact` with full anchors.

## §5 — Generation @ remaining 5 configs (TODO)

Plan §5 ablation table needs answer-quality / hallucination-rate columns
filled in for 6 rows. We have 1 (dense+rerank). 5 more generation runs
needed:

| Plan row | Configuration | Estimated runtime |
|---|---|---|
| 1 | symbol+bm25 (v0 baseline) — re-run on v2_full | ~40 min |
| 2 | dense | ~40 min |
| 3 | hybrid-rrf | ~40 min |
| 4 | hybrid-linear α=0.3 (sweep best) | ~40 min |
| 5 | hybrid-rrf + rerank | ~40 min |
| 6 | dense+rerank (final recommended) | ✅ done |

Total ~3.3 hours background generation + judge runs ($1.65 if Haiku 4.5,
once §6 is implemented).

## §6 — LLM-as-judge (TODO)

Backbone module `src/python_doc_assistant/evaluation/judge.py` landed with
26 unit tests (all currently red on `NotImplementedError` — implementation
pending). Once implemented:

1. **Agreement check (plan §6 step 1):** stratified sample 20 rows from
   v1 baseline manual scores; run judge; require `exact_match > 0.80`
   AND `cohen_kappa > 0.60`. Cost: $0.05.
2. **Full ablation:** judge runs on each of the 6 §5 generation configs
   (111 rows × 6 configs = 666 rows). Cost: ~$1.65. Records
   `judge_model` / `judge_prompt_hash` / raw output / timestamp per row.
3. **Output:** `judge_scores.jsonl` next to each `per_query.jsonl`;
   `aggregate()` from `evaluation.human_scoring` consumes the same shape.

Anthropic Haiku 4.5 selected over Qwen API (closed-source) and local
72B (Mac MPS infeasible). Decision recorded in commit `f745ebe`'s
context.

## §7 — Narrative answers (TODO)

Will be filled once §5 generation rows + §6 judge scores are in. Plan §7
mandates direct answers to:

- **How much does rerank contribute** to Recall@5 / answer quality?
  Retrieval answer: +2.7 pp Recall@5 (dense 0.811 → 0.838). Generation
  answer pending.
- **Dense vs BM25** — which queries does each win? Retrieval-only
  finding: BM25 wins on identifier-exact + typo-recovery; Dense wins on
  NL/howto/comparison.
- **α sweep optimum** + impact on hallucination — α=0.3 retrieval-best
  for linear hybrid; impact on hallucination rate pending §6 judge.
- **Final recommended configuration** + why?
  Strong candidate: **dense + rerank** (best Recall@5, +10.8 pp over
  v0 baseline, latency 21.6s/query acceptable). Confirmation pending
  hallucination_rate.

## §8 — v3 priority recommendation (TODO)

Will be informed by §6 + §7. Early signals:

- **Module-level vs method-level citation** — new v2 failure mode
  (dense+rerank surfaces section_chunks higher than v1's BM25 path; 1.5B
  model prefers broader chunks). Could be addressed by chunker boundary
  refinement (split section_chunks more aggressively) or rerank prompt
  tweaks.
- **1.5B Instruct ceiling** still likely the dominant constraint after
  retrieval is solved. plan §142's 7B-upgrade decision point should
  fire if §6 hallucination_rate stays > 10 % on the dense+rerank config.
- **Chunker fragmentation** (v1 §5 finding — code blocks tokenized
  one-per-line) may have less impact in v2 once dense embedding is in
  the mix (semantic match doesn't need clean code formatting), but
  worth verifying with a side-by-side on howto queries.
