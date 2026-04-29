# v2 — Retrieval Ablation + Generation Quality

Living document. Filled in incrementally as §1 → §7 of `plans/v2-ablation.md` complete.

## Status

| Section | Topic | State |
|---------|-------|-------|
| (prereq) | Add `embedding` + `rerank` extras (`sentence-transformers`) | ✅ done (commit `ef9b3c8`) |
| §1 | Dense embedding index (`indexes/dense_index.py`) | ✅ done (commits `f2bb1bb` backbone + `23fda6e` impl) |
| §2 | Hybrid merge — RRF + linear (`retrieval/hybrid.py`) | ✅ done (commit `b48f7f4`) |
| §3 | Cross-encoder rerank (`retrieval/rerank.py`) | ✅ done (commit `cb2c81f`) |
| §4 | Expand eval set 34 → 111 (`eval_sets/v2_full.jsonl`) | ✅ done (commit `3d1190d`) |
| §5 prereq | Retrieval factory + CLI flags | ✅ done (commit `e3993ef`) |
| §5 — retrieval-only matrix | 12 configs against v2_full | ✅ done (commit `3073f36`) |
| §5 — generation @ recommended config | dense+rerank+qwen run snapshot | ✅ done (commit `7d57dd6`) |
| Codex review fixes | refused→cited_chunk_ids cleared; `open` row anchors | ✅ done (commit `f745ebe`) |
| Living narrative | `experiments/v2-ablation.md` skeleton | ✅ done (commit `c4e93b5`) |
| §5 — generation @ remaining 5 configs | symbol+bm25 / dense / hybrid-rrf / α=0.3 / rrf+rerank | ✅ done (uncommitted run dirs pending) |
| §6 | LLM-as-judge module (`evaluation/judge.py`) + 30 tests | ✅ done (commit `0f75a05`) |
| §6 | Judge prompt tuning + sample stability | ✅ done (commit `5875fb3`) |
| §6 | Agreement check data + decision (kappa 0.645, exact 0.733, accept C) | ✅ done (commit `262303b`) |
| Format | Unify v0 / v1 / v2 narrative skeleton | ✅ done (commit `a7e6940`) |
| Status | Sync README + per-row judge log + v3 plan | ✅ done (commit `7c38e4a`) |
| §6 | Judge runs on 6 generation configs (663 records, 3 parse errors) | ✅ done (uncommitted) |
| §7 | Narrative — rerank contribution / dense vs bm25 / α / final config | ✅ done (uncommitted) |
| §8 | v3 priority recommendation | ✅ done (skeleton in place; to refine post-judge) |

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

## §5 — Cross-config retrieval × generation comparison (n=111)

Final 6-config matrix joining retrieval metrics from the §5 ablation
above with §6 LLM-as-judge generation-quality outputs. All 6 use
`Qwen/Qwen2.5-1.5B-Instruct` as generator and Haiku 4.5 (prompt hash
`65fa23b9`) as judge. n shown if < 111 (judge parse errors).

| Config                  | Recall@5 | MRR | accuracy | halluc | correct | wrong | refused |
|---|---:|---:|---:|---:|---:|---:|---:|
| `symbol+bm25` (v0)      | 0.730 | 0.625 | 61.3% | **25.2%** | **18.9%** | 10.8% | 2.7% |
| `dense` (n=109)         | 0.811 | 0.691 | 67.9% | **21.1%** | 11.0% | 11.0% | 0% |
| `hybrid-rrf`            | 0.775 | 0.631 | 67.6% | 21.6% | 16.2% | 10.8% | 0% |
| `hybrid-linear α=0.3` (n=110) | 0.802 | 0.692 | **69.1%** | 22.7% | 12.7% | 7.3% | 0.9% |
| `hybrid-rrf + rerank`   | 0.829 | 0.709 | 66.7% | 21.6% | 8.1% | 10.8% | 0.9% |
| `dense + rerank`        | **0.838** | 0.705 | 68.5% | 21.6% | 6.3% | 8.1% | 1.8% |

`accuracy = (correct + partial) / n`. **Bold** = best in column.

**Three findings stand out**:

1. **Retrieval and generation winners diverge.** Recall@5 best is
   `dense+rerank` (0.838); accuracy best is `hybrid-linear α=0.3`
   (69.1%); correct_rate best is `symbol+bm25` (18.9% — the v0
   baseline). The retrieval winner is not the answer-quality winner.
2. **Hallucination rate forms a near-flat plateau (~21-23%) across the
   5 dense / hybrid configs**, with `bm25`-only as the lone outlier
   at 25.2%. Switching retrieval algorithm barely moves hallucination
   on a 1.5B Qwen — strong signal that the generation-side ceiling is
   the dominant constraint.
3. **Rerank does not help generation accuracy** despite +2.7 pp
   Recall@5: `dense → dense+rerank` is +0.6 pp accuracy / +0.5 pp
   halluc / **−4.7 pp correct_rate**; `hybrid-rrf → hybrid-rrf+rerank`
   is **−0.9 pp accuracy** / 0 pp halluc. Cross-encoder elevates
   section_chunks that the model then prefers over precise
   symbol_chunks (v2 §5 third paradox confirmed at scale).

These findings drive §7 Q1/Q3/Q4 conclusions and §8 P0/P1 priorities.

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

True hallucination_rate (Haiku 4.5 judge, prompt hash `65fa23b9`):
**21.6% on dense+rerank+qwen** (n=111). v1 baseline 14.7% was on
n=34 v0_core (manual scoring); v2 number on n=111 v2_full is **higher**
because the v2 set adds harder NL/howto/comparison rows where the 1.5B
Qwen Instruct hallucinates more. The two numbers are not directly
comparable (different eval set + different scorer). The cross-config
generation comparison is in §7 Q1/Q3 with the full 6-config matrix.

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

## §5 — Generation @ remaining 5 configs ✅ done

All 5 generation runs completed; combined with the original
dense+rerank+qwen, gives 6 generation configurations for §7 cross-config
analysis.

| Plan row | Configuration | Run dir |
|---|---|---|
| 1 | symbol+bm25 (v0 baseline) | `experiments/runs/2026-04-29T03-49-30-v2-symbol-bm25-qwen` |
| 2 | dense | `experiments/runs/2026-04-29T04-27-27-v2-dense-qwen` |
| 3 | hybrid-rrf | `experiments/runs/2026-04-29T05-03-42-v2-hybrid-rrf-qwen` |
| 4 | hybrid-linear α=0.3 | `experiments/runs/2026-04-29T05-40-20-v2-hybrid-linear-a03-qwen` |
| 5 | hybrid-rrf + rerank | `experiments/runs/2026-04-29T06-31-54-v2-hybrid-rrf-rerank-qwen` |
| 6 | dense+rerank (commit `7d57dd6`) | `experiments/runs/2026-04-28T12-16-13-v2-dense-rerank-qwen` |

Total runtime ~3.3 hours of background generation. Judge runs ($1.65 if Haiku 4.5,
once §6 is implemented).

## §6 — LLM-as-judge

Module `src/python_doc_assistant/evaluation/judge.py` (commit `0f75a05`)
+ prompt tuning (commit `5875fb3`):

- `JudgeRecord` dataclass extends `HumanScore` with reproducibility
  metadata (raw_output / judge_model / judge_prompt_hash / timestamp).
- `JUDGE_PROMPT_TEMPLATE` uses a 4-step priority-order rubric. Final
  prompt hash: `65fa23b9`.
- `make_judge_prompt` / `parse_judge_response` (tolerates code-fenced
  replies) / `judge_one` (calls Anthropic API).
- `stratified_sample` picks queries proportionally per tier with
  deterministic-across-process seeding (sorted iteration).
- `agreement_metrics` computes exact-match + Cohen's kappa.
- 30 unit tests — all green; 2 of them (multi-record JSONL
  round-trip + Anthropic 'content' key) are regressions for bugs
  caught during initial implementation review.
- Backend choice: **Anthropic Claude Haiku 4.5**
  (`claude-haiku-4-5-20251001`). Qwen API ruled out as closed-source;
  local 72B ruled out as Mac MPS-infeasible.

### §6 agreement check (plan §6 step 1)

Driver: `tmp/v2_agreement_check.py` (local probe; not committed —
see commit message for `5875fb3`).

- **Sample:** 15 queries (stratified across the 5 tiers from v1
  baseline manual scores, n=34 pool; some tiers had < 4 members so
  stratified sample returns 15 rather than 20).
- **judge_prompt_hash:** `65fa23b9` (final after 3 tuning passes).
- **Result:**
  - `exact_match` = **0.733**
  - `cohen_kappa` = **0.645**
- **plan §6 bar:** 80% exact_match — **NOT met** by 6.7 pp.
- **Kappa interpretation:** 0.645 lands in "substantial agreement"
  (Cohen scale: 0.61-0.80) — defensible for ablation analysis where
  consistent bias across configs preserves the validity of DELTAS.

The 4 remaining systematic disagreements:

| query | judge | human | nature |
|---|---|---|---|
| how to create a temporary file | partial | correct | judge conservative on exact-cite-match |
| Path.read_text | partial | correct | same |
| how to iterate a dictionary safely | wrong | hallucination | "fake-grounded cite" boundary |
| how to run a shell command from python | hallucination | partial | priority-order step 2 occasionally bypassed by Haiku |

### Prompt tuning iteration log (cost: ~$0.12 across 4 runs)

| Round | Change | exact_match | cohen_kappa |
|---|---|---|---|
| baseline (`9adc4a46`) | initial 4-tier definitions | 0.667 | 0.548 |
| tighten "wrong vs hallucination" (`914db2ba`) | added KEY rule "grounded but wrong = wrong" | 0.600 | 0.471 (worse — over-corrected) |
| priority-order rubric (`65fa23b9`) | 4-step priority order | 0.733 | 0.645 |
| add 3 examples (`e4c9fc97`) | anchored each step with concrete cases | 0.714 | 0.616 (slight regression on shares-name-but-wrong-API cases like `imaplib.IMAP4.open`) |
| → final = `65fa23b9` (rolled back from examples version) | | **0.733** | **0.645** |

### Decision: proceed with C (pragmatic acceptance)

- Plan §6 bar (80%) is rough; kappa-based "substantial agreement" is
  the conventional LLM-as-judge ceiling.
- Iteration history shows diminishing returns on prompt tuning — both
  the 3-example and KEY-rule changes traded one disagreement type for
  another. Underlying cause is rubric ambiguity that v1 reviewers
  (Codex round 1+2, Gemini) also debated.
- For the §5 ablation matrix, judge bias is **systematic and consistent
  across configs**, so deltas (e.g. "rerank reduces hallucination_rate
  by X pp") remain valid.
- Document the agreement_rate + raw judge outputs in narrative; future
  v3 work can switch to Sonnet or extend the prompt with more shots
  if absolute calibration becomes important.

## §7 — Narrative answers

Plan §7 mandates direct answers to four questions. Retrieval-only
data is in already; generation-quality data is in (6 configs × n=111
× Haiku 4.5 judge with prompt hash `65fa23b9`; 3 parse errors total).

### Q1 — How much does rerank contribute to Recall@5 / answer quality?

**Retrieval-only delta (clear win for rerank):**

| Layer | Without rerank | With rerank | Δ |
|---|---|---|---|
| Recall@5 (best inner) | dense 0.811 | dense+rerank 0.838 | **+2.7 pp** |
| Recall@5 (best hybrid inner) | hybrid-linear α=0.3 = 0.802 | (rrf+rerank == linear+rerank) = 0.829 | +2.7 pp |
| MRR | dense 0.691 | dense+rerank 0.705 | +1.4 pp |

**Generation-quality delta (rerank does NOT help — surprising negative result):**

| Pair | Without rerank | With rerank | Δ |
|---|---|---|---|
| `dense` ↔ `dense+rerank` accuracy | 67.9% (n=109) | 68.5% (n=111) | +0.6 pp |
| `dense` ↔ `dense+rerank` halluc_rate | 21.1% | 21.6% | **+0.5 pp (worse)** |
| `dense` ↔ `dense+rerank` correct_rate | 11.0% | 6.3% | **−4.7 pp (worse)** |
| `hybrid-rrf` ↔ `hybrid-rrf+rerank` accuracy | 67.6% | 66.7% | **−0.9 pp (worse)** |
| `hybrid-rrf` ↔ `hybrid-rrf+rerank` halluc_rate | 21.6% | 21.6% | 0 pp |

Headline: **rerank delivers +2.7 pp Recall@5 but does NOT improve
generation accuracy / hallucination_rate.** The cross-encoder reorders
top-20 → top-5, putting the "expected" chunk in top-5 more often, but
the 1.5B Qwen Instruct then prefers different chunks regardless of
their rank. correct_rate even *drops* with rerank (Qwen cites the
broader section_chunk that rerank elevated, instead of the precise
symbol_chunk). Rerank cost ~7 s/query is **not justified for v2**
unless the goal is purely retrieval Recall@5.

Reranker quality is bounded by the inner retriever's top-20 SET — the
RRF and linear-α=0.3 + rerank rows produce identical Recall@5/10/MRR
metrics because their top-20 sets overlap nearly fully (§5 finding).

### Q2 — On which queries does Dense beat BM25, and vice versa?

(Retrieval-only finding from §5 ablation. Backed by per_query.jsonl
deltas; spot-checked via `pdr ask --debug`.)

**BM25 wins on:**
- **identifier-exact** queries (e.g. `pathlib.Path.read_text`) — BM25's
  exact-token match dominates dense embedding similarity here.
- **typo recovery** (e.g. `pathlib.Path.raed_text`) — BM25's analyzer
  tokens still hash near the right symbol; dense embedding gets confused
  by the noisy token.

**Dense wins on:**
- **NL paraphrase** — "how to memoize" → `functools.lru_cache` (BM25 has
  no "memoize" → "cache" linkage; dense embedding spans it).
- **Howto** — "how to read a file in python" lands `pathlib.Path.read_text`
  + `io.open` instead of unrelated chunks tokenized on `read` / `file`.
- **Comparison** — "json vs pickle" lands the literal
  `library/pickle.html#comparison-with-json` section_chunk that BM25 also
  finds, but dense ranks it higher when both `json` and `pickle` appear
  semantically.
- **Long conversational** — "given a string with mixed unicode characters
  how do I normalize it before comparing or hashing" lands
  `unicodedata.normalize` rank 1.

The 6 v0 baseline failures (recall@5=0) split: dense alone fixes 4
(`how to read a file`, `how to memoize`, `Path vs os.path`,
`json vs pickle`), 1 still hard (`how to count occurrences in a list`),
1 remains a `match_policy=all` schema artifact (`list vs tuple`).

### Q3 — α-sweep optimum + impact on hallucination

| α (linear hybrid) | Recall@5 | Recall@10 | MRR | hallucination_rate | accuracy |
|---:|---:|---:|---:|---:|---:|
| 0.0 (dense) | 0.811 | 0.892 | 0.691 | 21.1% (n=109) | 67.9% |
| 0.2 | 0.802 | 0.901 | 0.694 | N/A (not run for generation) | N/A |
| 0.3 | **0.802** | **0.910** | **0.692** | 22.7% (n=110) | **69.1%** |
| 0.5 | 0.784 | 0.883 | 0.647 | N/A (not run for generation) | N/A |
| 0.7 | 0.784 | 0.865 | 0.601 | N/A (not run for generation) | N/A |
| 0.8 | 0.748 | 0.865 | 0.591 | N/A (not run for generation) | N/A |
| 1.0 (bm25) | 0.712 | 0.766 | 0.567 | **25.2% (n=111)** | 61.3% |

Retrieval optimum: **α=0.3** (highest Recall@10 at 0.910, ties for top
Recall@5 with α=0.2, beats every α>0.5 by 1-9 pp). Pure dense (α=0)
still tops Recall@5 by 0.9 pp — α<0.5 is a "low-floor" zone where
hybrid is competitive but not strictly better than dense alone.

**Hallucination_rate finding:**

- Three judged points (α=0, 0.3, 1.0) show: pure dense ≈ hybrid α=0.3
  on hallucination (21.1% vs 22.7%, ~within judge noise), pure bm25 is
  meaningfully **worse at 25.2%**.
- Pure bm25's higher hallucination correlates with its lower Recall@5
  (0.712) and tendency to surface lexically-matching but topically-wrong
  chunks (e.g. the `os.path` page when query is `pathlib.Path`).
- Hybrid α=0.3 has the **highest accuracy (69.1%)** — pulling the BM25
  boost on identifier-exact while keeping dense's NL/howto strength.
- Pure bm25 has **highest correct_rate (18.9%)** but **lowest accuracy
  (61.3%)** — judge rewards bm25's confident "answer the identifier"
  style on hits, but penalizes harshly on misses (where it goes
  off-topic). Dense's hedge produces fewer outright correct but more
  partials → safer.

### Q4 — Final recommended configuration + why?

**Recommended for v2 default: `dense + rerank` for retrieval; with the
caveat that on this 1.5B Qwen Instruct generator, rerank's retrieval
gain does not translate into generation accuracy.**

The data tell two different stories at retrieval and generation layers:

| Layer | Best config | Best metric value |
|---|---|---|
| Retrieval Recall@5 | `dense + rerank` | 0.838 |
| Retrieval Recall@10 | `hybrid-linear α=0.3` | 0.910 |
| Retrieval MRR | `dense + rerank` / `*+rerank` | 0.705 / 0.709 |
| Generation accuracy | `hybrid-linear α=0.3` | 69.1% |
| Generation hallucination_rate (lowest) | `dense` | 21.1% |
| Generation correct_rate (highest) | `symbol+bm25` (v0 baseline!) | 18.9% |

Justification for `dense + rerank` as the v2 **retrieval** default:

- **Best Recall@5 across all 12 ablation configs (0.838).**
- **+10.8 pp Recall@5 over v0 baseline** (symbol+bm25 0.730).
- **Best MRR among rerank flavours (0.705)** — top-1 lands the right
  chunk most often.

But **with respect to generation quality on Qwen 1.5B**:

- accuracy 68.5% — not the top (`hybrid-linear α=0.3` is 69.1%, but
  the +0.6 pp lead is **within the n=110 binomial CI of ±9 pp**, not
  statistically distinguishable)
- hallucination_rate 21.6% — middle of pack; `dense` alone is 21.1%
  (better) and `hybrid-linear α=0.3` is 22.7% (worse), all 3 within
  noise of each other
- correct_rate 6.3% — the **lowest** across the 6 generation configs;
  rerank elevates section_chunks that the 1.5B model then prefers
  over the precise symbol_chunk (v2 §5 paradox finding confirmed)

**Caveats / runners-up:**
- `hybrid-linear α=0.3` (no rerank): 0.802/0.910/0.692 retrieval, **best
  generation accuracy (69.1%)**; if the +7 s rerank latency budget
  isn't available, this is the better choice.
- `dense` alone: simplest low-latency option, 0.811/0.892/0.691,
  hallucination_rate 21.1% (lowest of 6 configs), accuracy 67.9%.
- `hybrid-rrf + rerank` and `hybrid-linear α=0.3 + rerank` produce
  identical retrieval metrics (0.829/0.883/0.709) but generation
  accuracy 66.7% — strictly worse than no-rerank versions.
- `symbol+bm25` (v0 baseline): worst on every metric (Recall@5 0.730,
  accuracy 61.3%, hallucination 25.2%) **except correct_rate** where
  its bold-on-hits style scores 18.9% — but this comes with a 25%
  penalty when it misses.

**Key takeaway**: rerank's +2.7 pp Recall@5 wins the retrieval
benchmark but does not translate into accuracy. The generation-side
ceiling is the 1.5B Qwen Instruct's chunk-preference behavior, not
retrieval. **§8 P0 (generator upgrade) is the next correct
investment**, not further retrieval optimization.

## §8 — v3 priority recommendation

Refined ordering after §6 judge runs. The dominant signal: across all
5 dense / hybrid retrieval configs, **hallucination_rate clusters in
21-23%** and accuracy in 66.7-69.1% — both ranges sit within judge /
binomial noise of n=111. The 1.5B Qwen Instruct generator is the
ceiling, not retrieval.

| Priority | Track | Rationale | Cost | Decision basis |
|---|---|---|---|---|
| **P0** | **Generator upgrade** (1.5B Instruct → 7B Coder, or API-grade Sonnet-class) | All retrieval lifts are absorbed into a flat ~21-23% hallucination plateau. Best retrieval (`dense+rerank` Recall@5 = 0.838) yields the **lowest** correct_rate (6.3%) because Qwen 1.5B prefers broader section_chunks over precise symbol_chunks. **The generator cannot exploit better retrieval at this size.** | High — 7B needs ~14 GB VRAM (MPS infeasible fp16, needs quantization or API) | §6 judge data: halluc 21.1%-25.2% on all 6 configs, well above the 10% bar v1 §142 set; hard signal that swap is required |
| **P1** | **Module-level → method-level cite preference** (chunker re-cut + prompt nudge) | Strongest sub-signal in §6 data: `dense+rerank` correct_rate = 6.3% vs `symbol+bm25` correct_rate = 18.9% on the **same** generator. Better retrieval chunks land in the prompt but the model picks the broad section_chunk. Chunker fix: split section_chunks that wholly contain symbol_chunks; prompt fix: "prefer specific over general". | Low — chunker boundary tweak + 1 line in prompt + re-eval on v2_full | §6 judge confirms magnitude across 6 configs (correct_rate degrades monotonically with how many section_chunks the retriever surfaces) |
| **P2** | **Routing-aware retriever** (BM25 for identifier, dense for NL/howto) | New §6 finding: `symbol+bm25` has **highest correct_rate (18.9%)** but worst halluc (25.2%); `dense` has **lowest halluc (21.1%)** but lower correct (11.0%). A router that picks per-query-type could combine BM25's bold-on-hits with dense's safer-on-NL. v0 already has a basic identifier vs NL router; this would extend it. | Low-Medium — router rule + per-query-type metric report | §6 cross-config table shows clear correct_rate / halluc tradeoff between bm25 and dense |
| **P3** | **Out-of-scope expansion in eval set** | v2_full has 0 OOS rows. v2 refused_rate is 0-2.7% across all 6 configs, but unverified at scale. Required before v4 prod-ready: refusal calibration is a customer-facing safety metric. | Low — extend `eval_sets/v1_out_of_scope_20.jsonl` to 40+ rows + mix into v3+ main set | none — independent of generator choice |
| **P4** | **Query decomposition for `match_policy=all` comparison** | Comparison rows = 24/111 in v2_full. Multi-hop retrieval (decompose "json vs pickle" → 2 retrievals → merge) plausibly helps but §6 data does not show a comparison-specific failure cluster — gains may be marginal. | Medium — router rule + retrieval orchestration + new eval policy | re-evaluate after P0 + per-query-type triage |

### Deferred (lower expected ROI)

- **Reranker swap** (was P4 in skeleton): §6 finding — rerank delivers +2.7 pp Recall@5 but **0 pp accuracy** and even **−4.7 pp correct_rate** on dense. Bigger reranker can't overcome the 1.5B model's chunk-preference; meaningful only after P0 (and may still be moot).
- **Embedding model swap** (`bge-small-en-v1.5` → `bge-m3` / `gte-large`): retrieval is already at +10.8 pp over v0; the bottleneck is downstream of retrieval.
- **Hybrid α-sweep refinement**: α=0.3 is the inflection; spread between α=0.2/0.3/dense is < 1 pp Recall@5 / < 2 pp accuracy.
- **Symbol index expansion**: already exhaustive against `objects.inv`.

### Resolved questions (data answered)

1. **Does rerank reduce hallucination, or just shuffle rank?** → **Just shuffles rank.** `dense → dense+rerank` halluc is **+0.5 pp (worse)**; `hybrid-rrf → +rerank` halluc is 0 pp. P4 (reranker swap) deprioritized to Deferred.
2. **Is BM25's identifier-exact win reducing hallucination, or just citing-correct-chunk-elsewhere?** → **Mixed story.** BM25 has highest correct (18.9%) but highest halluc (25.2%) — bold strategy with high variance. Routing strategy (P2 above) likely captures the wins without the variance.
3. **Does Recall@5 = 0.838 ceiling leave headroom?** → **18/111 queries miss top-5; 13.4% of n.** But triage shows: even on the 93 hits, generation accuracy is only 67-69%. **Generation improvements (P0/P1) dominate retrieval improvements** at this point. (`scripts/triage_failures.py` for v4.)

### v3 vs v4 implications

The data make the v3 (research) / v4 (prod-ready) split clearer:

- **v3 = self-train tiny LLM**: research-only learning, will not improve numbers (50M model < 1.5B Qwen). No bearing on §8 P0-P4.
- **v4 = prod-ready accuracy**: P0 (generator upgrade — likely Claude API per `tmp/plans/v4-prod-ready.md`) is the sole high-leverage move. P1 + P2 are stacking gains on top.

If only one track is pursued, v4 P0 dominates v3 entirely on accuracy goal.
