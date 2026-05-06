# v4 — production-track accuracy (Qwen-only path), in progress

Active spec: `plans/v4-prod-ready.md` Revision 2026-05-05 — Qwen-only path.

This narrative documents work-in-progress on the v4 lift from the v2
baseline. Distinct from v3 / v3.1 (research side track), v4 stays on
the canonical pipeline (`pdr eval` + LLM-as-judge) and measures
quantitative accuracy / hallucination / refusal rates.

The Claude-path version of v4 was scoped at `accuracy ≥ 0.90`.
The Qwen-only revision targets `accuracy ≥ 0.78` on the same eval set
(`eval_sets/v2_full.jsonl`, n=111). We are documenting Week 0 + Week 1
checkpoints below; the doc will be updated as further sub-tasks land.

## Status

| Sub-task | Topic | State |
|---|---|---|
| 3' | Qwen 7B GGUF backend (`qwen_gguf_backend.py` + tests + CLI wiring) | ✅ done (commits `fc9894e`, `b24d6d7`, `67c8dd6`) |
| 0 | Baseline re-run on v2_full (symbol+bm25, then dense+rerank) | ✅ done |
| 1 | Refusal calibration (`grounded.py` SYSTEM_REFUSAL + EXAMPLES) | ✅ done (commit `81ff06e`) |
| 5d | Failure triage script | ✅ done (commit `14c1c37`) |
| 1' | Code-level query rewriter for typo'd identifiers | ✅ done (commit `01c0597`) |
| 2 | Retrieval miss recovery (HyDE + comparison decomp + chunker re-cut) | ⏳ next |
| 4 | Self-verify loop | ⏳ pending |
| 5b/c | Per-type metrics + refusal F1 | ⏳ pending |
| 6 | `pdr ask` interactive (already exists; gain `--backend qwen-gguf`) | ✅ wired |
| 1' | Anti-hallucination prompt (deferred) | ⏸️ deferred |
| 7 / 9 / 10 | Streaming / web UI / MCP | ⏸️ optional |

## Reproducibility

| Field | Value |
|---|---|
| docs_version | `3.12` |
| docs_served_version | `3.12.13` |
| docs_sha_short | `a5c1a35a5a02` |
| eval set | `eval_sets/v2_full.jsonl` (n=111) |
| generator | Qwen2.5-7B-Instruct Q4_K_M GGUF (~4.7 GB on disk) |
| inference stack | `llama-cpp-python` (Metal, n_gpu_layers=-1) |
| context | n_ctx=8192 (bumped from 4096 after dense+rerank prompts overflowed) |
| decoding | greedy (temperature=0.0, top_p=1.0, max_new_tokens=512) |
| judge | `claude-haiku-4-5-20251001` (prompt hash `65fa23b9`, same as v2 §6) |
| device | M1 Pro 16 GB |

## Week 0 — 7B Q4 swap (sub-task 3')

### Smoke (5 queries, sft_corpus.jsonl)

The `tmp/smoke_qwen7b_gguf.py` script confirmed:

- Model loads in ~15 s
- Throughput ~22.6 tok/s on M1 Metal
- 5 grounded RAG queries answered with `[1]` citations and a
  4-section structure ("signature -> brief description -> example ->
  source citation") matching the `grounded.py` template.

Sample output for `pathlib.Path.read_text`:

```
signature -> Path.read_text(encoding=None, errors=None) [1]
brief description -> Return the decoded contents of the pointed-to
file as a string. [1]
example -> >>> p = Path('my_text_file') >>> p.write_text('Text file
contents') 18 >>> p.read_text() 'Text file contents' [1]
source citation -> [1]
```

This is a different quality class from v3.1's 67M SFT outputs (which
under the same query produced "(encoding='UTF-8') returns the text
representation of the file, which is encoded in UTF-8. [1]").

### Baseline `pdr eval` runs (n=111)

Two retrievers tested on the unmodified `grounded.py` (matches v2
§9 baseline / follow-up setup):

| Run tag | Retriever | recall@5 | accuracy | correct | partial | wrong | halluc | refused |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `v4-qwen-gguf-baseline` | symbol+bm25 (k=5) | 0.730 | 0.703 | 0.225 | 0.477 | 0.054 | 0.054 | 0.189 |
| `v4-qwen-gguf-dense-rerank` | dense + rerank (top-20 → top-5) | 0.829 | 0.703 | 0.180 | 0.523 | 0.072 | 0.081 | 0.144 |

(`accuracy = (correct + partial) / n`; tier labels assigned by Haiku
4.5 judge, prompt hash `65fa23b9`.)

#### Comparison vs v2 baselines

| Run | retriever | accuracy | halluc | refused | recall@5 |
|---|---|---:|---:|---:|---:|
| v2 baseline (Qwen 1.5B transformers) | dense+rerank | 0.685 | 0.216 | 0.018 | 0.838 |
| v2 §9 follow-up (Claude capacity-class) | dense+rerank | 0.757 | 0.027 | 0.189 | 0.838 |
| **v4 (Qwen 7B Q4 GGUF)** | dense+rerank | **0.703** | **0.081** | **0.144** | **0.829** |

Two findings:

1. **The Qwen 1.5B → 7B Q4 swap is real.** Same retrieval pipeline,
   accuracy lifted from 0.685 to 0.703 (+1.8 pp) and hallucination_rate
   collapsed from 0.216 to 0.081 (-13.5 pp). This is the largest
   single delta in the whole v4 work plan and it cost half a day of
   coding plus a 5 GB model download.
2. **7B inherits Claude-class behavior on refusal.** `refused_rate`
   jumped from 0.018 (1.5B) to 0.144 (7B) — within the same range as
   Claude's 0.189 in v2 §9. The "Qwen barely refuses" observation that
   underwrote the original Qwen-only revision (and the rationale for
   deleting sub-task 1) was 1.5B-specific. At 7B, sub-task 1 is back
   on the table.

The two retriever choices produced **identical accuracy 0.703** but
different tier composition: symbol+bm25 has more confident
correct/refused outcomes; dense+rerank has more partial answers from
topically-related-but-not-direct chunks (see `correct=0.225 → 0.180`,
`partial=0.477 → 0.523`).

### A surprise — n_ctx=4096 overflowed on dense+rerank

The first dense+rerank run failed mid-eval with
`ValueError: Requested tokens (5348) exceed context window of 4096`.
Section chunks pulled by dense retrieval were larger than what
symbol+bm25 had returned. The default n_ctx was bumped to 8192
(commit `67c8dd6`) — Qwen2.5-7B was trained on 131k context, and Q4
KV cache at 8192 is ~3 GB on top of the ~4.7 GB model. The 16 GB M1
absorbs this with margin.

## Week 1 — sub-task 1 refusal calibration

### Triage

Pulled the 21 refused rows from `v4-qwen-gguf-baseline` (symbol+bm25)
and bucketed them by `hit_at_5`:

| Bucket | n | Description |
|---|---:|---|
| `hit_at_5=True`  | 11 | Chunks contain (or related to) the answer; refusal candidates |
| `hit_at_5=False` | 10 | Chunks unrelated to the question; refusal correct |

Inside the 11 `hit_at_5=True` rows, ~5 are clear false refusals (the
chunks contain the symbol and the model still refused — including
several typo queries like `pathlib.Path.raed_text`,
`subprocess.runn`, `json.loadss`, plus the exact-match query
`typing.Annotated`). The other ~6 are borderline (related chunks that
only support a partial answer). The full triage and proposed prompt
diff are in `tmp/refusal_calibration_draft.md`.

### Calibration applied

`src/python_doc_assistant/prompts/grounded.py` was modified
(commit `81ff06e`):

1. `SYSTEM_REFUSAL` softened: refuse only when chunks are
   "completely unrelated" to the question, prefer a partial answer
   otherwise.
2. `SYSTEM_HARD_RULES` adds an explicit "prefer partial over refusal"
   rule and tightens the refusal trigger language.
3. New `SYSTEM_REFUSAL_EXAMPLES` block with three calibration shots
   (typo query → ANSWER, partially-related chunks → PARTIAL,
   completely off-topic → REFUSE).

### Result

| Metric | Baseline (dense+rerank) | Calibrated | Δ |
|---|---:|---:|---:|
| **accuracy** | **0.703** | **0.730** | **+2.7 pp** |
| correct_rate | 0.180 | 0.225 | +4.5 pp |
| partial_rate | 0.523 | 0.505 | -1.8 pp |
| wrong_rate | 0.072 | 0.072 | 0 |
| **hallucination_rate** | **0.081** | **0.081** | **0** |
| **refused_rate** | **0.144** | **0.117** | **-2.7 pp** |
| tier counts (correct / refused) | 20 / 16 | **25 / 13** | +5 correct, -3 refused |

Notes:

- The calibration **did not let hallucination climb**. The risk that
  softening the refusal trigger would let the model invent
  partially-grounded answers was the principal worry going in; the
  observed `hallucination_rate=0.081` is identical pre/post.
- 3 refusals flipped to correct (16 → 13 refused, 20 → 25 correct).
  The pre-flight prediction was ~5 clear false refusals → correct;
  ~60% of the predicted lift materialized. The remaining 2 likely
  flipped from refused to partial (which is captured in the small
  partial drop, but partial counts also moved for unrelated reasons,
  so the per-row attribution is fuzzy without judge_scores diff).
- Per-query-latency rose from ~7 s/query to ~13 s/query post-calib —
  consistent with the model writing longer partial answers instead of
  emitting a single-token refusal marker.

### Cumulative pipeline lift

| Stage | Backend | Prompt | retriever | accuracy |
|---|---|---|---|---:|
| v2 §9 baseline | Qwen 1.5B transformers | strict (v1) | dense+rerank | 0.685 |
| v4 Week 0 | Qwen 7B Q4 GGUF | strict (v1) | dense+rerank | 0.703 (+1.8) |
| **v4 Week 1** | Qwen 7B Q4 GGUF | **calibrated** | dense+rerank | **0.730 (+4.5)** |
| Claude follow-up (target) | Claude capacity-class | strict (v1) | dense+rerank | 0.757 |

The v4 Qwen-only path now sits **2.7 pp below the Claude follow-up**.

## Honest gap analysis (Qwen 7B Q4 vs Claude)

Despite the same accuracy ceiling proximity (`0.730 vs 0.757`), the
two systems answer very differently:

| Axis | 7B Q4 (calibrated) | Claude follow-up | Gap |
|---|---:|---:|---|
| `correct_rate` | 0.225 | 0.370 | -14.5 pp (precision) |
| `partial_rate` | 0.505 | n/a (not in v2 §9 dump) | — |
| `hallucination_rate` | 0.081 | 0.027 | +5.4 pp (3× more hallucination) |
| `refused_rate` | 0.117 | 0.189 | -7.2 pp |

The 7B model trades precision for coverage: it gives "answer-shaped"
partial answers more often than Claude (which prefers refusal), but
3× as many of those answers contain unsupported claims. This matches
the v3.1 finding that small models are good at the **answer shape**
but struggle to maintain factual grounding consistently — except that
v3.1 was 67M and lost grounding hard, while 7B holds it well enough
for `(correct + partial) / n` to be only ~3 pp behind Claude.

## Week 2 — sub-task 1 follow-up: prompt R3 (failed) → code rewriter

### Triage of post-calibration refusals

`scripts/triage_failures.py` (added in `14c1c37`) bucketed the 13
post-calibration refusals from the `v4-qwen-gguf-dense-rerank-calib`
run by `judge tier × hit_at_5`:

| Bucket | n | Description |
|---|---:|---|
| `refused_hit5_yes` | 10 | Chunks contain (or relate to) the answer; model still refused |
| `refused_hit5_no`  |  3 | Chunks unrelated to the question; refusal correct |

Inside the 10 `hit_at_5=True` refusals, 5 are typo queries where the
right symbol chunk is in top-K but spelling does not match exactly:
`json.loadss`, `dict.fromKeys`, `dict.fromkeyss`, `subprocess.runn`,
`pathlib.Path.raed_text`. The other 5 are borderline natural-language
queries with related-but-not-direct chunks (the calibrated prompt
already converted most of those to partials; the residue is hard to
move without HyDE / reformulation).

### Round 3 prompt — aggressive typo block (failed)

`grounded.py` was modified (commit `5a2bcaf`) to add a dedicated
`SYSTEM_TYPO_TOLERANCE` block at the top of system content, with
imperative voice and 5 hardcoded typo→symbol mappings copying the
known false-refusal queries verbatim. Smoke on those 5 queries went
from 0/5 answered to 3/5 answered (json.loadss, dict.fromKeys,
subprocess.runn). Letter-swap cases (`raed_text`, `fromkeyss`)
continued to refuse.

`pdr eval` on n=111 with the R3 prompt:

| Metric | R1 calibrated | R3 prompt | Δ |
|---|---:|---:|---:|
| accuracy | 0.730 | 0.784 | +5.4 pp |
| correct_rate | 0.225 | 0.234 | +0.9 pp |
| partial_rate | 0.505 | 0.550 | +4.5 pp |
| **hallucination_rate** | **0.081** | **0.126** | **+4.5 pp** |
| refused_rate | 0.117 | 0.045 | -7.2 pp |

R3 hit the v4 accuracy target 0.78 but pushed hallucination_rate from
0.081 to 0.126 — past the 0.10 target, a 56% relative regression. The
per-tier story: of the 8 refusals R3 unblocked, ~3-5 became correct
or partial, but ~5 became hallucinations. The aggressive typo block
generalised into "any near-match → answer", so the model started
inventing content for queries that did not actually have a clean
near-match in the chunks.

R4 (the same typo block stripped of its hardcoded examples, leaving
only the abstract rule) regressed smoke-typos to 1/5 — the typo
example block was where the lift came from, but it was also where the
hallucination came from. Prompt-only path hit a ceiling.

### Code rewriter (final, commit `01c0597`)

`SYSTEM_TYPO_TOLERANCE` was reverted; the prompt is back to R1. A new
module `python_doc_assistant.retrieval.query_rewriter` runs between
retrieval and generation and rewrites the query to the canonical
symbol when one of the top-K chunks is an obvious typo match.

Algorithm: for each symbol chunk in the retrieved list, compute the
case-insensitive Levenshtein distance to the query. If exactly one
symbol minimises distance and that minimum is ≤ 2, rewrite the query
to that symbol. Otherwise leave the query alone. The disambiguation
guard (more than one symbol tied at the minimum distance) prevents
collapsing genuinely ambiguous queries; the ≤ 2 cap keeps unrelated
identifiers from being touched. Pure code, no extra dependency,
18 unit tests.

Two integration points: `evaluation/generation_eval.py` (eval
pipeline) and the `pdr ask` CLI command. The rewrite affects only
what the generator sees — the original query is preserved in
`per_query.jsonl` so the judge join stays stable.

Smoke on the 5 typo queries: **5/5 answered, 0 hallucinations**
(generator output cites the same chunk_ids it would for the
canonical query; behaviour is deterministic).

### Result on n=111

(`v4-qwen-gguf-dense-rerank-calib-rewriter`, 1 judge transient error
on `set vs frozenset`; n_judged=110.)

| Metric | R1 calibrated | R3 prompt | **Rewriter** | Δ vs R1 | Δ vs R3 |
|---|---:|---:|---:|---:|---:|
| accuracy | 0.730 | 0.784 | **0.773** | +4.3 pp | -1.1 pp |
| correct_rate | 0.225 | 0.234 | **0.255** | +3.0 pp | +2.1 pp |
| partial_rate | 0.505 | 0.550 | 0.518 | +1.3 pp | -3.2 pp |
| wrong_rate | 0.072 | 0.045 | 0.073 | +0.1 pp | +2.8 pp |
| **hallucination_rate** | **0.081** | **0.126** | **0.082** | +0.1 pp | **-4.4 pp** |
| refused_rate | 0.117 | 0.045 | 0.073 | -4.4 pp | +2.8 pp |

The rewriter recovers most of R3's accuracy (0.773 vs 0.784) without
R3's hallucination penalty (0.082 vs 0.126). Per-row attribution:
refused dropped from 13 to 8 = 5 typo queries flipped, of which 3
became correct, 1 became partial, and 1 ended in the judge-error
hole. No new hallucinations or wrongs were introduced — Levenshtein
matching is deterministic, and on near-misses where no clean rewrite
exists, the rewriter abstains and the model gets the original query
(and refuses, as before).

### Cumulative pipeline lift

| Stage | Backend | Prompt | Pre-gen rewrite | accuracy | hallucination |
|---|---|---|---|---:|---:|
| v2 §9 baseline | Qwen 1.5B transformers | strict (v1) | – | 0.685 | 0.216 |
| v4 Week 0 | Qwen 7B Q4 GGUF | strict (v1) | – | 0.703 | 0.081 |
| v4 Week 1 | Qwen 7B Q4 GGUF | calibrated R1 | – | 0.730 | 0.081 |
| v4 Week 2 (R3 failed) | Qwen 7B Q4 GGUF | aggressive R3 | – | 0.784 | 0.126 |
| **v4 Week 2 (rewriter)** | Qwen 7B Q4 GGUF | calibrated R1 | **lev≤2 rewriter** | **0.773** | **0.082** |
| Claude follow-up (target) | Claude capacity-class | strict (v1) | – | 0.757 | 0.027 |

The Qwen-only path now sits 1.6 pp above the Claude follow-up's
accuracy with 3× the hallucination rate, and 0.7 pp short of the v4
0.78 accuracy target.

### Lessons (sub-task 1)

1. **Prompt has a real, narrow ceiling for behaviour calibration.**
   R1 (soft hint) lifted 3 false-refusals at zero hallucination cost.
   R3 (aggressive examples) added 5 more lifts but introduced 5 new
   hallucinations 1:1. Once the prompt was strong enough to override
   the refusal reflex, it also overrode the grounding reflex.
2. **Code-level deterministic rewrites beat prompt nudges for
   surface-form normalisation.** The rewriter does not change the
   model's behaviour — it changes what the model sees. The model's
   "only use chunks" constraint is preserved, so hallucination does
   not move.
3. **Triage by judge_tier × hit_at_5 was the right framing.** The 5
   typo cases (refused_hit5_yes, near-match in chunks) and the 10
   refused_hit5_no cases (real retrieval miss) require different
   fixes. Sub-task 1 owned the first; sub-task 2 owns the second.

## What's next

- **Sub-task 2 (retrieval miss recovery)** — 8 of the remaining
  refusals are still `hit_at_5=False`. HyDE + comparison
  decomposition + chunker re-cut should recover several of those,
  lifting accuracy further toward and past the 0.78 target.
- **Sub-task 4 (self-verify loop)** — hallucination_rate is back at
  0.082, well below the 0.10 cap. Self-verify remains a fallback
  lever only if sub-task 2's accuracy lift comes with a hallucination
  cost we cannot pay otherwise.
- **Sub-task 5b/c (per-type metrics + refusal F1)** — defer until
  sub-task 2 lands; the per-type story is more interesting once
  retrieval misses have been triaged separately from refusals.

The v4 narrative will be amended as those sub-tasks land.
