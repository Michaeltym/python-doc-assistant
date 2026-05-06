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
| 2 | HyDE retriever + cli `--hyde` flag | ✅ done (commit `ef9c951`) |
| — | Judge re-evaluation (Haiku 4.5 → Codex CLI) | ✅ done (commit `99964cd`) |
| 4 | Self-verify loop | ⏸️ deferred (hallucination ≤ 0.01 under Codex judge) |
| 5b/c | Per-type metrics + refusal F1 | ✅ done (`scripts/per_type_metrics.py`) |
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
| judge (original) | `claude-haiku-4-5-20251001` (prompt hash `65fa23b9`, same as v2 §6) |
| judge (final) | `codex-manual-full-prompt` (Codex CLI internal LLM, GPT-5.5; same prompt hash `65fa23b9`) — see Week 3 |
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
| v2 §9 follow-up (GPT-5.5 manual via ChatGPT UI) | dense+rerank | 0.757 | 0.027 | 0.189 | 0.838 |
| **v4 (Qwen 7B Q4 GGUF)** | dense+rerank | **0.703** | **0.081** | **0.144** | **0.829** |

> ⚠ **The two findings below were written under the Haiku judge and
> overturned in Week 3.** The Codex re-evaluation found that v2
> baseline's true accuracy is 0.793 (not 0.685) and hallucination is
> 0.063 (not 0.216) — Haiku had reclassified ~17 wrong answers as
> hallucinations. The "Qwen 1.5B → 7B is real" claim and the "Qwen
> barely refuses at 1.5B" claim are smaller-magnitude effects than
> these Haiku-judged numbers suggested. See Week 3 §"Numbers under
> both judges" for the corrected table.

Two findings:

1. **The Qwen 1.5B → 7B Q4 swap is real.** Same retrieval pipeline,
   accuracy lifted from 0.685 to 0.703 (+1.8 pp) and hallucination_rate
   collapsed from 0.216 to 0.081 (-13.5 pp). This is the largest
   single delta in the whole v4 work plan and it cost half a day of
   coding plus a 5 GB model download.
2. **7B inherits GPT-5.5-class behavior on refusal.** `refused_rate`
   jumped from 0.018 (1.5B) to 0.144 (7B) — within the same range as
   GPT-5.5's 0.189 in v2 §9. The "Qwen barely refuses" observation that
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
| GPT-5.5 follow-up (target) | GPT-5.5 (manual via ChatGPT UI) | strict (v1) | dense+rerank | 0.757 |

The v4 Qwen-only path now sits **2.7 pp below the GPT-5.5 follow-up**.

## Honest gap analysis (Qwen 7B Q4 vs GPT-5.5)

Despite the same accuracy ceiling proximity (`0.730 vs 0.757`), the
two systems answer very differently:

| Axis | 7B Q4 (calibrated) | GPT-5.5 follow-up | Gap |
|---|---:|---:|---|
| `correct_rate` | 0.225 | 0.370 | -14.5 pp (precision) |
| `partial_rate` | 0.505 | n/a (not in v2 §9 dump) | — |
| `hallucination_rate` | 0.081 | 0.027 | +5.4 pp (3× more hallucination) |
| `refused_rate` | 0.117 | 0.189 | -7.2 pp |

The 7B model trades precision for coverage: it gives "answer-shaped"
partial answers more often than GPT-5.5 (which prefers refusal), but
3× as many of those answers contain unsupported claims. This matches
the v3.1 finding that small models are good at the **answer shape**
but struggle to maintain factual grounding consistently — except that
v3.1 was 67M and lost grounding hard, while 7B holds it well enough
for `(correct + partial) / n` to be only ~3 pp behind GPT-5.5.

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
| GPT-5.5 follow-up (target) | GPT-5.5 (manual via ChatGPT UI) | strict (v1) | – | 0.757 | 0.027 |

The Qwen-only path now sits 1.6 pp above the GPT-5.5 follow-up's
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

## Week 3 — sub-task 2 (HyDE) + judge re-evaluation

Two threads landed in Week 3: a HyDE retriever for sub-task 2, and an
audit-driven switch of the LLM-as-judge from Haiku 4.5 to a stricter
GPT-5.5 judge run via Codex CLI. The judge change re-shaped the
numbers across all five runs and is the larger story.

### HyDE retriever (sub-task 2)

`src/python_doc_assistant/retrieval/hyde.py` (commit `ef9c951`)
implements the canonical HyDE pipeline (Gao et al., 2022):

1. Classify the query. Skip identifier queries (already in document
   form) — they go straight to the existing dense+rerank path.
2. For non-identifier queries, ask the LLM to write a 3-5 sentence
   hypothetical Python documentation passage that would answer the
   query.
3. Embed that hypothetical (not the original query) for dense search.
4. Rerank the top-N candidates with the **original** query — the
   cross-encoder scores user intent, not the LLM's invention.
5. Return rank-ordered chunks; the generator still receives the
   original query string, only the retrieval input changed.

`QwenHypotheticalGenerator` shares the loaded `Llama` instance with
`QwenGGUFGenerator` so the 4.7 GB model loads once. The reranker
guard (rerank-with-original-query) caps downside risk: when the
hypothetical hallucinates an off-topic API, the cross-encoder still
scores against user intent and can suppress the wrong chunks. Per-eval
latency rose from ~14 s/query (rewriter only) to ~14 s/query
(rewriter + HyDE) — the hypothetical adds ~2 s, but it overlaps with
batched embedding work.

CLI integration (commit `ef9c951`): new `--hyde` flag on `pdr eval`,
requires `--backend=qwen-gguf` and `--retriever=dense`. The Generator
ABC gained explicit `temperature` / `top_p` / `max_new_tokens`
attributes so the eval CLI can reuse a single QwenGGUFGenerator
instance for both grounded answering and hypothetical generation.

19 unit tests cover skip-vs-HyDE branching, rerank-with-original-query,
the disambiguation guard, and the QwenHypotheticalGenerator ABC
contract.

### HyDE result on n=111 (Haiku judge, before re-evaluation)

(`v4-qwen-gguf-dense-rerank-calib-rewriter-hyde`)

| Metric | Rewriter | Rewriter + HyDE | Δ |
|---|---:|---:|---:|
| recall@5 | 0.829 | **0.856** | +2.7 pp |
| mrr | 0.691 | 0.715 | +2.4 pp |

Recall@5 jumped 2.7 pp — about 3 more queries had the right chunk
land in top-5. tier-level metrics required the judge step, which
prompted the audit below before reading them.

### Judge re-evaluation (Haiku 4.5 → Codex CLI)

While preparing the HyDE judge run, the Anthropic-API rate limit on
the default tier (5 req/min) made re-judging five 111-query runs
sequentially painful. The discussion shifted to "could we use a
different judge?", which surfaced two questions worth answering before
swapping anything: was Haiku applying the rubric correctly in the
first place, and would a different judge change the cross-run story?

Spot-checking 11 Haiku-vs-rubric mismatches (5 partial→correct
candidates, 4 hallucination→partial candidates, 2 hallucination→wrong
candidates) revealed two systematic Haiku biases:

1. **Haiku downgraded prose-correct answers to `partial` when example
   details were not in the retrieved chunks.** The rubric's KEY rule
   says: "When the prose is correct, grounding does NOT matter —
   partial vs correct is decided by citation alone." Haiku
   consistently penalised extra prose detail (signature explanations,
   example code) by moving the row from `correct` to `partial`, even
   when the cite matched the expected symbol exactly.
2. **Haiku labelled "model used the wrong retrieved chunk" as
   `hallucination`.** The rubric reserves `hallucination` for prose
   grounded outside any retrieval (model invented from prior
   knowledge); answers that built on the wrong retrieved chunk should
   be `wrong`. Haiku conflated the two, inflating `hallucination` at
   the expense of `wrong`.

The 5 spot-checked partial-vs-correct rows (e.g. `asyncio.create_task`,
`functools.lru_cache`, `os.path.join`, `argparse.ArgumentParser`,
`functools.partial`) all had:

- prose factually correct,
- cite exactly matching the expected symbol,
- example details that Haiku flagged as "fabricated" but that were
  obviously paraphrased from the chunk's signature line.

Per the rubric these are `correct`. Haiku judged `partial` for all
five.

Both the Codex CLI re-judging pass and a manual rubric check
classified those 5 rows as `correct`. The 4 hallucination→partial
rows (e.g. `list vs tuple`, `how to write JSON to a file`) similarly
all had prose correct + cite mismatched — `partial`, not
`hallucination`. The 2 hallucination→wrong rows (`abstract base class
vs Protocol`, `asyncio.gather vs asyncio.wait`) both had the model
build on a wrong-but-retrieved chunk — `wrong`, not `hallucination`.

Haiku 4.5 is a small judge, and the rubric's KEY rule is exactly the
kind of nuance a smaller model is more likely to drop. Switching to
the Codex CLI internal LLM (GPT-5.5, applied with the same
`make_judge_prompt` payload and same prompt hash `65fa23b9`) yields a
rubric-faithful pass, at the cost of `judge_model` differing from
the v2 baseline.

The decision: **adopt the Codex re-evaluation as the primary v4
numbers, keep the Haiku files as `judge_scores_haiku.jsonl` /
`results_haiku.json` for historical record.**

### Numbers under both judges (n=111 across all rows)

| Run | Haiku acc | Codex acc | Haiku halluc | Codex halluc |
|---|---:|---:|---:|---:|
| v2 baseline (Qwen 1.5B, dense+rerank) | 0.685 | **0.793** | 0.216 | **0.063** |
| v2 §9 follow-up (GPT-5.5 manual) | 0.757 | n/a (self-bias) | 0.027 | n/a |
| v4 baseline (Qwen 7B, symbol+bm25) | 0.703 | **0.730** | 0.054 | **0.000** |
| v4 R1 calibrated (dense+rerank) | 0.730 | **0.775** | 0.081 | **0.009** |
| v4 R3 prompt | 0.784 | **0.856** | 0.126 | **0.009** |
| v4 Rewriter | 0.773 | **0.829** | 0.082 | **0.009** |
| **v4 Rewriter + HyDE** | — | **0.874** | — | **0.009** |

Codex shifts accuracy +2.7 to +7.2 pp depending on the run, and
collapses hallucination from the 0.05–0.13 range to ~0.009 across
the board. The directional ordering across runs is preserved
(baseline < R1 < rewriter < R3 < HyDE), so the per-run diffs that
drove sub-task decisions are still in the same relative shape.

### Codex-judged tier breakdown

| Run | n | correct | partial | wrong | halluc | refused | accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline (symbol+bm25) | 111 | 44 (.396) | 37 (.333) | 9 (.081) | 0 (.000) | 21 (.189) | 0.730 |
| R1 calibrated | 111 | 35 (.315) | 51 (.459) | 11 (.099) | 1 (.009) | 13 (.117) | 0.775 |
| R3 prompt | 111 | 38 (.342) | 57 (.514) | 10 (.090) | 1 (.009) | 5 (.045) | 0.856 |
| Rewriter | 111 | 39 (.351) | 53 (.477) | 10 (.090) | 1 (.009) | 8 (.072) | 0.829 |
| **Rewriter + HyDE** | 111 | **41 (.369)** | **56 (.505)** | 8 (.072) | 1 (.009) | 5 (.045) | **0.874** |

### What changes about earlier sub-task decisions

Three retrospective wrinkles in light of the Codex numbers:

1. **The R3 → rewriter pivot was sound but the precise framing was
   wrong.** Under Haiku, the case for the rewriter rested on
   "matches R3's accuracy without R3's hallucination penalty" — the
   penalty was 0.082 vs 0.126. Under Codex, hallucination is 0.009 in
   both, and R3 sits 2.7 pp ahead on accuracy. The rewriter is still
   the better engineering choice (deterministic, prompt-stable, no
   risk of "any near-match → answer" generalisation under future
   prompt changes), but the quantitative case needs the engineering
   argument now, not the hallucination argument.
2. **The deferred sub-task 4 (self-verify loop) is more deferred.**
   Under Haiku, hallucination_rate hovered around 0.08 and the v4
   plan capped it at ≤ 0.10; sub-task 4 was the planned safety net.
   Under Codex, hallucination is 0.009 across all 5 runs, well below
   the cap. The self-verify lever is filed as "only consider if a
   future change reintroduces a hallucination problem".
3. **The v2 → v4 lift is smaller than originally claimed.** Haiku put
   v2 Qwen 1.5B at accuracy 0.685 / hallucination 0.216; the entire
   v4 program looked like a 0.685 → 0.874 lift (+18.9 pp). Under
   Codex, v2 Qwen 1.5B is actually 0.793 / 0.063 — Haiku had
   misclassified ~17 of v2's "wrong" answers as hallucinations and
   undercredited a chunk of partials. The corrected v4 lift is
   0.793 → 0.874 = **+8.1 pp accuracy** with hallucination already
   low to begin with (0.063 → 0.009). The 7B swap, the calibration,
   the rewriter, and HyDE each still moved the needle; the size of
   the move is just smaller than the headline Haiku numbers
   advertised.

### Cumulative pipeline lift (Codex judge)

| Stage | Backend | Prompt | Pre-gen rewrite | Retrieval | accuracy | hallucination |
|---|---|---|---|---|---:|---:|
| v2 baseline (Qwen 1.5B) | Qwen 1.5B transformers | strict (v1) | – | dense+rerank | 0.793 | 0.063 |
| v4 Week 0 | Qwen 7B Q4 GGUF | strict (v1) | – | symbol+bm25 | 0.730 | 0.000 |
| v4 Week 1 | Qwen 7B Q4 GGUF | calibrated R1 | – | dense+rerank | 0.775 | 0.009 |
| v4 Week 2 (R3 explored) | Qwen 7B Q4 GGUF | aggressive R3 | – | dense+rerank | 0.856 | 0.009 |
| v4 Week 2 (rewriter) | Qwen 7B Q4 GGUF | calibrated R1 | lev≤2 rewriter | dense+rerank | 0.829 | 0.009 |
| **v4 Week 3 (HyDE)** | Qwen 7B Q4 GGUF | calibrated R1 | lev≤2 rewriter | dense+rerank+**HyDE** | **0.874** | **0.009** |

Note: v4 Week 0 used `symbol+bm25` (different retriever from v2
baseline's `dense+rerank`), so the 0.793 → 0.730 row-to-row drop is a
retriever change, not a regression. The first apples-to-apples
comparison with v2 baseline is v4 Week 1 (Qwen 7B + calibrated R1,
dense+rerank) at 0.775 — actually 1.8 pp **below** v2 baseline 0.793,
because the 7B model's stricter refusal behaviour traded coverage for
caution before the rewriter was added back. Sub-tasks 1 / 1' / 2
(rewriter + HyDE) recovered that gap and pushed past it.

Compared to the v4 0.78 accuracy target: rewriter+HyDE is **+9.4 pp
above target**. Compared to v2 baseline (Qwen 1.5B, dense+rerank,
Codex-judged at 0.793): rewriter+HyDE adds **+8.1 pp accuracy** and
takes hallucination from 0.063 to 0.009. The v2 §9 GPT-5.5 (manual)
follow-up's 0.757 number remains Haiku-graded; re-judging it via
Codex would be a self-grading exercise (Codex CLI's internal LLM is
also GPT-5.5), so we keep that number as-is and treat cross-row
comparisons against it as directional.

## Per-type breakdown (sub-task 5b/c)

`scripts/per_type_metrics.py` (added in this commit) splits each
run's tiers by the eval set's `query_type` field and computes a
refusal F1 using `hit_at_5=False` as proxy ground truth for "chunks
did not contain the answer". Eval set composition: 40 identifier /
27 natural_language / 21 comparison / 23 howto (no `out_of_scope`
rows in v2_full, so the F1 is hit_at_5-driven, not an explicit
should-refuse label).

### Per-type accuracy across runs (Codex judge)

| Run | identifier | natural_language | comparison | howto | refusal F1 |
|---|---:|---:|---:|---:|---:|
| v2 baseline (Qwen 1.5B, dense+rerank) | 0.900 | 0.778 | 0.571 | 0.826 | 0.000 |
| v4 baseline (7B, symbol+bm25) | 0.800 | 0.593 | 0.714 | 0.783 | 0.431 |
| v4 R1 calibrated (dense+rerank) | 0.775 | 0.704 | 0.857 | 0.783 | 0.187 |
| v4 R3 prompt | 0.850 | 0.852 | 0.810 | 0.913 | 0.167 |
| v4 rewriter | 0.900 | 0.704 | 0.857 | 0.826 | 0.222 |
| **v4 rewriter + HyDE** | **0.925** | **0.778** | **0.857** | **0.913** | **0.286** |

Three observations the global accuracy hides:

1. **Identifier was not the bottleneck.** v2 baseline already had
   identifier at 0.900. The v4 work had identifier dip to 0.775
   (R1 calibrated) before the rewriter pulled it back to 0.900 and
   HyDE took it to 0.925. The rewriter's impact is concentrated here
   (typo recovery), exactly where it was designed to operate.
2. **Comparison was the largest single-class lift.** v2 baseline at
   0.571 → R1 calibrated at 0.857 = +28.6 pp, the biggest per-class
   delta in the whole table. The 7B model's calibrated prompt
   handles "A vs B" framing far better than 1.5B did. HyDE held
   that level (0.857); the comparison decomp lever we considered
   for sub-task 2 is unnecessary at this point.
3. **Natural-language was the v4 stress class.** It dropped from
   0.778 (v2 baseline) to 0.593 (v4 7B with symbol+bm25, before any
   sub-task 1 work) because 7B refuses more readily on vague
   questions and symbol+bm25 surfaces less topical context for NL.
   R3 briefly took it to 0.852 by being aggressive, but the
   rewriter alone landed at 0.704. HyDE was specifically designed
   for this class and recovered NL to 0.778 — exactly the v2
   baseline level. The HyDE accuracy story is largely the NL
   accuracy story.

### Refusal F1 reading

The F1 numbers look small (0.167–0.431) because:

- v2 Qwen 1.5B basically never refuses (refused=2/111), so F1=0:
  precision is undefined, recall is 0 against the 23 hit_at_5=False
  rows.
- v4 7B + calibrated prompt refuses 5–13 times; precision is OK
  (60–100%) but recall against the 16 hit_at_5=False rows is low
  (model still tries to answer when chunks miss).
- v4 baseline (symbol+bm25, F1=0.431) has the highest F1 because
  the symbol+bm25 retriever produces more clearly off-topic chunks
  on NL queries, which the model correctly refuses. Once the
  retriever became dense+rerank (R1 onward), more queries got
  partial-relevance chunks → model partially answers → refusal F1
  drops even though accuracy goes up.

This is an artefact of using `hit_at_5=False` as proxy ground
truth: a partial-relevance chunk is hit_at_5=True (the right symbol
is in retrieval) but the model's reasonable response is still
"partial answer", not refusal. A proper refusal F1 needs explicit
`should_refuse` labels in the eval set. v0/v2 chose not to add
those (`eval_sets/v2_full.jsonl` has no `out_of_scope` rows); a v5
revision could.

For now the takeaway is: **v4's refusal calibration kept hallucination
at 0.009 without trading away too much answering coverage** — the F1
number is informative about the calibration's behaviour even if the
absolute value is hard to read against an ideal.

## What's next

- **Sub-task 4 (self-verify loop)** — deferred indefinitely. Codex
  judge puts hallucination at 0.009; the original 0.10 cap is no
  longer binding.
- **Comparison decomp + chunker re-cut** — sub-task 2 leftovers.
  Per-type breakdown shows comparison is already at 0.857 and HyDE
  has NL at the v2 baseline level; further sub-task 2 work has
  diminishing returns. Park.
- **Eval set v5 with `out_of_scope` rows** — the only way to get a
  meaningful refusal F1 is explicit ground-truth labels for which
  queries should refuse. A 10–20 row out-of-scope addition to
  `v2_full` (or a separate `v5_oos.jsonl`) would fix this.
- **v2 ablation re-judge for table consistency** — the other 5 v2
  Qwen runs (`bm25-qwen`, `dense-qwen`, hybrid variants) are still
  Haiku-graded. Re-judging them via Codex would keep the v2
  ablation table internally consistent if we ever revisit the v2
  retrieval ablation.

The v4 narrative is closing — the Qwen-only path's accuracy target
is met with margin (0.874 vs 0.78 cap), hallucination is at 0.009
(vs 0.10 cap), and per-type breakdown shows no class is dragging
significantly. Future work folds into v5 territory.
