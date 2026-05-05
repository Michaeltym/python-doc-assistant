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
| 5d | Failure triage script | ⏳ next |
| 2 | Retrieval miss recovery (HyDE + comparison decomp + chunker re-cut) | ⏳ pending |
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

## What's next

- **Sub-task 5d (failure triage)** — automate the triage we did
  manually for sub-task 1; output failure categories from
  `judge_scores.jsonl` so future iterations can target the largest
  bucket.
- **Sub-task 2 (retrieval miss recovery)** — 10 of the remaining 13
  refused rows have `hit_at_5=False`. HyDE + comparison decomposition
  + chunker re-cut should recover several of those, lifting
  `accuracy` further toward the 0.78 target.
- **Sub-task 4 (self-verify loop)** — only worth the 2-3× compute if
  hallucination_rate becomes the dominant failure after sub-task 2.
  Currently 0.081 is already 3× lower than the v2 baseline 0.216, so
  this is not the priority lever right now.

The v4 narrative will be amended as those sub-tasks land.
