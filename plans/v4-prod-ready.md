# v4 — Production-Track Accuracy

**Parent:** [`../PLAN.md`](../PLAN.md) (v4 chapter to be added).

**Prerequisites:** v2 complete, including the §9 cross-generator follow-up
that established the generator-side hallucination ceiling.

**Estimated effort:** 4-6 weeks.

**Goal:** Lift answer accuracy on Python documentation queries from the
v2 baseline (`accuracy = 0.685` / `hallucination_rate = 0.216` on the
local Qwen 1.5B + `dense+rerank` stack) to **`accuracy ≥ 0.90` and
`hallucination_rate ≤ 0.03`** on a scaled eval set, while preserving
v0-v2's reproducibility, framework-light design, and local-first
operation.

---

## Non-goals

- Cloud deployment, multi-user / SaaS hosting, authentication.
- Replacing v0 / v1 / v2 implementations — v4 is layered on top, not a
  rewrite.
- Introducing managed RAG frameworks (LangChain, LlamaIndex, hosted
  vector databases, etc.).
- Discarding v0-v2 reproducibility primitives — sha-pinned corpus,
  manifest snapshots per run, and immutable historical run dirs all
  remain.

---

## Quantitative definition

The accuracy metric used throughout v4:

```
accuracy = (correct + partial) / n_total
```

Tier semantics match the v2 §6 four-tier rubric:

| Tier | Prose | Citation | Counts toward `accuracy`? |
|---|---|---|---|
| `correct` | accurate | precise (matches expected symbols / URL) | ✅ |
| `partial` | accurate | retrieved but imprecise | ✅ |
| `wrong` | inaccurate | (any) | ❌ |
| `hallucination` | references content not in retrieved chunks | — | ❌ |
| `refused` | model emitted `[INSUFFICIENT-CONTEXT]` | — | ❌ on in-scope queries |

The shift from `correct_rate` to `(correct + partial) / n` reflects v2
§6 calibration evidence: the LLM-as-judge systematically classifies
prose-correct, citation-imprecise answers as `partial`. From a
user-perceived correctness viewpoint, both tiers represent a usable
answer.

**Targets:**

- `accuracy ≥ 0.90` on the v4 eval set; stretch `≥ 0.95`
- `hallucination_rate ≤ 0.03`
- `wrong_rate ≤ 0.05`
- `refused_rate ≤ 0.05` on in-scope queries
- `latency_p50 ≤ 3 s` for interactive queries
- 95% confidence intervals reported on every aggregate accuracy claim

**Eval set requirements:**

- `n ≥ 300` in-scope queries
- Balanced across the four query types (identifier / natural_language /
  howto / comparison)
- 20-30 explicitly out-of-scope queries for refusal calibration
- Reproducibility: same `docs_version` + `docs_sha_short` keys as the
  rest of the corpus

---

## Baseline data (from v2 §9)

The v2 §9 follow-up replaced the local Qwen 1.5B generator with a
capacity-class generator on the same `dense+rerank` retrieval and the
same Haiku 4.5 judge (prompt hash `65fa23b9`). Results on `n=111`:

- `hallucination_rate`: **0.216 → 0.027** (already below the v4
  target of 0.03)
- `correct_rate`: 0.063 → 0.370
- `accuracy`: 0.685 → 0.757
- `refused_rate`: 0.018 → 0.189

Refusal triage on the 21 refused rows split as:

- **13 retrieval-miss rows** (`hit_at_5 = False`) — refusal was correct.
- **8 retrieval-hit rows** (`hit_at_5 = True`) — over-cautious refusals;
  the expected chunk was in the prompt but the generator emitted
  `[INSUFFICIENT-CONTEXT]`.

This breakdown drives the v4 sub-task ordering: the largest accuracy
lever is no longer raw generator capacity but **refusal calibration**
plus **retrieval-miss recovery**.

---

## Sub-tasks

### 1. Refusal calibration (P0)

Files:

- Modify `src/python_doc_assistant/prompts/grounded.py`
- Optionally new `src/python_doc_assistant/generation/refusal_calibration.py`

Mechanism:

- Soften the `[INSUFFICIENT-CONTEXT]` trigger so it requires both
  (a) chunks are unrelated, AND (b) no partial inference is possible
  from any chunk
- Add 3-5 in-prompt calibration examples illustrating
  "partially-grounded answer with cited `[N]` is preferred over
  `[INSUFFICIENT-CONTEXT]`"
- Optionally implement a two-stage retry: if the generator emits
  `[INSUFFICIENT-CONTEXT]` and `hit_at_5 = True`, re-prompt with a
  stronger directive to attempt an answer

Expected lift: **+5-7 pp accuracy** by closing 6 of the 8 false
refusals observed in v2 §9.

Effort: 1 day.

### 2. Retrieval-miss recovery (P1)

Three components addressing the 13 retrieval-miss rows + the
`Recall@5 = 0.838` ceiling:

#### 2a. Query rewrite (HyDE)

File: new `src/python_doc_assistant/retrieval/query_rewrite.py`.

For NL / howto queries:

1. LLM call: "Generate a hypothetical Python documentation excerpt
   that would answer this question" → hypothetical answer text
2. Embed the hypothetical answer (not the original query) for dense
   retrieval

Identifier queries bypass this rewrite (lexical match dominates).

Expected lift: NL/howto `Recall@5` +3-5 pp.

Effort: 1 day.

#### 2b. Comparison query decomposition

File: modify `src/python_doc_assistant/retrieval/router.py`.

For `match_policy=all` comparison queries (e.g. `Path vs os.path`):

1. LLM splits the query into two sub-queries
2. Each sub-query independently retrieves top-3
3. Merge + dedupe → final retrieved set

Expected lift: comparison `Recall@5` +5-10 pp.

Effort: 0.5 day.

#### 2c. Chunker re-cut

File: modify `src/python_doc_assistant/ingest/chunker.py`.

Two rule changes:

- When a `section_chunk` wholly contains one or more `symbol_chunk`s,
  strip the symbol blocks from the section's text rather than carry
  them in both chunks
- Cap section chunk length at 1500 characters; split on `<h2>` / `<h3>`
  boundaries when over

Re-ingest under a new `docs_sha_short` to preserve v0-v2 reproducibility.

Expected lift: identifier `cite_match_expected` +5-10 pp.

Effort: 1 day plus updated chunker fixture tests.

### 3. Generator backend integration (P1)

File: new `src/python_doc_assistant/generation/claude_backend.py`;
modifications to `src/python_doc_assistant/cli.py`.

`ClaudeGenerator(Generator)` implementation:

- Wraps the Anthropic Messages API (`client.messages.create()`)
- Default `model_id = "claude-sonnet-4-6"`; alternative
  `claude-haiku-4-5` for fast iteration
- Reuses the same grounded prompt template as `QwenGenerator`
- Reuses the existing `parse_response` for citation extraction
- Streaming output via `stream=True`

CLI surface:

```
pdr eval --set ... --backend claude --model claude-sonnet-4-6
pdr eval --set ... --backend qwen   --model Qwen/Qwen2.5-1.5B-Instruct  # unchanged
```

Validation: a fresh `pdr eval` run must reproduce the v2 §9 baseline
metrics within statistical noise, then improve under sub-tasks 1+2.

Effort: 1 day, including unit tests with a mocked client.

### 4. Self-verification loop (P2)

File: new `src/python_doc_assistant/generation/verify.py`.

Three-stage flow per query:

1. Generate the initial answer with the grounded prompt
2. Verify call: "For each claim in this answer, identify whether it is
   supported by the retrieved chunks. List unsupported claims."
3. Revise call (only when verify identifies unsupported claims):
   "Rewrite the answer, removing or rephrasing the unsupported claims."

CLI flag `--verify / --no-verify` (default on).

Expected lift: +3-5 pp accuracy by catching residual hallucination and
prose-wrong cases.

Cost: 2-3× API calls per query; remains sub-cent in absolute terms.

Effort: 1 day.

### 5. Eval set + diagnostic metrics (P1)

#### 5a. `eval_sets/v4_prod.jsonl`

Starting from `v2_full.jsonl` (`n=111`):

- 100 hand-written queries covering uncovered stdlib areas
  (`asyncio`, `typing` advanced features, `contextlib`,
  `functools.singledispatch`, `weakref`, etc.)
- 50 LLM-aided generated queries with manual review before inclusion
- 30 out-of-scope queries (cross-language / unrelated-domain)

Total: ≥ 300 in-scope + 30 OOS.

Effort: 2 days.

#### 5b. Per-query-type metrics

Files: modify `src/python_doc_assistant/evaluation/retrieval_metrics.py`,
`evaluation/run_writer.py`, `evaluation/judge.py`.

`results.json` gains a `per_type` block:

```json
"per_type": {
  "identifier":       { "n": 90, "recall@5": 0.94, "accuracy": 0.85, "hallucination_rate": 0.04 },
  "natural_language": { "n": 80, "recall@5": 0.78, "accuracy": 0.81, "hallucination_rate": 0.07 },
  "howto":            { "n": 70, "recall@5": 0.65, "accuracy": 0.69, "hallucination_rate": 0.12 },
  "comparison":       { "n": 60, "recall@5": 0.71, "accuracy": 0.75, "hallucination_rate": 0.09 }
}
```

This addresses the existing aggregate-only reporting that obscures
per-type strengths and weaknesses.

Effort: 0.5 day.

#### 5c. Refusal precision / recall metric

File: new `src/python_doc_assistant/evaluation/refusal_metrics.py`.

Definitions:

- `refused_recall_on_oos = refused_oos / total_oos`
- `refused_precision = (refused ∩ oos) / total_refused`
- `f1` of the two

`results.json` gains a `refusal` block:

```json
"refusal": {
  "oos_n": 30,
  "in_scope_n": 281,
  "refused_recall_on_oos": 0.93,
  "refused_precision": 0.78,
  "f1": 0.85
}
```

Effort: 0.5 day.

#### 5d. Failure-mode triage

File: new `scripts/triage_failures.py`.

For each `wrong` or `hallucination` row in `judge_scores.jsonl`:

- Categorize: `retrieval_miss` (`hit_at_5 = False`) /
  `cite_hallucination` (claims chunks not present) /
  `prose_wrong` (cites correctly but answer wrong) /
  `format_error` (parse_response failure)
- Output a markdown report: counts per category plus three sample
  queries per category

Effort: 1 day.

### 6. Interactive `pdr ask` subcommand (P0)

File: modify `src/python_doc_assistant/cli.py`.

Single-query entry point complementing the existing `pdr eval` batch
pipeline:

```
pdr ask "how to read a file in python"
pdr ask --backend claude --debug "Path vs os.path"
pdr ask --show-retrieved "subprocess.run"
```

Implementation:

- Reuses `build_retrieve_fn` and the `Generator` interface
- Streaming output (`--stream`, default on)
- Inline citation rendering: `[1]` / `[2]` are highlighted; a trailing
  `Sources:` block lists the canonical docs URLs
- `--debug` prints retrieval routing decisions, chunk scores, ranks
- `--show-retrieved` prints the top-K chunks before the answer
- Pre-flight checks for missing API key, missing index, or missing
  corpus → friendly error with the exact remediation command

Effort: 0.5 day.

### 7. Streaming + structured CLI output (P2)

Files: new `src/python_doc_assistant/cli/render.py`; modifications to
`pdr ask`.

`pyproject.toml` gains:

```toml
[project.optional-dependencies]
cli = ["rich"]
```

Components:

- `render_answer(text, citations, retrieved)` — Markdown rendering with
  highlighted citations and a URL footer
- Streaming token printer with cursor management
- Error formatter (red box plus actionable command)
- `--debug` retrieval table: chunk_id, score, rank, canonical URL

Effort: 1 day.

### 8. (Optional) Agentic / tool-use generator (P3)

Activated only if accuracy after sub-tasks 1-7 is below 88%.

File: new `src/python_doc_assistant/generation/agent_backend.py`.

Tools exposed to the generator via the Anthropic Messages tool-use
protocol:

1. `retrieve(query, k=5)` — secondary retrieval call from the corpus
2. `verify_code(snippet)` — execute Python in an isolated subprocess
   (bounded stdlib, no network, time + memory caps)
3. `ask_clarification(question)` — interactive turn (CLI only)

Expected lift: +3-5 pp on the residual hardest queries.

Effort: 1 week, including sandbox design and security review.

### 9. (Optional) HTTP API + web UI

File scope: new `src/python_doc_assistant/server/` (FastAPI), new
`frontend/` (React + TypeScript + Vite), new `pdr serve` CLI
subcommand.

API surface:

- `POST /ask {query, backend}` → server-sent-events stream of answer
  tokens followed by final citations
- `POST /search {query, k}` → JSON list of retrieved chunks
- `GET /runs/{tag}` → JSON snapshot of a run's `results.json`

Frontend depth tiers (calibrated to remaining effort budget):

- Tier A: single page (input + answer + citations)
- Tier B: + ablation visualization + per-query browser
- Tier C: + settings + saved query history

Effort: 1-2 weeks (decision on tier defers until sub-tasks 1-8
complete).

### 10. (Optional) MCP server

Activated in parallel with sub-task 9 to expose the same retrieval /
generation pipeline to MCP-compatible clients.

File: new `src/python_doc_assistant/mcp_server/` using the Python `mcp`
SDK.

Tools exposed:

- `search_python_docs(query, k)` → list of chunks
- `ask_python_docs(query, backend)` → answer + citations
- `get_eval_results(tag)` → results snapshot

Reuses sub-task 9's server-layer abstractions; only the transport
adapter (stdio / HTTP for MCP) differs.

Effort: 1-2 days.

---

## Roadmap

| Week | Deliverable | Expected accuracy |
|---|---|---|
| 1 | Sub-task 1 (refusal calibration) + sub-task 6 (`pdr ask`) | 78-85% |
| 2 | Sub-task 3 (Claude backend) + sub-task 4 (verify) + sub-task 7 (streaming / rich CLI) | 82-90% |
| 3 | Sub-task 2 (retrieval miss recovery: 2a/2b/2c) | 85-92% |
| 4 | Sub-task 5a/b/c/d (eval expansion + per-type + refusal + triage) | 88-93% with tighter CIs |
| 5-6 | (Conditional) sub-task 8 (agentic) if < 88%; otherwise narrative finalization + completion-criteria check | 92-96% / locked |
| 7+ | (Optional) sub-tasks 9 / 10 (HTTP API + web UI / MCP server) | UI-access milestone |

**Decision gates:**

- End of week 1: `accuracy < 0.78` → revisit prompt or retrieval before
  continuing; `0.78-0.88` → continue per plan; `> 0.88` → consider
  deferring sub-task 8
- End of week 4: `accuracy < 0.88` on `n ≥ 300` → sub-task 8 (agentic)
  is required to hit the 0.90 target
- End of week 6: `accuracy < 0.90` → narrative documents the partial
  achievement and proposes a v4.1 follow-up

---

## Risks / decision items

### R1. API cost

- Sonnet 4.6: ~$3/M input, ~$15/M output
- Per eval run (n=300, ~2k input + 500 output × 3 configs): ~$3-5
- Full week of iteration: ~$30-50
- Mitigation: use Haiku 4.5 for fast iteration; reserve Sonnet for
  validation runs

### R2. Judge methodology

- v2 §6 used Haiku 4.5 with prompt hash `65fa23b9` (Cohen's kappa
  0.645, substantial agreement)
- v4 reuses this judge for cross-config delta comparisons against v2
  baselines; absolute numbers shift only via the prompt-hash change
- Same-family bias risk if both generator and judge are Claude models
  → optionally evaluate a 30-50 row sample with a second-family judge
  (GPT, Gemini, etc.) to verify cross-family agreement

### R3. Eval set size vs confidence interval

- `n=300` + `accuracy = 0.92` → 95% CI ≈ ±3 pp
- A hard `≥ 0.95` claim requires `n ≥ 500-1000`
- Mitigation: report `accuracy 0.92 ± 0.03 on n=300`; expand to a v4.1
  follow-up if a tighter claim is needed

### R4. Corpus / chunker compatibility

- Sub-task 2c changes `chunks.jsonl` content under the same
  `docs_version`. Re-ingest produces a new `docs_sha_short` directory;
  v0-v2 narratives and run dirs remain unchanged because they pin the
  old `docs_sha_short`
- v4 narrative explicitly records the new `docs_sha_short` so v0-v2 vs
  v4 comparisons cite the corpus boundary

---

## Completion criteria

**Code:**

- [ ] `claude_backend.py` + unit tests with mocked client
- [ ] `query_rewrite.py` + unit tests
- [ ] `chunker.py` re-cut rules + new fixture tests
- [ ] `verify.py` + unit tests
- [ ] `refusal_metrics.py` + unit tests
- [ ] `triage_failures.py` script
- [ ] `pdr ask` subcommand + tests including pre-flight error paths
- [ ] CLI flags wired: `--backend {qwen, claude}`, `--verify / --no-verify`,
      `--stream`, `--debug`, `--show-retrieved`
- [ ] `per_type` block in `results.json` + tests
- [ ] `refusal` block in `results.json` + tests
- [ ] Streaming + rich CLI rendering + integration tests

**Data and narrative:**

- [ ] `eval_sets/v4_prod.jsonl` with `n ≥ 300` plus 30 OOS
- [ ] At least three v4 generation configs run end-to-end
      (e.g. Haiku, Sonnet, Sonnet+verify)
- [ ] `experiments/v4-prod-track.md` narrative with reproducibility
      table, results table, deltas vs v2 baseline, failure-mode triage
- [ ] Top-level `README.md` updated with v4 results

**Numerical thresholds:**

- [ ] `accuracy ≥ 0.90` on `eval_sets/v4_prod.jsonl`
- [ ] `hallucination_rate ≤ 0.03`
- [ ] `latency_p50 ≤ 3 s`
- [ ] 95% CIs reported on every aggregate accuracy claim

**Optional:**

- [ ] HTTP API + web UI (Tier A or higher)
- [ ] MCP server tested with at least one MCP-compatible client

---

## Connection to v0 / v1 / v2 / v3

| File | Role |
|---|---|
| [`v0-retrieval-eval.md`](v0-retrieval-eval.md) | Retrieval baseline; chunker + BM25 + symbol index |
| [`v1-qwen-generator.md`](v1-qwen-generator.md) | First grounded generator; baseline accuracy 0.685 |
| [`v2-ablation.md`](v2-ablation.md) | Retrieval ablation + LLM-as-judge calibration; cross-generator follow-up establishes the v4 generator-side ceiling |
| [`v3-tiny-llm.md`](v3-tiny-llm.md) | Research side track; not on the v4 critical path |
| **`v4-prod-ready.md`** (this document) | Production-track accuracy lift to `≥ 0.90` |
