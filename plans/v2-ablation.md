# v2 — Ablation & Optimization

**Parent doc:** [../PLAN.md](../PLAN.md) §7 v2

**Prerequisite:** v1 completed (baseline Qwen runs + full manual scoring of current eval queries in hand; expanding to 100–200 entries is part of this stage)

**Estimated duration:** about 1 week

**Core goal:** Add dense embedding + hybrid merge + cross-encoder rerank, and produce a **complete ablation table** that uses data to answer how much each optimization layer contributes.

---

## Prerequisites

- [ ] Expand extras: `uv sync ... --extra embedding --extra rerank`
- [ ] `bge-small-en-v1.5` smoke test: embed a sentence, confirm output is 384-dim
- [ ] `bge-reranker-base` smoke test: scores a `(query, doc)` pair

## Subtasks

### 1. Dense embedding index

File: `src/python_doc_assistant/indexes/dense_index.py`

- Use `sentence-transformers` to load `BAAI/bge-small-en-v1.5`
- Compute an embedding for each chunk and store as a numpy array (`data/indexes/<DOCS_VERSION>/<sha_short>/dense.npy`, sharing the same `<sha_short>` layering as v0's chunks / bm25)
- Retrieval: cosine similarity between the query embedding and all chunk embeddings
- At a scale of a few thousand chunks, numpy is sufficient — no need for FAISS

### 2. Hybrid merge

File: `src/python_doc_assistant/retrieval/hybrid.py`

Implement both merge methods and compare them in the ablation:

- **RRF**: `score = Σ 1 / (k + rank_i)` across multiple paths, typically `k=60`
- **Linear weighting**: `score = α * bm25_norm + (1-α) * dense_norm` (requires score normalization)

### 3. Cross-encoder rerank

File: `src/python_doc_assistant/retrieval/rerank.py`

- Top-20 candidates → cross-encoder rerank → final Top-5
- `BAAI/bge-reranker-base` scores `(query, chunk)` pairs
- Mind latency: cross-encoder is much slower than embedding, so batch process

### 4. Expand the eval set to 100–200 entries

File: `eval_sets/v2_full_200.jsonl`

- Built on top of v0's 30–50 entries
- Focus on adding: comparison-type, howto-type, long queries, queries with typos
- Keep the same multi-answer schema

### 5. Ablation matrix (core deliverable)

**Ablation constants (must be fixed across configurations, otherwise answer quality / hallucination rate cannot be causally attributed):**

- Generator: `Qwen2.5-1.5B-Instruct` (the default model selected in v1)
- Grounded prompt: v1 final version (prompt file + hash written into `results.json`)
- Decoding parameters: `temperature / top_p / top_k / max_new_tokens` fixed
- Retrieved K: the Top-K fed into the generator is fixed (e.g. fix K=5)
- Eval set: fixed at `v2_full_200.jsonl`
- Seed: all randomness (dense tie-break / sampling) is seeded

Run the following configurations, each producing an independent timestamped run directory `experiments/runs/<YYYY-MM-DDTHH-MM-SS>-v2-<config>/` (naming rules in PLAN.md §8 + plans/v0 §9; the constants above are written into the `ablation_constants` field of each `results.json`):

| Configuration              | Metrics collected                                      |
| -------------------------- | ------------------------------------------------------ |
| BM25-only (v0 baseline)    | Recall@5, MRR, answer quality, hallucination rate      |
| Dense-only                 | Same as above                                          |
| Hybrid-RRF                 | Same as above                                          |
| Hybrid-linear (α sweep)    | Find the best α                                        |
| Hybrid-RRF + rerank        | Same as above                                          |
| Hybrid-best + rerank       | Same as above (final recommended configuration)        |

### 6. LLM-as-judge

File: `src/python_doc_assistant/evaluation/answer_metrics.py` (extended)

**Judge backend candidates:**

- Claude API (e.g. Sonnet) — stable quality, factor in cost
- Qwen API (`qwen-max` etc., **closed-source API**) — low cost; note this is an API, not a local model
- Local `Qwen2.5-72B-Instruct` (only if hardware allows) — best reproducibility but high hardware barrier

**Reproducibility info that must be recorded for every judge run** (written into the `judge` field of `results.json`):

- `judge_model` (full model ID / API version string)
- `judge_prompt_hash` (SHA1 of the prompt template)
- `temperature` / `top_p` / `max_tokens`
- `raw_judge_output` for every sample (**raw unparsed output**, not the parsed tier)
- `timestamp`

**Agreement rate check:** First run 20 samples and compute the agreement rate against v1's manual scoring; expect > 80%. If it does not pass, switch judge or modify the prompt; only run the full ablation after it passes.

### 7. Experiment narrative document

File: `experiments/v2-ablation.md`

Must directly answer:

- How much does rerank contribute to Recall@5 / answer quality?
- On which queries does Dense beat BM25? And vice versa?
- The optimal value from the α sweep + its impact on hallucination rate
- What is the final recommended configuration? Why?

## Completion criteria

- [ ] All 6 rows of numbers in the ablation table are filled in
- [ ] LLM-as-judge agreement rate with humans > 80%
- [ ] `experiments/v2-ablation.md` can directly answer both "how much does rerank contribute" and "Dense vs BM25 scenario differences"
- [ ] Each configuration has its own independent timestamped run directory (`experiments/runs/<YYYY-MM-DDTHH-MM-SS>-v2-<config>/`) containing `results.json` + `per_query.jsonl`

## Decision points during execution (I will ask you)

- α sweep range (0.1–0.9 with step size 0.1? Or finer?)
- Whether rerank keeps raw scores or fully replaces them
- Which to use for LLM-as-judge: Claude API / Qwen API (closed-source) / local `Qwen2.5-72B-Instruct` (cost vs reproducibility vs hardware barrier)
- The specific queries for expanding the eval set (I will pick a batch for you to review)
