# v3 (optional) — In-house Tiny LLM Comparison Backend

**Parent doc:** [../PLAN.md](../PLAN.md) §7 v3

**Prerequisite:** v2 is complete; however, this is a **learning-oriented side track** and does not block the main project.

**Estimated duration:** +2 weeks

**Core goal:** Write a modern decoder-only LLM from scratch, wire it into the same RAG pipeline, and compare it side by side with Qwen as a Generator backend. **Does not aim to surpass Qwen in quality**; the purpose is to understand the architecture and walk through the full training-inference-evaluation pipeline.

---

## Non-goals (emphasized)

- Not a replacement for Qwen
- Does not chase generation quality
- Pretraining and SFT are **stretch goals**, not MVP pass criteria (see completion criteria below)

## Subtasks

### 1. Model architecture (hand-written, do not call `transformers.LlamaModel`)

File: `src/python_doc_assistant/generation/tinydocs/model.py`

Modern decoder-only:

- **RoPE** (rotary position embedding) — position encoding
- **RMSNorm** — normalization layer
- **SwiGLU FFN** — feedforward (FFN) activation
- **Weight tying** — share weights between the embedding layer and the output head (save parameters)
- **KV cache** — cache key/value at inference time to speed up autoregression
- Attention: start with pure PyTorch, do not bring in Flash Attention

Parameter count: roughly 50M–100M (depends on what hardware can fit)

### 2. Tokenizer

Two options:

- **Option A**: directly reuse the `Qwen2.5` tokenizer (simplest, aligns with the input distribution)
- **Option B**: train a BPE from scratch (high learning value but significant workload)

**MVP uses Option A; Option B goes into stretch goals**, only considered if there is time after the MVP is done.

### 3. Pretraining data mix

File: `data/pretrain/` (`.gitignore`)

- General-purpose text: a small slice of `FineWeb-Edu`
- Python documentation: chunks produced by v0 ingest
- Ratio: 80% general / 20% docs
- Slice with a fixed seed to keep it reproducible

### 4. Pretraining

File: `src/python_doc_assistant/generation/tinydocs/train.py`

- Optimizer: `AdamW`
- Learning rate: cosine with warmup
- Mixed precision: **smoke-test before training** (MPS support for `bf16` varies with the PyTorch version; cannot assume it works out of the box). Fallback order: `bf16` → `fp16` → `fp32` (`fp32` is slow but most stable, used as the fallback)
- Checkpoint every 1000 steps
- Write the loss curve to tensorboard

**Hardware note:** local MPS works but is slow. May need to switch to cloud GPU (Modal / Lambda / RunPod, rent on demand). Calculate cost before starting.

### 5. SFT (Supervised Fine-Tuning)

File: `src/python_doc_assistant/generation/tinydocs/sft.py`

- Data format: `(query, retrieved_chunks) → answer`
- Data source: v1/v2 Qwen outputs + manual filtering (essentially distillation)
- Scale: starting from a few thousand entries

### 6. TinyDocsGenerator adapter

File: `src/python_doc_assistant/generation/tinydocs_backend.py`

- Implement the `Generator` ABC
- Load the in-house checkpoint
- Wire into the same grounded prompt
- Add a `--model tinydocs` toggle to the CLI

### 7. Comparison evaluation

Run on the same eval set (the 100–200 entries from v2):

- `Qwen2.5-1.5B-Instruct` (v1/v2 baseline)
- `TinyDocsGenerator`

Comparison metrics: Recall@5 (retrieval layer is unchanged, should be identical), answer quality, hallucination rate, latency.

**Judge methodology — Opus 4.7 manual copy-paste (v3+ onward):**

- v2 §6 used Anthropic API + Haiku 4.5 (prompt hash `65fa23b9`, kappa 0.645)
- From v3 onward, the judge moves to **Opus 4.7 via Claude Code manual copy-paste workflow**: per_query.jsonl is rendered into batches of 20-50 cases, pasted into Claude, the JSON reply is parsed back into `judge_scores.jsonl`. Implementation scripts (`scripts/judge_render_batch.py`, `scripts/judge_parse_batch.py`) are part of v3 deliverables.
- New `judge_prompt_hash` (Opus 4.7 prompt may simplify vs Haiku); v2 vs v3+ hallucination_rate **not directly comparable** (different judge model)
- v3 vs v4 internal deltas remain valid (consistent judge model within one cohort)

### 8. Experiment narrative document

File: `experiments/v3-tinydocs-vs-qwen.md`

Record honestly:

- Architecture decisions + why RoPE / RMSNorm / SwiGLU were chosen
- Training data + hyperparameters
- Quality gap versus Qwen (expected to be much worse, present honestly, no sugarcoating)
- What was learned + what would change next time

## Completion criteria (MVP)

Under the practical constraints of 2 weeks + local MPS, the MVP bar is set at **pipeline wired end-to-end** rather than training a good model:

- [ ] Model architecture runs, can overfit on a single small batch (proves forward/backward is correct)
- [ ] Reuse the `Qwen2.5` tokenizer (training the tokenizer from scratch goes into stretch goals)
- [ ] `TinyDocsGenerator` adapter is wired into the `Generator` ABC and can run once through the RAG pipeline (what gets generated does not matter, as long as the pipeline runs)
- [ ] `experiments/v3-tinydocs-vs-qwen.md` is finished writing, honestly recording how far the work got

## Stretch goals (tackle if time permits, **NOT** a pass criterion)

- [ ] Small-scale pretraining with a reasonable loss curve (descending + converging)
- [ ] SFT distill from Qwen, starting from a few thousand entries
- [ ] End-to-end quality comparison data versus Qwen (Recall@5 unchanged / answer quality / hallucination rate / latency)

**Generation quality is never used as a pass criterion.** If MVP passes, v3 passes; stretch goals are completed as far as time allows.

## Decision points to make during execution (I will ask you)

**MVP stage:**

- Choose parameter count: 50M / 100M / larger (depends on what MPS can fit)
- Keep tokenizer Option A (reuse Qwen) or take the risk of trying B (train BPE from scratch)

**Stretch goal stage (only decided if time remains after MVP):**

- Whether to switch to cloud GPU (Modal / Lambda / RunPod, calculate the budget)
- Pretraining data mix ratio (general : docs)
- SFT data distill source (Qwen vs Claude, compliance + quality tradeoff)
- When the 2 weeks are up, "stop" stretch goals; do not drag down the main project
