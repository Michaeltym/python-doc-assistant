# v3 (optional) — In-house Tiny LLM Comparison Backend

**Parent doc:** [../PLAN.md](../PLAN.md) §7 v3

**Status:** ✅ MVP closed (commits `b839775` … `dc51663`). All four MVP completion
criteria met; experiment narrative committed at
`experiments/v3-tinydocs-vs-qwen.md`.

**Follow-up:** [`v3.1-fineweb-sft.md`](v3.1-fineweb-sft.md) re-opens v3 to address
the quality gap surfaced by §8 narrative (output is "doc-shaped noise"). v3.1
swaps the docs-only pretrain for a FineWeb-Edu mix and scales SFT distillation
from 122 to 1000+ samples. **Optional, learning-oriented**; does not block the
main project just like v3 itself.

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

**MVP scope: Python docs only.** Build a `corpus.jsonl` from the v0
chunks (`data/chunks/<docs_version>/<sha>/chunks.jsonl`), shuffled
with a fixed seed for reproducibility, plus a sibling `manifest.json`
recording the input path, seed, line count, total bytes, and build
timestamp.

Adding general-purpose text (e.g. a slice of `FineWeb-Edu` at an
80% general / 20% docs ratio) was the original plan, but is deferred:
at the v3 parameter budget (~50–100M) and on a single-machine training
target, the broader vocabulary coverage from a general-text mix does
not materially change the "pipeline-through" learning objective. A
docs-only corpus keeps §3 implementation small (~80 lines) and lets
v3 §4–§7 advance without an extra dataset dependency. Mixing general
text remains a sensible follow-up if v3 quality work is later
revisited.

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

---

## Why v3.1 was opened post-MVP

The v3 §8 narrative documented two undeniable shortcomings of the closed
MVP that the MVP scope explicitly accepted but that read poorly when seen
side-by-side with Qwen output:

1. **Output is "doc-shaped noise."** The 10k-step `run-10k-b8` ckpt produces
   text like `"- - name - file = foo to create a path to the archive.
   Changed in version 3.6"` — recognizable docs vocabulary stitched together
   without grammar. Qwen at the same task produces grammatical instructional
   prose with `[N]` citations. The gap is too large to attribute to scale
   alone; the MVP base never learned English structure because the corpus
   was 2.6 M tokens of one narrow domain.
2. **SFT alone cannot rescue the docs-only base.** A 122-sample SFT
   experiment in a post-MVP feasibility exploration produced 26 %
   qualitative wins, 18 % mode collapses, and 56 % no-change outcomes — net
   flat. The mechanism worked (one query produced a verbatim copy of the
   Qwen exemplar's literal), but the base could not produce coherent
   English answer prose to be SFT-tuned in the first place.

A 10k-step probe with a FineWeb-Edu mix at matched compute (same model size,
same step count) showed grammatical English emerged
(`"the city was in a very small area. The city was built in the early 18th
century..."`) where the docs-only baseline produced disconnected fragments.
That probe motivates v3.1.

v3.1 is **strictly optional**, like v3 itself — not a v4 prereq, not a
production deliverable. If v3.1 ships well, the v3 narrative gains a
compelling "before / after" comparison; if it stalls, v3 stays closed at
the MVP bar and the project moves on. See [`v3.1-fineweb-sft.md`](v3.1-fineweb-sft.md)
for the full plan.
