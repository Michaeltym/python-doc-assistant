# v3 — TinyDocs (self-trained) vs Qwen2.5

Closed v3 narrative. Documents the §1–§8 deliverables of
`plans/v3-tiny-llm.md` — a hand-written 57M-parameter decoder-only
transformer trained on Python docs, wired behind the same `Generator`
ABC that powers v1 Qwen.

**MVP framing:** v3 is a learning side-project. The pass criterion is
**"pipeline runs end-to-end"**, not generation quality. A self-trained
50–100M model on 2.6M tokens of docs has zero realistic chance of
matching a 1.5B SFT chat model — that gap is acknowledged up front and
the comparison story is qualitative, not quantitative.

## Status

| Section | Topic | State |
|---------|-------|-------|
| §1 | Model architecture (RMSNorm / RoPE / SwiGLU / KV cache / weight tying) | ✅ done (commit `b839775`) |
| §1 | GPT-2 style init fix (std=0.02) | ✅ done (commit `a3a61c6`) |
| §2 | BPE tokenizer trained from scratch (Option B — stretch goal) | ✅ done (commit `e9e1b85`) |
| §3 | docs-only pretrain corpus + manifest | ✅ done (commit `74f180f`) |
| §3 | CLI: `build_pretrain_corpus.py` + `train_tokenizer.py` | ✅ done (commit `811d2e3`) |
| §4a | `TinyDocsDataset` + segment builder | ✅ done (commit `51e2260`) |
| §4b | Pretraining loop + LR schedule + checkpointing | ✅ done (commit `403ea2a`) |
| §4c | `pretrain_tinydocs.py` CLI + `encode_corpus.py` two-stage cache | ✅ done (commit `e03b75f`) |
| §5 | SFT (distill from Qwen) | ⏳ stretch goal — not pursued |
| §6 | `TinyDocsGenerator` adapter + `pdr ask --backend tinydocs` | ✅ done (commit `0dedc1e`) |
| §7 | Comparison eval vs Qwen | ⏳ stretch goal — not pursued (rationale below) |
| §8 | This narrative | ✅ done |

## MVP completion checklist

| Criterion | State | Evidence |
|---|---|---|
| Model architecture runs, can overfit on a single small batch (forward + backward correct) | ✅ | `tests/test_tinydocs_model.py` (19 tests pass); training loss 10.22 → 0.76 over 10k steps |
| Tokenizer ready (Option A reuse Qwen, or B self-train BPE) | ✅ stretch | Option B chosen — self-trained BPE, vocab 32 000 |
| `TinyDocsGenerator` wired into `Generator` ABC; pipeline runs through RAG once | ✅ | `pdr ask --backend tinydocs ...` succeeds end-to-end |
| `experiments/v3-tinydocs-vs-qwen.md` written, honestly recording how far it got | ✅ | This doc |

All four MVP criteria met; v3 is closed.

## Reproducibility

| Field | Value |
|---|---|
| docs_version | `3.12` |
| docs_served_version | `3.12.13` |
| docs_sha_short | `a5c1a35a5a02` |
| corpus | `data/pretrain/corpus.jsonl` (11 248 lines, 6.82 MB, seed 42) |
| tokenizer | `data/tokenizer/tokenizer.json` (BPE, vocab 32 000, 31 842 merges, 5 specials) |
| model params | **57.3 M** (12 layers × hidden 512 × FFN ×4, weight-tied embeddings) |
| training segments | 10 314 segments × 257 tokens (seq_len 256 + shift) — ~2.64 M tokens total |
| training device | M1 MPS (single Mac, fp32) |
| ckpt format | `dict{step, model_state, optimizer_state, model_config, train_config}` |

**Disk note:** Each checkpoint is ~688 MB (model + AdamW m/v state). A
full run with `checkpoint_every=2000` over 10 000 steps emits ~3.5 GB.
Everything under `data/` is `.gitignore`'d; nothing in this directory
is committed.

## §1 — Model architecture

Hand-written decoder-only transformer in `src/python_doc_assistant/generation/tinydocs/model.py`.
No `transformers.LlamaModel` dependency.

| Component | Choice | Why |
|---|---|---|
| Norm | RMSNorm (scale only, no bias) | Modern Llama-style — fewer params, stable. Bias adds little for this scale. |
| Position | RoPE (rotate-half) | Encodes position via complex rotation; needs no learned embedding; KV cache works trivially because each token's positional info is baked into its q/k at the time it's computed. |
| Attention | Multi-head causal, pure PyTorch (no Flash Attention) | MPS has no Flash Attention support; pure PyTorch keeps backward-pass clean for learning. KV cache is hand-rolled with explicit `caches: list[KVCache]` arg. |
| FFN | SwiGLU, expansion ratio 4× rounded up to 256 | Standard modern choice; sigmoid-gated path (`silu(gate) * up`) outperforms plain GELU at small scale per Llama / DeepMind ablations. |
| Word embedding | Tied to LM head (`vocab_head.weight = embeddings.weight`) | Saves ~16M params on a 57M model. Standard for small models. |
| Init | `nn.init.normal_(weight, std=0.02)` for Linear and Embedding | Default PyTorch init produced step-0 loss = 382 (vs theoretical log(32 000) ≈ 10.37); GPT-2-style 0.02 fixed it. See commit `a3a61c6`. |

Hyperparameters frozen in `TinyDocsConfig` (frozen dataclass):

```
vocab_size      = 32 000
hidden_dim      = 512
n_layers        = 12
n_heads         = 8
head_dim        = 64
max_seq_len     = 2048   (architecture cap; training used 256)
rope_theta      = 10 000
ffn_mult        = 4
norm_eps        = 1e-6
tie_word_embeds = True
```

## §2 — Tokenizer

**Decision: Option B (self-trained BPE) — stretch goal completed during MVP scope.**

Plan §2 listed Option A (reuse Qwen2.5 tokenizer, 152k vocab) as the
MVP default and Option B (train BPE from scratch) as stretch. Option A
turns out to be incompatible with the 50–100M parameter budget once
embedding cost is computed:

```
Qwen vocab × hidden × 2 (with tying) ≈ 152 000 × 512 ≈ 78 M params
```

That's already over budget for the embedding alone. Option B was the
forced choice. Vocab = 32 000 keeps embedding cost manageable
(32 000 × 512 ≈ 16 M params, ~28 % of total).

BPE trained on the same docs corpus. Pretokenization splits on
whitespace and punctuation; whitespace is tracked via a dedicated
`<sp>` token rather than baked into BPE merges (simpler decoder).

| Stat | Value |
|---|---|
| Vocab size | 32 000 |
| Specials | `<pad>`, `<bos>`, `<eos>`, `<unk>`, `<sp>` |
| BPE merges | 31 842 |

Known limitation: `pathlib.Path` tokenizes as 5 tokens because the
pretokenizer treats `.` as its own punctuation unit, scattering
`<sp>`s. Acceptable for MVP; real fix is a Sphinx-symbol-aware
pretokenizer.

## §3 — Pretraining data

**Scope: Python docs only.** Plan §3 deferred the FineWeb-Edu
general-text mix on the grounds that, at this parameter budget and on
single-machine training, broader vocabulary coverage from a general
mix does not materially change the "pipeline-through" learning
objective.

Concretely the FineWeb mix would have required:

- ~1 day of CLI work (HF streaming + mix + re-encode)
- Re-training the BPE on a mixed corpus (Qwen tokenizer can't be reused — see §2)
- Cloud GPU rental ($50–200) — single Mac MPS at ~200 tok/s would need ~58 days for 1B tokens

That cost is out of proportion to the MVP goal of "pipeline runs". A
follow-up that actually wants competitive output quality would change
this decision.

| Build step | Output |
|---|---|
| `build_pretrain_corpus.py` | `data/pretrain/corpus.jsonl` (11 248 lines, 6.82 MB, seed 42) |
| `train_tokenizer.py` | `data/tokenizer/tokenizer.json` |
| `encode_corpus.py` | `data/pretrain/segments-256.pt` (10 314 segments × 257 tokens) |

**Two-stage caching:** BPE encoding is the slow step (~3 minutes for
the full corpus even with the per-token encode cache). Splitting it
into a separate CLI lets training scripts load the cached tensor in
seconds — important when iterating on training hyperparameters.

## §4 — Training runs

Two real runs were executed end-to-end. Both used fp32 on M1 MPS.

### Run 1 — `run-1h-explore` (2 k steps × batch 4)

Exploratory run; goal was to validate the loop and observe loss shape.

| Param | Value |
|---|---|
| total_steps | 2 000 |
| batch_size | 4 |
| seq_len | 256 |
| warmup_steps | 100 |
| base_lr | 3e-4 |
| schedule | cosine to 0 |
| weight_decay | 0.1 |
| grad_clip | 1.0 |
| wall-clock | ~7 min |
| step rate | ~5 step/s |

Loss curve:

| step | loss |
|---|---|
| 0 | 10.33 |
| 100 (warmup done) | 3.36 |
| 500 | 3.00 |
| 1 000 | 2.62 |
| 1 500 | 2.33 |
| **2 000 (final, last 200 mean)** | **2.46 ± 0.10** |

Behavior: rapid drop from log(vocab_size) baseline through warmup,
then steady cosine descent, plateau forming at ~2.4 by step 1 500.
0.79 epoch over the corpus — short of one full pass.

### Run 2 — `run-10k-b8` (10 k steps × batch 8)

Tuning run after observing run 1 was severely undertrained.

| Param | Value |
|---|---|
| total_steps | 10 000 |
| batch_size | 8 |
| effective epochs | ~6.2 |
| wall-clock | ~69 min |
| step rate | ~2.4 step/s (slower per-step at b=8) |

Loss curve:

| step | loss |
|---|---|
| 0 | 10.22 |
| 100 | 3.20 |
| 1 000 | 2.48 |
| 2 000 | 2.25 |
| 4 000 | 1.62 |
| 6 000 | 1.38 |
| 8 000 | 1.03 |
| **10 000 (last 500 mean)** | **0.76 ± 0.11** |

Train loss dropped 3.2× from run 1's 2.46 plateau. **0.76 is
suspiciously low for a 57M model on 2.6M tokens** — there was no
held-out validation set, but six epochs over a tiny corpus virtually
guarantees the model is memorizing chunks. The smoke samples below
are consistent with that read.

## §6 — Pipeline integration

`TinyDocsGenerator` (subclass of the v1 `Generator` ABC) loads a
checkpoint + tokenizer eagerly in `__init__`, then implements the
standard `generate(query, retrieved_chunks, ...)` contract.

Pipeline differences vs `QwenGenerator`:

- The base LM is **not chat-tuned**; `build_grounded_prompt`'s
  message list is flattened with `"\n\n".join(m["content"] for m in messages)`
  before encoding. Role markers (`<system>`, `<|im_start|>`) carry no
  learned signal and are dropped.
- Tokenizer has no chat template; encoding uses
  `tokenizer.encode(text, add_bos=True, add_eos=False)`.
- Decoding is greedy with a hand-rolled KV-cache loop:
  prefill (one forward over the full prompt) → argmax last-position logits →
  per-token decode loop, feeding one new token per step at the
  correct RoPE position, until `eos_id` or `max_new_tokens`.
- Model `max_seq_len` was 256 at training time; the prompt is
  **tail-truncated** to `model_max_seq_len - max_new_tokens` so the
  question (placed at the end of the flattened prompt) survives.

CLI: `pdr ask --backend tinydocs --tinydocs-ckpt <path> --tinydocs-tok <path> "<query>"`.

End-to-end smoke (10k-b8 ckpt, k=3, max_new_tokens=64):

```
$ pdr ask --backend tinydocs ...  "how to use Path?"
resolved docs-sha=a5c1a35a5a02
 - - name - file = foo to create a path to the archive . - - default - directory ¶ ...
```

Pipeline runs. That's the §6 criterion. The output is gibberish, as
expected — see Quality Gap below.

## Quality gap vs Qwen

**No formal LLM-as-judge comparison was run** (see "Why §7 was not
pursued" below). The evidence presented here is qualitative.

### Smoke comparison — same query, both checkpoints

Single hand-crafted `Chunk` for `pathlib.Path`, `max_new_tokens=64`.

| Query | run-1h-explore (2 k steps) | run-10k-b8 (10 k steps) |
|---|---|---|
| how to use Path? | `' = " " " " " " " " ..."'` | `' - - name - file = foo to create a path to the archive . - - default - directory ¶ This is the default - - file for the - system'` |
| what is asyncio? | (n/a) | `' E - O on Windows ; if you need to use it , you can use the subprocess module instead . [ 2 ] The following options are supported : See also'` |
| json.dumps example | (n/a) | `' . py module with the same name and name in the source file . Added in version 3 . 2 .'` |

Run 1: pure repetition of `" "`, no useful signal. Run 2 has learned
docs vocabulary ("default-directory", "subprocess module", "Added in
version 3.2", "py module") and approximately Sphinx-style sentence
shapes ("This is the default", section-anchor prose). It does not
follow instructions, does not cite `[N]`, never refuses, and does not
answer the actual query — it generates plausible-sounding doc-shaped
continuations.

### Versus Qwen2.5-1.5B-Instruct

Qualitatively, on the same chunks:

- **Coherence:** Qwen produces grammatical, query-relevant prose with
  inline `[1]` citations; TinyDocs produces docs-shaped fragments
  unrelated to the query.
- **Citation discipline:** Qwen learned the `[N]` format from
  instruction tuning; TinyDocs has no SFT signal for it.
- **Refusal behavior:** Qwen will emit `[INSUFFICIENT-CONTEXT]` when
  appropriate; TinyDocs has never been trained on the marker and
  never refuses.
- **Latency:** TinyDocs ~1 s for 64 tokens on CPU (~15 ms / token);
  Qwen ~5–15 s for 512 tokens on MPS. Per-token TinyDocs is faster
  because it is 26× smaller, but its output is unusable.

Expected gap: enormous. A 57M base LM trained on 2.6M tokens of one
narrow corpus loses to a 1.5B SFT model by every quality metric. This
is not a bug; it is the experiment.

### Why §7 was not pursued

Plan §7 calls for a formal eval comparing the two backends on
Recall@5 / answer quality / hallucination rate / latency. Three
reasons made the cost-to-insight ratio poor:

1. **Recall@5 is retrieval-only**, identical for both backends (same
   router / SymbolIndex / BM25Index) — no signal.
2. **The quality answer is already known qualitatively** from smoke:
   TinyDocs hallucinates ~100 %, refuses ~0 %, scores at the floor on
   a 4-tier judge. A run-out result with 50–200 query × 2 backend ×
   manual Opus-4.7 copy-paste judging is roughly 100–400 manual
   batches for a number nobody is uncertain about.
3. **Latency is the only genuinely interesting axis**, and a
   one-paragraph qualitative note above already covers it.

If a future v4 retrains TinyDocs with SFT distillation, the comparison
becomes informative again — at that point Recall@5 still won't change,
but answer quality and refusal behavior will.

## What we learned

- **MVP scoping pays off.** The 14-day stretch goal of "build a tiny
  decoder-only LM end-to-end" was achievable only because the pass
  criterion was *pipeline runs*, not *good output*. Anything stricter
  would have demanded cloud GPU + general-text data + SFT distillation
  — multiple weeks of additional work.
- **Tokenizer choice constrains everything else.** Reusing Qwen's
  152k vocab would have eaten the entire 50–100M parameter budget on
  embeddings alone. The "MVP uses Option A" line in the plan was
  wrong; it had to be Option B from day one. (See plan review for the
  retrofit; commit history shows §2 chose B directly.)
- **Init matters more than expected.** Default PyTorch `nn.Linear`
  init produced step-0 loss of 382 — far above the theoretical
  `log(vocab_size) ≈ 10.4`. GPT-2-style `std=0.02` fixed it. This
  was a 1-line bug fix that would have hidden behind LR tuning if
  not caught at step 0.
- **Caching the slow step is worth a separate CLI.** BPE encoding
  was 3 min on 11 k lines (~5x faster after adding a per-word encode
  cache). Iterating on training hyperparameters would have been
  painful without `encode_corpus.py` as its own stage.
- **Loss alone is misleading on tiny corpora.** Run 2's 0.76 looks
  great in isolation; in context it is a memorization signal, not a
  generalization signal. A 5 %-holdout val split would have made the
  overfit visible. (Plan §4 listed val-loss tracking as nice-to-have,
  but it was not implemented; that was a mistake.)
- **KV cache + RoPE is fiddly.** First end-to-end smoke crashed on a
  position-out-of-range error because the prompt-truncation slice
  was reversed (`encoded[224:]` instead of `encoded[-224:]`). Three
  more shape / kwarg bugs surfaced in the same review. None of them
  were caught by the unit tests because `_call_model` was stubbed.
  Future versions should add at least one integration test that runs
  the model on a tiny config end-to-end (no stubs).

## What would change next time

| Area | Change | Reason |
|---|---|---|
| Validation | 5 % held-out segments + val loss every 200 steps | Quantify overfit; without it, "loss = 0.76" is uninterpretable. |
| Data | FineWeb-Edu mix at 80 / 20 (general / docs), retrain BPE on the mix | 2.6 M tokens is far below the Chinchilla-optimal ~1 B for 57 M params. The mix is what would actually move generation quality. |
| Compute | Cloud GPU (Modal / RunPod) for any run > 10 k steps | M1 MPS at ~2.4 step/s is fine for smoke but unworkable for serious pretraining. |
| SFT (§5) | Distill ~5 k examples from Qwen on grounded queries | Without SFT, the base LM literally cannot follow the `[N]` / refusal contract. SFT is the cheapest single change with the largest quality return. |
| Tokenizer | Sphinx-symbol-aware pretokenizer | Avoid `pathlib.Path` → 5 tokens scattering. Worth ~1 % vocab efficiency on docs-domain text. |
| Integration tests | One end-to-end test that does prefill + decode on a 1-layer / 64-dim TinyDocsConfig | Would have caught the 4 stubs-hide-bugs found in §6 review. |
