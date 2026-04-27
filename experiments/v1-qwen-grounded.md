# v1 — Qwen2.5 Grounded Generation

Living document. Filled in incrementally as §3 → §7 of `plans/v1-qwen-generator.md` complete.

## Status

| Section | Topic | State |
|---------|-------|-------|
| §1 | Generator ABC + Answer dataclass | ✅ done (commit `b14129f`) |
| §2 | grounded prompt template + parser | ✅ done (commit `b14129f`) |
| §3 | QwenGenerator wiring | ✅ done (commit `8b440ba`) |
| §4 | Qwen-Coder side-by-side | ⏳ TODO |
| §5 | CLI `--debug` extension | ⏳ TODO |
| §6 | 4-tier human scoring on full eval set | ⏳ TODO |
| §7 | Refusal rate on out-of-scope eval set | ⏳ TODO |
| §8 | v2 priority recommendation | ⏳ TODO |

## Reproducibility (current)

- **model_id:** `Qwen/Qwen2.5-1.5B-Instruct`
- **decoding:** greedy — `temperature=0`, `top_p=1`, `max_new_tokens=512`, `do_sample=False`
- **device:** auto-detect (`cuda > mps > cpu`); Mac smoke run used `mps`
- **prompt format:** numeric `[N]` citations, HARD RULES, system/user role split, no few-shot
- **chat template:** `tokenizer.apply_chat_template(..., tokenize=False, add_generation_prompt=True)` (Qwen2.5 default)

Reproducibility fields below (`docs_version`, `docs_served_version`, `docs_sha_short`, `ingest_manifest`) will be filled once §6 runs against `eval_sets/v0_core_30.jsonl`. §3 smoke used hand-crafted `Chunk` objects, not the ingested corpus.

## §3 Smoke findings (2 hand-crafted queries)

Goal: validate end-to-end wiring before running on the full eval set. Not a statistical sample — qualitative signal only.

### Test 1 — grounded howto query

- **Query:** "How do I read a text file with pathlib?"
- **Chunks:** 1 — `symbol:pathlib.Path.read_text` (signature + 1-line description)
- **Output:** prose answer using `read_text` correctly, ending with `[1]`
- **Result:** `cited_chunk_ids=('symbol:pathlib.Path.read_text',)`, `refused=False`, latency ~7.6 s on MPS
- **Issue:** code example hallucinates a chained call —  `with file_path.open(...) as file: file_content = file.read_text()` — `read_text` is a `Path` method, not chainable on a file handle. Chunk content was read but mixed with prior knowledge.

### Test 2 — out-of-scope refusal query

- **Query:** "What is the capital of France?"
- **Chunks:** 0 (empty `retrieved_chunks` list)
- **Output:** `[1] Paris`
- **Result:** `cited_chunk_ids=()` (out-of-range index dropped), `refused=False`, latency ~0.6 s
- **Expected:** model emits `[INSUFFICIENT-CONTEXT]` alone per HARD RULES
- **Actual failure:** model invented citation `[1]` and answered from prior knowledge

### Baseline failure modes (carry into §6/§7)

1. **Citation compliance partial** — `[N]` appears once at answer end, not at every factual sentence as HARD RULES require. §6 4-tier scoring rubric must distinguish "no citation" from "wrong citation"; this 1.5B baseline is the former.
2. **Refusal compliance 0% on empty CONTEXT** — §7 `refusal_rate` will be near zero. Model invents a citation index and answers.
3. **Code example fidelity poor** — model mixes chunk content with prior knowledge, producing chains the docs do not support. Affects "wrong" vs "hallucination" §6 tier classification.

## §3 Prompt iteration log

Four prompt configurations were smoke-tested before settling on the current state. Each tweak traded citation compliance against refusal compliance — none was strictly better than the previous, so the final pick is "least-worst on Test 1".

| Round | Config | Test 1 citation | Test 2 refusal |
|-------|--------|-----------------|----------------|
| 1 | `[#chunk_id]` markers + system/user split + HARD RULES + greedy | 0% | ✅ refused |
| 2 | Round 1 + 2-shot multi-turn (positive citation + refusal demos) | 0% (markdown-link hallucination `[pathlib...](URL)`) | ❌ "Paris is the capital..." |
| D-min | Numeric `[N]` markers (replaces `[#chunk_id]`); no explicit empty CONTEXT block | ✅ `[1]` at end | ❌ "[1] Paris" |
| D-strict | Round D-min + always-emit `CONTEXT:\n(no chunks provided)` + "NEVER invent [N]" rule | regressed to 0% | ❌ "[1] Paris" |

**Final state = Round D-min.** Reverting Round D-strict back to D-min restored Test 1 citation; Test 2 refusal stayed broken in every config except Round 1.

### Why iteration stopped

- Each tweak surfaced a new tradeoff rather than a strict improvement → diminishing returns from prompt-only iteration on a 1.5B model.
- Plan §6 explicitly measures these failure modes on the full eval set with 4-tier human scoring — pre-iterating before that data exists pre-commits to the wrong baseline.
- Plan §142 has an explicit "upgrade to 7B" decision point if §6 confirms 1.5B cannot meet the hallucination<10% target.

## TODO

- **§4** — add `--model qwen2.5-coder-1.5b-instruct` flag; run both models on the eval set; record per-query citation / refusal / latency / hallucination flags
- **§5** — extend `pdr ask` with `--debug` (chunks + scores + final prompt + answer + citation match against retrieved set)
- **§6** — run on `eval_sets/v0_core_30.jsonl`; 4-tier human scoring; compute hallucination rate; populate `experiments/runs/<ts>-v1-qwen/` with `results.json` + `per_query.jsonl` + `human_scores.jsonl`
- **§7** — author `eval_sets/v1_out_of_scope_20.jsonl` (20 queries); compute refusal rate; pin baseline number
- **§8** — recommend v2 priority (rerank-first vs dense-first) based on whether retrieval misses or generation hallucination dominates the failure data
