# v1 — Qwen2.5 Grounded Generation

Living document. Filled in incrementally as §3 → §7 of `plans/v1-qwen-generator.md` complete.

## Status

| Section | Topic | State |
|---------|-------|-------|
| §1 | Generator ABC + Answer dataclass | ✅ done (commit `b14129f`) |
| §2 | grounded prompt template + parser | ✅ done (commit `b14129f`) |
| §3 | QwenGenerator wiring | ✅ done (commit `8b440ba`) |
| §4 | Qwen-Coder side-by-side | ✅ done (commit `ecc3e53` + this doc) |
| §5 | CLI `--debug` extension | ⏳ TODO |
| §6 | 4-tier human scoring on full eval set | ⏳ TODO |
| §7 | Refusal rate on out-of-scope eval set | ⏳ TODO |
| §8 | v2 priority recommendation | partially answered in §4 — confirmed below in §6/§8 |

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

## §4 — Qwen vs Qwen-Coder side-by-side

Two `pdr eval` runs on `eval_sets/v0_core_30.jsonl` (n=34 — set name kept from v0; actual size is 34), same retrieval pipeline, different generator.

### Reproducibility

- **docs_version / docs_served_version / docs_sha_short:** `3.12` / `3.12.13` / `a5c1a35a5a02`
- **decoding_params (both runs):** `temperature=0`, `top_p=1`, `max_new_tokens=512`, `do_sample=False` (greedy)
- **k_for_generation:** 5 (top-5 retrieved chunks fed to LLM)
- **device:** MPS (Mac)
- **runs:**
  - `experiments/runs/2026-04-28T00-08-32-v1-qwen/` — `Qwen/Qwen2.5-1.5B-Instruct`
  - `experiments/runs/2026-04-28T00-23-02-v1-coder/` — `Qwen/Qwen2.5-Coder-1.5B-Instruct`

### Aggregate metrics

| Dimension | Base (Instruct) | Coder | Δ |
|---|---|---|---|
| Recall@5 (retrieval) | 0.824 | 0.824 | identical (same retrieval pipeline) |
| Recall@10 (retrieval) | 0.853 | 0.853 | identical |
| MRR (retrieval) | 0.674 | 0.674 | identical |
| ≥1 citation emitted | 23/34 (68 %) | 25/34 (74 %) | Coder +6 pp |
| **Citation matches expected** | 13/34 (38 %) | 14/34 (41 %) | Coder +3 pp |
| Cited but not in expected (likely hallucination) | 10/34 (29 %) | 11/34 (32 %) | Coder marginally worse |
| No citation emitted | 11/34 (32 %) | 8/34 (24 %) | Coder −8 pp |
| Refused (`refused=true`) | 0/34 | 1/34 | Coder caught 1 retrieval miss |
| Echo of `signature -> ...` hint in answer | 11/34 (32 %) | 14/34 (41 %) | both leak |
| Mean generation latency | 14.1 s | 18.1 s | Coder +28 % |
| Total generation wall time | 480 s | 614 s | — |

Citation accuracy is computed as: did the cited `chunk_id` (after stripping the `symbol:` prefix) match any of the row's `expected_symbols`. False positives are possible when a tangentially-related chunk happens to expose the right answer; manual scoring in §6 will refine this.

### Headline finding — retrieval is the bottleneck, not generation

Six natural-language queries in the eval set route through BM25 (no exact symbol match) and pull lexically-similar but semantically-wrong chunks. The generator then either invents an answer from prior knowledge or cites the wrong chunk:

| Query | Top-3 retrieved | Expected |
|---|---|---|
| how to read a file in python | `section:library/gzip#examples-of-usage`, `platform.libc_ver`, `argparse...convert_arg_line_to_args` | `pathlib.Path.read_text` / `io.open` / `open` |
| how to iterate a dictionary safely | `sys.modules`, `dict.popitem`, `select.devpoll.unregister` | `dict.items` / `dict.keys` / `dict.values` |
| how to count occurrences in a list | `bytearray.replace`, `bytes.replace`, `str.replace` | `collections.Counter` |
| how to memoize a function | `section:library/copyreg#example`, `pstats.Stats.get_stats_profile`, `logging.getLogRecordFactory` | `functools.lru_cache` / `functools.cache` |
| how to flatten a nested list with itertools | `itertools.product`, `itertools.repeat`, `pyclbr.readmodule_ex` | `itertools.chain` / `itertools.chain.from_iterable` |
| how to compile a regex pattern | `fnmatch.translate`, `re.Pattern.flags`, `unittest.TestCase.assertRegex` | `re.compile` / `re.Pattern` |

In **every** case the answer is in the corpus, but BM25 keyword overlap surfaces a different module first. No amount of generation tuning fixes this — **v2 priority is dense embeddings + rerank**.

### The "Coder catches a miss, Base hallucinates" example

```
Query:    how to count occurrences in a list
Expected: collections.Counter
Retrieved (both runs): symbol:bytearray.replace, symbol:bytes.replace, symbol:str.replace

Coder:  refused=True, model_output_text=""
        ✅ correct — chunks do NOT contain the answer; refusal is honest.

Base:   refused=False, cited=[symbol:multiprocessing.shared_memory.ShareableList.count]
        text: "The `count` function can be used to count occurrences in a list.
               Here is an example: my_list.count('apple') ... [5]"
        ❌ Three failures stacked:
           1. Used prior knowledge about list.count (not in chunks).
           2. Cited [5] which mapped to ShareableList.count — not what the
              prose described.
           3. Created the appearance of grounding while everything was
              fabricated.
```

This single query shows why a refusal mechanism matters even when most queries are positive — the failure mode under bad retrieval is invisible "looks-cited" hallucination, not visible "I don't know".

### Tier-style breakdown (n=34, by hand)

These tiers anticipate plan §6 4-tier scoring; numbers are estimates from the head-to-head jq scan, not formal scoring (that's §6).

| Tier | Base | Coder |
|---|---|---|
| Correct (cite hits expected, prose grounded in chunk) | 13 (38 %) | 14 (41 %) |
| Partial (cite hits but prose mixes prior knowledge) | ~5 | ~5 |
| Wrong (cite misses, prose still readable but ungrounded) | ~10 | ~11 |
| Hallucination (no cite or invented cite + ungrounded prose) | ~6 | ~3 |
| Refused | 0 | 1 |

Coder marginally better on hallucination (3 vs 6 estimated); §6 will pin numbers with a real rubric and inter-rater agreement.

### Style differences (qualitative)

- **Coder leans "code-formatted"** — outputs raw signatures with parameter defaults (`re. compile ( pattern , flags = 0 )`); training distribution shows.
- **Base leans more "prose-explanatory"** — fewer code blocks, more sentences.
- Both echo the `signature -> brief description -> example -> source citation` structure hint verbatim 32 % / 41 % of the time. The hint is meant as guidance but the model treats it as a "format string to begin with". v1 §3 prompt could reword (e.g. "Format: ..." vs "Use this answer structure: ...") to reduce leakage; deferred since §6 measures this directly.

### §4 verdict

- **Default for v1 = Coder.** Marginal gains on citation discipline (+6 pp ≥1 citation, +3 pp citation-match-expected) and 1 correct refusal. Cost: 28 % slower generation, ~3 pp more echo of the structure hint. Net signal favours Coder, but n=34 is small.
- **The 38–41 % citation-match-expected ceiling is set by retrieval, not generation.** Six NL queries deterministically retrieve the wrong chunks; the model can only cite what it sees. **v2 P0 = retrieval upgrade (dense + rerank)**, not prompt iteration or model swap.
- **Update default in `config.toml`?** Not yet — §6 4-tier human scoring + §7 refusal eval should confirm before pinning Coder as the v1 default in code. Current `qwen_backend.py` `DEFAULT_MODEL_ID` stays `Qwen/Qwen2.5-1.5B-Instruct`.

## TODO

- **§5** — extend `pdr ask` with `--debug` (chunks + scores + final prompt + answer + citation match against retrieved set)
- **§6** — run on `eval_sets/v0_core_30.jsonl`; 4-tier human scoring; compute hallucination rate; populate `experiments/runs/<ts>-v1-qwen/` with `human_scores.jsonl`
- **§7** — author `eval_sets/v1_out_of_scope_20.jsonl` (20 queries); compute refusal rate; pin baseline number
- **§8** — recommend v2 priority (rerank-first vs dense-first) — §4 already points strongly at "retrieval first"; §6/§7 numbers should confirm or reject
