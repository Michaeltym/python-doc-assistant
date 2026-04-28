# v1 — Qwen2.5 Grounded Generation

Living document. Filled in incrementally as §3 → §7 of `plans/v1-qwen-generator.md` complete.

## Status

| Section | Topic | State |
|---------|-------|-------|
| §1 | Generator ABC + Answer dataclass | ✅ done (commit `b14129f`) |
| §2 | grounded prompt template + parser | ✅ done (commit `b14129f`) |
| §3 | QwenGenerator wiring | ✅ done (commit `8b440ba`) |
| §4 | Qwen-Coder side-by-side | ✅ done (commit `ecc3e53` + this doc) |
| §5 | CLI `--debug` extension (`pdr ask`) | ✅ done (commit + this doc) |
| §6 | 4-tier human scoring on full eval set | ✅ done (this doc + `human_scores.jsonl` per run) |
| §7 | Refusal rate on out-of-scope eval set | ✅ done (this doc + 2 oos run snapshots) |
| §8 | v2 priority recommendation | ✅ done (this doc, see §8 below) |

## Reproducibility (current)

- **model_id:** `Qwen/Qwen2.5-1.5B-Instruct`
- **decoding:** greedy — `temperature=0`, `top_p=1`, `max_new_tokens=512`, `do_sample=False`
- **device:** auto-detect (`cuda > mps > cpu`); Mac smoke run used `mps`
- **prompt format:** numeric `[N]` citations, HARD RULES, system/user role split, no few-shot
- **chat template:** `tokenizer.apply_chat_template(..., tokenize=False, add_generation_prompt=True)` (Qwen2.5 default)

Reproducibility fields below (`docs_version`, `docs_served_version`, `docs_sha_short`, `ingest_manifest`) will be filled once §6 runs against `eval_sets/v0_core.jsonl`. §3 smoke used hand-crafted `Chunk` objects, not the ingested corpus.

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

Two `pdr eval` runs on `eval_sets/v0_core.jsonl` (n = 34), same retrieval pipeline, different generator.

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

## §5 — `pdr ask` CLI + debug discovery

`pdr ask <query> [--debug] [--model <hf_id>] [--k 5]` runs a single
grounded retrieval + generation in one shot. `--debug` prints four blocks
per plan §5: retrieved chunks (rank + score + chunk_id), final chat
template messages, the answer, and citation validation (each cited
chunk_id annotated with `in_retrieved=yes|no (not in top-K)`).

### Smoke run on `"How to read a text file with pathlib?"`

Retrieval (k=5):

```
rank=1  score=1.000  id=section:library/gzip#examples-of-usage      ← wrong domain
rank=2  score=0.500  id=symbol:zipfile.Path.read_text                ← related
rank=3  score=0.333  id=symbol:sqlite3.Connection.text_factory       ← unrelated
rank=4  score=0.250  id=symbol:pathlib.Path.read_text                ← target
rank=5  score=0.200  id=symbol:importlib.resources.read_text         ← related
```

Target chunk lands at rank 4. Generator output: a clean prose answer +
correct code example using `Path.read_text()`, but `cited_chunk_ids=()` —
**no `[N]` emitted at all.** Same "no citation" failure mode §4
quantified at 32 %. The model wrote from prior knowledge instead of
grounding in chunk [4].

### Headline finding (NEW, beyond §5 scope) — chunker fragments code blocks

The `--debug` prompt dump exposed a chunker quality bug. The first chunk
shipped to the LLM contains:

```
[1] Examples of usage
Example of how to read a compressed file:
import
gzip
with
gzip
.
open
(
'/home/joe/file.txt.gz'
,
'rb'
)
as
f
:
file_content
=
f
.
read
()
```

Every Python token sits on its own line. Cause: the chunker processes
the Sphinx `<pre>` blocks span-by-span instead of text-by-text, so each
`<span class="...">token</span>` becomes a line in `chunk.text`. On
identifier queries this is mostly cosmetic, but on natural-language
howto queries the LLM has to mentally reassemble the code before
deciding whether the chunk is relevant — and frequently it gives up
and writes from prior knowledge.

This connects two §4 findings into one mechanism:
- **No-citation rate (32 %)** is partly chunker-driven, not just model-
  weakness — even when retrieval is right, the chunk is hard to read.
- **The "looks-cited" hallucination rate** (cite-but-not-expected, 29-32 %)
  shrinks if chunks are grounded enough for the model to actually use.

### v2 priority update (extends §4)

Original §4 verdict: v2 P0 = retrieval upgrade (dense + rerank).

After §5, the priority list is:

1. **Retrieval upgrade** (dense + rerank) — fixes the 6 wrong-routed NL
   queries; lifts the 38-41 % citation-match ceiling.
2. **Chunker code-block fix** — collapse `<pre>`/`<code>` token-spans
   back into single text lines so chunks are LLM-readable. Likely
   bigger generation-quality lift than another prompt round.
3. **Reranker** — re-orders the top-N retrieved set by semantic
   similarity to query; complements (1).
4. Prompt tweaks (few-shot, structure-hint rewording) — only after
   (1)-(3) ship, since they confound measurements today.

### `pdr ask` test coverage

7 hermetic tests in `tests/test_cli.py` (no real Qwen loaded, stubs
`QwenGenerator` via monkeypatch):

- prints answer text without `--debug`
- refused answer (empty `text`) prints `[INSUFFICIENT-CONTEXT]` fallback
- `--model` flag reaches the generator constructor
- `--debug` prints retrieved chunks with `rank=`/`score=`/`id=`
- `--debug` prints final prompt (`--- system ---` / `--- user ---` blocks)
- `--debug` annotates each citation with `in_retrieved=yes|no`
- without `--debug` no `[debug] *` blocks appear

## §6 — 4-tier human scoring on full eval set

### Methodology

- **Eval set:** `eval_sets/v0_core.jsonl` (n = 34).
- **Tiers** (plan §6 + `refused` for the marker case): `correct`, `partial`, `wrong`, `hallucination`, `refused`. Definitions in `src/python_doc_assistant/evaluation/human_scoring.py` docstring.
- **Tooling:** `evaluation/human_scoring.py` validates and aggregates `human_scores.jsonl` (one entry per query). Tests in `tests/test_human_scoring.py`.
- **Process** — three review passes:
  1. **Heuristic draft:** `tmp/scores-draft/<tag>-human-scores-draft.jsonl` generated from cited-vs-expected overlap + retrieval presence.
  2. **Claude review:** 12 reclassifications targeting boundary cases (no-cite-but-correct → `partial` not `wrong`; cite-grounded-but-wrong-symbol disambiguation).
  3. **Codex review (rounds 1 + 2) + Gemini review:** 14 further reclassifications. Codex tightened the rubric: *"reserve `hallucination` for content not present in retrieved chunks or unrelated citations made to look grounded; grounded-but-wrong-API answers are `wrong`."* Final hallucination rate dropped from heuristic 26-29 % to scored 14.7 % per run.
- **Final scores** committed at `experiments/runs/<run>/human_scores.jsonl`.

### Aggregate (after multi-reviewer pass)

| Tier | v1-qwen (Base Instruct) | v1-coder | Notes |
|---|---|---|---|
| correct | 12 (35.3 %) | 10 (29.4 %) | facts + citations both right |
| partial | 14 (41.2 %) | 16 (47.1 %) | answer right but wrong/missing citation, OR cite ok but content thin |
| wrong | 3 (8.8 %) | 2 (5.9 %) | factually wrong (regardless of citation) |
| **hallucination** | **5 (14.7 %)** | **5 (14.7 %)** | content not in retrieved chunks OR unrelated cite faking grounding |
| refused | 0 | 1 (2.9 %) | model emitted `[INSUFFICIENT-CONTEXT]` |
| **hallucination_rate** | **0.147** | **0.147** | plan §6 target was < 0.10 — **not met** |
| **correct_rate** | 0.353 | 0.294 | |

### Headline finding — both models tie on hallucination, but for different reasons

This is the §6 result that drives the v2 plan more than the §4 numbers did:

> **Coder = "lazy grounding"** — cites the correct chunk but produces empty / structure-only output (e.g. `functools.wraps` answer was just `definition -> [1] context -> [1] example -> [1]` with zero substantive prose). Coder treats RAG as a retrieval-only task and over-extracts.
>
> **Base = "fake grounding"** — cites unrelated retrieved chunks to dress up prior-knowledge prose (e.g. `how to count occurrences in a list` → cited `multiprocessing.shared_memory.ShareableList.count` while writing about generic `list.count`).
>
> Both failure modes converge on the same hallucination rate (14.7 %), but the mechanisms differ. Coder needs prompt pressure to actually use the chunk; Base needs prompt pressure to refuse rather than fabricate.

(Mechanism analysis credit: Codex + Gemini reviewer notes.)

### Specific failure cases worth keeping

| query | tier (v1-qwen / v1-coder) | what happened |
|---|---|---|
| how to count occurrences in a list | partial / refused | Base cited `ShareableList.count` (paraphrased its description) but answered with prior-knowledge `list.count` example. Coder correctly refused because `collections.Counter` was not retrieved. |
| collections.Counter | partial / partial | Both produced mostly-correct definitions but Base falsely claimed `Counter` supports `fromkeys` — the cited docs explicitly state it does NOT. Codex caught this. |
| how to convert a string to bytes | wrong / wrong | Both cited `urllib.parse.unquote_to_bytes` (URL decoder) — wrong API for the asked-for `str.encode` task. |
| json vs pickle | wrong / partial | Base reversed the security guidance (claimed JSON is vulnerable, pickle is secure — opposite of truth). Coder hedged. |
| how to use functools wraps | correct / partial | Coder: empty echo of the structure hint with `[1]` placeholders, no real prose despite cite-match-expected. The clearest single instance of "lazy grounding". |
| how to join two paths | partial / wrong | Coder degenerated into `join_thread() join_thread() join_thread() ...` token-loop; chunker fragmentation (§5 finding) likely contributed. |

### vs plan §6 completion bar

- **Target:** hallucination_rate < 0.10
- **Result:** 0.147 on both models (n = 34). **Target NOT met.**
- **Distance:** 4.7 pp; means roughly 2 fewer hallucinations on the same set would clear the bar — fragile boundary, more eval data would tighten the estimate.

Plan §142 lists "upgrade to 7B" as the explicit decision point when 1.5B fails the < 10 % target. **§6 says trigger that decision** — alongside the §4 + §5 retrieval / chunker fixes that drive the bar lower without needing more parameters.

## §7 — Refusal rate on out-of-scope eval set

### Eval set

- **File:** `eval_sets/v1_out_of_scope_20.jsonl` (n = 20)
- **Schema:** every row has `expected_symbols=[]`, `expected_urls=[]`, `query_type="out_of_scope"`. Retrieval recall@K is 0 by construction; the metric of interest is the generator's refusal behavior.
- **Topic spread:** ML / DL frameworks (transformer, gradient descent, PyTorch DataLoader), other languages (JS, Rust, Go, Ruby, C++, TypeScript), web/SaaS (Django REST, FastAPI, Vue.js, Rails), DevOps/cloud (K8s, AWS Lambda, Docker), datastores/protocols (PostgreSQL window functions, Redis pub/sub, MQTT QoS, B-tree, HTTP/2). All clearly outside the Python stdlib docs corpus.

### Reproducibility

Same as §4 / §6: greedy decoding, k_for_generation=5, MPS, docs sha `a5c1a35a5a02`.

- `experiments/runs/2026-04-28T02-04-35-v1-qwen-oos/` — `Qwen/Qwen2.5-1.5B-Instruct`
- `experiments/runs/2026-04-28T02-10-13-v1-coder-oos/` — `Qwen/Qwen2.5-Coder-1.5B-Instruct`

### Aggregate

| Dimension | v1-qwen-oos | v1-coder-oos |
|---|---|---|
| n queries | 20 | 20 |
| Refused (`refused == true`) | 14 | 14 |
| **refusal_rate** | **0.70** | **0.70** |
| Mean generation latency | 4.3 s | 6.3 s |
| Total wall time | 85 s | 126 s |

(Latency is much lower than §4 / §6 because refusing emits `[INSUFFICIENT-CONTEXT]` and stops — no full 512-token answer.)

### Headline finding — refusal works on OBVIOUS misses, not SUBTLE ones

This is the key §7 result that re-frames §4 + §6:

| Retrieval miss type | Example | Refusal behavior |
|---|---|---|
| **Obvious cross-domain** | "Rust ownership rules", "Kubernetes pod autoscaling" → returns `tkinter.font` / `imghdr.what` chunks | model identifies chunks as irrelevant, fires HARD-RULES refusal (70 % rate) |
| **Subtle same-domain wrong-API** | "how to count occurrences in a list" → returns `bytes.replace` / `str.replace` chunks | model treats chunks as related, hallucinates an answer + invents a citation (§4 / §6) |

Plan §7 implicitly assumed refusal is a single dimension; it's actually two. The bar that matters for v2 is **subtle-miss refusal**, where the §4 / §6 numbers (~0 %) hold and the §7 number does NOT generalize.

### What survived refusal — 6 leaked queries per model

The 6 queries each model "answered" rather than refusing:

**v1-qwen-oos:**
1. "what is gradient descent" — prior-knowledge prose, no citation
2. "what is a B-tree index" — prior-knowledge prose, no citation
3. "what is HTTP/2 multiplexing" — prior-knowledge prose, no citation
4. "Django REST framework serializers" — bracket leakage `[Django REST framework serializers]` then prose
5. "Vue.js reactive state" — prose + invented `[symbol:codecs.IncrementalEncoder.setstate]` citation (clearly fabricated)
6. "Kubernetes pod autoscaling" — **degenerate token loop** `[1] [2] [3] [4] ... [26] ...` (retrieval returned no chunks; same empty-CONTEXT failure as §3 smoke Test 2 "Paris")

**v1-coder-oos:**
1. "what is gradient descent" — prior knowledge
2. "what is a B-tree index" — prior knowledge, with `[BTreeIndex]` bracket prefix
3. "what is HTTP/2 multiplexing" — prior knowledge
4. "how to train a transformer model" — generic ML steps, no citation
5. "AWS Lambda cold start" — prior knowledge
6. "Ruby on Rails routing" — prior knowledge

### Patterns

1. **Three queries leak on both models** (gradient descent / B-tree / HTTP/2 multiplexing). All are `"what is X"` definitional queries. The model's prior on `"what is X"` → "produce a definition" beats the HARD-RULES "refuse if not in chunks" instruction.
2. **Asymmetric leakage on the other 3+3** — Base leaked Django / Vue / K8s, Coder leaked transformer / Lambda / Rails. Aligns with §6 mechanism contrast (Coder over-extracts when training distribution covers the topic; Base "fake-grounds" with prior knowledge prose).
3. **Empty retrieval is brittle.** "Kubernetes pod autoscaling" returned 0 chunks → Base degenerated into bracket-number spam (`[1] [2] ... [26]`). This is the exact failure plan §3 Round D-strict tried to fix and failed; the bug is still there for empty CONTEXT, just rare.

### Versus plan §7 completion

- Plan §7 asks "compute the fraction of correct refusals". Result: **0.70 on both models**.
- The narrative §7 wanted ("refusal works") is partially true and partially misleading. **§4 + §6 + §7 together** show that 70 % refusal is the upper bound on a domain so foreign that even the model can tell the chunks are unrelated; refusal collapses on subtle misses where retrieval surfaces lexically-similar wrong-API chunks.
- v2 priority unchanged: retrieval quality (which determines whether the model sees obvious-miss or subtle-miss) drives generation behavior more than prompt or model choice.

## §8 — v2 priority recommendation

After §3-§7, the v2 priority list is:

1. **Retrieval upgrade — dense + rerank.** This is the lever with the most leverage:
   - Lifts the §4 citation-match-expected ceiling from 38-41 % toward 80 %+ (informed guess based on closing the 6 NL queries that route wrong).
   - Converts subtle-miss situations into obvious-miss → §7 refusal_rate of 70 % becomes the floor not ceiling.
   - The §6 "lazy grounding" / "fake grounding" failure modes both shrink when chunks are correct.
2. **Chunker code-block fix** (§5 finding). Sphinx `<pre>`/`<code>` blocks currently fragment one Python token per line; LLM cannot read them. Generation hallucinates more on howto queries with code answers as a result.
3. **Optional 7B upgrade** (plan §142 decision). Only worthwhile if (1) + (2) leave hallucination_rate above 10 %. Cost: ~14 GB download, MPS inference 30-60 s/query. Coder vs Base tied at 14.7 % hallucination; the gap to <10 % is small enough that retrieval/chunker fixes alone may close it.
4. **Few-shot or prompt rewording** — defer until (1)-(3) ship; current prompt iterations confound measurement (see §3 prompt iteration log).

### Out of scope for v1; logged for future

- `query_id` field on `EvalQuery` for stable joins (PLAN.md §8 review noted; deferred).
- Aggregator that updates `results.json` in place with `human_scores` + `refusal_rate` aggregates (currently in narrative only; could move into `evaluation/run_writer.py`).
- Larger eval set (n = 100-200 per plan §6 stretch goal).
- Out-of-scope schema: `notes` field describing why each query is out of scope; today included but not validated.
