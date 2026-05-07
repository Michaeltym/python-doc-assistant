# v1 — Wire Up Qwen Generator

**Parent doc:** [../PLAN.md](../PLAN.md) §7 v1

**Prerequisite:** v0 completed (retrieval end-to-end loop + eval set in hand)

**Estimated duration:** ~1 week

**Core goal:** On top of v0 retrieval, feed Top-K chunks into Qwen2.5 to produce **grounded answers** (enforce citations, refuse when out of scope).

---

## Prerequisites

- [ ] Expand extras: `uv sync --extra dev --extra ingest --extra retrieval --extra generation` (this step starts installing `torch` + `transformers`)
- [ ] Qwen smoke test: load `Qwen2.5-1.5B-Instruct` and generate a "Hello world"
- [ ] Confirm MPS is available and latency is acceptable (if > 10 seconds/query, consider switching to 0.5B or a `llama.cpp` quantized version)

## Subtasks

### 1. Generator ABC interface

File: `src/python_doc_assistant/generation/interface.py`

```python
class Generator(ABC):
    def generate(
        self,
        query: str,
        retrieved_chunks: list[Chunk],
        *,
        query_type: QueryType | None = None,  # prompt template selection depends on this field; when None, the backend falls back
        stream: bool = False,
    ) -> Answer:
        ...
```

`Answer` is a dataclass: answer text + cited chunk ids + refusal flag + generation latency.

**Why `query_type` is part of the interface:** v1 prompts are conditioned on `query_type` (identifier / comparison / howto / natural_language). If it isn't passed through the interface, the generator can only guess from the raw query and loses the information already labeled in the eval schema. v0 eval entries naturally carry `query_type`, so passing it through directly is cleanest.

**Why abstract the interface now and not in v0?** v0 had no notion of a generator, so abstracting the interface early would be over-engineering. Starting in v1 we need both Qwen and (later) TinyDocs implementations, which is when an ABC has value.

### 2. Grounded prompt template (conditioned on query_type)

File: `src/python_doc_assistant/prompts/grounded.py`

**Common requirements:**

- Enforce using only the provided chunks
- If the answer is not found, **explicitly refuse** (output a fixed marker)
- Citation format: `[#chunk_id]`

**Answer structure templated by `query_type`:**

| query_type         | Structure                                                                          |
| ------------------ | ---------------------------------------------------------------------------------- |
| `identifier`       | signature → brief description → example → source                                   |
| `comparison`       | bullet list of points for each side → differences → recommended scenarios → source |
| `howto`            | steps → code example → source                                                      |
| `natural_language` | definition → background → example → source                                         |

`query_type` is already explicitly labeled in the v0 eval schema, and v1 uses it to pick the template. In the online inference stage we can pass it manually for now, and add a classifier in v2 or later that infers it from the raw query.

Write 2 versions of the prompt first (simple version + with few-shot); after v1 is done, compare which is more stable.

### 3. QwenGenerator implementation

File: `src/python_doc_assistant/generation/qwen_backend.py`

- Load `Qwen2.5-1.5B-Instruct` (default)
- Stuff query + chunks into the grounded prompt
- Call `transformers` generate
- Parse output (extract cited chunk ids + refusal marker)
- Configurable: `max_new_tokens`, `temperature`, `top_p`

### 4. Qwen-Coder side-by-side comparison

- Add `--model qwen2.5-coder-1.5b-instruct` flag
- Run both models on the same eval set, save per-query results
- `experiments/v1-qwen-vs-coder.md` for the comparison (Python docs are code-adjacent, so the Coder version should in theory be more suitable)

### 5. CLI --debug extension

File: `src/python_doc_assistant/cli.py` (extension)

```text
pdr ask "How to read a file?" --k 5 --debug
```

`--debug` outputs:

- Score and ID of each retrieved chunk
- Full text of the final prompt
- The generated answer
- The chunk ids cited by the model (and whether they are in the retrieved set)

### 6. Human scoring of generation quality

- **Cover all currently in eval queries** (the 30–50 from v0); expanding the eval set to 100–200 is left for v2
- **4-tier human scoring:**
  - **correct**: facts correct + citations correct
  - **partial**: main answer correct but missing details or wrong citations
  - **wrong**: factually wrong
  - **hallucination**: uses content not present in the chunks

**Run directory layout** (human scoring and generation eval share one timestamped directory, both following the run naming rules in PLAN.md §8 / plans/v0 §9):

```
experiments/runs/<YYYY-MM-DDTHH-MM-SS>-v1-qwen/
├── results.json           # summary: config + DOCS_VERSION + docs_served_version + docs_sha_short + ingest_manifest snapshot + model + prompt_hash + decoding params + aggregate human scores
├── per_query.jsonl        # per-query retrieved chunks + model output + citations + refusal flag
└── human_scores.jsonl     # 4-tier scores, joined back to per_query.jsonl by query_id
```

### 7. Refusal rate

- Deliberately craft out-of-scope queries (e.g. "how to train a transformer", "JavaScript array methods")
- File: `eval_sets/v1_out_of_scope_20.jsonl` (20 entries)
- Compute the fraction of correct refusals

### 8. Experiment narrative document

File: `experiments/v1-qwen-grounded.md`

Answer:

- Distribution across the 4 generation-quality tiers
- Hallucination rate
- Refusal rate
- Qwen vs Qwen-Coder: which is more suitable for this domain
- Priority for v2 (add rerank first or dense first?)

## Completion criteria

- [ ] `Generator` ABC + `QwenGenerator` implementation
- [ ] Both Qwen and Qwen-Coder models run end-to-end via `pdr ask`
- [ ] All current eval queries have been human-scored
- [ ] Hallucination rate < 10%
- [ ] `experiments/v1-qwen-grounded.md` written
