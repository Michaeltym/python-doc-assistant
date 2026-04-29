"""LLM-as-judge for v2 §6 generation quality scoring.

Calls the Anthropic API (Claude Haiku 4.5 by default) on each row of a
`per_query.jsonl` and emits a `human_scores.jsonl`-compatible file —
same schema as v1 manual scoring (`evaluation.human_scoring`), so the
existing `aggregate()` consumes both interchangeably.

Reproducibility (plan §6 contract — written into results.json `judge`
field by the eventual CLI subcommand):
    judge_model           full model id (e.g. "claude-haiku-4-5-20251001")
    judge_prompt_hash     SHA1 of `JUDGE_PROMPT_TEMPLATE` (8-hex prefix
                          stored alongside the records)
    temperature / max_tokens
    raw_output            UNPARSED Claude reply, per row
    timestamp             ISO 8601 UTC, per row

Workflow:
    1. agreement check — pick a 20-row stratified sample of queries that
       have v1 manual scores; run judge; compute Cohen's kappa + exact-
       match agreement vs the human label. plan §6 bar: > 0.80 exact
       agreement OR > 0.60 weighted-kappa equivalent.
    2. if pass → run judge on the full per_query.jsonl; write
       `judge_scores.jsonl` next to it.
    3. v2 hallucination_rate = aggregate(judge_scores).hallucination_rate.

NOTE: `anthropic` is *lazily* imported inside the method that calls the
API. Importing this module without the `judge` extra installed must NOT
raise. (Tests rely on this.)
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from math import ceil
from pathlib import Path
from typing import Any, Final

from python_doc_assistant.evaluation.human_scoring import VALID_TIERS, HumanScore
from python_doc_assistant.ingest.chunker import Chunk

# ------------------------------------------------------------------
# Defaults (plan §6 + decision log: Haiku 4.5 / temperature=0)
# ------------------------------------------------------------------

DEFAULT_JUDGE_MODEL: Final[str] = "claude-haiku-4-5-20251001"
DEFAULT_TEMPERATURE: Final[float] = 0.0  # deterministic judge
DEFAULT_MAX_TOKENS: Final[int] = 200  # tier + 1-sentence reason


# ------------------------------------------------------------------
# Prompt template (any wording change → new SHA1 → new judge_prompt_hash)
# ------------------------------------------------------------------

JUDGE_PROMPT_TEMPLATE: Final[str] = """\
You are scoring a Python-stdlib documentation assistant's answer.

TIER DEFINITIONS:
- correct       — facts correct AND citation matches expected_symbols
- partial       — answer substantively correct but missing details OR
                  has wrong/missing/module-level citation when a method-
                  level one was expected
- wrong         — factually wrong (regardless of citation)
- hallucination — uses content NOT present in the retrieved chunks
                  (most severe; reserve for invented/unsupported claims)
- refused       — model output was [INSUFFICIENT-CONTEXT] alone
                  (or refused == true with empty text)

INPUTS:
- query                  : {query}
- expected_symbols       : {expected_symbols}
- model refused?         : {refused}
- model's cited chunks   : {cited_chunk_ids}
- retrieved chunks (top-K, with text):
{retrieved_block}

MODEL OUTPUT:
\"\"\"
{model_output_text}
\"\"\"

Respond with EXACTLY one line of JSON, no code fences, no prose:
{{"tier": "<one of correct|partial|wrong|hallucination|refused>", "reason": "<one short sentence>"}}
"""


# ------------------------------------------------------------------
# Exceptions
# ------------------------------------------------------------------


class JudgeError(Exception):
    """Bad judge response (unparseable JSON, invalid tier, etc.)."""


# ------------------------------------------------------------------
# Data records
# ------------------------------------------------------------------


@dataclass(frozen=True)
class JudgeRecord:
    """One judge output. Extends HumanScore with reproducibility metadata.

    Persisted as JSONL alongside `per_query.jsonl`; `aggregate()` from
    `evaluation.human_scoring` consumes the (query, tier, notes) subset.
    """

    query: str
    tier: str  # one of VALID_TIERS
    notes: str  # judge's one-sentence reason
    raw_output: str  # full unparsed Claude reply (plan §6 reproducibility)
    judge_model: str
    judge_prompt_hash: str  # SHA1[:8] of JUDGE_PROMPT_TEMPLATE
    timestamp: str  # ISO 8601 UTC


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def judge_prompt_hash() -> str:
    """SHA1[:8] of the JUDGE_PROMPT_TEMPLATE.

    Used as a reproducibility fingerprint — any prompt-wording change
    should bump the hash so old judge runs can be told apart from new.
    Returns an 8-hex-char prefix to keep result.json small.
    """
    return _sha1_8(JUDGE_PROMPT_TEMPLATE)


def make_judge_prompt(
    *,
    query: str,
    expected_symbols: tuple[str, ...],
    retrieved_chunks: list[Chunk],
    cited_chunk_ids: tuple[str, ...],
    refused: bool,
    model_output_text: str,
) -> str:
    """Fill JUDGE_PROMPT_TEMPLATE with one row's data.

    `retrieved_chunks` carries the top-K full Chunk objects so the judge
    can verify whether the prose is supported by the retrieved text. The
    'retrieved_block' placeholder is built as:
        [#chunk_id_1] title_1
        text_1[:600]
        ---
        [#chunk_id_2] title_2
        ...

    Args:
        query: user query string.
        expected_symbols: tuple of expected stdlib symbols (eval row).
        retrieved_chunks: top-K Chunk objects (title + text used).
        cited_chunk_ids: tuple of chunk_ids the model cited.
        refused: whether the model emitted the refusal marker.
        model_output_text: the model's full prose reply.

    Returns:
        The complete prompt string ready to send to the judge model.
    """
    return JUDGE_PROMPT_TEMPLATE.format(
        query=query,
        expected_symbols=", ".join(expected_symbols),
        retrieved_block=_format_retrieved_block(retrieved_chunks),
        cited_chunk_ids=", ".join(cited_chunk_ids),
        refused=refused,
        model_output_text=model_output_text,
    )


def parse_judge_response(raw: str) -> tuple[str, str]:
    """Parse the judge's JSON reply → (tier, reason).

    Tolerant of leading/trailing whitespace; tolerant of code fences if
    Claude leaks them (strip ``` blocks before json.loads). Raise
    `JudgeError` if:
        - JSON is invalid
        - top-level is not an object
        - 'tier' missing or not in VALID_TIERS
        - 'reason' missing or not a string

    Returns:
        (tier, reason)
    """
    text = raw.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        raise JudgeError(f"Invalid json: {e.msg}") from e

    if not isinstance(obj, dict):
        raise JudgeError(f"expected JSON object, got {type(obj).__name__}")
    tier = obj.get("tier")
    reason = obj.get("reason")
    if not isinstance(tier, str) or tier not in VALID_TIERS:
        raise JudgeError(f"invalid tier: {tier!r}; expected one of {sorted(VALID_TIERS)}")

    if not isinstance(reason, str):
        raise JudgeError(f"invalid reason: expected str, got {type(reason).__name__}")

    return tier, reason


def judge_one(
    query: str,
    expected_symbols: tuple[str, ...],
    retrieved_chunks: list[Chunk],
    cited_chunk_ids: tuple[str, ...],
    refused: bool,
    model_output_text: str,
    *,
    client: Any,
    model_id: str = DEFAULT_JUDGE_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> JudgeRecord:
    """Score one row via the Anthropic API; return a JudgeRecord.

    Implementation outline:
        1. prompt = make_judge_prompt(...)
        2. response = client.messages.create(
               model=model_id,
               max_tokens=max_tokens,
               temperature=temperature,
               messages=[{"role": "user", "content": prompt}],
           )
        3. raw = response.content[0].text
        4. tier, reason = parse_judge_response(raw)
        5. return JudgeRecord(
               query=query,
               tier=tier,
               notes=reason,
               raw_output=raw,
               judge_model=model_id,
               judge_prompt_hash=judge_prompt_hash(),
               timestamp=datetime.now(timezone.utc).isoformat(),
           )

    Lets `JudgeError` from `parse_judge_response` propagate so the caller
    can decide whether to retry / skip / fail the run.
    """
    prompt = make_judge_prompt(
        query=query,
        expected_symbols=expected_symbols,
        retrieved_chunks=retrieved_chunks,
        cited_chunk_ids=cited_chunk_ids,
        refused=refused,
        model_output_text=model_output_text,
    )
    response = client.messages.create(
        model=model_id,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text
    tier, reason = parse_judge_response(raw)
    return JudgeRecord(
        query=query,
        tier=tier,
        notes=reason,
        raw_output=raw,
        judge_model=model_id,
        judge_prompt_hash=judge_prompt_hash(),
        timestamp=_now_utc_iso(),
    )


def stratified_sample(
    human_scores: list[HumanScore],
    *,
    n: int = 20,
    seed: int = 42,
) -> list[str]:
    """Pick ~`n` queries spread evenly across tiers (for agreement check).

    Plan §6 step 1 wants 20 samples for the human-vs-judge agreement
    check before scaling. Stratified sampling protects against the
    trivial pitfall where the random 20 are all 'correct' and judge
    looks artificially good.

    Args:
        human_scores: pool to sample from (e.g. v1 manual scores).
        n: target sample size; per-tier slot is `ceil(n / len(VALID_TIERS))`.
        seed: random seed for reproducibility.

    Returns:
        List of `query` strings (the join key against per_query.jsonl).
        Length may be < `n` if some tiers have fewer than the per-tier
        slot count.
    """
    if len(human_scores) < n:
        return [s.query for s in human_scores]
    total_queries_per_tier = ceil(n / len(VALID_TIERS))
    scores_by_tier: dict[str, list[HumanScore]] = {}
    for s in human_scores:
        scores_by_tier.setdefault(s.tier, [])
        scores_by_tier[s.tier].append(s)
    queries: list[str] = []
    rng = random.Random(seed)
    for t in VALID_TIERS:
        scores = scores_by_tier.get(t, [])
        queries_per_tier = [s.query for s in scores]
        if len(scores) <= total_queries_per_tier:
            queries.extend(queries_per_tier)
        else:
            queries.extend(rng.sample(queries_per_tier, k=total_queries_per_tier))
    return queries


def agreement_metrics(
    judge_scores: dict[str, str],
    human_scores: dict[str, str],
) -> dict[str, float]:
    """Compare two label maps {query: tier} → agreement metrics.

    Returns:
        {
            "n":             int,    # queries present in both maps
            "exact_match":   float,  # fraction of exact tier agreement
            "cohen_kappa":   float,  # chance-adjusted agreement
        }

    plan §6 bar:
        - exact_match > 0.80   AND
        - cohen_kappa > 0.60   (4-tier classification convention)

    Implementation outline:
        1. common = sorted(set(judge_scores) & set(human_scores))
        2. n = len(common); if n == 0 → all-zeros result
        3. agree = sum(judge_scores[q] == human_scores[q] for q in common)
           exact_match = agree / n
        4. cohen_kappa:
              p_observed = exact_match
              for each tier in VALID_TIERS:
                  p_judge = (count where judge_scores[q]==tier) / n
                  p_human = (count where human_scores[q]==tier) / n
                  p_chance += p_judge * p_human
              kappa = (p_observed - p_chance) / (1 - p_chance) if p_chance != 1 else 1.0
    """
    overlap_queries = set(judge_scores) & set(human_scores)
    n = len(overlap_queries)
    if n == 0:
        return {"n": 0, "exact_match": 0.0, "cohen_kappa": 0.0}
    total_exact_match = sum(1 for q in overlap_queries if judge_scores[q] == human_scores[q])
    exact_match = total_exact_match / n
    p_chance = 0.0
    for tier in VALID_TIERS:
        judge_tier_count = sum(1 for q in overlap_queries if judge_scores[q] == tier)
        human_tier_count = sum(1 for q in overlap_queries if human_scores[q] == tier)
        p_chance += (judge_tier_count / n) * (human_tier_count / n)
    if p_chance == 1.0:
        kappa = 1.0
    else:
        kappa = (exact_match - p_chance) / (1 - p_chance)
    return {"n": n, "exact_match": exact_match, "cohen_kappa": kappa}


def judge_records_to_human_scores(records: list[JudgeRecord]) -> list[HumanScore]:
    """Project JudgeRecords into `HumanScore` shape so `aggregate()` works.

    The reproducibility metadata (raw_output / judge_model / hash /
    timestamp) is dropped — `aggregate()` only reads tier + query.
    """
    return [HumanScore(query=r.query, tier=r.tier, notes=r.notes) for r in records]


def write_judge_records(records: list[JudgeRecord], path: Path) -> None:
    """Write JSONL of full JudgeRecord (with all reproducibility fields).

    Path conventionally `<run_dir>/judge_scores.jsonl`. Caller pre-creates
    the parent dir.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            obj = json.dumps(asdict(r))
            f.write(obj + "\n")


def load_judge_records(path: Path) -> list[JudgeRecord]:
    """Inverse of `write_judge_records`. Skips blank lines.

    Raises:
        FileNotFoundError if `path` is missing.
        JudgeError on malformed JSON or missing required fields.
    """
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    records: list[JudgeRecord] = []
    REQUIRED_FIELDS = [
        "query",
        "tier",
        "notes",
        "raw_output",
        "judge_model",
        "judge_prompt_hash",
        "timestamp",
    ]
    with path.open("r", encoding="utf-8") as f:
        for index, line in enumerate(f, start=1):
            stripped_line = line.strip()
            if not stripped_line:
                continue
            try:
                obj = json.loads(stripped_line)
                for field in REQUIRED_FIELDS:
                    if field not in obj:
                        raise JudgeError(f"Line No. {index}: missing field {field}")
                    if field == "tier" and obj[field] not in VALID_TIERS:
                        raise JudgeError(f"Line No. {index}: invalid tier {obj[field]}")
                records.append(
                    JudgeRecord(
                        query=obj["query"],
                        tier=obj["tier"],
                        notes=obj["notes"],
                        raw_output=obj["raw_output"],
                        judge_model=obj["judge_model"],
                        judge_prompt_hash=obj["judge_prompt_hash"],
                        timestamp=obj["timestamp"],
                    )
                )
            except json.JSONDecodeError as e:
                raise JudgeError(f"Line No. {index}: invalid json str") from e
    return records


# ------------------------------------------------------------------
# Helpers (private)
# ------------------------------------------------------------------


def _format_retrieved_block(chunks: list[Chunk], *, max_chars: int = 600) -> str:
    """Render top-K chunks as the retrieved_block placeholder content.

    Each chunk:
        [#<chunk_id>] <title>
        <text truncated to max_chars chars>
        ---
    Trailing `---` on every chunk, blank line between chunks. Truncating
    each chunk to ~600 chars caps the prompt at ~3-4k tokens for K=5
    chunks plus prose — keeps Haiku token cost predictable.
    """
    retrieved_block = [
        f"[#{chunk.chunk_id}] {chunk.title}\n{chunk.text[:max_chars]}\n" for chunk in chunks
    ]
    return "---\n".join(retrieved_block)


def _now_utc_iso() -> str:
    """Current UTC time, ISO 8601 — wrapped for monkeypatching in tests."""
    return datetime.now(timezone.utc).isoformat()


def _sha1_8(s: str) -> str:
    """SHA1 of `s` truncated to 8 hex chars."""
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:8]


# ------------------------------------------------------------------
# Re-exports for convenience
# ------------------------------------------------------------------

__all__ = [
    "DEFAULT_JUDGE_MODEL",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_TEMPERATURE",
    "JUDGE_PROMPT_TEMPLATE",
    "JudgeError",
    "JudgeRecord",
    "VALID_TIERS",
    "agreement_metrics",
    "judge_one",
    "judge_prompt_hash",
    "judge_records_to_human_scores",
    "load_judge_records",
    "make_judge_prompt",
    "parse_judge_response",
    "stratified_sample",
    "write_judge_records",
]


# Reference json import — used by parse_judge_response / load / write
# Listed here so the linter sees its purpose.
_ = json
