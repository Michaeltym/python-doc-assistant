"""Human scoring helpers for v1 §6.

Workflow (manual; no CLI subcommand):

    experiments/runs/<ts>-<tag>/
        results.json        (written by `pdr eval`)
        per_query.jsonl     (written by `pdr eval`)
        human_scores.jsonl  (hand-written by reviewer — see schema below)

human_scores.jsonl schema (one JSON object per line):

    {
      "query": "<query string copied from per_query.jsonl>",
      "tier":  "correct" | "partial" | "wrong" | "hallucination" | "refused",
      "notes": "<optional reviewer note>"
    }

Tier definitions (plan §6 + this module):
    correct       — facts correct AND citations correct
    partial       — answer correct in substance but missing details OR
                    wrong/missing citation
    wrong         — factually wrong (regardless of citation)
    hallucination — uses content NOT present in the chunks (most severe)
    refused       — model emitted the [INSUFFICIENT-CONTEXT] marker
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

VALID_TIERS: Final[frozenset[str]] = frozenset(
    {"correct", "partial", "wrong", "hallucination", "refused"}
)


# ------------------------------------------------------------------
# Data records
# ------------------------------------------------------------------


@dataclass(frozen=True)
class HumanScore:
    """One line of human_scores.jsonl, parsed."""

    query: str  # join key against per_query.jsonl
    tier: str  # one of VALID_TIERS
    notes: str | None = None


class HumanScoreError(Exception):
    """Schema violation in human_scores.jsonl."""


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def load_human_scores(path: Path) -> list[HumanScore]:
    """Read JSONL at `path` into a list of HumanScore records.

    Validation:
        - each line must be a JSON object with `query` (non-empty str) and
          `tier` (str in VALID_TIERS).
        - `notes` is optional (str | null).
        - blank lines are skipped (so trailing newlines don't break parsing).
        - `query` strings must be unique across the file (raise on duplicate).

    Raises:
        HumanScoreError on any schema violation (missing key, wrong type,
        invalid tier, duplicate query).
    """
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    human_scores: dict[str, HumanScore] = {}
    with path.open("r", encoding="utf-8") as f:
        for index, line in enumerate(f, start=1):
            stripped_line = line.strip()
            if not stripped_line:
                continue
            obj = _parse_jsonl_line(stripped_line, line_no=index)
            query = obj.get("query")
            tier = obj.get("tier")
            notes = obj.get("notes")
            if not query:
                raise HumanScoreError(f"line {index}: invalid query")
            if human_scores.get(query):
                raise HumanScoreError(f"line {index}: duplicated query")
            if not tier or tier not in VALID_TIERS:
                raise HumanScoreError(f"line {index}: invalid tier")
            human_scores[query] = HumanScore(query=query, tier=tier, notes=notes)
    return list(human_scores.values())


def aggregate(scores: list[HumanScore]) -> dict[str, Any]:
    """Compute tier counts + headline rates.

    Returns a dict with these keys (all tiers always present, even when
    count is zero — keeps downstream consumers from needing `.get` defaults):

        {
            "n": int,
            "tier_counts": {tier: int, ...},
            "tier_rates":  {tier: float, ...},     # tier_count / n; 0 when n==0
            "hallucination_rate": float,           # tier_counts["hallucination"] / n
            "correct_rate":       float,           # tier_counts["correct"] / n
        }

    `hallucination_rate` and `correct_rate` are surfaced top-level because
    they're plan §6 / §8 headline numbers (hallucination_rate < 10 % is the
    v1 completion target).
    """
    n = len(scores)
    tier_counts = {
        "correct": 0,
        "partial": 0,
        "wrong": 0,
        "hallucination": 0,
        "refused": 0,
    }
    for s in scores:
        tier_counts[s.tier] += 1
    return {
        "n": n,
        "tier_counts": tier_counts,
        "tier_rates": {k: v / n if n > 0 else 0.0 for k, v in tier_counts.items()},
        "hallucination_rate": tier_counts["hallucination"] / n if n > 0 else 0.0,
        "correct_rate": tier_counts["correct"] / n if n > 0 else 0.0,
    }


# ------------------------------------------------------------------
# Helpers (private)
# ------------------------------------------------------------------


def _parse_jsonl_line(raw: str, *, line_no: int) -> dict[str, Any]:
    """Parse one JSONL line; HumanScoreError on JSON error or non-object."""
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HumanScoreError(f"line {line_no}: invalid JSON ({exc.msg})") from exc
    if not isinstance(obj, dict):
        raise HumanScoreError(f"line {line_no}: expected JSON object, got {type(obj).__name__}")
    return obj
