"""Tests for python_doc_assistant.evaluation.human_scoring."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from python_doc_assistant.evaluation.human_scoring import (
    VALID_TIERS,
    HumanScore,
    HumanScoreError,
    aggregate,
    load_human_scores,
)

# ------------------------------------------------------------------
# Fixture builders
# ------------------------------------------------------------------


def _write_jsonl(path: Path, records: list[dict]) -> None:
    """Write JSONL fixture; one record per line."""
    path.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
    )


# ------------------------------------------------------------------
# Constants + dataclass
# ------------------------------------------------------------------


def test_valid_tiers_exact_set() -> None:
    """Plan §6 4-tier + 'refused' for the marker-emitting case."""
    assert VALID_TIERS == frozenset(
        {"correct", "partial", "wrong", "hallucination", "refused"}
    )


def test_human_score_is_frozen() -> None:
    s = HumanScore(query="q", tier="correct")
    with pytest.raises(Exception):
        s.tier = "wrong"  # type: ignore[misc]


def test_human_score_notes_optional() -> None:
    assert HumanScore(query="q", tier="correct").notes is None
    s = HumanScore(query="q", tier="correct", notes="ok")
    assert s.notes == "ok"


# ------------------------------------------------------------------
# load_human_scores
# ------------------------------------------------------------------


def test_load_human_scores_valid(tmp_path: Path) -> None:
    p = tmp_path / "scores.jsonl"
    _write_jsonl(
        p,
        [
            {"query": "q1", "tier": "correct"},
            {"query": "q2", "tier": "wrong", "notes": "missed citation"},
        ],
    )
    scores = load_human_scores(p)
    assert scores == [
        HumanScore(query="q1", tier="correct"),
        HumanScore(query="q2", tier="wrong", notes="missed citation"),
    ]


def test_load_human_scores_invalid_tier_raises(tmp_path: Path) -> None:
    p = tmp_path / "scores.jsonl"
    _write_jsonl(p, [{"query": "q", "tier": "almost"}])
    with pytest.raises(HumanScoreError, match="tier"):
        load_human_scores(p)


def test_load_human_scores_missing_query_raises(tmp_path: Path) -> None:
    p = tmp_path / "scores.jsonl"
    _write_jsonl(p, [{"tier": "correct"}])
    with pytest.raises(HumanScoreError):
        load_human_scores(p)


def test_load_human_scores_missing_tier_raises(tmp_path: Path) -> None:
    p = tmp_path / "scores.jsonl"
    _write_jsonl(p, [{"query": "q"}])
    with pytest.raises(HumanScoreError):
        load_human_scores(p)


def test_load_human_scores_duplicate_query_raises(tmp_path: Path) -> None:
    p = tmp_path / "scores.jsonl"
    _write_jsonl(
        p,
        [
            {"query": "q", "tier": "correct"},
            {"query": "q", "tier": "wrong"},
        ],
    )
    with pytest.raises(HumanScoreError, match="duplicate"):
        load_human_scores(p)


def test_load_human_scores_skips_blank_lines(tmp_path: Path) -> None:
    """Trailing or blank lines must not error (common when files are hand-edited)."""
    p = tmp_path / "scores.jsonl"
    p.write_text(
        json.dumps({"query": "q1", "tier": "correct"})
        + "\n\n"
        + json.dumps({"query": "q2", "tier": "partial"})
        + "\n",
        encoding="utf-8",
    )
    scores = load_human_scores(p)
    assert len(scores) == 2


def test_load_human_scores_empty_query_raises(tmp_path: Path) -> None:
    """Empty query string is not a valid join key."""
    p = tmp_path / "scores.jsonl"
    _write_jsonl(p, [{"query": "", "tier": "correct"}])
    with pytest.raises(HumanScoreError):
        load_human_scores(p)


def test_load_human_scores_invalid_json_raises(tmp_path: Path) -> None:
    p = tmp_path / "scores.jsonl"
    p.write_text("{not valid json\n", encoding="utf-8")
    with pytest.raises(HumanScoreError):
        load_human_scores(p)


# ------------------------------------------------------------------
# aggregate
# ------------------------------------------------------------------


def test_aggregate_empty_scores() -> None:
    out = aggregate([])
    assert out["n"] == 0
    assert all(out["tier_counts"][t] == 0 for t in VALID_TIERS)
    assert out["hallucination_rate"] == 0.0
    assert out["correct_rate"] == 0.0


def test_aggregate_counts_and_rates() -> None:
    scores = [
        HumanScore("q1", "correct"),
        HumanScore("q2", "correct"),
        HumanScore("q3", "wrong"),
        HumanScore("q4", "hallucination"),
    ]
    out = aggregate(scores)
    assert out["n"] == 4
    assert out["tier_counts"]["correct"] == 2
    assert out["tier_counts"]["wrong"] == 1
    assert out["tier_counts"]["hallucination"] == 1
    assert out["tier_counts"]["partial"] == 0
    assert out["tier_counts"]["refused"] == 0
    assert out["correct_rate"] == 0.5
    assert out["hallucination_rate"] == 0.25


def test_aggregate_includes_all_tiers_in_dicts() -> None:
    """Even tiers with 0 count must appear in tier_counts/tier_rates."""
    out = aggregate([HumanScore("q", "correct")])
    for tier in VALID_TIERS:
        assert tier in out["tier_counts"]
        assert tier in out["tier_rates"]


def test_aggregate_rates_sum_to_one() -> None:
    """Sanity: tier_rates should sum to 1.0 when n > 0."""
    scores = [
        HumanScore(f"q{i}", t)
        for i, t in enumerate(["correct", "partial", "wrong", "hallucination", "refused"])
    ]
    out = aggregate(scores)
    total = sum(out["tier_rates"].values())
    assert abs(total - 1.0) < 1e-9
