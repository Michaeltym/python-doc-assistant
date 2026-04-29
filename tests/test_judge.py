"""Tests for python_doc_assistant.evaluation.judge (plan v2 §6).

Hermetic — no real Anthropic API calls. A `_FakeAnthropicClient` returns
canned JSON strings so the prompt → API → parse → JudgeRecord pipeline
is exercised end-to-end without network or API key.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from python_doc_assistant.evaluation.human_scoring import HumanScore
from python_doc_assistant.evaluation.judge import (
    DEFAULT_JUDGE_MODEL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    JUDGE_PROMPT_TEMPLATE,
    JudgeError,
    JudgeRecord,
    agreement_metrics,
    judge_one,
    judge_prompt_hash,
    judge_records_to_human_scores,
    load_judge_records,
    make_judge_prompt,
    parse_judge_response,
    stratified_sample,
    write_judge_records,
)
from python_doc_assistant.ingest.chunker import Chunk

# ------------------------------------------------------------------
# Fixture builders
# ------------------------------------------------------------------


def _chunk(chunk_id: str, *, title: str | None = None, text: str = "BODY") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        chunk_type="symbol",
        docs_version="3.12",
        title=title or chunk_id,
        text=text,
        symbols=(chunk_id,),
        canonical_url=f"library/foo.html#{chunk_id}",
        anchor=chunk_id,
        parent_module=None,
        source_path="library/foo.html",
        source_hash="sha256:abc",
    )


class _FakeContentBlock:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeUsage:
    def __init__(self, input_tokens: int = 100, output_tokens: int = 30) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.content = [_FakeContentBlock(text)]
        self.usage = _FakeUsage()


class _FakeMessages:
    """Stub for `anthropic.Anthropic().messages` — just `.create()`."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> _FakeResponse:
        self.calls.append(kwargs)
        text = self._replies.pop(0) if self._replies else "{}"
        return _FakeResponse(text)


class _FakeClient:
    """Stub for `anthropic.Anthropic()`."""

    def __init__(self, replies: list[str]) -> None:
        self.messages = _FakeMessages(replies)


# ------------------------------------------------------------------
# Constants + dataclass
# ------------------------------------------------------------------


def test_default_constants() -> None:
    assert DEFAULT_JUDGE_MODEL == "claude-haiku-4-5-20251001"
    assert DEFAULT_TEMPERATURE == 0.0
    assert DEFAULT_MAX_TOKENS == 200


def test_judge_record_is_frozen() -> None:
    r = JudgeRecord(
        query="q",
        tier="correct",
        notes="ok",
        raw_output="raw",
        judge_model="m",
        judge_prompt_hash="h",
        timestamp="2026-04-29T00:00:00+00:00",
    )
    with pytest.raises(Exception):
        r.tier = "wrong"  # type: ignore[misc]


def test_judge_prompt_hash_stable() -> None:
    """Same template → same hash; calling twice yields same value."""
    h1 = judge_prompt_hash()
    h2 = judge_prompt_hash()
    assert h1 == h2
    # 8 hex chars
    assert len(h1) == 8
    assert all(c in "0123456789abcdef" for c in h1)


def test_judge_prompt_hash_changes_when_template_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hash must reflect the live template — sanity for the reproducibility claim."""
    h_orig = judge_prompt_hash()
    monkeypatch.setattr(
        "python_doc_assistant.evaluation.judge.JUDGE_PROMPT_TEMPLATE",
        JUDGE_PROMPT_TEMPLATE + " EXTRA",
    )
    h_new = judge_prompt_hash()
    assert h_orig != h_new


# ------------------------------------------------------------------
# make_judge_prompt
# ------------------------------------------------------------------


def test_make_judge_prompt_fills_required_fields() -> None:
    prompt = make_judge_prompt(
        query="how to read a file",
        expected_symbols=("pathlib.Path.read_text",),
        retrieved_chunks=[_chunk("symbol:pathlib.Path.read_text", text="Read file")],
        cited_chunk_ids=("symbol:pathlib.Path.read_text",),
        refused=False,
        model_output_text="Use Path.read_text [1].",
    )
    assert "how to read a file" in prompt
    assert "pathlib.Path.read_text" in prompt
    assert "Use Path.read_text [1]." in prompt
    # Tier definitions must be in the prompt (helps Claude classify)
    assert "correct" in prompt
    assert "hallucination" in prompt
    assert "refused" in prompt


def test_make_judge_prompt_includes_retrieved_chunk_text() -> None:
    """Judge needs chunk TEXT (not just ids) to verify grounding."""
    prompt = make_judge_prompt(
        query="q",
        expected_symbols=(),
        retrieved_chunks=[_chunk("c1", title="C1", text="distinctive_chunk_body_xyz")],
        cited_chunk_ids=(),
        refused=False,
        model_output_text="some prose",
    )
    assert "distinctive_chunk_body_xyz" in prompt
    assert "C1" in prompt


# ------------------------------------------------------------------
# parse_judge_response
# ------------------------------------------------------------------


def test_parse_judge_response_basic() -> None:
    raw = '{"tier": "correct", "reason": "facts and citation match"}'
    tier, reason = parse_judge_response(raw)
    assert tier == "correct"
    assert reason == "facts and citation match"


def test_parse_judge_response_strips_whitespace_and_fences() -> None:
    """Tolerate leading/trailing whitespace and code fences."""
    raw = '```json\n{"tier": "partial", "reason": "no cite"}\n```'
    tier, reason = parse_judge_response(raw)
    assert tier == "partial"
    assert reason == "no cite"


def test_parse_judge_response_invalid_json_raises() -> None:
    with pytest.raises(JudgeError):
        parse_judge_response("{not valid json")


def test_parse_judge_response_unknown_tier_raises() -> None:
    raw = '{"tier": "almost", "reason": "..."}'
    with pytest.raises(JudgeError, match="tier"):
        parse_judge_response(raw)


def test_parse_judge_response_missing_tier_raises() -> None:
    raw = '{"reason": "no tier"}'
    with pytest.raises(JudgeError):
        parse_judge_response(raw)


def test_parse_judge_response_missing_reason_raises() -> None:
    raw = '{"tier": "correct"}'
    with pytest.raises(JudgeError):
        parse_judge_response(raw)


# ------------------------------------------------------------------
# judge_one
# ------------------------------------------------------------------


def test_judge_one_builds_record_with_metadata() -> None:
    client = _FakeClient(['{"tier": "correct", "reason": "good cite"}'])
    record = judge_one(
        query="q1",
        expected_symbols=("foo",),
        retrieved_chunks=[_chunk("symbol:foo")],
        cited_chunk_ids=("symbol:foo",),
        refused=False,
        model_output_text="Use foo() [1].",
        client=client,
    )
    assert isinstance(record, JudgeRecord)
    assert record.query == "q1"
    assert record.tier == "correct"
    assert record.notes == "good cite"
    # Reproducibility metadata
    assert record.judge_model == DEFAULT_JUDGE_MODEL
    assert len(record.judge_prompt_hash) == 8
    assert record.timestamp  # non-empty ISO 8601
    assert record.raw_output == '{"tier": "correct", "reason": "good cite"}'


def test_judge_one_passes_decoding_params_to_client() -> None:
    client = _FakeClient(['{"tier": "correct", "reason": "ok"}'])
    judge_one(
        query="q",
        expected_symbols=(),
        retrieved_chunks=[_chunk("c1")],
        cited_chunk_ids=(),
        refused=False,
        model_output_text="x",
        client=client,
        model_id="claude-foo",
        temperature=0.5,
        max_tokens=100,
    )
    assert len(client.messages.calls) == 1
    call = client.messages.calls[0]
    assert call["model"] == "claude-foo"
    assert call["temperature"] == 0.5
    assert call["max_tokens"] == 100


def test_judge_one_uses_content_key_in_messages() -> None:
    """messages[0] must use 'content' (Anthropic API contract), not 'prompt'.

    Locks in the API schema requirement: a {role: 'user', prompt: '...'} payload
    triggers a 400 BadRequestError in production. Tests with a permissive stub
    that only records kwargs do not catch this — explicit field check needed.
    """
    client = _FakeClient(['{"tier": "correct", "reason": "ok"}'])
    judge_one(
        query="q",
        expected_symbols=(),
        retrieved_chunks=[_chunk("c1")],
        cited_chunk_ids=(),
        refused=False,
        model_output_text="x",
        client=client,
    )
    call = client.messages.calls[0]
    msg = call["messages"][0]
    assert msg["role"] == "user"
    assert "content" in msg, f"expected 'content' key, got {list(msg.keys())}"
    assert "prompt" not in msg, "Anthropic API uses 'content', not 'prompt'"
    assert isinstance(msg["content"], str)
    assert len(msg["content"]) > 0


def test_judge_one_propagates_parse_errors() -> None:
    client = _FakeClient(["not JSON at all"])
    with pytest.raises(JudgeError):
        judge_one(
            query="q",
            expected_symbols=(),
            retrieved_chunks=[_chunk("c1")],
            cited_chunk_ids=(),
            refused=False,
            model_output_text="x",
            client=client,
        )


# ------------------------------------------------------------------
# stratified_sample
# ------------------------------------------------------------------


def test_stratified_sample_distributes_across_tiers() -> None:
    """Each tier with non-zero count should appear in the sample."""
    pool = (
        [HumanScore(f"c{i}", "correct") for i in range(10)]
        + [HumanScore(f"p{i}", "partial") for i in range(10)]
        + [HumanScore(f"w{i}", "wrong") for i in range(5)]
        + [HumanScore(f"h{i}", "hallucination") for i in range(5)]
        + [HumanScore(f"r{i}", "refused") for i in range(2)]
    )
    sampled_queries = stratified_sample(pool, n=20, seed=42)
    assert len(sampled_queries) <= 20

    sampled_tiers = {q[0] for q in sampled_queries}  # first char encodes tier
    # Every tier with members in pool should be represented at least once
    assert "c" in sampled_tiers
    assert "p" in sampled_tiers
    assert "w" in sampled_tiers
    assert "h" in sampled_tiers
    assert "r" in sampled_tiers


def test_stratified_sample_seeded_is_deterministic() -> None:
    pool = [HumanScore(f"q{i}", "correct") for i in range(100)]
    a = stratified_sample(pool, n=10, seed=42)
    b = stratified_sample(pool, n=10, seed=42)
    assert a == b


def test_stratified_sample_handles_small_pool() -> None:
    """Asking for n=20 from pool of 5 returns all 5 — no over-sampling."""
    pool = [HumanScore(f"q{i}", "correct") for i in range(5)]
    sampled = stratified_sample(pool, n=20, seed=42)
    assert len(sampled) == 5


# ------------------------------------------------------------------
# agreement_metrics
# ------------------------------------------------------------------


def test_agreement_metrics_perfect_agreement() -> None:
    judge = {"q1": "correct", "q2": "wrong", "q3": "partial"}
    human = {"q1": "correct", "q2": "wrong", "q3": "partial"}
    out = agreement_metrics(judge, human)
    assert out["n"] == 3
    assert out["exact_match"] == 1.0
    assert out["cohen_kappa"] == pytest.approx(1.0)


def test_agreement_metrics_zero_agreement() -> None:
    judge = {"q1": "correct", "q2": "correct"}
    human = {"q1": "wrong", "q2": "hallucination"}
    out = agreement_metrics(judge, human)
    assert out["n"] == 2
    assert out["exact_match"] == 0.0
    assert out["cohen_kappa"] <= 0.0  # below chance


def test_agreement_metrics_partial() -> None:
    judge = {"q1": "correct", "q2": "wrong", "q3": "partial", "q4": "correct"}
    human = {"q1": "correct", "q2": "wrong", "q3": "wrong", "q4": "partial"}
    out = agreement_metrics(judge, human)
    assert out["n"] == 4
    assert out["exact_match"] == 0.5  # 2 of 4 match
    # kappa < exact_match because chance agreement is non-zero
    assert out["cohen_kappa"] <= out["exact_match"]


def test_agreement_metrics_uses_only_common_keys() -> None:
    """Queries present in only one map are ignored — sample/eval mismatch."""
    judge = {"q1": "correct", "q2": "wrong"}
    human = {"q2": "wrong", "q3": "partial"}
    out = agreement_metrics(judge, human)
    assert out["n"] == 1  # only q2 is in both
    assert out["exact_match"] == 1.0


def test_agreement_metrics_empty_returns_zeros() -> None:
    out = agreement_metrics({}, {})
    assert out["n"] == 0
    assert out["exact_match"] == 0.0
    assert out["cohen_kappa"] == 0.0


# ------------------------------------------------------------------
# judge_records_to_human_scores
# ------------------------------------------------------------------


def test_judge_records_to_human_scores_drops_metadata() -> None:
    records = [
        JudgeRecord(
            query="q1",
            tier="correct",
            notes="ok",
            raw_output="...",
            judge_model="m",
            judge_prompt_hash="h",
            timestamp="ts",
        ),
        JudgeRecord(
            query="q2",
            tier="partial",
            notes="no cite",
            raw_output="...",
            judge_model="m",
            judge_prompt_hash="h",
            timestamp="ts",
        ),
    ]
    out = judge_records_to_human_scores(records)
    assert all(isinstance(s, HumanScore) for s in out)
    assert [s.query for s in out] == ["q1", "q2"]
    assert [s.tier for s in out] == ["correct", "partial"]


# ------------------------------------------------------------------
# write / load round-trip
# ------------------------------------------------------------------


def test_write_load_round_trip(tmp_path: Path) -> None:
    records = [
        JudgeRecord(
            query="q1",
            tier="correct",
            notes="ok",
            raw_output='{"tier":"correct","reason":"ok"}',
            judge_model="claude-haiku-4-5-20251001",
            judge_prompt_hash="abc12345",
            timestamp="2026-04-29T12:00:00+00:00",
        ),
    ]
    path = tmp_path / "judge_scores.jsonl"
    write_judge_records(records, path)
    assert path.exists()
    loaded = load_judge_records(path)
    assert loaded == records


def test_write_load_round_trip_multiple_records(tmp_path: Path) -> None:
    """Multiple records must round-trip — each on its own line (JSONL contract).

    Locks in newline-separation: a single-record round-trip will pass even
    when write_judge_records concatenates without `\\n`, because the file
    parses as 1 row. This test catches that regression.
    """
    records = [
        JudgeRecord(
            query=f"q{i}",
            tier=t,
            notes=f"reason {i}",
            raw_output=f"raw{i}",
            judge_model="claude-haiku-4-5-20251001",
            judge_prompt_hash="abc12345",
            timestamp="2026-04-29T12:00:00+00:00",
        )
        for i, t in enumerate(["correct", "partial", "wrong"])
    ]
    path = tmp_path / "judge_scores.jsonl"
    write_judge_records(records, path)
    loaded = load_judge_records(path)
    assert len(loaded) == 3
    assert loaded == records
    # File on disk must have at least 2 newlines separating 3 records
    text = path.read_text(encoding="utf-8")
    assert text.count("\n") >= 2


def test_load_judge_records_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_judge_records(tmp_path / "missing.jsonl")


def test_load_judge_records_skips_blank_lines(tmp_path: Path) -> None:
    """Trailing blank line shouldn't error."""
    path = tmp_path / "judge_scores.jsonl"
    record_dict = {
        "query": "q1",
        "tier": "correct",
        "notes": "ok",
        "raw_output": "raw",
        "judge_model": "m",
        "judge_prompt_hash": "h",
        "timestamp": "ts",
    }
    path.write_text(json.dumps(record_dict) + "\n\n", encoding="utf-8")
    records = load_judge_records(path)
    assert len(records) == 1


def test_load_judge_records_invalid_tier_raises(tmp_path: Path) -> None:
    path = tmp_path / "judge_scores.jsonl"
    bad = {
        "query": "q1",
        "tier": "almost",
        "notes": "ok",
        "raw_output": "raw",
        "judge_model": "m",
        "judge_prompt_hash": "h",
        "timestamp": "ts",
    }
    path.write_text(json.dumps(bad) + "\n", encoding="utf-8")
    with pytest.raises(JudgeError):
        load_judge_records(path)
