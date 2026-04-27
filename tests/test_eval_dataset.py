"""Tests for python_doc_assistant.evaluation.dataset.

Hermetic — JSONL written under tmp_path; no real eval files needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from python_doc_assistant.evaluation.dataset import (
    DEFAULT_MATCH_POLICY,
    DEFAULT_URL_MATCH,
    EvalQuery,
    EvalSchemaError,
    load_eval_set,
    parse_eval_query,
)

# ------------------------------------------------------------------
# Fixture builders
# ------------------------------------------------------------------


def _full_entry(**overrides: Any) -> dict[str, Any]:
    """A schema-valid entry; tests override specific fields to exercise edge cases."""
    base: dict[str, Any] = {
        "query": "Path vs os.path",
        "query_type": "comparison",
        "expected_symbols": ["pathlib.Path", "os.path"],
        "expected_urls": ["library/pathlib.html", "library/os.path.html"],
        "match_policy": "all",
        "url_match": "strip_anchor",
        "notes": "multi-hop / comparison",
    }
    base.update(overrides)
    return base


def _write_jsonl(path: Path, entries: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(e) for e in entries) + "\n",
        encoding="utf-8",
    )


# ------------------------------------------------------------------
# parse_eval_query — happy path
# ------------------------------------------------------------------


def test_parse_eval_query_full_schema() -> None:
    entry = _full_entry()
    q = parse_eval_query(entry)
    assert q == EvalQuery(
        query="Path vs os.path",
        query_type="comparison",
        expected_symbols=("pathlib.Path", "os.path"),
        expected_urls=("library/pathlib.html", "library/os.path.html"),
        match_policy="all",
        url_match="strip_anchor",
        notes="multi-hop / comparison",
    )


def test_parse_eval_query_applies_match_policy_default() -> None:
    entry = _full_entry()
    entry.pop("match_policy")
    q = parse_eval_query(entry)
    assert q.match_policy == DEFAULT_MATCH_POLICY


def test_parse_eval_query_applies_url_match_default() -> None:
    entry = _full_entry()
    entry.pop("url_match")
    q = parse_eval_query(entry)
    assert q.url_match == DEFAULT_URL_MATCH


def test_parse_eval_query_notes_optional() -> None:
    entry = _full_entry()
    entry.pop("notes")
    q = parse_eval_query(entry)
    assert q.notes is None


def test_parse_eval_query_out_of_scope_allows_empty_expected() -> None:
    """out_of_scope queries have no expected matches by definition."""
    entry = _full_entry(
        query_type="out_of_scope",
        expected_symbols=[],
        expected_urls=[],
    )
    q = parse_eval_query(entry)
    assert q.expected_symbols == ()
    assert q.expected_urls == ()


# ------------------------------------------------------------------
# parse_eval_query — required field violations
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    "missing_field",
    ["query", "query_type", "expected_symbols", "expected_urls"],
)
def test_parse_eval_query_missing_required_raises(missing_field: str) -> None:
    entry = _full_entry()
    entry.pop(missing_field)
    with pytest.raises(EvalSchemaError):
        parse_eval_query(entry)


# ------------------------------------------------------------------
# parse_eval_query — enum violations
# ------------------------------------------------------------------


def test_parse_eval_query_invalid_query_type_raises() -> None:
    with pytest.raises(EvalSchemaError):
        parse_eval_query(_full_entry(query_type="bogus"))


def test_parse_eval_query_invalid_match_policy_raises() -> None:
    with pytest.raises(EvalSchemaError):
        parse_eval_query(_full_entry(match_policy="maybe"))


def test_parse_eval_query_invalid_url_match_raises() -> None:
    with pytest.raises(EvalSchemaError):
        parse_eval_query(_full_entry(url_match="fuzzy"))


# ------------------------------------------------------------------
# parse_eval_query — type checks
# ------------------------------------------------------------------


def test_parse_eval_query_expected_symbols_must_be_list() -> None:
    with pytest.raises(EvalSchemaError):
        parse_eval_query(_full_entry(expected_symbols="pathlib.Path"))


def test_parse_eval_query_query_must_be_string() -> None:
    with pytest.raises(EvalSchemaError):
        parse_eval_query(_full_entry(query=42))


# ------------------------------------------------------------------
# load_eval_set
# ------------------------------------------------------------------


def test_load_eval_set_reads_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "v0.jsonl"
    _write_jsonl(path, [_full_entry(), _full_entry(query="another")])
    loaded = load_eval_set(path)
    assert len(loaded) == 2
    assert loaded[0].query == "Path vs os.path"
    assert loaded[1].query == "another"


def test_load_eval_set_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "v0.jsonl"
    path.write_text(
        f"{json.dumps(_full_entry())}\n\n\n{json.dumps(_full_entry(query='b'))}\n",
        encoding="utf-8",
    )
    loaded = load_eval_set(path)
    assert len(loaded) == 2


def test_load_eval_set_malformed_json_raises(tmp_path: Path) -> None:
    path = tmp_path / "v0.jsonl"
    path.write_text(
        f"{json.dumps(_full_entry())}\nNOT JSON\n",
        encoding="utf-8",
    )
    with pytest.raises(EvalSchemaError):
        load_eval_set(path)


def test_load_eval_set_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_eval_set(tmp_path / "missing.jsonl")


def test_load_eval_set_propagates_schema_error(tmp_path: Path) -> None:
    """A schema violation on a single line bubbles up as EvalSchemaError."""
    bad = _full_entry()
    bad.pop("query")
    path = tmp_path / "v0.jsonl"
    _write_jsonl(path, [_full_entry(), bad])
    with pytest.raises(EvalSchemaError):
        load_eval_set(path)
