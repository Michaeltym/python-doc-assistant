"""Eval set schema + JSONL loader.

See plans/v0-retrieval-eval.md §8 and PLAN.md §8 for the contract.

A v0 eval set is a JSONL file. Each line is one EvalQuery:

    {
      "query": "Path vs os.path",
      "query_type": "comparison",
      "expected_symbols": ["pathlib.Path", "os.path"],
      "expected_urls": ["library/pathlib.html", "library/os.path.html"],
      "match_policy": "all",       # optional, default "any"
      "url_match": "strip_anchor", # optional, default "strip_anchor"
      "notes": "multi-hop"         # optional
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

# ------------------------------------------------------------------
# Enums (validated by parse_eval_query)
# ------------------------------------------------------------------

QUERY_TYPES: Final[frozenset[str]] = frozenset(
    {"identifier", "natural_language", "comparison", "howto", "out_of_scope"}
)
MATCH_POLICIES: Final[frozenset[str]] = frozenset({"any", "all"})
URL_MATCHES: Final[frozenset[str]] = frozenset({"exact", "strip_anchor", "prefix"})

DEFAULT_MATCH_POLICY: Final[str] = "any"
DEFAULT_URL_MATCH: Final[str] = "strip_anchor"
REQUIRED: Final[list[str]] = ["query", "query_type", "expected_symbols", "expected_urls"]


# ------------------------------------------------------------------
# Exceptions
# ------------------------------------------------------------------


class EvalSchemaError(Exception):
    """An eval entry violates the schema (missing field / bad enum / wrong type)."""


# ------------------------------------------------------------------
# Data records
# ------------------------------------------------------------------


@dataclass(frozen=True)
class EvalQuery:
    """One row from an eval JSONL set (PLAN.md §8 schema)."""

    query: str
    query_type: str  # one of QUERY_TYPES
    expected_symbols: tuple[str, ...]  # may be empty for out_of_scope
    expected_urls: tuple[str, ...]  # may be empty for out_of_scope
    match_policy: str = DEFAULT_MATCH_POLICY  # one of MATCH_POLICIES
    url_match: str = DEFAULT_URL_MATCH  # one of URL_MATCHES
    notes: str | None = None


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def parse_eval_query(data: dict[str, Any]) -> EvalQuery:
    """Validate `data` against the schema and return an EvalQuery.

    Required keys: query, query_type, expected_symbols, expected_urls.
    Optional keys: match_policy, url_match, notes.

    Raises EvalSchemaError with a descriptive message on any violation.
    """
    for key in REQUIRED:
        if key not in data:
            raise EvalSchemaError(f"Missing required field {key}")
    query = data["query"]
    query_type = data["query_type"]
    expected_symbols = data["expected_symbols"]
    expected_urls = data["expected_urls"]
    match_policy = data.get("match_policy", DEFAULT_MATCH_POLICY)
    url_match = data.get("url_match", DEFAULT_URL_MATCH)
    notes = data.get("notes")
    _validate_str("query", query)
    _validate_str("query_type", query_type)
    _validate_list_str("expected_symbols", expected_symbols)
    _validate_list_str("expected_urls", expected_urls)
    _validate_str("match_policy", match_policy)
    _validate_str("url_match", url_match)
    _validate_str("notes", notes)
    _validate_value("query_type", query_type, QUERY_TYPES)
    _validate_value("match_policy", match_policy, MATCH_POLICIES)
    _validate_value("url_match", url_match, URL_MATCHES)
    return EvalQuery(
        query=query,
        query_type=query_type,
        expected_symbols=tuple(expected_symbols),
        expected_urls=tuple(expected_urls),
        match_policy=match_policy,
        url_match=url_match,
        notes=notes,
    )


def load_eval_set(path: Path) -> list[EvalQuery]:
    """Read a JSONL eval set from `path` and return a list of EvalQuery records.

    - Skips blank lines.
    - Raises EvalSchemaError on malformed JSON or schema violations
      (with the offending line number in the message).
    - Raises FileNotFoundError if `path` does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    eval_queries: list[EvalQuery] = []
    with path.open(encoding="utf-8") as f:
        for index, line in enumerate(f, start=1):
            striped_line = line.strip()
            if not striped_line:
                continue
            try:
                data = json.loads(striped_line)
            except json.JSONDecodeError as e:
                raise EvalSchemaError(f"line {index}: fail to load data") from e
            try:
                eval_query = parse_eval_query(data)
                eval_queries.append(eval_query)
            except EvalSchemaError as e:
                raise EvalSchemaError(f"line {index}: {e}") from e
    return eval_queries


def _validate_str(key: str, value: Any) -> None:
    if value is None:
        return
    if not isinstance(value, str):
        raise EvalSchemaError(f"Invalid {key} type, expect str")


def _validate_list_str(key: str, value: Any) -> None:
    if not isinstance(value, list) or any(not isinstance(s, str) for s in value):
        raise EvalSchemaError(f"Invalid {key} type, expect list[str]")


def _validate_value(key: str, value: Any, allowed: frozenset[str]) -> None:
    if value is None:
        return
    if value not in allowed:
        raise EvalSchemaError(f"Invalid {key}: {value}, expect one of {', '.join(allowed)}")
