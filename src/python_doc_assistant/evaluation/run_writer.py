"""Write a single eval run to experiments/runs/<timestamp>-<tag>/.

See plans/v0-retrieval-eval.md §9 and PLAN.md §8 §10 for the contract.

Writes two files inside the run dir:
    results.json      — aggregate metrics + config + manifest snapshot
    per_query.jsonl   — one line per EvalQuery: retrieved chunks + hit detail

Default behavior refuses to overwrite an existing run dir (plan §9).
Pass overwrite=True to force.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

from python_doc_assistant.evaluation.retrieval_metrics import EvalRunResult

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

DEFAULT_EXPERIMENTS_ROOT: Final[Path] = Path("experiments/runs")
RESULTS_JSON_NAME: Final[str] = "results.json"
PER_QUERY_JSONL_NAME: Final[str] = "per_query.jsonl"


# ------------------------------------------------------------------
# Exceptions
# ------------------------------------------------------------------


class RunWriterError(Exception):
    """Run directory already exists and overwrite=False, or other write failure."""


# ------------------------------------------------------------------
# Data records
# ------------------------------------------------------------------


@dataclass(frozen=True)
class RunMetadata:
    """Reproducibility fields written into results.json (PLAN.md §4 / AGENTS.md §Eval Rules)."""

    docs_version: str  # "3.12"
    docs_served_version: str  # "3.12.13"
    docs_sha_short: str  # 12-char sha
    ingest_manifest: dict[str, Any]  # full manifest snapshot
    config: dict[str, Any]  # retrieval mode, k, eval set path, etc.
    tag: str  # run tag (e.g. "v0-bm25")
    command: str  # invocation, e.g. "pdr eval --set ... --tag ..."


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def write_run(
    out_dir: Path,
    run_result: EvalRunResult,
    metadata: RunMetadata,
    *,
    overwrite: bool = False,
) -> None:
    """Write results.json + per_query.jsonl into `out_dir`.

    Refuses to write into an existing directory unless overwrite=True
    (plan §9 overwrite-protection rule).
    """
    _ensure_writable(out_dir, overwrite)
    out_dir.mkdir(parents=True, exist_ok=overwrite)
    _serialize_results_json(run_result, metadata, out_dir / RESULTS_JSON_NAME)
    _serialize_per_query(run_result, out_dir / PER_QUERY_JSONL_NAME)


def make_run_dir(
    tag: str,
    *,
    experiments_root: Path | None = None,
    now: datetime | None = None,
) -> Path:
    """Build a `<root>/<YYYY-MM-DDTHH-MM-SS>-<tag>/` Path.

    `now` is overridable for tests. The path is computed but NOT created on disk.
    """
    root = experiments_root if experiments_root is not None else DEFAULT_EXPERIMENTS_ROOT
    ts = _format_timestamp(now or _utc_now())
    return root / f"{ts}-{tag}"


# ------------------------------------------------------------------
# Helpers (private)
# ------------------------------------------------------------------


def _format_timestamp(dt: datetime) -> str:
    """ISO 8601 to-second precision with `-` separators inside the time portion.

    Example: 2026-04-27T15-30-45 (filesystem-safe; no `:` because Windows balks).
    """
    return dt.strftime("%Y-%m-%dT%H-%M-%S")


def _ensure_writable(out_dir: Path, overwrite: bool) -> None:
    """Raise RunWriterError if out_dir exists and overwrite=False."""
    if out_dir.exists() and not overwrite:
        raise RunWriterError(f"{out_dir} already exists; pass overwrite=True to force")


def _serialize_per_query(run_result: EvalRunResult, path: Path) -> None:
    """Write one JSON object per line: PerQueryResult fields."""
    with path.open("w", encoding="utf-8") as f:
        for query in run_result.queries:
            q = asdict(query)
            f.write(json.dumps(q) + "\n")


def _serialize_results_json(run_result: EvalRunResult, metadata: RunMetadata, path: Path) -> None:
    """Write aggregate metrics + metadata as a single JSON object."""
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                **asdict(metadata),
                "recall_at_5": run_result.recall_at_5,
                "recall_at_10": run_result.recall_at_10,
                "mrr": run_result.mrr,
                "n_queries": run_result.n_queries,
            },
            f,
            indent=2,
        )


def _utc_now() -> datetime:
    """Current UTC time. Wrapped for monkeypatching in tests."""
    return datetime.now(timezone.utc)
