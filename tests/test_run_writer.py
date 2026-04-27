"""Tests for python_doc_assistant.evaluation.run_writer.

Hermetic — runs written under tmp_path; timestamp pinned via monkeypatch.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from python_doc_assistant.evaluation.retrieval_metrics import (
    EvalRunResult,
    PerQueryResult,
    RetrievedChunk,
)
from python_doc_assistant.evaluation.run_writer import (
    PER_QUERY_JSONL_NAME,
    RESULTS_JSON_NAME,
    RunMetadata,
    RunWriterError,
    make_run_dir,
    write_run,
)

# ------------------------------------------------------------------
# Fixture builders
# ------------------------------------------------------------------


def _per_query() -> PerQueryResult:
    return PerQueryResult(
        query="pathlib.Path.read_text",
        query_type="identifier",
        match_policy="any",
        url_match="strip_anchor",
        expected_symbols=("pathlib.Path.read_text",),
        expected_urls=("library/pathlib.html",),
        retrieved=(
            RetrievedChunk(
                chunk_id="symbol:pathlib.Path.read_text",
                score=12.5,
                rank=1,
                canonical_url="library/pathlib.html#pathlib.Path.read_text",
                symbols=("pathlib.Path.read_text",),
            ),
        ),
        hit_at_5=True,
        hit_at_10=True,
        rank_for_mrr=1,
    )


def _run_result() -> EvalRunResult:
    pq = _per_query()
    return EvalRunResult(
        queries=(pq,),
        recall_at_5=1.0,
        recall_at_10=1.0,
        mrr=1.0,
        n_queries=1,
    )


def _metadata() -> RunMetadata:
    return RunMetadata(
        docs_version="3.12",
        docs_served_version="3.12.13",
        docs_sha_short="a5c1a35a5a02",
        ingest_manifest={
            "docs_version": "3.12",
            "docs_served_version": "3.12.13",
            "docs_url": "https://example/x.tar.bz2",
            "docs_archive_sha256": "a5c1a35a5a02..." + "0" * 50,
            "ingest_timestamp": "2026-04-25T09:55:36Z",
        },
        config={"retrieval_mode": "bm25+symbol", "k": 10, "eval_set": "v0_core_30.jsonl"},
        tag="v0-bm25",
        command="pdr eval --set eval_sets/v0_core_30.jsonl --tag v0-bm25",
    )


# ------------------------------------------------------------------
# make_run_dir
# ------------------------------------------------------------------


def test_make_run_dir_format(tmp_path: Path) -> None:
    fixed_now = datetime(2026, 4, 27, 15, 30, 45, tzinfo=timezone.utc)
    out = make_run_dir("v0-bm25", experiments_root=tmp_path, now=fixed_now)
    assert out == tmp_path / "2026-04-27T15-30-45-v0-bm25"


def test_make_run_dir_does_not_create(tmp_path: Path) -> None:
    """make_run_dir builds a Path but does NOT touch the filesystem."""
    fixed_now = datetime(2026, 4, 27, 15, 30, 45, tzinfo=timezone.utc)
    out = make_run_dir("test", experiments_root=tmp_path, now=fixed_now)
    assert not out.exists()


# ------------------------------------------------------------------
# write_run — file shape
# ------------------------------------------------------------------


def test_write_run_creates_results_json_and_per_query_jsonl(tmp_path: Path) -> None:
    out_dir = tmp_path / "run-1"
    write_run(out_dir, _run_result(), _metadata())
    assert (out_dir / RESULTS_JSON_NAME).is_file()
    assert (out_dir / PER_QUERY_JSONL_NAME).is_file()


def test_write_run_results_json_includes_metadata(tmp_path: Path) -> None:
    out_dir = tmp_path / "run-1"
    write_run(out_dir, _run_result(), _metadata())
    data = json.loads((out_dir / RESULTS_JSON_NAME).read_text(encoding="utf-8"))
    # Reproducibility fields per PLAN.md §4 / AGENTS.md §Eval Rules
    assert data["docs_version"] == "3.12"
    assert data["docs_served_version"] == "3.12.13"
    assert data["docs_sha_short"] == "a5c1a35a5a02"
    assert "ingest_manifest" in data
    assert data["tag"] == "v0-bm25"
    # Metrics
    assert data["recall_at_5"] == 1.0
    assert data["recall_at_10"] == 1.0
    assert data["mrr"] == 1.0
    assert data["n_queries"] == 1


def test_write_run_per_query_jsonl_has_one_line_per_query(tmp_path: Path) -> None:
    out_dir = tmp_path / "run-1"
    write_run(out_dir, _run_result(), _metadata())
    lines = (out_dir / PER_QUERY_JSONL_NAME).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    pq = json.loads(lines[0])
    assert pq["query"] == "pathlib.Path.read_text"
    assert pq["hit_at_5"] is True
    assert pq["rank_for_mrr"] == 1
    # Retrieved chunks serialized
    assert len(pq["retrieved"]) == 1
    assert pq["retrieved"][0]["chunk_id"] == "symbol:pathlib.Path.read_text"


# ------------------------------------------------------------------
# write_run — overwrite protection (plan §9)
# ------------------------------------------------------------------


def test_write_run_refuses_existing_dir(tmp_path: Path) -> None:
    out_dir = tmp_path / "run-1"
    out_dir.mkdir()
    with pytest.raises(RunWriterError):
        write_run(out_dir, _run_result(), _metadata())


def test_write_run_overwrite_force(tmp_path: Path) -> None:
    out_dir = tmp_path / "run-1"
    out_dir.mkdir()
    (out_dir / "leftover.txt").write_text("old", encoding="utf-8")
    write_run(out_dir, _run_result(), _metadata(), overwrite=True)
    assert (out_dir / RESULTS_JSON_NAME).is_file()
