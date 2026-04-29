"""Tests for python_doc_assistant.cli.

Hermetic — uses click's CliRunner; underlying ingest/parse/index calls are mocked
via monkeypatch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from python_doc_assistant.cli import (
    _load_chunks,
    _resolve_docs_sha,
    _resolve_docs_version,
    _save_chunks,
    main,
)
from python_doc_assistant.ingest.chunker import Chunk

CLI_MODULE = "python_doc_assistant.cli"


# ------------------------------------------------------------------
# Fixture builders
# ------------------------------------------------------------------


def _symbol_chunk(name: str) -> Chunk:
    return Chunk(
        chunk_id=f"symbol:{name}",
        chunk_type="symbol",
        docs_version="3.12",
        title=name.rsplit(".", 1)[-1],
        text=f"Stub doc for {name}",
        symbols=(name,),
        canonical_url=f"library/foo.html#{name}",
        anchor=name,
        parent_module=name.rsplit(".", 1)[0] if "." in name else None,
        source_path="library/foo.html",
        source_hash="sha256:abc",
    )


# ------------------------------------------------------------------
# _resolve_docs_version
# ------------------------------------------------------------------


def test_resolve_docs_version_override_wins(tmp_path: Path) -> None:
    cfg = tmp_path / "config.toml"
    cfg.write_text('DOCS_VERSION = "3.12"\n', encoding="utf-8")
    assert _resolve_docs_version("3.13", config_path=cfg) == "3.13"


def test_resolve_docs_version_reads_config(tmp_path: Path) -> None:
    cfg = tmp_path / "config.toml"
    cfg.write_text('DOCS_VERSION = "3.12"\n', encoding="utf-8")
    assert _resolve_docs_version(None, config_path=cfg) == "3.12"


def test_resolve_docs_version_missing_config_raises(tmp_path: Path) -> None:
    import click

    cfg = tmp_path / "missing.toml"
    with pytest.raises(click.ClickException):
        _resolve_docs_version(None, config_path=cfg)


# ------------------------------------------------------------------
# _resolve_docs_sha
# ------------------------------------------------------------------


def test_resolve_docs_sha_override_wins(tmp_path: Path) -> None:
    assert _resolve_docs_sha("3.12", "abcdef123456", data_root=tmp_path) == "abcdef123456"


def test_resolve_docs_sha_reads_current_txt(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    current = tmp_path / "docs" / "3.12" / "current.txt"
    current.parent.mkdir(parents=True)
    current.write_text("a1b2c3d4e5f6\n", encoding="utf-8")
    assert _resolve_docs_sha("3.12", None, data_root=tmp_path) == "a1b2c3d4e5f6"
    captured = capsys.readouterr()
    assert "a1b2c3d4e5f6" in captured.out  # plan §7 prints resolved sha


def test_resolve_docs_sha_no_source_raises(tmp_path: Path) -> None:
    import click

    with pytest.raises(click.ClickException):
        _resolve_docs_sha("3.12", None, data_root=tmp_path)


# ------------------------------------------------------------------
# _save_chunks + _load_chunks round-trip
# ------------------------------------------------------------------


def test_save_load_chunks_round_trip(tmp_path: Path) -> None:
    original = [
        _symbol_chunk("pathlib.Path"),
        _symbol_chunk("dict.fromkeys"),
    ]
    path = tmp_path / "chunks.jsonl"
    _save_chunks(original, path)
    assert path.is_file()

    restored = _load_chunks(path)
    assert restored == original


def test_save_chunks_creates_parent_dirs(tmp_path: Path) -> None:
    chunks = [_symbol_chunk("foo")]
    path = tmp_path / "deep" / "nested" / "chunks.jsonl"
    _save_chunks(chunks, path)
    assert path.is_file()


# ------------------------------------------------------------------
# CLI: ingest
# ------------------------------------------------------------------


def test_cli_ingest_invokes_ingest_docs(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class FakeManifest:
        docs_version = "3.12"
        docs_served_version = "3.12.13"

    class FakeResult:
        sha_short = "abcdef123456"
        docs_dir = Path("data/docs/3.12/abcdef123456")
        manifest_path = Path("data/docs/3.12/abcdef123456/ingest_manifest.json")
        manifest = FakeManifest()
        skipped = False

    def fake_ingest(version: str, *, force_switch: bool = False, **kwargs: Any) -> FakeResult:
        captured["version"] = version
        captured["force_switch"] = force_switch
        return FakeResult()

    monkeypatch.setattr(f"{CLI_MODULE}.ingest_docs", fake_ingest)

    runner = CliRunner()
    result = runner.invoke(main, ["ingest", "--version", "3.12"])
    assert result.exit_code == 0, result.output
    assert captured == {"version": "3.12", "force_switch": False}


def test_cli_ingest_force_switch_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_ingest(version: str, *, force_switch: bool = False, **kwargs: Any) -> Any:
        captured["force_switch"] = force_switch

        class R:
            sha_short = "x"
            docs_dir = Path(".")
            manifest_path = Path(".")
            skipped = False

        return R()

    monkeypatch.setattr(f"{CLI_MODULE}.ingest_docs", fake_ingest)
    runner = CliRunner()
    result = runner.invoke(main, ["ingest", "--version", "3.12", "--force-switch"])
    assert result.exit_code == 0, result.output
    assert captured["force_switch"] is True


# ------------------------------------------------------------------
# CLI: build-index
# ------------------------------------------------------------------


def test_cli_build_index_writes_artifacts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """build-index calls parse_objects_inv + build_chunks; writes chunks.jsonl + bm25.pkl."""
    chunks = [_symbol_chunk("pathlib.Path"), _symbol_chunk("pathlib.Path.read_text")]

    monkeypatch.setattr(f"{CLI_MODULE}.parse_objects_inv", lambda _docs_dir: [])
    monkeypatch.setattr(
        f"{CLI_MODULE}.build_chunks",
        lambda _dd, _ver, _syms: chunks,
    )

    # Pre-create config + current.txt under tmp_path so resolve helpers succeed.
    cfg = tmp_path / "config.toml"
    cfg.write_text('DOCS_VERSION = "3.12"\n', encoding="utf-8")
    current = tmp_path / "data" / "docs" / "3.12" / "current.txt"
    current.parent.mkdir(parents=True)
    current.write_text("abcdef123456\n", encoding="utf-8")

    monkeypatch.setattr(f"{CLI_MODULE}.DEFAULT_CONFIG_PATH", cfg)
    monkeypatch.setattr(f"{CLI_MODULE}.DEFAULT_DATA_ROOT", tmp_path / "data")

    runner = CliRunner()
    result = runner.invoke(main, ["build-index"])
    assert result.exit_code == 0, result.output

    chunks_path = tmp_path / "data" / "chunks" / "3.12" / "abcdef123456" / "chunks.jsonl"
    bm25_path = tmp_path / "data" / "indexes" / "3.12" / "abcdef123456" / "bm25.pkl"
    assert chunks_path.is_file()
    assert bm25_path.is_file()


# ------------------------------------------------------------------
# CLI: search
# ------------------------------------------------------------------


def test_cli_search_prints_chunk_ids(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """search should print one chunk_id per line for the routed result."""
    from python_doc_assistant.indexes.bm25_index import BM25Index

    # Need >= 5 chunks for BM25 IDF to be non-degenerate (small corpora produce <= 0 scores).
    chunks = [
        _symbol_chunk("pathlib.Path"),
        _symbol_chunk("pathlib.Path.read_text"),
        _symbol_chunk("pathlib.Path.write_text"),
        _symbol_chunk("dict.fromkeys"),
        _symbol_chunk("os.path.join"),
        _symbol_chunk("io.open"),
    ]

    cfg = tmp_path / "config.toml"
    cfg.write_text('DOCS_VERSION = "3.12"\n', encoding="utf-8")
    current = tmp_path / "data" / "docs" / "3.12" / "current.txt"
    current.parent.mkdir(parents=True)
    current.write_text("abcdef123456\n", encoding="utf-8")

    chunks_path = tmp_path / "data" / "chunks" / "3.12" / "abcdef123456" / "chunks.jsonl"
    chunks_path.parent.mkdir(parents=True)
    _save_chunks(chunks, chunks_path)

    bm25_path = tmp_path / "data" / "indexes" / "3.12" / "abcdef123456" / "bm25.pkl"
    BM25Index(chunks).save(bm25_path)

    monkeypatch.setattr(f"{CLI_MODULE}.parse_objects_inv", lambda _docs_dir: [])
    monkeypatch.setattr(f"{CLI_MODULE}.DEFAULT_CONFIG_PATH", cfg)
    monkeypatch.setattr(f"{CLI_MODULE}.DEFAULT_DATA_ROOT", tmp_path / "data")

    runner = CliRunner()
    result = runner.invoke(main, ["search", "Path.read_text", "--k", "3"])
    assert result.exit_code == 0, result.output
    # At least one chunk_id should print
    assert "symbol:pathlib.Path" in result.output


def test_cli_search_debug_emits_extra_info(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """--debug surfaces query_type / docs-sha / tokens."""
    from python_doc_assistant.indexes.bm25_index import BM25Index

    chunks = [_symbol_chunk("pathlib.Path")]
    cfg = tmp_path / "config.toml"
    cfg.write_text('DOCS_VERSION = "3.12"\n', encoding="utf-8")
    current = tmp_path / "data" / "docs" / "3.12" / "current.txt"
    current.parent.mkdir(parents=True)
    current.write_text("abcdef123456\n", encoding="utf-8")
    chunks_path = tmp_path / "data" / "chunks" / "3.12" / "abcdef123456" / "chunks.jsonl"
    chunks_path.parent.mkdir(parents=True)
    _save_chunks(chunks, chunks_path)
    bm25_path = tmp_path / "data" / "indexes" / "3.12" / "abcdef123456" / "bm25.pkl"
    BM25Index(chunks).save(bm25_path)

    monkeypatch.setattr(f"{CLI_MODULE}.parse_objects_inv", lambda _docs_dir: [])
    monkeypatch.setattr(f"{CLI_MODULE}.DEFAULT_CONFIG_PATH", cfg)
    monkeypatch.setattr(f"{CLI_MODULE}.DEFAULT_DATA_ROOT", tmp_path / "data")

    runner = CliRunner()
    result = runner.invoke(main, ["search", "pathlib.Path", "--debug"])
    assert result.exit_code == 0, result.output
    assert "query_type" in result.output
    assert "docs_sha" in result.output or "abcdef" in result.output


# ------------------------------------------------------------------
# CLI: eval (plan §9)
# ------------------------------------------------------------------


def _eval_set_jsonl(path: Path) -> None:
    """Write a tiny valid eval set for CLI tests."""
    path.write_text(
        '{"query": "pathlib.Path.read_text", "query_type": "identifier", '
        '"expected_symbols": ["pathlib.Path.read_text"], '
        '"expected_urls": ["library/pathlib.html"]}\n',
        encoding="utf-8",
    )


def _setup_index_artifacts(tmp_path: Path, sha: str = "abcdef123456") -> None:
    """Pre-create the chunks.jsonl + bm25.pkl + ingest_manifest.json artifacts.

    Uses >= 5 chunks so BM25 IDF is non-degenerate (small corpora collapse to <= 0
    scores and get filtered out by `if score > 0`).
    """
    from python_doc_assistant.cli import _save_chunks
    from python_doc_assistant.indexes.bm25_index import BM25Index

    chunks = [
        _symbol_chunk("pathlib.Path"),
        _symbol_chunk("pathlib.Path.read_text"),
        _symbol_chunk("pathlib.Path.write_text"),
        _symbol_chunk("dict.fromkeys"),
        _symbol_chunk("os.path.join"),
        _symbol_chunk("io.open"),
    ]
    chunks_path = tmp_path / "data" / "chunks" / "3.12" / sha / "chunks.jsonl"
    chunks_path.parent.mkdir(parents=True)
    _save_chunks(chunks, chunks_path)

    bm25_path = tmp_path / "data" / "indexes" / "3.12" / sha / "bm25.pkl"
    BM25Index(chunks).save(bm25_path)

    docs_dir = tmp_path / "data" / "docs" / "3.12" / sha
    docs_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "docs_version": "3.12",
        "docs_served_version": "3.12.13",
        "docs_url": "https://example/x.tar.bz2",
        "docs_archive_sha256": sha + "0" * (64 - len(sha)),
        "ingest_timestamp": "2026-04-25T09:55:36Z",
    }
    import json as _json

    (docs_dir / "ingest_manifest.json").write_text(_json.dumps(manifest), encoding="utf-8")


# ------------------------------------------------------------------
# CLI: judge (plan v2 §6)
# ------------------------------------------------------------------


def _judge_run_dir(tmp_path: Path, *, with_judge_scores: bool = False) -> Path:
    """Build a minimal run dir for `pdr judge` tests.

    Layout:
        run_dir/
          results.json        (docs_version + docs_sha_short + ablation fields)
          per_query.jsonl     (1 row by default; tests can append)
          judge_scores.jsonl  (optional, only when with_judge_scores=True)
    """
    run_dir = tmp_path / "experiments" / "runs" / "ts-judge-test"
    run_dir.mkdir(parents=True)

    import json as _json

    results = {
        "docs_version": "3.12",
        "docs_served_version": "3.12.13",
        "docs_sha_short": "abcdef123456",
        "ingest_manifest": {},
        "config": {"retriever": "dense", "k": 10, "eval_set": "v2_full.jsonl"},
        "tag": "v2-dense-qwen",
        "command": "pdr eval ...",
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "decoding_params": {"temperature": 0.0, "top_p": 1.0, "max_new_tokens": 512},
        "recall_at_5": 0.802,
        "recall_at_10": 0.883,
        "mrr": 0.682,
        "n_queries": 1,
    }
    (run_dir / "results.json").write_text(_json.dumps(results, indent=2), encoding="utf-8")

    retrieved_chunk = {
        "chunk_id": "symbol:pathlib.Path.read_text",
        "score": 0.9,
        "rank": 1,
        "canonical_url": "library/pathlib.html#pathlib.Path.read_text",
        "symbols": ["pathlib.Path.read_text"],
    }
    per_query = [
        {
            "query": "pathlib.Path.read_text",
            "query_type": "identifier",
            "match_policy": "any",
            "url_match": "strip_anchor",
            "expected_symbols": ["pathlib.Path.read_text"],
            "expected_urls": ["library/pathlib.html"],
            "retrieved": [retrieved_chunk],
            "hit_at_5": True,
            "hit_at_10": True,
            "rank_for_mrr": 1,
            "model_output_text": "Use Path.read_text() [1].",
            "cited_chunk_ids": ["symbol:pathlib.Path.read_text"],
            "refused": False,
            "generation_latency_seconds": 12.3,
        }
    ]
    (run_dir / "per_query.jsonl").write_text(
        "\n".join(_json.dumps(r) for r in per_query) + "\n", encoding="utf-8"
    )

    if with_judge_scores:
        (run_dir / "judge_scores.jsonl").write_text("", encoding="utf-8")

    return run_dir


def _stub_judge_one(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace evaluation.judge.judge_one with a capturing stub.

    Returns the captured-state dict so tests can inspect what the CLI
    passed to judge_one (model_id, temperature, max_tokens, etc.).
    """
    from python_doc_assistant.evaluation.judge import JudgeRecord

    captured: dict[str, Any] = {"calls": []}

    def fake_judge_one(
        query: str,
        expected_symbols: Any,
        retrieved_chunks: Any,
        cited_chunk_ids: Any,
        refused: bool,
        model_output_text: str,
        **kwargs: Any,
    ) -> JudgeRecord:
        # Mirror real judge_one: 6 positional + keyword-only client/model_id/etc.
        captured["calls"].append({
            "query": query,
            "expected_symbols": expected_symbols,
            "retrieved_chunks": retrieved_chunks,
            "cited_chunk_ids": cited_chunk_ids,
            "refused": refused,
            "model_output_text": model_output_text,
            **kwargs,
        })
        return JudgeRecord(
            query=query,
            tier="correct",
            notes="stubbed",
            raw_output='{"tier":"correct","reason":"stubbed"}',
            judge_model=kwargs.get("model_id", "stub-model"),
            judge_prompt_hash="abcd1234",
            timestamp="2026-04-29T12:00:00+00:00",
        )

    monkeypatch.setattr("python_doc_assistant.evaluation.judge.judge_one", fake_judge_one)
    # Also intercept anthropic.Anthropic() so no API call is attempted.
    import sys
    import types

    fake_module = types.ModuleType("anthropic")

    class _FakeClient:
        def __init__(self) -> None:
            pass

    fake_module.Anthropic = _FakeClient  # type: ignore[attr-defined]
    fake_module.APIError = Exception  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", fake_module)

    return captured


def test_cli_judge_writes_judge_scores_jsonl(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`pdr judge --run-dir X` writes judge_scores.jsonl + updates results.json."""
    import json as _json

    run_dir = _judge_run_dir(tmp_path)
    captured = _stub_judge_one(monkeypatch)

    # Point chunks.jsonl at an empty fixture so judge_one stub doesn't depend on it
    chunks_path = tmp_path / "data" / "chunks" / "3.12" / "abcdef123456" / "chunks.jsonl"
    chunks_path.parent.mkdir(parents=True)
    chunks_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(f"{CLI_MODULE}.DEFAULT_DATA_ROOT", tmp_path / "data")

    runner = CliRunner()
    result = runner.invoke(main, ["judge", "--run-dir", str(run_dir)])
    assert result.exit_code == 0, result.output

    # judge_scores.jsonl created
    judge_path = run_dir / "judge_scores.jsonl"
    assert judge_path.exists()
    records = [
        _json.loads(line)
        for line in judge_path.read_text().splitlines()
        if line.strip()
    ]
    assert len(records) == 1
    assert records[0]["query"] == "pathlib.Path.read_text"
    assert records[0]["tier"] == "correct"

    # results.json updated with judge + judge_aggregate keys
    results = _json.loads((run_dir / "results.json").read_text())
    assert "judge" in results
    assert results["judge"]["judge_model"] == "claude-haiku-4-5-20251001"
    assert "judge_aggregate" in results
    assert results["judge_aggregate"]["n"] == 1

    # judge_one received the per_query row's fields
    assert len(captured["calls"]) == 1
    call = captured["calls"][0]
    assert call["query"] == "pathlib.Path.read_text"
    assert call["model_id"] == "claude-haiku-4-5-20251001"
    assert call["temperature"] == 0.0
    assert call["max_tokens"] == 200


def test_cli_judge_skips_existing_unless_rerun_existing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """If judge_scores.jsonl already exists, skip silently — caller must opt in."""
    run_dir = _judge_run_dir(tmp_path, with_judge_scores=True)
    captured = _stub_judge_one(monkeypatch)
    chunks_path = tmp_path / "data" / "chunks" / "3.12" / "abcdef123456" / "chunks.jsonl"
    chunks_path.parent.mkdir(parents=True)
    chunks_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(f"{CLI_MODULE}.DEFAULT_DATA_ROOT", tmp_path / "data")

    runner = CliRunner()
    # Without flag → skip
    result = runner.invoke(main, ["judge", "--run-dir", str(run_dir)])
    assert result.exit_code == 0, result.output
    assert "SKIP" in result.output
    assert len(captured["calls"]) == 0

    # With flag → re-judge
    result2 = runner.invoke(main, ["judge", "--run-dir", str(run_dir), "--rerun-existing"])
    assert result2.exit_code == 0, result2.output
    assert len(captured["calls"]) == 1


def test_cli_judge_max_rows_caps_judge_calls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """--max-rows 2 limits the per_query rows judged."""
    import json as _json

    run_dir = _judge_run_dir(tmp_path)
    # Append more rows
    pq_path = run_dir / "per_query.jsonl"
    extra = []
    for i in range(2, 6):
        extra.append({
            "query": f"q{i}",
            "query_type": "identifier",
            "match_policy": "any",
            "url_match": "strip_anchor",
            "expected_symbols": [],
            "expected_urls": [],
            "retrieved": [],
            "hit_at_5": False,
            "hit_at_10": False,
            "rank_for_mrr": None,
            "model_output_text": "x",
            "cited_chunk_ids": [],
            "refused": False,
            "generation_latency_seconds": 1.0,
        })
    with pq_path.open("a", encoding="utf-8") as f:
        for r in extra:
            f.write(_json.dumps(r) + "\n")

    captured = _stub_judge_one(monkeypatch)
    chunks_path = tmp_path / "data" / "chunks" / "3.12" / "abcdef123456" / "chunks.jsonl"
    chunks_path.parent.mkdir(parents=True)
    chunks_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(f"{CLI_MODULE}.DEFAULT_DATA_ROOT", tmp_path / "data")

    runner = CliRunner()
    result = runner.invoke(main, ["judge", "--run-dir", str(run_dir), "--max-rows", "2"])
    assert result.exit_code == 0, result.output
    # judge_one called only twice despite 5 rows in per_query.jsonl
    assert len(captured["calls"]) == 2


def test_cli_judge_passes_decoding_overrides(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """--judge-model / --temperature / --max-tokens reach judge_one."""
    run_dir = _judge_run_dir(tmp_path)
    captured = _stub_judge_one(monkeypatch)
    chunks_path = tmp_path / "data" / "chunks" / "3.12" / "abcdef123456" / "chunks.jsonl"
    chunks_path.parent.mkdir(parents=True)
    chunks_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(f"{CLI_MODULE}.DEFAULT_DATA_ROOT", tmp_path / "data")

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "judge", "--run-dir", str(run_dir),
            "--judge-model", "claude-sonnet-4-5",
            "--temperature", "0.2",
            "--max-tokens", "300",
        ],
    )
    assert result.exit_code == 0, result.output
    call = captured["calls"][0]
    assert call["model_id"] == "claude-sonnet-4-5"
    assert call["temperature"] == 0.2
    assert call["max_tokens"] == 300


# ------------------------------------------------------------------
# CLI: build-index --with-dense (v2 §5 prereq)
# ------------------------------------------------------------------


def test_cli_build_index_with_dense_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """--with-dense triggers DenseIndex(chunks).save(<.../dense.npy>)."""
    saved: list[Path] = []

    class _StubDenseIndex:
        def __init__(self, chunks: list[Chunk], **_: Any) -> None:
            self.chunks = chunks

        def save(self, path: Path) -> None:
            saved.append(path)

    cfg = tmp_path / "config.toml"
    cfg.write_text('DOCS_VERSION = "3.12"\n', encoding="utf-8")
    current = tmp_path / "data" / "docs" / "3.12" / "current.txt"
    current.parent.mkdir(parents=True, exist_ok=True)
    current.write_text("abcdef123456\n", encoding="utf-8")
    docs_dir = tmp_path / "data" / "docs" / "3.12" / "abcdef123456"
    docs_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(f"{CLI_MODULE}.DEFAULT_CONFIG_PATH", cfg)
    monkeypatch.setattr(f"{CLI_MODULE}.DEFAULT_DATA_ROOT", tmp_path / "data")
    monkeypatch.setattr(f"{CLI_MODULE}.parse_objects_inv", lambda _docs_dir: [])
    monkeypatch.setattr(
        f"{CLI_MODULE}.build_chunks",
        lambda *_args, **_kwargs: [_symbol_chunk("pathlib.Path.read_text")],
    )
    monkeypatch.setattr(
        "python_doc_assistant.indexes.dense_index.DenseIndex", _StubDenseIndex
    )

    runner = CliRunner()
    result = runner.invoke(main, ["build-index", "--with-dense"])
    assert result.exit_code == 0, result.output
    assert len(saved) == 1
    assert saved[0].name == "dense.npy"
    assert "dense ->" in result.output


# ------------------------------------------------------------------
# CLI: eval --retriever / --rerank (v2 §5 prereq)
# ------------------------------------------------------------------


def _stub_build_eval_retrieve_fn(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    """Replace _build_eval_retrieve_fn with a capturing stub.

    Returns a dict that will hold the kwargs the CLI passed in.
    """
    captured: dict[str, Any] = {}

    def fake_build(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return lambda q, k: []  # empty retrieve_fn → no chunks retrieved

    monkeypatch.setattr(f"{CLI_MODULE}._build_eval_retrieve_fn", fake_build)
    return captured


def test_cli_eval_default_retriever_is_symbol_bm25(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No --retriever flag → defaults to v0 'symbol+bm25' router."""
    captured = _stub_build_eval_retrieve_fn(monkeypatch)

    cfg = tmp_path / "config.toml"
    cfg.write_text('DOCS_VERSION = "3.12"\n', encoding="utf-8")
    current = tmp_path / "data" / "docs" / "3.12" / "current.txt"
    current.parent.mkdir(parents=True, exist_ok=True)
    current.write_text("abcdef123456\n", encoding="utf-8")

    _setup_index_artifacts(tmp_path)
    eval_path = tmp_path / "v0.jsonl"
    _eval_set_jsonl(eval_path)

    monkeypatch.setattr(f"{CLI_MODULE}.parse_objects_inv", lambda _docs_dir: [])
    monkeypatch.setattr(f"{CLI_MODULE}.DEFAULT_CONFIG_PATH", cfg)
    monkeypatch.setattr(f"{CLI_MODULE}.DEFAULT_DATA_ROOT", tmp_path / "data")
    monkeypatch.setattr(
        "python_doc_assistant.evaluation.run_writer.DEFAULT_EXPERIMENTS_ROOT",
        tmp_path / "experiments" / "runs",
    )

    runner = CliRunner()
    result = runner.invoke(main, ["eval", "--set", str(eval_path), "--tag", "v2-default"])
    assert result.exit_code == 0, result.output
    assert captured["retriever"] == "symbol+bm25"
    assert captured["rerank"] is False


def test_cli_eval_dense_retriever_kwarg_threading(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """--retriever dense reaches the dispatch helper."""
    captured = _stub_build_eval_retrieve_fn(monkeypatch)

    cfg = tmp_path / "config.toml"
    cfg.write_text('DOCS_VERSION = "3.12"\n', encoding="utf-8")
    current = tmp_path / "data" / "docs" / "3.12" / "current.txt"
    current.parent.mkdir(parents=True, exist_ok=True)
    current.write_text("abcdef123456\n", encoding="utf-8")

    _setup_index_artifacts(tmp_path)
    eval_path = tmp_path / "v0.jsonl"
    _eval_set_jsonl(eval_path)

    monkeypatch.setattr(f"{CLI_MODULE}.parse_objects_inv", lambda _docs_dir: [])
    monkeypatch.setattr(f"{CLI_MODULE}.DEFAULT_CONFIG_PATH", cfg)
    monkeypatch.setattr(f"{CLI_MODULE}.DEFAULT_DATA_ROOT", tmp_path / "data")
    monkeypatch.setattr(
        "python_doc_assistant.evaluation.run_writer.DEFAULT_EXPERIMENTS_ROOT",
        tmp_path / "experiments" / "runs",
    )

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["eval", "--set", str(eval_path), "--tag", "v2-dense", "--retriever", "dense"],
    )
    assert result.exit_code == 0, result.output
    assert captured["retriever"] == "dense"


def test_cli_eval_rerank_flag_threads_through(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """--rerank + --rerank-candidates reach the dispatch helper."""
    captured = _stub_build_eval_retrieve_fn(monkeypatch)

    cfg = tmp_path / "config.toml"
    cfg.write_text('DOCS_VERSION = "3.12"\n', encoding="utf-8")
    current = tmp_path / "data" / "docs" / "3.12" / "current.txt"
    current.parent.mkdir(parents=True, exist_ok=True)
    current.write_text("abcdef123456\n", encoding="utf-8")

    _setup_index_artifacts(tmp_path)
    eval_path = tmp_path / "v0.jsonl"
    _eval_set_jsonl(eval_path)

    monkeypatch.setattr(f"{CLI_MODULE}.parse_objects_inv", lambda _docs_dir: [])
    monkeypatch.setattr(f"{CLI_MODULE}.DEFAULT_CONFIG_PATH", cfg)
    monkeypatch.setattr(f"{CLI_MODULE}.DEFAULT_DATA_ROOT", tmp_path / "data")
    monkeypatch.setattr(
        "python_doc_assistant.evaluation.run_writer.DEFAULT_EXPERIMENTS_ROOT",
        tmp_path / "experiments" / "runs",
    )

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "eval", "--set", str(eval_path), "--tag", "v2-rerank",
            "--retriever", "hybrid-rrf",
            "--rerank",
            "--rerank-candidates", "30",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["retriever"] == "hybrid-rrf"
    assert captured["rerank"] is True
    assert captured["rerank_candidates"] == 30


def test_cli_eval_alpha_flag_threads_through(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """--alpha 0.7 reaches the dispatch helper."""
    captured = _stub_build_eval_retrieve_fn(monkeypatch)

    cfg = tmp_path / "config.toml"
    cfg.write_text('DOCS_VERSION = "3.12"\n', encoding="utf-8")
    current = tmp_path / "data" / "docs" / "3.12" / "current.txt"
    current.parent.mkdir(parents=True, exist_ok=True)
    current.write_text("abcdef123456\n", encoding="utf-8")

    _setup_index_artifacts(tmp_path)
    eval_path = tmp_path / "v0.jsonl"
    _eval_set_jsonl(eval_path)

    monkeypatch.setattr(f"{CLI_MODULE}.parse_objects_inv", lambda _docs_dir: [])
    monkeypatch.setattr(f"{CLI_MODULE}.DEFAULT_CONFIG_PATH", cfg)
    monkeypatch.setattr(f"{CLI_MODULE}.DEFAULT_DATA_ROOT", tmp_path / "data")
    monkeypatch.setattr(
        "python_doc_assistant.evaluation.run_writer.DEFAULT_EXPERIMENTS_ROOT",
        tmp_path / "experiments" / "runs",
    )

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "eval", "--set", str(eval_path), "--tag", "v2-linear",
            "--retriever", "hybrid-linear",
            "--alpha", "0.7",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["retriever"] == "hybrid-linear"
    assert captured["alpha"] == pytest.approx(0.7)


# ------------------------------------------------------------------
# CLI: ask (plan v1 §5)
# ------------------------------------------------------------------


class _StubAskGenerator:
    """Stub for QwenGenerator used by `pdr ask` tests.

    Records last call so tests can assert on what was passed in. The class
    matches the call shape `QwenGenerator(model_id)` and exposes a
    `.generate(query, chunks, *, query_type=None, stream=False)` method.
    """

    def __init__(self, model_id: str = "stub", **_: Any) -> None:
        self.model_id = model_id
        self.last_query: str | None = None
        self.last_chunks: list[Chunk] | None = None
        # Response payload — overridable per-test by mutating these:
        self.text: str = "Use `read_text()` [1]."
        self.cited_chunk_ids: tuple[str, ...] = ("symbol:pathlib.Path.read_text",)
        self.refused: bool = False
        self.latency_seconds: float = 0.5

    def generate(
        self,
        query: str,
        retrieved_chunks: list[Chunk],
        **_: Any,
    ) -> Any:
        from python_doc_assistant.generation.interface import Answer

        self.last_query = query
        self.last_chunks = retrieved_chunks
        return Answer(
            text=self.text,
            cited_chunk_ids=self.cited_chunk_ids,
            refused=self.refused,
            latency_seconds=self.latency_seconds,
        )


def _ask_setup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _StubAskGenerator:
    """Common fixture for pdr ask tests: config + indexes + stub generator.

    Returns the stub generator instance the CLI will use, so tests can mutate
    its response payload before invoking the command.
    """
    cfg = tmp_path / "config.toml"
    cfg.write_text('DOCS_VERSION = "3.12"\n', encoding="utf-8")
    current = tmp_path / "data" / "docs" / "3.12" / "current.txt"
    current.parent.mkdir(parents=True, exist_ok=True)
    current.write_text("abcdef123456\n", encoding="utf-8")

    _setup_index_artifacts(tmp_path)

    monkeypatch.setattr(f"{CLI_MODULE}.parse_objects_inv", lambda _docs_dir: [])
    monkeypatch.setattr(f"{CLI_MODULE}.DEFAULT_CONFIG_PATH", cfg)
    monkeypatch.setattr(f"{CLI_MODULE}.DEFAULT_DATA_ROOT", tmp_path / "data")

    # Patch QwenGenerator at its source module — the CLI lazy-imports it
    # at call time, so we must intercept the source attribute (not a name
    # already bound in cli.py).
    stub = _StubAskGenerator()

    def factory(model_id: str = "stub", **kwargs: Any) -> _StubAskGenerator:
        stub.model_id = model_id
        return stub

    monkeypatch.setattr(
        "python_doc_assistant.generation.qwen_backend.QwenGenerator", factory
    )
    return stub


def test_cli_ask_prints_answer_text(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Without --debug, only the model's answer text is printed."""
    stub = _ask_setup(monkeypatch, tmp_path)
    stub.text = "The read_text method returns the file contents [1]."

    runner = CliRunner()
    result = runner.invoke(main, ["ask", "Path.read_text"])
    assert result.exit_code == 0, result.output
    assert "The read_text method returns the file contents [1]." in result.output


def test_cli_ask_refused_prints_refusal_marker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Refused answer (empty text) → user sees [INSUFFICIENT-CONTEXT] rather than blank."""
    stub = _ask_setup(monkeypatch, tmp_path)
    stub.text = ""
    stub.cited_chunk_ids = ()
    stub.refused = True

    runner = CliRunner()
    result = runner.invoke(main, ["ask", "what is graphql"])
    assert result.exit_code == 0, result.output
    assert "[INSUFFICIENT-CONTEXT]" in result.output


def test_cli_ask_passes_model_id_through(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """--model flag reaches QwenGenerator constructor."""
    stub = _ask_setup(monkeypatch, tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        main, ["ask", "Path.read_text", "--model", "Qwen/Qwen2.5-Coder-1.5B-Instruct"]
    )
    assert result.exit_code == 0, result.output
    assert stub.model_id == "Qwen/Qwen2.5-Coder-1.5B-Instruct"


def test_cli_ask_debug_prints_retrieved_chunks_with_scores(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """--debug shows rank + score + chunk_id for each retrieved chunk."""
    _ask_setup(monkeypatch, tmp_path)

    runner = CliRunner()
    result = runner.invoke(main, ["ask", "Path.read_text", "--debug"])
    assert result.exit_code == 0, result.output
    # At least one chunk shown with rank= score= id= format
    assert "rank=" in result.output
    assert "score=" in result.output
    # The retrieved chunk id appears (symbol:pathlib.Path.read_text is in fixture)
    assert "symbol:pathlib.Path.read_text" in result.output


def test_cli_ask_debug_prints_final_prompt_messages(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """--debug dumps the chat-template messages (system + user blocks)."""
    _ask_setup(monkeypatch, tmp_path)

    runner = CliRunner()
    result = runner.invoke(main, ["ask", "Path.read_text", "--debug"])
    assert result.exit_code == 0, result.output
    # System block delimiter
    assert "--- system ---" in result.output
    # User block delimiter
    assert "--- user ---" in result.output
    # System content fragment proves the prompt was rendered
    assert "documentation chunks provided below" in result.output


def test_cli_ask_debug_validates_citations_against_retrieved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """--debug flags each cited chunk_id with whether it was in the top-K set."""
    stub = _ask_setup(monkeypatch, tmp_path)
    stub.cited_chunk_ids = (
        "symbol:pathlib.Path.read_text",  # likely IS in retrieved
        "symbol:totally:made:up",  # NOT in retrieved
    )

    runner = CliRunner()
    result = runner.invoke(main, ["ask", "Path.read_text", "--debug"])
    assert result.exit_code == 0, result.output
    assert "in_retrieved=yes" in result.output
    assert "in_retrieved=no" in result.output


def test_cli_ask_no_debug_omits_debug_blocks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Without --debug, none of the debug section headers appear."""
    _ask_setup(monkeypatch, tmp_path)

    runner = CliRunner()
    result = runner.invoke(main, ["ask", "Path.read_text"])
    assert result.exit_code == 0, result.output
    assert "[debug] retrieved" not in result.output
    assert "[debug] prompt" not in result.output
    assert "[debug] citations" not in result.output


# ------------------------------------------------------------------
# CLI: eval (plan §9)
# ------------------------------------------------------------------


def test_cli_eval_writes_run_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """eval should produce a run dir under DEFAULT_EXPERIMENTS_ROOT with both files."""
    cfg = tmp_path / "config.toml"
    cfg.write_text('DOCS_VERSION = "3.12"\n', encoding="utf-8")
    current = tmp_path / "data" / "docs" / "3.12" / "current.txt"
    current.parent.mkdir(parents=True, exist_ok=True)
    current.write_text("abcdef123456\n", encoding="utf-8")

    _setup_index_artifacts(tmp_path)

    eval_path = tmp_path / "v0.jsonl"
    _eval_set_jsonl(eval_path)

    monkeypatch.setattr(f"{CLI_MODULE}.parse_objects_inv", lambda _docs_dir: [])
    monkeypatch.setattr(f"{CLI_MODULE}.DEFAULT_CONFIG_PATH", cfg)
    monkeypatch.setattr(f"{CLI_MODULE}.DEFAULT_DATA_ROOT", tmp_path / "data")
    # Pin run dir under tmp to keep tests hermetic
    experiments_root = tmp_path / "experiments" / "runs"
    monkeypatch.setattr(
        "python_doc_assistant.evaluation.run_writer.DEFAULT_EXPERIMENTS_ROOT",
        experiments_root,
    )

    runner = CliRunner()
    result = runner.invoke(main, ["eval", "--set", str(eval_path), "--tag", "v0-bm25"])
    assert result.exit_code == 0, result.output

    # Exactly one run dir created
    run_dirs = list(experiments_root.iterdir())
    assert len(run_dirs) == 1
    rd = run_dirs[0]
    assert rd.name.endswith("-v0-bm25")
    assert (rd / "results.json").is_file()
    assert (rd / "per_query.jsonl").is_file()


def test_cli_eval_records_nonzero_mrr_on_top1_hit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When the eval query hits at rank 1, MRR must be 1.0 (not 0.0).

    Catches the rank=0 placeholder bug in retrieve_fn — if rank stays 0,
    match_rank returns 0 and MRR collapses to 0 even on a top-1 hit.
    """
    import json as _json

    cfg = tmp_path / "config.toml"
    cfg.write_text('DOCS_VERSION = "3.12"\n', encoding="utf-8")
    current = tmp_path / "data" / "docs" / "3.12" / "current.txt"
    current.parent.mkdir(parents=True, exist_ok=True)
    current.write_text("abcdef123456\n", encoding="utf-8")

    _setup_index_artifacts(tmp_path)
    eval_path = tmp_path / "v0.jsonl"
    _eval_set_jsonl(eval_path)

    monkeypatch.setattr(f"{CLI_MODULE}.parse_objects_inv", lambda _docs_dir: [])
    monkeypatch.setattr(f"{CLI_MODULE}.DEFAULT_CONFIG_PATH", cfg)
    monkeypatch.setattr(f"{CLI_MODULE}.DEFAULT_DATA_ROOT", tmp_path / "data")
    experiments_root = tmp_path / "experiments" / "runs"
    monkeypatch.setattr(
        "python_doc_assistant.evaluation.run_writer.DEFAULT_EXPERIMENTS_ROOT",
        experiments_root,
    )

    runner = CliRunner()
    result = runner.invoke(main, ["eval", "--set", str(eval_path), "--tag", "v0-bm25"])
    assert result.exit_code == 0, result.output

    rd = next(iter(experiments_root.iterdir()))
    data = _json.loads((rd / "results.json").read_text(encoding="utf-8"))
    # Single-query eval set, target chunk is in BM25 corpus → must be top-K hit
    assert data["recall_at_5"] == 1.0
    # MRR must be > 0 (rank-derived from chunk position, not stuck at 0)
    assert data["mrr"] > 0.0


def test_cli_eval_prints_summary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """eval stdout includes recall@5 / recall@10 / mrr summary line."""
    cfg = tmp_path / "config.toml"
    cfg.write_text('DOCS_VERSION = "3.12"\n', encoding="utf-8")
    current = tmp_path / "data" / "docs" / "3.12" / "current.txt"
    current.parent.mkdir(parents=True, exist_ok=True)
    current.write_text("abcdef123456\n", encoding="utf-8")

    _setup_index_artifacts(tmp_path)
    eval_path = tmp_path / "v0.jsonl"
    _eval_set_jsonl(eval_path)

    monkeypatch.setattr(f"{CLI_MODULE}.parse_objects_inv", lambda _docs_dir: [])
    monkeypatch.setattr(f"{CLI_MODULE}.DEFAULT_CONFIG_PATH", cfg)
    monkeypatch.setattr(f"{CLI_MODULE}.DEFAULT_DATA_ROOT", tmp_path / "data")
    monkeypatch.setattr(
        "python_doc_assistant.evaluation.run_writer.DEFAULT_EXPERIMENTS_ROOT",
        tmp_path / "experiments" / "runs",
    )

    runner = CliRunner()
    result = runner.invoke(main, ["eval", "--set", str(eval_path), "--tag", "v0-bm25"])
    assert result.exit_code == 0, result.output
    assert "recall@5" in result.output
    assert "mrr" in result.output


def test_cli_eval_without_model_records_null_generation_fields(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Retrieval-only run → results.json has null model + null decoding_params."""
    import json as _json

    cfg = tmp_path / "config.toml"
    cfg.write_text('DOCS_VERSION = "3.12"\n', encoding="utf-8")
    current = tmp_path / "data" / "docs" / "3.12" / "current.txt"
    current.parent.mkdir(parents=True, exist_ok=True)
    current.write_text("abcdef123456\n", encoding="utf-8")

    _setup_index_artifacts(tmp_path)
    eval_path = tmp_path / "v0.jsonl"
    _eval_set_jsonl(eval_path)

    monkeypatch.setattr(f"{CLI_MODULE}.parse_objects_inv", lambda _docs_dir: [])
    monkeypatch.setattr(f"{CLI_MODULE}.DEFAULT_CONFIG_PATH", cfg)
    monkeypatch.setattr(f"{CLI_MODULE}.DEFAULT_DATA_ROOT", tmp_path / "data")
    experiments_root = tmp_path / "experiments" / "runs"
    monkeypatch.setattr(
        "python_doc_assistant.evaluation.run_writer.DEFAULT_EXPERIMENTS_ROOT",
        experiments_root,
    )

    runner = CliRunner()
    result = runner.invoke(main, ["eval", "--set", str(eval_path), "--tag", "v0-bm25"])
    assert result.exit_code == 0, result.output

    rd = next(iter(experiments_root.iterdir()))
    data = _json.loads((rd / "results.json").read_text(encoding="utf-8"))
    assert data["model"] is None
    assert data["decoding_params"] is None


def test_cli_eval_with_model_dispatches_to_generation_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """--model flag routes through the generation dispatch and surfaces
    the model_id + decoding_params in results.json.

    Hermetic: monkeypatches `_run_eval_with_optional_generation` so no real
    Qwen is loaded. Verifies the dispatch passes model_id through.
    """
    import json as _json

    from python_doc_assistant.evaluation.retrieval_metrics import EvalRunResult

    cfg = tmp_path / "config.toml"
    cfg.write_text('DOCS_VERSION = "3.12"\n', encoding="utf-8")
    current = tmp_path / "data" / "docs" / "3.12" / "current.txt"
    current.parent.mkdir(parents=True, exist_ok=True)
    current.write_text("abcdef123456\n", encoding="utf-8")

    _setup_index_artifacts(tmp_path)
    eval_path = tmp_path / "v0.jsonl"
    _eval_set_jsonl(eval_path)

    monkeypatch.setattr(f"{CLI_MODULE}.parse_objects_inv", lambda _docs_dir: [])
    monkeypatch.setattr(f"{CLI_MODULE}.DEFAULT_CONFIG_PATH", cfg)
    monkeypatch.setattr(f"{CLI_MODULE}.DEFAULT_DATA_ROOT", tmp_path / "data")
    experiments_root = tmp_path / "experiments" / "runs"
    monkeypatch.setattr(
        "python_doc_assistant.evaluation.run_writer.DEFAULT_EXPERIMENTS_ROOT",
        experiments_root,
    )

    captured: dict[str, Any] = {}
    canned_run = EvalRunResult(
        queries=(), recall_at_5=0.0, recall_at_10=0.0, mrr=0.0, n_queries=0
    )
    canned_decoding = {"temperature": 0.0, "top_p": 1.0, "max_new_tokens": 512}

    def fake_dispatch(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return canned_run, kwargs["model_id"], canned_decoding

    monkeypatch.setattr(
        f"{CLI_MODULE}._run_eval_with_optional_generation", fake_dispatch
    )

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "eval",
            "--set",
            str(eval_path),
            "--tag",
            "v1-qwen",
            "--model",
            "Qwen/Qwen2.5-1.5B-Instruct",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["model_id"] == "Qwen/Qwen2.5-1.5B-Instruct"

    rd = next(iter(experiments_root.iterdir()))
    data = _json.loads((rd / "results.json").read_text(encoding="utf-8"))
    assert data["model"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert data["decoding_params"] == canned_decoding


def test_cli_eval_refuses_existing_run_dir_without_overwrite(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Second invocation with same timestamp should fail unless --overwrite."""
    cfg = tmp_path / "config.toml"
    cfg.write_text('DOCS_VERSION = "3.12"\n', encoding="utf-8")
    current = tmp_path / "data" / "docs" / "3.12" / "current.txt"
    current.parent.mkdir(parents=True, exist_ok=True)
    current.write_text("abcdef123456\n", encoding="utf-8")

    _setup_index_artifacts(tmp_path)
    eval_path = tmp_path / "v0.jsonl"
    _eval_set_jsonl(eval_path)
    experiments_root = tmp_path / "experiments" / "runs"

    monkeypatch.setattr(f"{CLI_MODULE}.parse_objects_inv", lambda _docs_dir: [])
    monkeypatch.setattr(f"{CLI_MODULE}.DEFAULT_CONFIG_PATH", cfg)
    monkeypatch.setattr(f"{CLI_MODULE}.DEFAULT_DATA_ROOT", tmp_path / "data")
    monkeypatch.setattr(
        "python_doc_assistant.evaluation.run_writer.DEFAULT_EXPERIMENTS_ROOT",
        experiments_root,
    )
    # Pin the timestamp so two invocations collide.
    from datetime import datetime, timezone

    fixed = datetime(2026, 4, 27, 15, 30, 45, tzinfo=timezone.utc)
    monkeypatch.setattr("python_doc_assistant.evaluation.run_writer._utc_now", lambda: fixed)

    runner = CliRunner()
    first = runner.invoke(main, ["eval", "--set", str(eval_path), "--tag", "v0-bm25"])
    assert first.exit_code == 0, first.output

    second = runner.invoke(main, ["eval", "--set", str(eval_path), "--tag", "v0-bm25"])
    assert second.exit_code != 0, "expected failure on existing run dir"

    third = runner.invoke(
        main, ["eval", "--set", str(eval_path), "--tag", "v0-bm25", "--overwrite"]
    )
    assert third.exit_code == 0, third.output
