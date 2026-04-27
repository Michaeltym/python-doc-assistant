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
