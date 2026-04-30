"""Tests for v3 §3 pretrain corpus builder."""

from __future__ import annotations

import json
from pathlib import Path

from python_doc_assistant.generation.tinydocs.data_mix import build_corpus_from_chunks


def _write_chunks_fixture(path: Path, n: int = 10) -> None:
    """Write a tiny chunks.jsonl fixture with `n` synthetic chunks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i in range(n):
            chunk = {
                "chunk_id": f"symbol:test.module.func_{i}",
                "chunk_type": "symbol",
                "title": f"func_{i}",
                "text": f"This is test chunk {i}. " * 5,
                "source_path": f"library/test.html#func_{i}",
                "canonical_url": f"library/test.html#func_{i}",
                "docs_version": "3.12",
                "source_hash": "fakehashfakehash",
                "anchor": f"func_{i}",
                "symbols": [f"test.module.func_{i}"],
            }
            f.write(json.dumps(chunk) + "\n")


def test_corpus_has_one_line_per_chunk(tmp_path: Path) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    out_path = tmp_path / "corpus.jsonl"
    _write_chunks_fixture(chunks_path, n=10)

    build_corpus_from_chunks(chunks_path, out_path)

    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 10


def test_corpus_each_line_has_text_and_source(tmp_path: Path) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    out_path = tmp_path / "corpus.jsonl"
    _write_chunks_fixture(chunks_path, n=5)

    build_corpus_from_chunks(chunks_path, out_path)

    for line in out_path.read_text(encoding="utf-8").splitlines():
        obj = json.loads(line)
        assert "text" in obj
        assert obj["source"] == "python-docs"
        assert isinstance(obj["text"], str)
        assert len(obj["text"]) > 0


def test_corpus_is_deterministic(tmp_path: Path) -> None:
    """Same seed → byte-identical corpus.jsonl."""
    chunks_path = tmp_path / "chunks.jsonl"
    out_a = tmp_path / "corpus_a.jsonl"
    out_b = tmp_path / "corpus_b.jsonl"
    _write_chunks_fixture(chunks_path, n=20)

    build_corpus_from_chunks(chunks_path, out_a, seed=42)
    build_corpus_from_chunks(chunks_path, out_b, seed=42)

    assert out_a.read_bytes() == out_b.read_bytes()


def test_corpus_seed_changes_order(tmp_path: Path) -> None:
    """Different seed → different first line (high probability with n=20)."""
    chunks_path = tmp_path / "chunks.jsonl"
    out_a = tmp_path / "corpus_a.jsonl"
    out_b = tmp_path / "corpus_b.jsonl"
    _write_chunks_fixture(chunks_path, n=20)

    build_corpus_from_chunks(chunks_path, out_a, seed=42)
    build_corpus_from_chunks(chunks_path, out_b, seed=1)

    first_a = out_a.read_text(encoding="utf-8").splitlines()[0]
    first_b = out_b.read_text(encoding="utf-8").splitlines()[0]
    assert first_a != first_b


def test_manifest_records_reproducibility_fields(tmp_path: Path) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    out_path = tmp_path / "corpus.jsonl"
    manifest_path = tmp_path / "manifest.json"
    _write_chunks_fixture(chunks_path, n=10)

    manifest = build_corpus_from_chunks(chunks_path, out_path, seed=42, manifest_path=manifest_path)

    # Returned manifest dict
    assert manifest["chunks_path"] == str(chunks_path)
    assert manifest["seed"] == 42
    assert manifest["n_lines"] == 10
    assert manifest["total_bytes"] > 0
    assert "build_timestamp" in manifest

    # File on disk matches returned dict
    on_disk = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert on_disk == manifest


def test_corpus_skips_empty_chunks(tmp_path: Path) -> None:
    """Chunks with empty text should be skipped (e.g. section_chunk with no body)."""
    chunks_path = tmp_path / "chunks.jsonl"
    out_path = tmp_path / "corpus.jsonl"

    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    with chunks_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"chunk_id": "1", "text": "real text"}) + "\n")
        f.write(json.dumps({"chunk_id": "2", "text": ""}) + "\n")
        f.write(json.dumps({"chunk_id": "3", "text": "   "}) + "\n")  # whitespace-only
        f.write(json.dumps({"chunk_id": "4", "text": "another real"}) + "\n")

    build_corpus_from_chunks(chunks_path, out_path)

    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2  # only the two non-empty chunks
