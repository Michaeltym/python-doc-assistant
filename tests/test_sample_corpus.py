"""Tests for python_doc_assistant.generation.tinydocs.sample.

Two-pass Bernoulli byte-target sampler used by v3.1 §1.1 for BPE training.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from python_doc_assistant.generation.tinydocs.sample import (
    count_bytes,
    sample_lines_to_byte_target,
)

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


def _make_corpus(tmp_path: Path, n_lines: int, line_size: int) -> Path:
    """Write a jsonl corpus with `n_lines` lines, each ~`line_size` bytes.

    Each line is a `{"text": "..."}` JSON record padded so the encoded
    line (including trailing `\\n`) is approximately `line_size` bytes.
    Lines are deterministic so tests can compare byte counts exactly.
    """
    path = tmp_path / "corpus.jsonl"
    # Padding length so each {"text": "..."}\n is ~line_size.
    overhead = len('{"text": ""}\n')
    pad = max(1, line_size - overhead)
    with path.open("w", encoding="utf-8") as f:
        for i in range(n_lines):
            payload = (str(i) * pad)[:pad]
            f.write(f'{{"text": "{payload}"}}\n')
    return path


# ------------------------------------------------------------------
# count_bytes
# ------------------------------------------------------------------


def test_count_bytes_matches_file_size_for_dense_corpus(tmp_path: Path) -> None:
    """For a corpus with no blank lines, count_bytes equals file size."""
    path = _make_corpus(tmp_path, n_lines=100, line_size=200)
    file_size = path.stat().st_size
    assert count_bytes(path) == file_size


def test_count_bytes_skips_blank_lines(tmp_path: Path) -> None:
    """Blank / whitespace lines are excluded from the count."""
    path = tmp_path / "corpus.jsonl"
    with path.open("w", encoding="utf-8") as f:
        f.write('{"text": "hello"}\n')
        f.write("\n")  # blank
        f.write("   \n")  # whitespace
        f.write('{"text": "world"}\n')
    expected = len('{"text": "hello"}\n'.encode("utf-8")) + len(
        '{"text": "world"}\n'.encode("utf-8")
    )
    assert count_bytes(path) == expected


# ------------------------------------------------------------------
# sample_lines_to_byte_target
# ------------------------------------------------------------------


def test_sample_size_within_tolerance(tmp_path: Path) -> None:
    """Output bytes are within ±20 % of target on a corpus where target ≈ 25 % of total.

    Bernoulli variance dominates at smaller corpora; ±20 % is loose-but-honest
    for n_lines=200 with target_ratio=0.25 (binomial stdev ≈ 6 %, 3σ ~ 18 %).
    """
    path = _make_corpus(tmp_path, n_lines=200, line_size=200)
    total = count_bytes(path)
    target = total // 4

    written = 0
    for line in sample_lines_to_byte_target(path, target, seed=42):
        written += len(line.encode("utf-8"))

    lower = int(target * 0.8)
    upper = int(target * 1.2)
    assert lower <= written <= upper, (
        f"sampled {written} not in [{lower}, {upper}] (target {target})"
    )


def test_deterministic_with_seed(tmp_path: Path) -> None:
    """Two runs with same seed → byte-identical concatenated output."""
    path = _make_corpus(tmp_path, n_lines=200, line_size=200)
    target = count_bytes(path) // 4

    def run() -> str:
        return "".join(sample_lines_to_byte_target(path, target, seed=42))

    a = run()
    b = run()
    assert hashlib.sha256(a.encode()).hexdigest() == hashlib.sha256(b.encode()).hexdigest()


def test_different_seeds_diverge(tmp_path: Path) -> None:
    """Different seeds produce different samples (non-trivial check)."""
    path = _make_corpus(tmp_path, n_lines=200, line_size=200)
    target = count_bytes(path) // 4

    a = "".join(sample_lines_to_byte_target(path, target, seed=42))
    b = "".join(sample_lines_to_byte_target(path, target, seed=43))
    assert a != b


def test_target_larger_than_input_yields_all_lines(tmp_path: Path) -> None:
    """When target ≥ total, every non-empty line is yielded exactly once."""
    path = _make_corpus(tmp_path, n_lines=50, line_size=100)
    total = count_bytes(path)

    lines = list(sample_lines_to_byte_target(path, total * 2, seed=42))
    assert len(lines) == 50
    assert sum(len(line.encode("utf-8")) for line in lines) == total


def test_lines_preserved_with_trailing_newline(tmp_path: Path) -> None:
    """Yielded lines end with `\\n` so the caller can write straight to a file."""
    path = _make_corpus(tmp_path, n_lines=100, line_size=200)
    total = count_bytes(path)

    for line in sample_lines_to_byte_target(path, total, seed=42):
        assert line.endswith("\n")
