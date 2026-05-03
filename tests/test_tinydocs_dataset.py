"""Tests for v3 §4a Dataset (segments + TinyDocsDataset)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from python_doc_assistant.generation.tinydocs.dataset import (  # noqa: E402
    TinyDocsDataset,
    build_segments,
)
from python_doc_assistant.generation.tinydocs.tokenizer import TinyDocsTokenizer  # noqa: E402
from python_doc_assistant.generation.tinydocs.tokenizer_train import train_bpe  # noqa: E402

SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<unk>", "<sp>")


def _tiny_tokenizer() -> TinyDocsTokenizer:
    """Train a tiny BPE on a pangram corpus so most letters are single tokens."""
    corpus = [
        "the quick brown fox jumps over the lazy dog",
        "pack my box with five dozen liquor jugs",
        "how vexingly quick daft zebras jump",
    ] * 20
    vocab, merges = train_bpe(corpus, vocab_size=64, special_tokens=SPECIAL_TOKENS)
    return TinyDocsTokenizer(vocab=vocab, merges=merges, special_tokens=SPECIAL_TOKENS)


def _write_corpus(tmp_path: Path, texts: list[str]) -> Path:
    p = tmp_path / "corpus.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for t in texts:
            f.write(json.dumps({"text": t, "source": "python-docs"}) + "\n")
    return p


# ------------------------------------------------------------------
# build_segments
# ------------------------------------------------------------------


def test_build_segments_shape_and_dtype(tmp_path: Path) -> None:
    """Output tensor must be 2-D with shape (n_segments, seq_len + 1) and dtype long."""
    tok = _tiny_tokenizer()
    corpus = _write_corpus(tmp_path, ["the quick brown fox jumps over the lazy dog"] * 30)

    seq_len = 16
    segments = build_segments(corpus, tok, seq_len=seq_len)

    assert segments.dim() == 2
    assert segments.shape[1] == seq_len + 1
    assert segments.dtype == torch.long
    assert segments.shape[0] >= 1


def test_build_segments_concatenates_across_docs_with_eos(tmp_path: Path) -> None:
    """All docs should be concatenated; <eos> ID should appear in the output."""
    tok = _tiny_tokenizer()
    corpus = _write_corpus(tmp_path, ["the dog", "the fox"] * 30)

    segments = build_segments(corpus, tok, seq_len=16)
    flat = segments.flatten().tolist()

    # eos should appear (between docs)
    assert tok.eos_id in flat


def test_build_segments_drops_partial_tail(tmp_path: Path) -> None:
    """If the total token count is not a multiple of (seq_len + 1), the
    incomplete tail is dropped (caller should not rely on partial segments)."""
    tok = _tiny_tokenizer()
    corpus = _write_corpus(tmp_path, ["the quick brown"])  # very short → one short doc

    segments = build_segments(corpus, tok, seq_len=64)

    # Either 0 segments (too short for one full segment) or 1 segment, no partial 2nd
    assert segments.shape[1] == 65
    assert segments.shape[0] in (0, 1)


# ------------------------------------------------------------------
# build_segments — n_workers (v3.1 §1.3 wiring)
# ------------------------------------------------------------------


def test_build_segments_n_workers_default_matches_pre_v31(tmp_path: Path) -> None:
    """Default n_workers (no kwarg) must keep the v3 §4a sequential behaviour."""
    tok = _tiny_tokenizer()
    corpus = _write_corpus(tmp_path, ["the quick brown fox jumps over the lazy dog"] * 30)

    segments = build_segments(corpus, tok, seq_len=16)

    # Sanity: produces at least one segment with the right shape (existing
    # behaviour). The point of this test is just to assert there's no
    # regression when callers don't pass n_workers.
    assert segments.dim() == 2
    assert segments.shape[1] == 17
    assert segments.shape[0] >= 1


def test_build_segments_n_workers_parallel_matches_sequential(tmp_path: Path) -> None:
    """n_workers=2 must produce the same tensor (byte-identical) as n_workers=1.

    The encode_batch_parallel path under the hood must preserve order;
    if it shuffles batches, segments diverge. This test catches that.
    """
    tok = _tiny_tokenizer()
    # Need enough docs to exercise the parallel batching path. With ~30 docs
    # and chunksize default the Pool sees real work, but stays fast in CI.
    corpus = _write_corpus(
        tmp_path,
        [
            "the quick brown fox jumps over the lazy dog",
            "pack my box with five dozen liquor jugs",
            "how vexingly quick daft zebras jump",
        ]
        * 10,
    )

    seq_len = 16
    seq_segments = build_segments(corpus, tok, seq_len=seq_len, n_workers=1)
    par_segments = build_segments(corpus, tok, seq_len=seq_len, n_workers=2)

    assert torch.equal(seq_segments, par_segments)


def test_build_segments_n_workers_skips_blank_lines(tmp_path: Path) -> None:
    """Blank / whitespace lines must still be filtered when running parallel."""
    tok = _tiny_tokenizer()
    corpus_path = tmp_path / "corpus.jsonl"
    with corpus_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"text": "the quick brown fox"}) + "\n")
        f.write("\n")  # blank line — must be skipped (matches sequential)
        f.write("   \n")  # whitespace
        f.write(json.dumps({"text": "jumps over the lazy dog"}) + "\n")

    seq_segments = build_segments(corpus_path, tok, seq_len=8, n_workers=1)
    par_segments = build_segments(corpus_path, tok, seq_len=8, n_workers=2)

    assert torch.equal(seq_segments, par_segments)


def test_build_segments_n_workers_propagates_bos_eos(tmp_path: Path) -> None:
    """Parallel path must still emit <bos>/<eos> markers between docs."""
    tok = _tiny_tokenizer()
    corpus = _write_corpus(tmp_path, ["the dog", "the fox"] * 20)

    par_segments = build_segments(corpus, tok, seq_len=16, n_workers=2)
    flat = par_segments.flatten().tolist()
    # eos must appear at doc boundaries — same invariant as sequential
    assert tok.eos_id in flat
    assert tok.bos_id in flat


def test_build_segments_n_workers_invalid_or_one_uses_sequential_path(
    tmp_path: Path,
) -> None:
    """n_workers <= 1 must still produce correct output (sequential fallback)."""
    tok = _tiny_tokenizer()
    corpus = _write_corpus(tmp_path, ["the quick brown fox"] * 5)

    out_default = build_segments(corpus, tok, seq_len=8)
    out_one = build_segments(corpus, tok, seq_len=8, n_workers=1)
    out_zero = build_segments(corpus, tok, seq_len=8, n_workers=0)

    # All three should be byte-identical (default == 1, 0 falls through to fallback).
    assert torch.equal(out_default, out_one)
    assert torch.equal(out_zero, out_one)


# ------------------------------------------------------------------
# TinyDocsDataset
# ------------------------------------------------------------------


def test_dataset_len_matches_segments() -> None:
    segments = torch.randint(0, 100, (10, 17), dtype=torch.long)
    ds = TinyDocsDataset(segments)
    assert len(ds) == 10


def test_dataset_getitem_returns_shifted_input_target_pair() -> None:
    """input_ids = segment[:-1], target_ids = segment[1:], both shape (seq_len,)."""
    seq_len = 16
    segments = torch.arange(0, 17, dtype=torch.long).unsqueeze(0)  # one segment [0..16]
    ds = TinyDocsDataset(segments)

    input_ids, target_ids = ds[0]

    assert input_ids.shape == (seq_len,)
    assert target_ids.shape == (seq_len,)
    # input_ids = [0, 1, ..., 15]; target_ids = [1, 2, ..., 16]
    assert torch.equal(input_ids, torch.arange(0, seq_len))
    assert torch.equal(target_ids, torch.arange(1, seq_len + 1))


def test_dataset_getitem_target_is_input_shifted_by_one() -> None:
    """For any segment, target[i] must equal input[i + 1]."""
    segments = torch.randint(0, 100, (5, 17), dtype=torch.long)
    ds = TinyDocsDataset(segments)

    for idx in range(len(ds)):
        input_ids, target_ids = ds[idx]
        assert torch.equal(input_ids[1:], target_ids[:-1])


def test_dataset_dataloader_compatible() -> None:
    """Dataset must work with torch.utils.data.DataLoader for batching."""
    from torch.utils.data import DataLoader

    segments = torch.randint(0, 100, (8, 17), dtype=torch.long)
    ds = TinyDocsDataset(segments)
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)

    batches = list(loader)
    assert len(batches) == 2
    input_batch, target_batch = batches[0]
    assert input_batch.shape == (4, 16)
    assert target_batch.shape == (4, 16)
