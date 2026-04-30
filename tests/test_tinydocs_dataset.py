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
