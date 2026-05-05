"""Tests for v3.1 §6 SFT fine-tuning loop."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from python_doc_assistant.generation.tinydocs.config import TinyDocsConfig  # noqa: E402
from python_doc_assistant.generation.tinydocs.sft_train import (  # noqa: E402
    LOSS_IGNORE_INDEX,
    SFTConfig,
    SFTDataset,
    SFTRecord,
    build_sft_record,
    build_sft_records,
    collate_sft,
    load_chunk_lookup,
    train_sft,
)
from python_doc_assistant.generation.tinydocs.tokenizer import TinyDocsTokenizer  # noqa: E402
from python_doc_assistant.generation.tinydocs.tokenizer_train import train_bpe  # noqa: E402
from python_doc_assistant.ingest.chunker import Chunk  # noqa: E402

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def tiny_tokenizer() -> TinyDocsTokenizer:
    """Train a tiny BPE on toy text so the test corpus has predictable ids."""
    sentences = ["the quick brown fox jumps over the lazy dog"] * 4
    specials = ("<pad>", "<bos>", "<eos>", "<unk>", "<sp>")
    vocab, merges = train_bpe(sentences, vocab_size=200, special_tokens=specials)
    return TinyDocsTokenizer(vocab=vocab, merges=merges, special_tokens=specials)


@pytest.fixture
def chunks_jsonl(tmp_path: Path) -> Path:
    """Write a 2-chunk jsonl file matching real chunks.jsonl schema."""
    path = tmp_path / "chunks.jsonl"
    rows = [
        {
            "chunk_id": "symbol:pathlib.Path.read_text",
            "chunk_type": "symbol",
            "docs_version": "3.12",
            "title": "Path.read_text",
            "text": "Read the text contents of the path.",
            "symbols": ["pathlib.Path.read_text"],
            "canonical_url": "library/pathlib.html#pathlib.Path.read_text",
            "anchor": "pathlib.Path.read_text",
            "parent_module": "pathlib",
            "source_path": "library/pathlib.html",
            "source_hash": "sha256:deadbeef",
        },
        {
            "chunk_id": "symbol:pathlib.Path.write_text",
            "chunk_type": "symbol",
            "docs_version": "3.12",
            "title": "Path.write_text",
            "text": "Open the path in text mode and write data to it.",
            "symbols": ["pathlib.Path.write_text"],
            "canonical_url": "library/pathlib.html#pathlib.Path.write_text",
            "anchor": "pathlib.Path.write_text",
            "parent_module": "pathlib",
            "source_path": "library/pathlib.html",
            "source_hash": "sha256:deadbeef",
        },
    ]
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return path


@pytest.fixture
def sft_jsonl(tmp_path: Path) -> Path:
    """Three SFT rows: one happy, one refusal (must skip), one missing chunk."""
    path = tmp_path / "sft_corpus.jsonl"
    rows = [
        {
            "query": "How do I read a file?",
            "query_type": "natural_language",
            "source": "synthetic",
            "retrieved_chunk_ids": ["symbol:pathlib.Path.read_text"],
            "qwen_answer": "Use Path.read_text [1].",
            "qwen_refused": False,
            "qwen_latency_seconds": 1.2,
        },
        {
            "query": "How do I cure cancer?",
            "query_type": "natural_language",
            "source": "synthetic",
            "retrieved_chunk_ids": ["symbol:pathlib.Path.read_text"],
            "qwen_answer": "[INSUFFICIENT-CONTEXT]",
            "qwen_refused": True,
            "qwen_latency_seconds": 0.5,
        },
        {
            "query": "What is foo?",
            "query_type": "natural_language",
            "source": "synthetic",
            "retrieved_chunk_ids": ["symbol:does.not.exist"],
            "qwen_answer": "Foo is a thing [1].",
            "qwen_refused": False,
            "qwen_latency_seconds": 1.0,
        },
    ]
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return path


# ------------------------------------------------------------------
# load_chunk_lookup
# ------------------------------------------------------------------


def test_load_chunk_lookup_indexes_by_id(chunks_jsonl: Path) -> None:
    lookup = load_chunk_lookup(chunks_jsonl)
    assert "symbol:pathlib.Path.read_text" in lookup
    assert "symbol:pathlib.Path.write_text" in lookup
    chunk = lookup["symbol:pathlib.Path.read_text"]
    assert isinstance(chunk, Chunk)
    assert chunk.title == "Path.read_text"


def test_load_chunk_lookup_rejects_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "dup.jsonl"
    row = {
        "chunk_id": "symbol:foo",
        "chunk_type": "symbol",
        "docs_version": "3.12",
        "title": "foo",
        "text": "x",
        "symbols": [],
        "canonical_url": "library/foo.html",
        "anchor": None,
        "parent_module": None,
        "source_path": "library/foo.html",
        "source_hash": "sha256:0",
    }
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
        f.write(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_chunk_lookup(path)


# ------------------------------------------------------------------
# build_sft_record
# ------------------------------------------------------------------


def test_build_sft_record_masks_prompt_in_labels(
    chunks_jsonl: Path,
    tiny_tokenizer: TinyDocsTokenizer,
) -> None:
    """labels must equal LOSS_IGNORE_INDEX on prompt positions, real ids on answer."""
    lookup = load_chunk_lookup(chunks_jsonl)
    row = {
        "query": "the fox",
        "query_type": "natural_language",
        "source": "eval",
        "retrieved_chunk_ids": ["symbol:pathlib.Path.read_text"],
        "qwen_answer": "fox",
        "qwen_refused": False,
        "qwen_latency_seconds": 1.0,
    }
    record = build_sft_record(row, lookup, tiny_tokenizer, max_seq_len=256)
    assert record is not None
    # Prompt portion all -100
    prompt_labels = record.labels[: record.prompt_len]
    assert (prompt_labels == LOSS_IGNORE_INDEX).all().item()
    # Answer portion has real ids matching input_ids on the same positions
    answer_labels = record.labels[record.prompt_len :]
    answer_ids = record.input_ids[record.prompt_len :]
    assert torch.equal(answer_labels, answer_ids)
    # Length invariant
    assert record.input_ids.shape == record.labels.shape


def test_build_sft_record_skips_refused(
    chunks_jsonl: Path,
    tiny_tokenizer: TinyDocsTokenizer,
) -> None:
    lookup = load_chunk_lookup(chunks_jsonl)
    row = {
        "query": "x",
        "query_type": "natural_language",
        "source": "synthetic",
        "retrieved_chunk_ids": ["symbol:pathlib.Path.read_text"],
        "qwen_answer": "[INSUFFICIENT-CONTEXT]",
        "qwen_refused": True,
        "qwen_latency_seconds": 0.5,
    }
    assert build_sft_record(row, lookup, tiny_tokenizer, max_seq_len=256) is None


def test_build_sft_record_skips_missing_chunk(
    chunks_jsonl: Path,
    tiny_tokenizer: TinyDocsTokenizer,
) -> None:
    lookup = load_chunk_lookup(chunks_jsonl)
    row = {
        "query": "x",
        "query_type": "natural_language",
        "source": "synthetic",
        "retrieved_chunk_ids": ["symbol:does.not.exist"],
        "qwen_answer": "y [1].",
        "qwen_refused": False,
        "qwen_latency_seconds": 0.5,
    }
    assert build_sft_record(row, lookup, tiny_tokenizer, max_seq_len=256) is None


# ------------------------------------------------------------------
# build_sft_records (loader)
# ------------------------------------------------------------------


def test_build_sft_records_filters_invalid(
    chunks_jsonl: Path,
    sft_jsonl: Path,
    tiny_tokenizer: TinyDocsTokenizer,
) -> None:
    records = build_sft_records(sft_jsonl, chunks_jsonl, tiny_tokenizer, max_seq_len=256)
    # Only the happy row survives (3 input rows, 1 refused, 1 missing chunk)
    assert len(records) == 1
    assert all(isinstance(r, SFTRecord) for r in records)


# ------------------------------------------------------------------
# collate_sft
# ------------------------------------------------------------------


def test_collate_sft_pads_to_longest(tiny_tokenizer: TinyDocsTokenizer) -> None:
    """Pad input_ids with pad_token_id, labels with LOSS_IGNORE_INDEX."""
    pad = tiny_tokenizer.pad_id
    short_ids = torch.tensor([1, 2, 3], dtype=torch.long)
    short_lab = torch.tensor([LOSS_IGNORE_INDEX, LOSS_IGNORE_INDEX, 3], dtype=torch.long)
    long_ids = torch.tensor([4, 5, 6, 7, 8], dtype=torch.long)
    long_lab = torch.tensor([LOSS_IGNORE_INDEX, 5, 6, 7, 8], dtype=torch.long)
    batch_ids, batch_lab = collate_sft(
        [(short_ids, short_lab), (long_ids, long_lab)],
        pad_token_id=pad,
    )
    assert batch_ids.shape == (2, 5)
    assert batch_lab.shape == (2, 5)
    # short row right-padded
    assert batch_ids[0, 3].item() == pad
    assert batch_ids[0, 4].item() == pad
    assert batch_lab[0, 3].item() == LOSS_IGNORE_INDEX
    assert batch_lab[0, 4].item() == LOSS_IGNORE_INDEX
    # long row unchanged
    assert torch.equal(batch_ids[1], long_ids)
    assert torch.equal(batch_lab[1], long_lab)


# ------------------------------------------------------------------
# SFTDataset
# ------------------------------------------------------------------


def test_sft_dataset_returns_record_tensors() -> None:
    rec = SFTRecord(
        input_ids=torch.tensor([1, 2, 3], dtype=torch.long),
        labels=torch.tensor([LOSS_IGNORE_INDEX, 2, 3], dtype=torch.long),
        prompt_len=1,
    )
    ds = SFTDataset([rec])
    assert len(ds) == 1
    ids, lab = ds[0]
    assert torch.equal(ids, rec.input_ids)
    assert torch.equal(lab, rec.labels)


def test_sft_dataset_index_out_of_range_raises() -> None:
    ds = SFTDataset([])
    with pytest.raises(IndexError):
        ds[0]


# ------------------------------------------------------------------
# train_sft (smoke — runs 1 epoch on 2 records, no NaN)
# ------------------------------------------------------------------


def _save_init_ckpt(tmp_path: Path, model_config: TinyDocsConfig) -> Path:
    """Save a minimal init ckpt so train_sft can load fresh state."""
    from dataclasses import asdict

    from python_doc_assistant.generation.tinydocs.model import TinyDocsModel

    model = TinyDocsModel(model_config)
    ckpt = {
        "step": 0,
        "model_state": model.state_dict(),
        "optimizer_state": {},
        "model_config": asdict(model_config),
    }
    path = tmp_path / "init.pt"
    torch.save(ckpt, path)
    return path


def test_train_sft_smoke(
    tmp_path: Path,
    tiny_tokenizer: TinyDocsTokenizer,
) -> None:
    """One-epoch run on 2 tiny records: ckpt + log written, loss finite."""
    model_config = TinyDocsConfig(
        vocab_size=tiny_tokenizer.vocab_size,
        max_seq_len=64,
        hidden_dim=32,
        n_layers=2,
        n_heads=2,
        n_kv_heads=2,
        head_dim=16,
    )
    init_ckpt_path = _save_init_ckpt(tmp_path, model_config)

    rec_a = SFTRecord(
        input_ids=torch.tensor([1, 2, 3, 4, 5], dtype=torch.long),
        labels=torch.tensor([LOSS_IGNORE_INDEX, LOSS_IGNORE_INDEX, 3, 4, 5], dtype=torch.long),
        prompt_len=2,
    )
    rec_b = SFTRecord(
        input_ids=torch.tensor([2, 3, 4, 5], dtype=torch.long),
        labels=torch.tensor([LOSS_IGNORE_INDEX, 3, 4, 5], dtype=torch.long),
        prompt_len=1,
    )
    dataset = SFTDataset([rec_a, rec_b])
    sft_config = SFTConfig(
        base_lr=1e-4,
        epochs=1,
        batch_size=2,
        warmup_steps=0,
        log_every=1,
        checkpoint_every_epochs=1,
        device="cpu",
        seed=42,
    )
    ckpt_dir = tmp_path / "run-sft-smoke"
    log_path = ckpt_dir / "loss.jsonl"

    train_sft(
        model_config,
        sft_config,
        dataset,
        init_ckpt_path=init_ckpt_path,
        ckpt_dir=ckpt_dir,
        log_path=log_path,
        pad_token_id=tiny_tokenizer.pad_id,
    )

    # Final ckpt produced
    assert (ckpt_dir / "step_final.pt").exists()
    # Log has at least one line and loss is finite
    lines = log_path.read_text().strip().splitlines()
    assert lines
    entry = json.loads(lines[0])
    assert "loss" in entry
    assert torch.isfinite(torch.tensor(entry["loss"])).item()
