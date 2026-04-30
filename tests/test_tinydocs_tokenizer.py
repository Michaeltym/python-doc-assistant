"""Tests for TinyDocs BPE tokenizer (v3 §2)."""

from __future__ import annotations

from pathlib import Path

from python_doc_assistant.generation.tinydocs.tokenizer import TinyDocsTokenizer
from python_doc_assistant.generation.tinydocs.tokenizer_train import (
    pretokenize,
    train_bpe,
)

# ------------------------------------------------------------------
# pretokenize
# ------------------------------------------------------------------


def test_pretokenize_splits_whitespace() -> None:
    out = pretokenize("hello world  foo")
    assert out == ["hello", "world", "foo"]


def test_pretokenize_separates_punctuation() -> None:
    out = pretokenize("hello, world!")
    assert "," in out
    assert "!" in out
    assert "hello" in out
    assert "world" in out


# ------------------------------------------------------------------
# train_bpe
# ------------------------------------------------------------------


SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<unk>", "<sp>")


def test_train_bpe_reaches_target_vocab_size() -> None:
    corpus = (
        ["hello world how are you today"] * 10
        + ["the quick brown fox jumps over the lazy dog"] * 5
        + ["python programming language tutorial guide"] * 5
    )
    vocab, merges = train_bpe(corpus, vocab_size=64, special_tokens=SPECIAL_TOKENS)
    assert len(vocab) == 64


def test_train_bpe_special_tokens_first() -> None:
    corpus = ["hello world"]
    vocab, _ = train_bpe(corpus, vocab_size=32, special_tokens=SPECIAL_TOKENS)
    assert vocab[0] == "<pad>"
    assert vocab[1] == "<bos>"
    assert vocab[2] == "<eos>"
    assert vocab[3] == "<unk>"
    assert vocab[4] == "<sp>"


def test_train_bpe_returns_merges_in_order() -> None:
    corpus = ["aaaa bbbb"] * 100
    vocab, merges = train_bpe(corpus, vocab_size=20, special_tokens=SPECIAL_TOKENS)
    # First merge should be either (a, a) or (b, b) since those are most frequent
    assert merges[0] in {("a", "a"), ("b", "b")}


# ------------------------------------------------------------------
# TinyDocsTokenizer
# ------------------------------------------------------------------


def _train_small_tokenizer() -> TinyDocsTokenizer:
    corpus = [
        "the quick brown fox jumps over the lazy dog",
        "pack my box with five dozen liquor jugs",
        "how vexingly quick daft zebras jump",
        "sphinx of black quartz judge my vow",
        "the five boxing wizards jump quickly",
    ] * 20
    vocab, merges = train_bpe(corpus, vocab_size=64, special_tokens=SPECIAL_TOKENS)
    return TinyDocsTokenizer(vocab=vocab, merges=merges, special_tokens=SPECIAL_TOKENS)


def test_tokenizer_round_trip_single_word() -> None:
    """A frequent in-vocab word should round-trip cleanly (single token expected)."""
    tok = _train_small_tokenizer()
    text = "the"  # appears 60 times in the pangram corpus → likely a single merged token
    ids = tok.encode(text)
    out = tok.decode(ids)
    assert out == text


def test_tokenizer_round_trip_multi_word() -> None:
    """Multi-word text should round-trip via `<sp>` token between words."""
    tok = _train_small_tokenizer()
    text = "the the"  # both words fully merged + <sp> in the middle
    ids = tok.encode(text)
    out = tok.decode(ids)
    assert out == text


def test_tokenizer_inserts_sp_between_words() -> None:
    """`<sp>` ID must appear between every pair of word tokens."""
    tok = _train_small_tokenizer()
    ids = tok.encode("the quick")
    # `<sp>` should appear at least once (between the two words)
    assert tok.sp_id in ids


def test_tokenizer_vocab_size_property() -> None:
    tok = _train_small_tokenizer()
    assert tok.vocab_size == 64


def test_tokenizer_special_token_ids() -> None:
    tok = _train_small_tokenizer()
    assert tok.pad_id == 0
    assert tok.bos_id == 1
    assert tok.eos_id == 2
    assert tok.unk_id == 3
    assert tok.sp_id == 4
    # all distinct + within vocab range
    ids = {tok.pad_id, tok.bos_id, tok.eos_id, tok.unk_id, tok.sp_id}
    assert len(ids) == 5
    assert all(0 <= i < tok.vocab_size for i in ids)


def test_tokenizer_add_bos_eos() -> None:
    tok = _train_small_tokenizer()
    ids = tok.encode("the", add_bos=True, add_eos=True)
    assert ids[0] == tok.bos_id
    assert ids[-1] == tok.eos_id


def test_tokenizer_encode_is_deterministic() -> None:
    tok = _train_small_tokenizer()
    a = tok.encode("the quick")
    b = tok.encode("the quick")
    assert a == b


def test_tokenizer_save_load_round_trip(tmp_path: Path) -> None:
    """Save + load should reproduce the same tokenizer behavior."""
    tok = _train_small_tokenizer()
    path = tmp_path / "tokenizer.json"
    tok.save(path)
    loaded = TinyDocsTokenizer.load(path)
    assert loaded.vocab_size == tok.vocab_size
    text = "the quick"
    assert loaded.encode(text) == tok.encode(text)
