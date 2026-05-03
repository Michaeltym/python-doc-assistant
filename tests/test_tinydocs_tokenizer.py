"""Tests for TinyDocs BPE tokenizer (v3 §2)."""

from __future__ import annotations

from pathlib import Path

from python_doc_assistant.generation.tinydocs.tokenizer import TinyDocsTokenizer
from python_doc_assistant.generation.tinydocs.tokenizer_train import (
    pretokenize,
    train_bpe,
    train_bpe_incremental,
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
# train_bpe_incremental (v3.1 §1.2)
#
# Equivalence with naive `train_bpe` is the primary correctness gate.
# We compare on small frequency-unambiguous corpora so tie-breaking
# (which can diverge between naive and incremental — see docstring on
# train_bpe_incremental) doesn't muddy the comparison.
# ------------------------------------------------------------------


def test_incremental_matches_naive_simple_corpus() -> None:
    """Same (vocab, merges) as naive on a small unambiguous corpus."""
    corpus = ["aaaa bbbb"] * 100
    vocab_n, merges_n = train_bpe(corpus, vocab_size=12, special_tokens=SPECIAL_TOKENS)
    vocab_i, merges_i = train_bpe_incremental(
        corpus, vocab_size=12, special_tokens=SPECIAL_TOKENS
    )
    assert vocab_i == vocab_n
    assert merges_i == merges_n


def test_incremental_matches_naive_repeated_pairs() -> None:
    """Word with the same pair appearing multiple times (e.g., 'aaaa')."""
    corpus = ["aaaaa"] * 50
    vocab_n, merges_n = train_bpe(corpus, vocab_size=10, special_tokens=SPECIAL_TOKENS)
    vocab_i, merges_i = train_bpe_incremental(
        corpus, vocab_size=10, special_tokens=SPECIAL_TOKENS
    )
    assert vocab_i == vocab_n
    assert merges_i == merges_n


def test_incremental_matches_naive_multiword_corpus() -> None:
    """Mixed corpus, target small vocab to force several merges."""
    corpus = (
        ["hello world hello"] * 10
        + ["foo bar foo bar foo"] * 5
        + ["alpha beta gamma"] * 3
    )
    vocab_n, merges_n = train_bpe(corpus, vocab_size=24, special_tokens=SPECIAL_TOKENS)
    vocab_i, merges_i = train_bpe_incremental(
        corpus, vocab_size=24, special_tokens=SPECIAL_TOKENS
    )
    assert set(vocab_i) == set(vocab_n)
    # Merge multisets must match (order may differ on freq ties)
    assert sorted(merges_i) == sorted(merges_n)


def test_incremental_terminates_when_no_pairs_left() -> None:
    """If target vocab size > base + all possible merges, loop should exit."""
    corpus = ["a"] * 5  # only one char, no pairs to merge
    vocab, merges = train_bpe_incremental(
        corpus, vocab_size=100, special_tokens=SPECIAL_TOKENS
    )
    assert merges == []
    # vocab = 5 specials + 1 unique char = 6
    assert len(vocab) == 6


def test_incremental_special_tokens_first() -> None:
    """Same vocab ordering convention as naive: specials, base chars, merges."""
    corpus = ["hello world"]
    vocab, _ = train_bpe_incremental(
        corpus, vocab_size=20, special_tokens=SPECIAL_TOKENS
    )
    assert vocab[: len(SPECIAL_TOKENS)] == list(SPECIAL_TOKENS)


def test_incremental_reaches_target_vocab_size() -> None:
    """Hits the exact target size when corpus has enough pairs to merge."""
    corpus = (
        ["hello world how are you today"] * 10
        + ["the quick brown fox jumps over the lazy dog"] * 5
    )
    vocab, _ = train_bpe_incremental(
        corpus, vocab_size=48, special_tokens=SPECIAL_TOKENS
    )
    assert len(vocab) == 48


# ------------------------------------------------------------------
# encode_batch_parallel (v3.1 §1.3)
# ------------------------------------------------------------------


def _build_test_tokenizer() -> TinyDocsTokenizer:
    """Train a small tokenizer for parallel-encode tests (avoids re-training each test)."""
    corpus = [
        "the quick brown fox jumps over the lazy dog",
        "pack my box with five dozen liquor jugs",
        "how vexingly quick daft zebras jump",
        "sphinx of black quartz judge my vow",
    ] * 5
    vocab, merges = train_bpe(corpus, vocab_size=80, special_tokens=SPECIAL_TOKENS)
    return TinyDocsTokenizer(vocab=vocab, merges=merges, special_tokens=SPECIAL_TOKENS)


def test_parallel_matches_sequential() -> None:
    """Parallel output must equal sequential output element-wise."""
    tok = _build_test_tokenizer()
    texts = [
        "the quick brown fox",
        "pack my box with five",
        "how vexingly quick",
        "sphinx of black quartz",
    ] * 10
    seq = [tok.encode(t) for t in texts]
    par = tok.encode_batch_parallel(texts, n_workers=2)
    assert par == seq


def test_parallel_preserves_order() -> None:
    """`output[i]` corresponds to `texts[i]` (no reordering)."""
    tok = _build_test_tokenizer()
    texts = [f"text number {i} here" for i in range(40)]
    par = tok.encode_batch_parallel(texts, n_workers=2)
    for i, encoded in enumerate(par):
        assert encoded == tok.encode(texts[i])


def test_parallel_forwards_add_bos_eos() -> None:
    """`add_bos` / `add_eos` flags reach worker."""
    tok = _build_test_tokenizer()
    texts = ["hello", "world"] * 5
    par = tok.encode_batch_parallel(texts, n_workers=2, add_bos=True, add_eos=True)
    seq = [tok.encode(t, add_bos=True, add_eos=True) for t in texts]
    assert par == seq


def test_parallel_falls_back_to_sequential_for_small_inputs() -> None:
    """Few texts → sequential path (avoids Pool spawn overhead)."""
    tok = _build_test_tokenizer()
    texts = ["hello"]
    out = tok.encode_batch_parallel(texts, n_workers=4)
    assert out == [tok.encode("hello")]


def test_parallel_n_workers_1_uses_sequential() -> None:
    """`n_workers=1` skips multiprocessing entirely."""
    tok = _build_test_tokenizer()
    texts = ["hello world"] * 10
    out = tok.encode_batch_parallel(texts, n_workers=1)
    assert out == [tok.encode(t) for t in texts]


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
