"""BPE training algorithm — pure functions.

See plans/v3-tiny-llm.md §2. Public entry point is `train_bpe`.
"""

from __future__ import annotations

import re
from collections.abc import Iterable


def pretokenize(text: str) -> list[str]:
    """Split text into word units. MVP: whitespace + simple punctuation."""
    return re.findall(r"[\w']+|[^\w\s']", text)


def merge_pair(chars: list[str], pair: tuple[str, str]) -> list[str]:
    new_chars = []
    i = 0
    while i < len(chars):
        if i < len(chars) - 1 and (chars[i], chars[i + 1]) == pair:
            new_chars.append(chars[i] + chars[i + 1])
            i += 2
        else:
            new_chars.append(chars[i])
            i += 1
    return new_chars


def train_bpe(
    corpus: Iterable[str],
    *,
    vocab_size: int,
    special_tokens: tuple[str, ...],
) -> tuple[list[str], list[tuple[str, str]]]:
    """Train BPE on a corpus iterator.

    Returns:
        (vocab, merges) where:
            vocab: list of token strings, ordered (special tokens first,
                then base chars, then merged tokens by merge order).
            merges: list of (left, right) pairs in training order.
    """
    vocab: list[str] = []
    vocab.extend(special_tokens)
    vocab_freqs_dict: dict[str, int] = {}
    merges: list[tuple[str, str]] = []
    for sentence in corpus:
        tokens = pretokenize(sentence)
        for token in tokens:
            vocab_freqs_dict.setdefault(token, 0)
            vocab_freqs_dict[token] += 1
    # { "abc": (["a, b, c"], 100) }
    splits: dict[str, tuple[list[str], int]] = {
        token: (list(token), freq) for token, freq in vocab_freqs_dict.items()
    }
    unique_chars = sorted(set("".join([token for token, _ in vocab_freqs_dict.items()])))
    vocab.extend(unique_chars)

    while len(vocab) < vocab_size:
        pair_freqs: dict[tuple[str, str], int] = {}
        for token, (chars, freq) in splits.items():
            for i in range(len(chars) - 1):
                pair = (chars[i], chars[i + 1])
                pair_freqs.setdefault(pair, 0)
                pair_freqs[pair] += freq
        if not pair_freqs:
            break  # no more pairs to merge
        best_pair = max(pair_freqs, key=pair_freqs.__getitem__)
        new_token = best_pair[0] + best_pair[1]
        splits = {
            token: _merge_chars(chars_with_freq, best_pair)
            for token, chars_with_freq in splits.items()
        }
        merges.append(best_pair)
        vocab.append(new_token)
    return vocab, merges


def _merge_chars(
    chars_with_freq: tuple[list[str], int], best_pair: tuple[str, str]
) -> tuple[list[str], int]:
    chars, freq = chars_with_freq
    return merge_pair(chars, best_pair), freq


# ------------------------------------------------------------------
# Incremental BPE (v3.1 §1.2)
# ------------------------------------------------------------------


def train_bpe_incremental(
    corpus: Iterable[str],
    *,
    vocab_size: int,
    special_tokens: tuple[str, ...],
    show_progress: bool = False,
) -> tuple[list[str], list[tuple[str, str]]]:
    """Same contract as `train_bpe`, but maintains pair-frequency state across
    merges so each iteration only updates the words affected by the merged
    pair, not the full corpus.

    Algorithm (v3.1 §1.2):
      1. Pretokenize + count word frequencies (same as naive `train_bpe`).
      2. Initialise base vocab with special tokens + unique characters.
      3. Build TWO indexes (kept up-to-date for the rest of the run):
           pair_freqs: dict[(str, str), int]
             — global frequency of each adjacent char pair
           pair_to_words: dict[(str, str), set[str]]
             — inverted index: pair → set of word keys containing it
      4. Loop until len(vocab) reaches vocab_size or no pairs remain:
           a. Find best_pair = max(pair_freqs, key=pair_freqs.__getitem__)
           b. Take a snapshot of pair_to_words[best_pair] (the affected words)
           c. For each affected word w (freq f) — keep this loop tight:
                - Read current `chars` from splits[w]
                - Subtract every (pair, count) of `chars` from pair_freqs and
                  prune pair_to_words[pair].discard(w) when count drops
                - Apply merge_pair(chars, best_pair) to get new_chars
                - Update splits[w] = (new_chars, f)
                - Add every (pair, count) of `new_chars` back to pair_freqs
                  and pair_to_words[pair].add(w)
           d. Append best_pair → merges, "".join(best_pair) → vocab

    Equivalence with naive `train_bpe`:
      - Both algorithms produce the same set of merges in the same order
        for unambiguous frequency rankings.
      - Tie-breaking depends on dict insertion order (Python 3.7+); naive
        rebuilds pair_freqs each iteration so pairs sort by current word-
        iteration order. Incremental keeps the dict alive: new pairs created
        by merges land at the end of insertion order, so when freq ties occur
        the two algorithms can pick *different* pairs. For strict equivalence
        on test corpora, choose inputs without low-frequency ties (or compare
        as sets — the tests cover both modes).

    Expected speedup: 10–30 × on real corpora (sentencepiece / fastBPE
    benchmarks), since each merge only touches ~|pair_to_words[pair]|
    words instead of all unique words.
    """
    vocab: list[str] = []
    vocab.extend(special_tokens)
    merges: list[tuple[str, str]] = []
    vocab_freqs_dict: dict[str, int] = {}
    for sentence in corpus:
        tokens = pretokenize(sentence)
        for token in tokens:
            vocab_freqs_dict.setdefault(token, 0)
            vocab_freqs_dict[token] += 1
    tokens_str = ""
    # before: { (a, b): 10, (b, c): 30, (c, d): 70, (d, e): 50 }
    # after: { (a, b): 10, (b, c): 10, (b, cd): 20, (cd, e): 50 }
    pair_freqs: dict[tuple[str, str], int] = {}
    # before: { (a, b): { "abc" }, (b, c): { "abc", "bcd" },
    #           (c, d): { "bcd", "cde" }, (d, e): { "cde" } }
    # after:  { (a, b): { "abc" }, (b, cd): { "bcd" }, (cd, e): { "cde" } }
    pair_to_words: dict[tuple[str, str], set[str]] = {}
    # before: { abc: (["a", "b", "c"], 10), bcd: (["b", "c", "d"], 20), cde: (["c", "d", "e"], 50) }
    splits: dict[str, tuple[list[str], int]] = {}
    for token, freq in vocab_freqs_dict.items():
        tokens_str += token
        splits[token] = (list(token), freq)
        for i in range(len(list(token)) - 1):
            pair = (token[i], token[i + 1])
            pair_freqs.setdefault(pair, 0)
            pair_freqs[pair] += freq
            pair_to_words.setdefault(pair, set())
            pair_to_words[pair].add(token)

    unique_chars = sorted(set(tokens_str))
    vocab.extend(unique_chars)

    pbar = None
    if show_progress:
        from tqdm import tqdm

        pbar = tqdm(
            total=vocab_size,
            initial=len(vocab),
            desc="BPE merges",
            unit="merge",
        )

    while len(vocab) < vocab_size:
        if not pair_freqs:
            break
        best_pair = max(pair_freqs, key=pair_freqs.__getitem__)
        new_token = best_pair[0] + best_pair[1]
        affected_tokens = list(pair_to_words[best_pair])
        for token in affected_tokens:
            chars, freq = splits[token]
            for pair, count in _word_pairs(chars).items():
                pair_freqs[pair] -= count * freq
                if pair_freqs[pair] <= 0:
                    del pair_freqs[pair]
                    pair_to_words[pair].discard(token)
                    if not pair_to_words[pair]:
                        del pair_to_words[pair]
                else:
                    pair_to_words[pair].discard(token)

            new_chars = merge_pair(chars, best_pair)
            splits[token] = (new_chars, freq)

            for pair, count in _word_pairs(new_chars).items():
                pair_freqs[pair] = pair_freqs.get(pair, 0) + count * freq
                pair_to_words.setdefault(pair, set()).add(token)

        merges.append(best_pair)
        vocab.append(new_token)
        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()
    return vocab, merges


def _word_pairs(chars: list[str]) -> dict[tuple[str, str], int]:
    """Count each adjacent pair occurrence in chars."""
    counts: dict[tuple[str, str], int] = {}
    for i in range(len(chars) - 1):
        pair = (chars[i], chars[i + 1])
        counts[pair] = counts.get(pair, 0) + 1
    return counts
