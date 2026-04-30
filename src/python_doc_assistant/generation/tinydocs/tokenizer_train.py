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
