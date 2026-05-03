"""TinyDocs tokenizer — encode / decode / save / load wrapper.

Built on top of (vocab, merges) from `tokenizer_train.train_bpe`.
See plans/v3-tiny-llm.md §2.
"""

from __future__ import annotations

import json
from pathlib import Path

from python_doc_assistant.generation.tinydocs.tokenizer_train import merge_pair, pretokenize


class TinyDocsTokenizer:
    """BPE tokenizer with merges-in-order encoding and special-token support."""

    def __init__(
        self,
        vocab: list[str],
        merges: list[tuple[str, str]],
        special_tokens: tuple[str, ...] = ("<pad>", "<bos>", "<eos>", "<unk>", "<sp>"),
    ) -> None:
        self.vocab = vocab
        self.token_to_id = {token: i for i, token in enumerate(vocab)}
        self.merges = merges
        self.merge_priority = {pair: i for i, pair in enumerate(merges)}
        self.special_tokens = special_tokens
        self._pad_id = self.token_to_id["<pad>"]
        self._bos_id = self.token_to_id["<bos>"]
        self._eos_id = self.token_to_id["<eos>"]
        self._unk_id = self.token_to_id["<unk>"]
        self._sp_id = self.token_to_id["<sp>"]
        self._encode_cache: dict[str, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_id(self) -> int:
        return self._pad_id

    @property
    def bos_id(self) -> int:
        return self._bos_id

    @property
    def eos_id(self) -> int:
        return self._eos_id

    @property
    def unk_id(self) -> int:
        return self._unk_id

    @property
    def sp_id(self) -> int:
        return self._sp_id

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = []
        tokens = pretokenize(text)
        if add_bos:
            ids.append(self.bos_id)
        for i, token in enumerate(tokens):
            if i > 0:
                ids.append(self._sp_id)
            ids.extend(self._encode_token(token))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        parts = []
        for i in ids:
            if i in [self.bos_id, self.eos_id, self.pad_id]:
                continue
            if i == self._sp_id:
                parts.append(" ")
                continue
            parts.append(self.vocab[i])

        return "".join(parts)

    def save(self, path: Path) -> None:
        """Write vocab + merges + special tokens to a single JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "vocab": self.vocab,
                    "merges": [list(m) for m in self.merges],
                    "special_tokens": self.special_tokens,
                },
                f,
            )

    @classmethod
    def load(cls, path: Path) -> TinyDocsTokenizer:
        """Read a saved tokenizer JSON file."""
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist")
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            vocab=data["vocab"],
            merges=[tuple(m) for m in data["merges"]],
            special_tokens=tuple(data["special_tokens"]),
        )

    def _encode_token(self, token: str) -> list[int]:
        if token in self._encode_cache:
            return self._encode_cache[token]
        chars = list(token)
        sentinel = len(self.merge_priority) + 1
        while len(chars) > 1:
            pairs = [(chars[i], chars[i + 1]) for i in range(len(chars) - 1)]
            best_pair = min(pairs, key=lambda p: self.merge_priority.get(p, sentinel))
            if self.merge_priority.get(best_pair, sentinel) >= sentinel:
                break
            chars = merge_pair(chars, best_pair)
        result = [self.token_to_id.get(c, self.unk_id) for c in chars]
        self._encode_cache[token] = result
        return result

    def encode_batch_parallel(
        self,
        texts: list[str],
        *,
        n_workers: int = 4,
        add_bos: bool = False,
        add_eos: bool = False,
        chunksize: int = 100,
    ) -> list[list[int]]:
        """Encode `texts` across `n_workers` processes via multiprocessing.Pool.

        Used by v3.1 §1.3 to speed up encoding the 2.4 GB FineWeb mix corpus
        from ~3 h single-thread to ~1 h on M1's 4 performance cores.

        Each worker:
          - Receives a pickled snapshot of the tokenizer (vocab + merges)
          - Maintains its own per-word `_encode_cache` (no cross-process sharing)
          - Encodes its assigned chunk of texts via the same `encode()` path

        Order is preserved: `output[i] == self.encode(texts[i], add_bos=...)`.

        For small inputs (or n_workers <= 1) falls back to sequential
        encoding to avoid `Pool` startup overhead (~1 s per worker on
        macOS spawn).

        Args:
            texts: list of input strings to encode.
            n_workers: number of worker processes. <= 1 → sequential fallback.
            add_bos: forwarded to `encode()`.
            add_eos: forwarded to `encode()`.
            chunksize: per-worker batch size for `Pool.map`. Larger values
                amortize IPC overhead but worsen load balancing on
                heterogeneous text lengths. 100 is a good default for ~3 KB
                lines (FineWeb scale).

        Returns:
            list[list[int]] — same length and order as `texts`.
        """
        if n_workers <= 1 or len(texts) < n_workers * chunksize:
            return [self.encode(text, add_bos=add_bos, add_eos=add_eos) for text in texts]
        from functools import partial
        from multiprocessing import Pool

        fn = partial(_worker_encode, tokenizer=self, add_bos=add_bos, add_eos=add_eos)
        with Pool(n_workers) as pool:
            results = pool.map(fn, texts, chunksize=chunksize)
        return results


def _worker_encode(
    text: str,
    *,
    tokenizer: TinyDocsTokenizer,
    add_bos: bool,
    add_eos: bool,
) -> list[int]:
    """Module-level helper for `multiprocessing.Pool.map` (must be picklable).

    Bound methods can't be sent to workers directly on macOS `spawn` start
    method; this thin wrapper sidesteps the issue. Callers pass the
    tokenizer instance via `functools.partial(_worker_encode, tokenizer=...)`.
    """
    return tokenizer.encode(text, add_bos=add_bos, add_eos=add_eos)
