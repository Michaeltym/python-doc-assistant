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
        for pair in self.merges:
            chars = merge_pair(chars, pair)
        result = [self.token_to_id.get(c, self.unk_id) for c in chars]
        self._encode_cache[token] = result
        return result
