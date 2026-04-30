"""Pre-tokenized corpus → fixed-length segments → torch Dataset.

See plans/v3-tiny-llm.md §4a.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset

from python_doc_assistant.generation.tinydocs.tokenizer import TinyDocsTokenizer


def build_segments(
    corpus_path: Path,
    tokenizer: TinyDocsTokenizer,
    *,
    seq_len: int,
) -> Tensor:
    """Read corpus.jsonl, encode every text, concatenate (with BOS/EOS markers
    between docs), and split into segments of length `seq_len + 1`.

    Returns a Tensor of shape (n_segments, seq_len + 1), dtype torch.long.
    The trailing token in each segment is the target for the previous one
    (caller does the shift).
    """
    token_ids: list[int] = []
    if not corpus_path.exists():
        raise FileNotFoundError(f"{corpus_path} does not exist")
    with corpus_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            stripped_line = line.strip()
            if not stripped_line:
                continue
            try:
                data = json.loads(stripped_line)
                text = data["text"]
                ids = tokenizer.encode(text, add_bos=True, add_eos=True)
                token_ids.extend(ids)
            except json.JSONDecodeError as e:
                raise ValueError(f"Line {i}: invalid json object") from e
    n_segments = len(token_ids) // (seq_len + 1)

    return torch.tensor(token_ids[: n_segments * (seq_len + 1)], dtype=torch.long).reshape(
        n_segments, seq_len + 1
    )


class TinyDocsDataset(Dataset[tuple[Tensor, Tensor]]):
    """Wrap pre-tokenized segments as a torch Dataset of (input, target) pairs."""

    def __init__(self, segments: Tensor) -> None:
        self.segments = segments

    def __len__(self) -> int:
        return self.segments.shape[0]

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Return (input_ids, target_ids), each shape (seq_len,).

        target_ids is input_ids shifted by 1: target[i] = input[i+1].
        """
        if idx >= self.__len__():
            raise ValueError(f"Invalid idx {idx}")
        segment = self.segments[idx, :]
        return (segment[:-1], segment[1:])
