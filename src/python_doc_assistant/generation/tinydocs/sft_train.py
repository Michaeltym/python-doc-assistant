"""SFT distillation loop for TinyDocs (v3.1 §6).

Fine-tunes a pretrained TinyDocs ckpt on (query, retrieved_chunks, qwen_answer)
triples produced by `scripts/build_sft_corpus.py`. Loss is masked to answer
tokens only — prompt tokens contribute zero gradient.

Schema (one JSON per line in sft_corpus.jsonl):
    {
        "query": str,
        "query_type": str,
        "source": "eval" | "synthetic",
        "retrieved_chunk_ids": list[str],
        "qwen_answer": str,
        "qwen_refused": bool,
        "qwen_latency_seconds": float,
    }

Chunk text lookup happens against `data/chunks/<docs_version>/<sha_short>/chunks.jsonl`
keyed by `chunk_id`. The prompt is rebuilt with `build_grounded_prompt` so
the trainer sees the same template the inference path uses.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import TypedDict

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from python_doc_assistant.generation.tinydocs.config import TinyDocsConfig
from python_doc_assistant.generation.tinydocs.model import TinyDocsModel
from python_doc_assistant.generation.tinydocs.tokenizer import TinyDocsTokenizer
from python_doc_assistant.ingest.chunker import Chunk
from python_doc_assistant.prompts.grounded import build_grounded_prompt
from python_doc_assistant.retrieval.router import QueryType

# Sentinel for CrossEntropyLoss to skip prompt tokens.
LOSS_IGNORE_INDEX = -100


@dataclass
class SFTConfig:
    """Hyperparameters for SFT fine-tuning."""

    base_lr: float = 5e-6
    epochs: int = 8
    batch_size: int = 4
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    warmup_steps: int = 20
    log_every: int = 10
    checkpoint_every_epochs: int = 4
    device: str = "mps"
    seed: int = 42


@dataclass
class SFTRecord:
    """One encoded SFT example.

    Attributes:
        input_ids: prompt tokens followed by answer tokens (1-D LongTensor).
        labels: same length as input_ids; LOSS_IGNORE_INDEX on prompt
            positions, real token id on answer positions. Shifted-by-one
            convention is applied by the loss path, not here.
        prompt_len: int, length of the prompt portion (used for invariant checks).
    """

    input_ids: Tensor
    labels: Tensor
    prompt_len: int


class SFTRow(TypedDict):
    query: str
    query_type: str
    source: str
    retrieved_chunk_ids: list[str]
    qwen_answer: str
    qwen_refused: bool
    qwen_latency_seconds: float


def load_chunk_lookup(chunks_path: Path) -> dict[str, Chunk]:
    """Build chunk_id → Chunk dict from chunks.jsonl.

    Reads every JSON line, materializes a Chunk, indexes by chunk_id.
    Raises if duplicate chunk_ids exist (dedup must happen upstream).
    """
    if not chunks_path.exists():
        raise FileNotFoundError(f"chunks_path {chunks_path} does not exist")
    chunks_by_id: dict[str, Chunk] = {}
    with chunks_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            try:
                data = json.loads(line.strip())
                if data["chunk_id"] in chunks_by_id:
                    raise ValueError(f"Line {i}: duplicated chunk_id {data['chunk_id']}")
                chunks_by_id[data["chunk_id"]] = Chunk(**data)
            except json.JSONDecodeError as e:
                raise ValueError(f"Line {i}: invalid json object") from e
    return chunks_by_id


def build_sft_record(
    row: SFTRow,
    chunk_lookup: dict[str, Chunk],
    tokenizer: TinyDocsTokenizer,
    *,
    max_seq_len: int,
) -> SFTRecord | None:
    """Encode one SFT row into an SFTRecord.

    Steps:
        1. Resolve `row["retrieved_chunk_ids"]` against chunk_lookup. Drop
           the row (return None) if any chunk_id is missing.
        2. Skip rows where `row["qwen_refused"]` is True (no answer to fit).
        3. Build the prompt with build_grounded_prompt(query, chunks,
           query_type=parsed_type) where parsed_type is None unless the
           row supplies a recognized QueryType string.
        4. Flatten messages with the same "\\n\\n".join pattern used by
           TinyDocsGenerator._flatten_messages.
        5. tokenizer.encode(prompt_text, add_bos=True, add_eos=False) →
           prompt_ids; tokenizer.encode(answer_text, add_bos=False,
           add_eos=True) → answer_ids.
        6. If len(prompt_ids) + len(answer_ids) > max_seq_len: tail-truncate
           the prompt (preserve answer; question sits at prompt tail per
           build_grounded_prompt convention).
        7. Construct input_ids = prompt_ids + answer_ids (LongTensor);
           labels = [LOSS_IGNORE_INDEX] * len(prompt_ids) + answer_ids.
        8. Return SFTRecord with prompt_len = len(prompt_ids).

    Returns None when the row should be skipped (missing chunk, refused,
    or empty answer).
    """
    chunk_ids = row["retrieved_chunk_ids"]
    if any(chunk_id not in chunk_lookup for chunk_id in chunk_ids):
        return None
    if row["qwen_refused"]:
        return None

    chunks = [chunk_lookup[chunk_id] for chunk_id in chunk_ids]
    query_type = QueryType(row["query_type"]) if row["query_type"] else None
    query = row["query"]
    messages = build_grounded_prompt(query, chunks, query_type=query_type)
    prompt_text = "\n\n".join(m["content"] for m in messages) + "\n"
    prompt_ids = tokenizer.encode(prompt_text, add_bos=True, add_eos=False)
    answer_text = row["qwen_answer"]
    answer_ids = tokenizer.encode(answer_text, add_bos=False, add_eos=True)
    if len(answer_ids) > max_seq_len:
        return None
    if len(prompt_ids) + len(answer_ids) > max_seq_len:
        prompt_ids = prompt_ids[(len(answer_ids) - max_seq_len) :]
    return SFTRecord(
        input_ids=torch.tensor(prompt_ids + answer_ids, dtype=torch.long),
        labels=torch.tensor(
            [LOSS_IGNORE_INDEX] * len(prompt_ids) + answer_ids,
            dtype=torch.long,
        ),
        prompt_len=len(prompt_ids),
    )


def build_sft_records(
    sft_jsonl: Path,
    chunks_path: Path,
    tokenizer: TinyDocsTokenizer,
    *,
    max_seq_len: int,
) -> list[SFTRecord]:
    """Iterate sft_jsonl rows, build records, drop Nones.

    Surfaces the count of skipped rows via a print line so the operator
    can sanity-check the data pipeline before the long fine-tune run.
    """
    if not sft_jsonl.exists():
        raise FileNotFoundError(f"sft_jsonl {sft_jsonl} does not exist")

    with sft_jsonl.open("r", encoding="utf-8") as f:
        sft_records: list[SFTRecord] = []
        chunk_lookup = load_chunk_lookup(chunks_path)
        for i, line in enumerate(f):
            if not line.strip():
                continue
            try:
                data = json.loads(line.strip())
                record = build_sft_record(data, chunk_lookup, tokenizer, max_seq_len=max_seq_len)
                if record is not None:
                    sft_records.append(record)
            except json.JSONDecodeError as e:
                raise ValueError(f"Line {i}: invalid json object") from e
    print(f"[sft] kept {len(sft_records)} records from {sft_jsonl}")
    return sft_records


class SFTDataset(Dataset[tuple[Tensor, Tensor]]):
    """Dataset of (input_ids, labels) for the SFT loop.

    Dynamic padding handled by the collate fn. Variable-length sequences
    are NOT padded here — __getitem__ returns raw tensors.
    """

    def __init__(self, records: list[SFTRecord]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        if not (0 <= idx < len(self.records)):
            raise IndexError(f"SFTDataset index {idx} out of range")
        record = self.records[idx]
        return record.input_ids, record.labels


def collate_sft(
    batch: Iterable[tuple[Tensor, Tensor]],
    *,
    pad_token_id: int,
) -> tuple[Tensor, Tensor]:
    """Right-pad a batch of (input_ids, labels) to the longest example.

    Padding for input_ids uses pad_token_id; padding for labels uses
    LOSS_IGNORE_INDEX so padded positions contribute zero loss.

    Returns (input_ids_BT, labels_BT) — both shape (batch, max_len).
    """
    items = list(batch)
    input_id_list = [ids for ids, _ in items]
    labels_list = [labels for _, labels in items]
    input_ids = pad_sequence(input_id_list, batch_first=True, padding_value=pad_token_id)
    labels = pad_sequence(labels_list, batch_first=True, padding_value=LOSS_IGNORE_INDEX)
    return input_ids, labels


def train_sft(
    model_config: TinyDocsConfig,
    sft_config: SFTConfig,
    dataset: SFTDataset,
    *,
    init_ckpt_path: Path,
    ckpt_dir: Path,
    log_path: Path,
    pad_token_id: int,
) -> None:
    """Run the SFT loop.

    - Load model state from init_ckpt_path (don't restore optimizer state —
      fresh AdamW with sft_config.base_lr).
    - DataLoader with collate_sft (closure over pad_token_id).
    - For each epoch / step:
        * forward → cross_entropy(logits.view(-1, vocab),
                                  labels.view(-1),
                                  ignore_index=LOSS_IGNORE_INDEX)
        * backward, grad_clip, optimizer.step
        * log (step, loss, lr) to log_path as JSON Lines
    - Linear warmup over warmup_steps to base_lr; cosine decay after to 0
      across remaining (epochs * len(dataset) // batch_size) steps.
    - Save ckpt every `checkpoint_every_epochs` epochs to
      ckpt_dir/step_<N>.pt; final ckpt to ckpt_dir/step_final.pt.
    """

    if not init_ckpt_path.exists():
        raise FileNotFoundError(f"init_ckpt_path {init_ckpt_path} does not exist")
    model = TinyDocsModel(model_config)
    checkpoint = torch.load(init_ckpt_path, weights_only=False, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    model.to(sft_config.device)
    optimizer = optim.AdamW(
        model.parameters(), lr=sft_config.base_lr, weight_decay=sft_config.weight_decay
    )
    dataloader = DataLoader(
        dataset,
        batch_size=sft_config.batch_size,
        shuffle=True,
        collate_fn=partial(collate_sft, pad_token_id=pad_token_id),
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    loss_func = nn.CrossEntropyLoss(ignore_index=LOSS_IGNORE_INDEX)
    epochs = sft_config.epochs
    torch.manual_seed(sft_config.seed)
    model.train()
    total_steps = epochs * len(dataloader)
    step = 0
    with log_path.open("w", encoding="utf-8") as log_f:
        for epoch in range(epochs):
            print(f"[{_now_clock()}] === epoch {epoch + 1} started ===")
            for input_ids, labels in dataloader:
                input_ids, labels = input_ids.to(sft_config.device), labels.to(sft_config.device)
                optimizer.zero_grad(set_to_none=True)
                logits, _ = model(input_ids)
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                loss = loss_func(
                    shift_logits.view(-1, model_config.vocab_size), shift_labels.view(-1)
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), sft_config.grad_clip)
                lr = get_lr(
                    step,
                    base_lr=sft_config.base_lr,
                    warmup_steps=sft_config.warmup_steps,
                    total_steps=total_steps,
                )
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
                optimizer.step()
                if step % sft_config.log_every == 0:
                    entry = {"step": step, "loss": loss.item(), "lr": lr}
                    log_f.write(json.dumps(entry) + "\n")
                    log_f.flush()
                step += 1

            if (epoch + 1) % sft_config.checkpoint_every_epochs == 0:
                ckpt_file = ckpt_dir / f"step_{step}.pt"
                print(f"[{_now_clock()}] saving ckpt → {ckpt_file}")
                save_checkpoint(
                    model,
                    optimizer,
                    step,
                    ckpt_file,
                    model_config=model_config,
                    sft_config=sft_config,
                )
    save_checkpoint(
        model,
        optimizer,
        total_steps,
        ckpt_dir / "step_final.pt",
        model_config=model_config,
        sft_config=sft_config,
    )


def _now_clock() -> str:
    return datetime.now().strftime("%H:%M:%S")


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    path: Path,
    *,
    model_config: TinyDocsConfig | None = None,
    sft_config: SFTConfig | None = None,
) -> None:
    """Persist model + optimizer state + step counter to a single .pt file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    if model_config is not None:
        checkpoint["model_config"] = asdict(model_config)
    if sft_config is not None:
        checkpoint["sft_config"] = asdict(sft_config)
    torch.save(checkpoint, path)


def get_lr(
    step: int,
    *,
    base_lr: float,
    warmup_steps: int,
    total_steps: int,
) -> float:
    """Cosine schedule with linear warmup.

    - step < warmup_steps: lr ramps linearly from 0 to base_lr
    - step >= warmup_steps: lr decays cosine from base_lr to 0 by total_steps
    """
    if step < warmup_steps:
        return base_lr * (step / warmup_steps)
    else:
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return base_lr * 0.5 * (1 + math.cos(progress * math.pi))
