"""Pretraining loop for TinyDocs (v3 §4b).

Pure orchestration: forward → loss → backward → optimizer step → log →
checkpoint. Mixed precision deferred (MVP runs fp32 on MPS).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, Dataset

from python_doc_assistant.generation.tinydocs.config import TinyDocsConfig
from python_doc_assistant.generation.tinydocs.model import TinyDocsModel


@dataclass
class TrainConfig:
    """Hyperparameters for the pretraining loop."""

    base_lr: float = 3e-4
    warmup_steps: int = 100
    total_steps: int = 5000
    batch_size: int = 4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    checkpoint_every: int = 1000
    log_every: int = 10
    device: str = "mps"
    dtype: str = "fp32"
    seed: int = 42


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


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    path: Path,
    *,
    model_config: TinyDocsConfig | None = None,
    train_config: TrainConfig | None = None,
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
    if train_config is not None:
        checkpoint["train_config"] = asdict(train_config)
    torch.save(checkpoint, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """Restore model + optimizer state from path. Returns the persisted step."""
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint {path} does not exist")
    checkpoint = torch.load(path, weights_only=False, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    step: int = checkpoint["step"]
    return step


def train_tinydocs(
    model_config: TinyDocsConfig,
    train_config: TrainConfig,
    dataset: Dataset[tuple[torch.Tensor, torch.Tensor]],
    *,
    ckpt_dir: Path,
    log_path: Path,
) -> None:
    """Run the full pretraining loop.

    - Builds the model, RoPE module, AdamW optimizer
    - Iterates DataLoader for `total_steps` updates
    - Logs (step, loss, lr) to log_path as JSON Lines
    - Saves checkpoints to ckpt_dir/step_<N>.pt every `checkpoint_every`
    """
    device = torch.device(train_config.device)
    model = TinyDocsModel(model_config).to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=train_config.base_lr, weight_decay=train_config.weight_decay
    )
    dataloader = DataLoader(dataset, batch_size=train_config.batch_size, shuffle=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    loss_func = nn.CrossEntropyLoss()

    epoch = 0
    step = 0
    model.train()
    with log_path.open("w", encoding="utf-8") as log_f:
        while step < train_config.total_steps:
            print(f"##### Epoch {epoch + 1} started #####")
            for input_ids, target_ids in dataloader:
                input_ids = input_ids.to(device)
                target_ids = target_ids.to(device)
                optimizer.zero_grad(set_to_none=True)

                logits, _ = model(input_ids)
                loss = loss_func(
                    logits.view(-1, model_config.vocab_size),
                    target_ids.view(-1),
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
                lr = get_lr(
                    step,
                    base_lr=train_config.base_lr,
                    warmup_steps=train_config.warmup_steps,
                    total_steps=train_config.total_steps,
                )
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
                optimizer.step()
                if step % train_config.log_every == 0:
                    entry = {"step": step, "loss": loss.item(), "lr": lr}
                    log_f.write(json.dumps(entry) + "\n")
                    log_f.flush()
                    print(
                        f"step {step}/{train_config.total_steps}  "
                        f"loss={loss.item():.4f}  lr={lr:.6f}"
                    )
                if step > 0 and step % train_config.checkpoint_every == 0:
                    save_checkpoint(
                        model,
                        optimizer,
                        step,
                        ckpt_dir / f"step_{step}.pt",
                        model_config=model_config,
                        train_config=train_config,
                    )
                step += 1
                if step >= train_config.total_steps:
                    break
            epoch += 1
    save_checkpoint(
        model,
        optimizer,
        step,
        ckpt_dir / "step_final.pt",
        model_config=model_config,
        train_config=train_config,
    )
