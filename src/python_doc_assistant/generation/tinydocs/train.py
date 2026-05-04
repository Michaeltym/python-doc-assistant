"""Pretraining loop for TinyDocs (v3 §4b).

Pure orchestration: forward → loss → backward → optimizer step → log →
checkpoint. Mixed precision deferred (MVP runs fp32 on MPS).
"""

from __future__ import annotations

import json
import math
import statistics
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, Dataset

from python_doc_assistant.generation.tinydocs.config import TinyDocsConfig
from python_doc_assistant.generation.tinydocs.model import TinyDocsModel


def _fmt_duration(seconds: float) -> str:
    """Format seconds as h/m/s for log lines."""
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _fmt_count(n: int) -> str:
    """Compact human-readable count for log lines."""
    if n >= 1_000_000_000:
        return f"{n / 1e9:.2f}G"
    if n >= 1_000_000:
        return f"{n / 1e6:.2f}M"
    if n >= 1_000:
        return f"{n / 1e3:.1f}k"
    return str(n)


def _now_clock() -> str:
    return datetime.now().strftime("%H:%M:%S")


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
    milestone_every: int = 1000
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
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = optim.AdamW(
        model.parameters(), lr=train_config.base_lr, weight_decay=train_config.weight_decay
    )
    dataloader = DataLoader(dataset, batch_size=train_config.batch_size, shuffle=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    loss_func = nn.CrossEntropyLoss()

    # ---- Banner
    seq_len = model_config.max_seq_len
    tokens_per_step = train_config.batch_size * seq_len
    n_segments = len(dataset)  # type: ignore[arg-type]
    total_corpus_tokens = n_segments * seq_len
    total_train_tokens = tokens_per_step * train_config.total_steps
    epochs_planned = (
        total_train_tokens / total_corpus_tokens if total_corpus_tokens else 0.0
    )
    print(f"[{_now_clock()}] model: {_fmt_count(n_params)} params  "
          f"(vocab {_fmt_count(model_config.vocab_size)} + transformer)")
    print(
        f"[{_now_clock()}] device={train_config.device}  "
        f"batch={train_config.batch_size}  seq_len={seq_len}  "
        f"total_steps={train_config.total_steps:,}  "
        f"attention_impl={model_config.attention_impl}"
    )
    print(
        f"[{_now_clock()}] tokens/step={_fmt_count(tokens_per_step)}  "
        f"plan: {_fmt_count(total_train_tokens)} train tokens "
        f"(≈ {epochs_planned:.2f} epochs over {_fmt_count(total_corpus_tokens)} corpus)"
    )
    print(f"[{_now_clock()}] === starting training ===\n")

    epoch = 0
    step = 0
    losses_recent: deque[float] = deque(
        maxlen=max(train_config.milestone_every // max(train_config.log_every, 1), 1)
    )
    rolling_step_rates: deque[float] = deque(maxlen=1000)

    start_time = time.time()
    last_log_time = start_time
    last_log_step = 0

    model.train()
    with log_path.open("w", encoding="utf-8") as log_f:
        while step < train_config.total_steps:
            print(f"[{_now_clock()}] === epoch {epoch + 1} started ===")
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

                # ---- Per-log-step
                if step % train_config.log_every == 0:
                    now = time.time()
                    elapsed = now - start_time
                    interval = now - last_log_time
                    interval_steps = step - last_log_step
                    rate = interval_steps / interval if interval > 0 else 0.0
                    rolling_step_rates.append(rate)
                    rolling_rate = (
                        sum(rolling_step_rates) / len(rolling_step_rates)
                        if rolling_step_rates
                        else 0.0
                    )
                    eta = (
                        (train_config.total_steps - step) / rolling_rate
                        if rolling_rate > 0
                        else 0
                    )
                    pct = 100 * step / train_config.total_steps
                    losses_recent.append(loss.item())

                    entry = {"step": step, "loss": loss.item(), "lr": lr}
                    log_f.write(json.dumps(entry) + "\n")
                    log_f.flush()

                    print(
                        f"[{_now_clock()}] step {step:>6}/{train_config.total_steps} "
                        f"({pct:5.2f} %)  loss {loss.item():6.3f}  lr {lr:.2e}  "
                        f"{rolling_rate:5.2f} step/s  "
                        f"elapsed {_fmt_duration(elapsed)}  "
                        f"eta {_fmt_duration(eta)}"
                    )
                    last_log_time = now
                    last_log_step = step

                # ---- Milestone block
                if step > 0 and step % train_config.milestone_every == 0:
                    if losses_recent:
                        loss_mean = statistics.mean(losses_recent)
                        loss_stdev = (
                            statistics.stdev(losses_recent)
                            if len(losses_recent) > 1
                            else 0.0
                        )
                    else:
                        loss_mean = loss_stdev = 0.0
                    pct = 100 * step / train_config.total_steps
                    elapsed = time.time() - start_time
                    rolling_rate = (
                        sum(rolling_step_rates) / len(rolling_step_rates)
                        if rolling_step_rates
                        else 0.0
                    )
                    eta = (
                        (train_config.total_steps - step) / rolling_rate
                        if rolling_rate > 0
                        else 0
                    )
                    finish_at = (
                        datetime.now() + timedelta(seconds=eta)
                    ).strftime("%Y-%m-%d %H:%M")
                    tokens_seen = step * tokens_per_step
                    epoch_pct = (
                        100 * tokens_seen / total_corpus_tokens
                        if total_corpus_tokens
                        else 0.0
                    )
                    print()
                    print(
                        f"=== milestone step {step}/{train_config.total_steps} "
                        f"({pct:.2f} %) ==="
                    )
                    print(
                        f"  loss recent {len(losses_recent) * train_config.log_every}: "
                        f"mean={loss_mean:.3f}  stdev={loss_stdev:.3f}"
                    )
                    print(
                        f"  tokens seen: {_fmt_count(tokens_seen)} / "
                        f"{_fmt_count(total_corpus_tokens)} ({epoch_pct:.1f} % of corpus)"
                    )
                    print(f"  step rate: {rolling_rate:.2f} step/s (rolling 1000)")
                    print(f"  elapsed: {_fmt_duration(elapsed)}")
                    print(f"  eta: {_fmt_duration(eta)}  (≈ finish {finish_at})")
                    print()

                if step > 0 and step % train_config.checkpoint_every == 0:
                    ckpt_file = ckpt_dir / f"step_{step}.pt"
                    print(f"[{_now_clock()}] saving ckpt → {ckpt_file}")
                    save_checkpoint(
                        model,
                        optimizer,
                        step,
                        ckpt_file,
                        model_config=model_config,
                        train_config=train_config,
                    )
                step += 1
                if step >= train_config.total_steps:
                    break
            print(
                f"[{_now_clock()}] === epoch {epoch + 1} complete (step {step}) ==="
            )
            epoch += 1

    final_path = ckpt_dir / "step_final.pt"
    print(f"\n[{_now_clock()}] saving final ckpt → {final_path}")
    save_checkpoint(
        model,
        optimizer,
        step,
        final_path,
        model_config=model_config,
        train_config=train_config,
    )
    elapsed = time.time() - start_time
    print(f"[{_now_clock()}] === training complete ===")
    print(f"  total elapsed: {_fmt_duration(elapsed)}")
    print(f"  steps: {step}, epochs: {epoch}")
    print(f"  log: {log_path}")
