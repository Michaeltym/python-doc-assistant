"""Two-pass byte-target Bernoulli sampler for jsonl pretrain corpora.

Used by v3.1 §1.1 to extract a representative ~100 MB sample of the 2.4 GB
mix corpus for BPE training. Picking the BPE training set on the full
corpus is unnecessary (BPE merges follow Zipf, so a 4 % sample captures
all high-frequency merges) and would make pure-Python BPE training
infeasibly slow.

Algorithm:
    1. Pass 1: count total UTF-8 bytes of all non-empty lines.
    2. Pass 2: for each line, keep with probability p = target / total.
       Stop emitting once kept bytes ≥ target.

Two-pass Bernoulli (vs single-pass reservoir) was chosen because:
  - With known total, p is deterministic; output size lands within ±5 %
    of target deterministically given a fixed seed.
  - Reservoir sampling on byte-weighted streams requires more bookkeeping
    (variable-size items don't fit "Algorithm L" cleanly).
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from pathlib import Path


def count_bytes(path: Path) -> int:
    """Sum UTF-8-encoded bytes of every non-empty line in `path`.

    Empty / whitespace-only lines are ignored (they would also be ignored
    in the sample pass, so counting them would skew the Bernoulli rate).

    Returns:
        Total UTF-8 byte count across non-empty lines.
    """
    if not path.exists():
        raise FileNotFoundError(f"Input path not found: {path}")
    total = 0
    with path.open("rb") as f:
        for line in f:
            if not line.strip():
                continue
            total += len(line)
    return total


def sample_lines_to_byte_target(
    input_path: Path,
    target_bytes: int,
    *,
    seed: int = 42,
) -> Iterator[str]:
    """Yield lines from `input_path` until kept-byte total ≥ `target_bytes`.

    Behaviour:
      - Two-pass: first pass counts total bytes (`count_bytes`), second pass
        Bernoulli-samples each line with probability `min(target / total, 1.0)`.
      - Empty / whitespace-only lines are skipped (in both passes).
      - Lines are yielded **with their trailing newline** (so the caller
        can write them straight to an output file without re-adding `\\n`).
      - Sampling is deterministic given `seed`: same seed + same input
        file → byte-identical output (modulo OS-level line-ending
        differences on the input).
      - If `target_bytes >= total` of the input, every non-empty line is
        yielded once (the rate clamps to 1.0).

    Args:
        input_path: jsonl path. Each non-empty line is sampled independently.
        target_bytes: desired total bytes of yielded lines (cumulative,
            UTF-8). The actual emitted total may overshoot by at most one
            line (we stop *after* reaching the target, not before).
        seed: random seed for the Bernoulli decisions.

    Yields:
        Each kept line as a string (with trailing newline).
    """
    written_bytes = 0
    rng = random.Random(seed)
    total_bytes = count_bytes(input_path)
    if total_bytes == 0:
        return
    probability = min(target_bytes / total_bytes, 1.0)
    with input_path.open("rb") as f:
        for line in f:
            if written_bytes >= target_bytes:
                break
            if not line.strip():
                continue
            if rng.random() < probability:
                written_bytes += len(line)
                yield line.decode("utf-8")
