"""CLI entry point: train BPE tokenizer on a corpus.

Usage:
    uv run python scripts/train_tokenizer.py \\
        --corpus data/pretrain/corpus.jsonl \\
        --vocab-size 32000 \\
        --out data/tokenizer/tokenizer.json
"""

from __future__ import annotations

import sys


def main() -> int:
    raise NotImplementedError("v3 §2")


if __name__ == "__main__":
    sys.exit(main())
