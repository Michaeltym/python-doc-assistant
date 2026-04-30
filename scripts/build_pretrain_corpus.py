"""CLI entry point: build pretrain corpus from Python docs chunks.

Usage:
    uv run python scripts/build_pretrain_corpus.py \\
        --chunks data/chunks/3.12/a5c1a35a5a02/chunks.jsonl \\
        --out data/pretrain/corpus.jsonl \\
        --seed 42
"""

from __future__ import annotations

import sys


def main() -> int:
    raise NotImplementedError("v3 §3")


if __name__ == "__main__":
    sys.exit(main())
