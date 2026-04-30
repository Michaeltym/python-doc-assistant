"""Build pretrain corpus from Python docs chunks.

See plans/v3-tiny-llm.md §3. MVP: only Python docs (no FineWeb mix).
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def build_corpus_from_chunks(
    chunks_path: Path,
    out_path: Path,
    *,
    seed: int = 42,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Read chunks.jsonl, shuffle with seed, write corpus.jsonl + manifest.json.

    Each output line is:
        {"text": "<chunk text>", "source": "python-docs"}

    Returns the manifest dict.
    """
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks path {chunks_path} does not exist")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output: list[dict[str, str]] = []
    with chunks_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()
    random.Random(seed).shuffle(lines)
    for i, line in enumerate(lines):
        stripped_line = line.strip()
        if not stripped_line:
            continue
        try:
            chunk = json.loads(stripped_line)
            chunk_text = chunk["text"].strip()
            if not chunk_text:
                continue
            output.append({"text": chunk_text, "source": "python-docs"})
        except json.JSONDecodeError:
            raise ValueError(f"Line {i}: invalid json object")
    with out_path.open("w", encoding="utf-8") as f:
        for entry in output:
            f.write(json.dumps(entry) + "\n")
    manifest = {
        "chunks_path": str(chunks_path),
        "seed": seed,
        "n_lines": len(output),
        "total_bytes": out_path.stat().st_size,
        "build_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if manifest_path is not None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(
                manifest,
                f,
            )
    return manifest
