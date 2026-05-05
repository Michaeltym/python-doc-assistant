"""Triage non-correct rows from a `pdr eval` run + judge.

Reads `<run_dir>/per_query.jsonl` (retrieval + generation) and
`<run_dir>/judge_scores.jsonl` (LLM-judge tiers), joins by query
string, and bucketizes every non-correct row into one of:

- `wrong`            — judge tier=wrong (cited correctly, prose wrong)
- `hallucination`    — judge tier=hallucination (claims not in chunks)
- `refused_hit5_yes` — judge tier=refused AND hit_at_5=True
                       (false refusal candidate; chunks contain the
                       answer but model refused)
- `refused_hit5_no`  — judge tier=refused AND hit_at_5=False
                       (retrieval miss; refusal correct, retrieval
                       broke)
- `partial`          — judge tier=partial (kept for completeness;
                       counts only, no samples printed unless
                       --verbose)

Output: a Markdown report on stdout with counts per category + up to
N sample queries per category. Use this to direct sub-task effort:
big `refused_hit5_yes` bucket → tighten refusal calibration; big
`refused_hit5_no` → push HyDE / chunker re-cut.

Usage:
    uv run python scripts/triage_failures.py \\
        --run-dir experiments/runs/2026-05-05T10-43-34-v4-qwen-gguf-dense-rerank-calib

    # JSON output for piping into other tools:
    uv run python scripts/triage_failures.py --run-dir <path> --format json

    # See the partial bucket in samples too:
    uv run python scripts/triage_failures.py --run-dir <path> --verbose
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import click


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _classify(per_query_row: dict, judge_row: dict) -> str:
    tier = judge_row.get("tier", "?")
    hit5 = bool(per_query_row.get("hit_at_5", False))
    if tier == "refused":
        return "refused_hit5_yes" if hit5 else "refused_hit5_no"
    return tier  # correct / partial / wrong / hallucination


def _format_markdown(buckets: dict, samples_per_bucket: int, verbose: bool) -> str:
    order = [
        "correct",
        "partial",
        "wrong",
        "hallucination",
        "refused_hit5_yes",
        "refused_hit5_no",
    ]
    total = sum(len(rows) for rows in buckets.values())
    lines: list[str] = []
    lines.append("# Triage report")
    lines.append("")
    lines.append(f"Total rows judged: **{total}**")
    lines.append("")
    lines.append("| Category | Count | Rate |")
    lines.append("|---|---:|---:|")
    for cat in order:
        n = len(buckets.get(cat, []))
        rate = n / total if total else 0.0
        lines.append(f"| {cat} | {n} | {rate:.3f} |")
    lines.append("")
    skip_default = {"correct", "partial"} if not verbose else set()
    for cat in order:
        if cat in skip_default:
            continue
        rows = buckets.get(cat, [])
        if not rows:
            continue
        lines.append(f"## {cat} (showing up to {samples_per_bucket})")
        for r in rows[:samples_per_bucket]:
            q = r["query"]
            ids = [c["chunk_id"] for c in r.get("retrieved", [])][:3]
            out = (r.get("model_output_text") or "").replace("\n", " ").strip()
            if len(out) > 140:
                out = out[:137] + "..."
            reason = r.get("_judge_reason", "")
            if len(reason) > 140:
                reason = reason[:137] + "..."
            lines.append(f"- query: `{q}`")
            lines.append(f"  - top chunks: {ids}")
            if out:
                lines.append(f"  - output: {out}")
            if reason:
                lines.append(f"  - judge reason: {reason}")
        lines.append("")
    return "\n".join(lines)


@click.command()
@click.option(
    "--run-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Eval run directory (must contain per_query.jsonl + judge_scores.jsonl).",
)
@click.option(
    "--samples", type=int, default=3, help="Sample rows per failure bucket (default 3)."
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["markdown", "json"]),
    default="markdown",
    help="Output format.",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Also include correct/partial samples in the markdown output.",
)
def main(run_dir: Path, samples: int, fmt: str, verbose: bool) -> None:
    """Bucketize a run's non-correct rows by failure mode."""
    pq_path = run_dir / "per_query.jsonl"
    jud_path = run_dir / "judge_scores.jsonl"
    if not pq_path.exists():
        raise click.UsageError(f"missing {pq_path}")
    if not jud_path.exists():
        raise click.UsageError(
            f"missing {jud_path}; run `pdr judge --run-dir {run_dir}` first"
        )

    pq_rows = _load_jsonl(pq_path)
    jud_rows = _load_jsonl(jud_path)
    jud_by_query = {row["query"]: row for row in jud_rows}

    buckets: dict[str, list[dict]] = defaultdict(list)
    for pq in pq_rows:
        q = pq["query"]
        jud = jud_by_query.get(q)
        if jud is None:
            continue
        cat = _classify(pq, jud)
        buckets[cat].append({
            **pq,
            "_judge_tier": jud.get("tier"),
            "_judge_reason": jud.get("reason", ""),
        })

    if fmt == "json":
        click.echo(
            json.dumps(
                {cat: [r["query"] for r in rows] for cat, rows in buckets.items()},
                indent=2,
            )
        )
    else:
        click.echo(_format_markdown(buckets, samples, verbose))


if __name__ == "__main__":
    main()
