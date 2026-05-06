"""Per-query_type metrics + refusal F1 for an eval run.

Reads `<run_dir>/per_query.jsonl` (retrieval + generation) and
`<run_dir>/judge_scores.jsonl` (LLM-judge tiers), groups rows by
`query_type`, and computes per-type tier breakdowns plus a refusal F1
using `hit_at_5` as proxy ground truth for "chunks contained the
answer".

Output: a Markdown report on stdout (default) or JSON. Use this to
spot per-type weaknesses (e.g. NL queries dragging down accuracy)
that a global aggregate hides.

Refusal F1 framework:
    Positive class = "should refuse" = chunks did NOT contain the
    answer (hit_at_5=False).
    - TP: model refused AND chunks missed (correct refusal)
    - FP: model refused AND chunks had it (false refusal)
    - FN: model answered AND chunks missed (risky answer)
    - TN: model answered AND chunks had it (good answer)

Usage:
    uv run python scripts/per_type_metrics.py \\
        --run-dir experiments/runs/2026-05-06T07-37-10-v4-qwen-gguf-dense-rerank-calib-rewriter-hyde
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import click

TIER_ORDER = ["correct", "partial", "wrong", "hallucination", "refused"]
QUERY_TYPE_ORDER = [
    "identifier",
    "natural_language",
    "comparison",
    "howto",
    "out_of_scope",
]


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _per_type_breakdown(
    per_query: list[dict], judges: dict[str, dict]
) -> dict[str, dict]:
    """Group rows by query_type, return {qt: {n, tiers: {tier: count}}}."""
    out: dict[str, dict] = {}
    for pq in per_query:
        q = pq["query"]
        j = judges.get(q)
        if j is None:
            continue
        qt = pq.get("query_type", "unknown")
        bucket = out.setdefault(qt, {"n": 0, "tiers": defaultdict(int)})
        bucket["n"] += 1
        bucket["tiers"][j["tier"]] += 1
    for bucket in out.values():
        bucket["tiers"] = dict(bucket["tiers"])
    return out


def _refusal_f1(per_query: list[dict], judges: dict[str, dict]) -> dict:
    """Refusal precision/recall/F1 using hit_at_5=False as 'should refuse'."""
    tp = fp = fn = tn = 0
    for pq in per_query:
        q = pq["query"]
        j = judges.get(q)
        if j is None:
            continue
        refused = j["tier"] == "refused"
        should_refuse = not pq.get("hit_at_5", False)
        if refused and should_refuse:
            tp += 1
        elif refused and not should_refuse:
            fp += 1
        elif not refused and should_refuse:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _format_markdown(
    buckets: dict[str, dict], refusal: dict, run_dir: Path, judge_model: str
) -> str:
    lines: list[str] = []
    lines.append(f"# Per-type metrics — `{run_dir.name}`")
    lines.append("")
    lines.append(f"Judge: `{judge_model}`")
    lines.append("")
    lines.append("## Per query_type")
    lines.append("")
    header = "| query_type | n | " + " | ".join(TIER_ORDER) + " | accuracy | halluc% |"
    sep = "|---|---:|" + "---:|" * len(TIER_ORDER) + "---:|---:|"
    lines.append(header)
    lines.append(sep)
    seen = set()
    for qt in QUERY_TYPE_ORDER:
        if qt not in buckets:
            continue
        seen.add(qt)
        b = buckets[qt]
        n = b["n"]
        tiers = b["tiers"]
        cells = [str(tiers.get(t, 0)) for t in TIER_ORDER]
        acc = (tiers.get("correct", 0) + tiers.get("partial", 0)) / n if n else 0
        hr = tiers.get("hallucination", 0) / n if n else 0
        lines.append(
            f"| {qt} | {n} | " + " | ".join(cells) + f" | {acc:.3f} | {hr * 100:.1f}% |"
        )
    extras = [qt for qt in buckets if qt not in seen]
    for qt in sorted(extras):
        b = buckets[qt]
        n = b["n"]
        tiers = b["tiers"]
        cells = [str(tiers.get(t, 0)) for t in TIER_ORDER]
        acc = (tiers.get("correct", 0) + tiers.get("partial", 0)) / n if n else 0
        hr = tiers.get("hallucination", 0) / n if n else 0
        lines.append(
            f"| {qt} | {n} | " + " | ".join(cells) + f" | {acc:.3f} | {hr * 100:.1f}% |"
        )
    lines.append("")
    lines.append("## Refusal F1 (hit_at_5=False as proxy for 'should refuse')")
    lines.append("")
    lines.append(f"- TP (correct refusals): **{refusal['tp']}**")
    lines.append(f"- FP (false refusals — chunks had answer, model refused): **{refusal['fp']}**")
    lines.append(
        f"- FN (missed refusals — chunks didn't have answer, model answered): **{refusal['fn']}**"
    )
    lines.append(f"- TN (good answers): **{refusal['tn']}**")
    lines.append("")
    lines.append(f"- **Precision**: {refusal['precision']:.3f}")
    lines.append(f"- **Recall**: {refusal['recall']:.3f}")
    lines.append(f"- **F1**: {refusal['f1']:.3f}")
    return "\n".join(lines)


@click.command()
@click.option(
    "--run-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Eval run directory (must contain per_query.jsonl + judge_scores.jsonl).",
)
@click.option(
    "--out-format",
    type=click.Choice(["markdown", "json"]),
    default="markdown",
    help="Output format.",
)
def main(run_dir: Path, out_format: str) -> None:
    """Per-query_type metrics + refusal F1 for a single run."""
    pq_path = run_dir / "per_query.jsonl"
    judge_path = run_dir / "judge_scores.jsonl"
    if not pq_path.exists():
        raise click.UsageError(f"missing {pq_path}")
    if not judge_path.exists():
        raise click.UsageError(f"missing {judge_path}; run `pdr judge --run-dir {run_dir}` first")

    pq_rows = _load_jsonl(pq_path)
    judges = {r["query"]: r for r in _load_jsonl(judge_path)}

    results_path = run_dir / "results.json"
    judge_model = "?"
    if results_path.exists():
        results = json.loads(results_path.read_text())
        judge_model = results.get("judge", {}).get("judge_model", "?")

    buckets = _per_type_breakdown(pq_rows, judges)
    refusal = _refusal_f1(pq_rows, judges)

    if out_format == "json":
        click.echo(
            json.dumps(
                {
                    "run_dir": str(run_dir),
                    "judge_model": judge_model,
                    "per_type": buckets,
                    "refusal_f1": refusal,
                },
                indent=2,
            )
        )
    else:
        click.echo(_format_markdown(buckets, refusal, run_dir, judge_model))


if __name__ == "__main__":
    main()
