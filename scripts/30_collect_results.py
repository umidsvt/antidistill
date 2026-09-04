#!/usr/bin/env python
"""Collect every eval output under outputs/ into one comparison table.

Walks outputs/<condition>/<model>/<benchmark>/*.jsonl, recomputes the metrics from the
stored per-problem verdicts (rather than trusting a number scraped from a log), and prints
a markdown table plus the Kim et al. reference values for the Qwen2.5-7B row.

Usage:  .venv-infer/bin/python scripts/30_collect_results.py [--out results/table.md]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Kim et al. / proposal Table (Section 2.2), Qwen2.5-7B row, AIME24 greedy pass@1.
REFERENCE = {"base": 13.3, "limo": 26.7, "hindsight": 3.3}

FNAME = re.compile(r"t(?P<temp>[\d.]+)_k(?P<k>\d+)_s\d+_e\d+\.jsonl$")


def summarize(path: Path) -> dict | None:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        return None
    m = FNAME.search(path.name)
    n = len(rows)
    n_pass = sum(bool(r["is_correct"]) for r in rows)
    out = {
        "temp": m.group("temp") if m else "?",
        "k": int(m.group("k")) if m else 1,
        "n": n,
        "pass": n_pass,
        "pass_pct": 100.0 * n_pass / n,
        "avg_pct": None,
        "mean_tokens": None,
    }
    if "avg_at_n" in rows[0]:
        out["avg_pct"] = 100.0 * sum(r["avg_at_n"] for r in rows) / n
    if "avg_response_token_length" in rows[0]:
        out["mean_tokens"] = sum(r["avg_response_token_length"] for r in rows) / n
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "results" / "eval_table.md"))
    args = ap.parse_args()

    outputs = REPO / "outputs"
    records = []
    for f in sorted(outputs.rglob("*.jsonl")):
        rel = f.relative_to(outputs)
        if len(rel.parts) < 3:
            continue
        condition = rel.parts[0]
        benchmark = rel.parts[-2]
        s = summarize(f)
        if s:
            records.append({"condition": condition, "benchmark": benchmark, **s})

    lines = ["# Eval results (recomputed from stored per-problem verdicts)\n"]
    lines += [
        "| condition | benchmark | temp | k | pass@k | avg@k | mean resp tokens |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in sorted(records, key=lambda r: (r["benchmark"], r["condition"], r["temp"])):
        avg = f"{r['avg_pct']:.2f}%" if r["avg_pct"] is not None else "—"
        tok = f"{r['mean_tokens']:.0f}" if r["mean_tokens"] is not None else "—"
        lines.append(
            f"| {r['condition']} | {r['benchmark']} | {r['temp']} | {r['k']} | "
            f"{r['pass']}/{r['n']} = {r['pass_pct']:.2f}% | {avg} | {tok} |"
        )

    lines += [
        "",
        "## Reference — Kim et al. / proposal, Qwen2.5-7B, AIME24 greedy pass@1\n",
        "| base | LIMO | hindsight |",
        "| --- | --- | --- |",
        f"| {REFERENCE['base']}% (4/30) | {REFERENCE['limo']}% (8/30) | {REFERENCE['hindsight']}% (1/30) |",
        "",
        "> Absolute levels are not directly comparable: greedy decoding differs across GPU",
        "> architectures (see `results/m1_base.md`). Judge on ordering and on avg@k, and use",
        "> our own base row as the baseline.",
    ]

    text = "\n".join(lines) + "\n"
    print(text)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
