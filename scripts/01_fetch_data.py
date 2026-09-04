#!/usr/bin/env python
"""Task 0.3 — build the undefended trace pool and characterise it.

1. Download `GAIR/LIMO-v2` and convert to the LLaMA-Factory alpaca format used by
   Kim et al. (`_kim_make_limo_dataset.py`), writing `data/raw/limo_v2.json`.
2. Assert the pool has exactly 800 rows.
3. Tokenize every (prompt + response) under the student tokenizer and report the
   length distribution and, crucially, **what fraction exceeds `cutoff_len`**.
   Silently truncated traces are a real confound for the replication and we want
   the number on record before training, not after.

Run:  .venv-infer/bin/python scripts/01_fetch_data.py
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Verbatim from make_limo_dataset.py / eval/prompts/qwen-instruct/*.py.
SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="GAIR/LIMO-v2")
    p.add_argument("--split", default="train")
    p.add_argument("--output", default=str(REPO / "data" / "raw" / "limo_v2.json"))
    p.add_argument("--expect_rows", type=int, default=800)
    p.add_argument("--tokenizer", default="Qwen/Qwen2.5-7B")
    p.add_argument("--cutoff_len", type=int, default=16384,
                   help="Must match cutoff_len in the training yaml.")
    p.add_argument("--report", default=str(REPO / "results" / "task0.3_pool_stats.md"))
    return p.parse_args()


def build(args: argparse.Namespace) -> list[dict]:
    from datasets import load_dataset

    print(f"Loading {args.dataset} (split={args.split}) ...")
    ds = load_dataset(args.dataset, split=args.split)
    print(f"  columns: {ds.column_names}")

    rows = [
        {
            "instruction": r["question"],
            "input": "",
            "output": r["solution"],
            "system": SYSTEM_PROMPT,
        }
        for r in ds
    ]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(rows)} rows -> {out.relative_to(REPO)}")
    return rows


def characterise(rows: list[dict], args: argparse.Namespace) -> list[str]:
    """Token-length distribution under the student tokenizer, in training format."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    total_lens, out_lens = [], []
    for r in rows:
        # Mirror how LLaMA-Factory builds an SFT example: chat-templated prompt + response.
        prompt = tok.apply_chat_template(
            [
                {"role": "system", "content": r["system"]},
                {"role": "user", "content": r["instruction"]},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        n_prompt = len(tok(prompt, add_special_tokens=False)["input_ids"])
        n_out = len(tok(r["output"], add_special_tokens=False)["input_ids"])
        total_lens.append(n_prompt + n_out)
        out_lens.append(n_out)

    n_over = sum(1 for n in total_lens if n > args.cutoff_len)
    pct_over = 100.0 * n_over / len(total_lens)

    def q(xs: list[int], p: float) -> int:
        s = sorted(xs)
        return s[min(len(s) - 1, int(p * len(s)))]

    lines = [
        "# Task 0.3 — LIMO-v2 trace pool statistics\n",
        f"- dataset: `{args.dataset}` split `{args.split}`",
        f"- rows: **{len(rows)}** (expected {args.expect_rows})",
        f"- tokenizer: `{args.tokenizer}`",
        f"- training `cutoff_len`: **{args.cutoff_len}**\n",
        "## Sequence length (chat-templated prompt + response), tokens\n",
        "| stat | prompt+response | response only |",
        "| --- | --- | --- |",
        f"| min | {min(total_lens)} | {min(out_lens)} |",
        f"| p25 | {q(total_lens,0.25)} | {q(out_lens,0.25)} |",
        f"| median | {int(statistics.median(total_lens))} | {int(statistics.median(out_lens))} |",
        f"| mean | {int(statistics.mean(total_lens))} | {int(statistics.mean(out_lens))} |",
        f"| p75 | {q(total_lens,0.75)} | {q(out_lens,0.75)} |",
        f"| p95 | {q(total_lens,0.95)} | {q(out_lens,0.95)} |",
        f"| max | {max(total_lens)} | {max(out_lens)} |",
        "",
        f"## Truncation at cutoff_len={args.cutoff_len}\n",
        f"**{n_over}/{len(total_lens)} = {pct_over:.2f}%** of traces exceed the cutoff "
        f"and will be truncated during SFT.",
        "",
        "> This is inherited from the LIMO default config, which Kim et al. state they used "
        "unchanged. We keep it for the replication; the number is recorded here so any later "
        "change to `cutoff_len` is a documented deviation rather than a silent one.",
    ]
    return lines


def main() -> int:
    args = parse_args()
    rows = build(args)

    if len(rows) != args.expect_rows:
        print(f"WARNING: expected {args.expect_rows} rows, got {len(rows)}")

    lines = characterise(rows, args)
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines[4:]))
    print(f"\nwrote {report.relative_to(REPO)}")
    return 0 if len(rows) == args.expect_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
