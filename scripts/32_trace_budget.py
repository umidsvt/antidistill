#!/usr/bin/env python
"""Token budget and stop-signal audit for a set of SFT datasets.

Motivation: the M6a supplementation result (results/m6_supplementation.md) turns on the
difference between what fraction of *problems* carry epistemic traces and what fraction of
*trained tokens* they carry. LIMO traces are ~9x longer than hindsight ones, so 50% of problems
is 90% of tokens. Reporting only the problem fraction is misleading.

It also measures the share of trained tokens belonging to examples that lose their trailing
<|im_end|> to `cutoff_len` — the mechanism behind the non-monotonic termination behaviour.
Example-weighted truncation understates this badly (mix10: 4.6% of examples, 33% of tokens),
because the truncated examples are by construction the longest ones.

Usage:
    scripts/32_trace_budget.py                       # all five datasets, LIMO as the epistemic source
    scripts/32_trace_budget.py --cutoff-len 32768    # what the control run would look like
"""

import argparse
import json
import os
import sys

DEFAULT_SETS = [
    ("LIMO", "data/raw/limo_v2.json"),
    ("mix50", "data/curated/limo_mixed_50.json"),
    ("mix25", "data/curated/limo_mixed_25.json"),
    ("mix10", "data/curated/limo_mixed_10.json"),
    ("hindsight", "data/defended/limo_hindsight_ds32b.json"),
]

# The qwen template as LLaMA-Factory renders it, with LIMO's system prompt.
SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."
PROMPT = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n{instruction}<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--cutoff-len", type=int, default=16384)
    ap.add_argument("--epistemic-source", default="data/raw/limo_v2.json",
                    help="dataset whose (instruction, output) pairs count as epistemic")
    ap.add_argument("--sets", nargs="*", default=None,
                    help="name=path pairs; defaults to the five study datasets")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    im_end = tok.convert_tokens_to_ids("<|im_end|>")

    sets = DEFAULT_SETS
    if args.sets:
        sets = [tuple(s.split("=", 1)) for s in args.sets]
    sets = [(n, p) for n, p in sets if os.path.exists(p)]
    if not sets:
        print("no datasets found — run from the repo root", file=sys.stderr)
        return 1

    # Match on the full (instruction, output) pair: LIMO-v2 contains 6 problems whose
    # instruction appears twice with different solutions, so keying on instruction alone
    # miscounts 8 of the 800 rows.
    epistemic = set()
    if os.path.exists(args.epistemic_source):
        epistemic = {(r["instruction"], r["output"])
                     for r in json.load(open(args.epistemic_source))}

    cache: dict[tuple[str, str], tuple[int, bool]] = {}

    def budget(instruction: str, output: str) -> tuple[int, bool]:
        """(trained target tokens after truncation, lost its stop token)."""
        key = (instruction, output)
        if key in cache:
            return cache[key]
        prompt = PROMPT.format(system=SYSTEM, instruction=instruction)
        n_prompt = len(tok(prompt, add_special_tokens=False)["input_ids"])
        target = tok(output, add_special_tokens=False)["input_ids"] + [im_end]
        total = n_prompt + len(target)
        if total <= args.cutoff_len:
            val = (len(target), False)
        else:
            # supervised.py does target_ids[:target_len], which drops the trailing <|im_end|>.
            val = (max(0, args.cutoff_len - n_prompt), True)
        cache[key] = val
        return val

    print(f"cutoff_len = {args.cutoff_len}, tokenizer = {args.model}\n")
    hdr = (f"{'dataset':10s} {'n':>4s} {'truncated':>16s} {'tokens':>10s} "
           f"{'mean':>7s} {'no-stop tok':>12s} {'epistemic tok':>14s}")
    print(hdr)
    print("-" * len(hdr))
    for name, path in sets:
        rows = json.load(open(path))
        n_trunc = tokens = trunc_tokens = epi_tokens = n_epi = 0
        for r in rows:
            kept, lost_stop = budget(r["instruction"], r["output"])
            tokens += kept
            if lost_stop:
                n_trunc += 1
                trunc_tokens += kept
            if (r["instruction"], r["output"]) in epistemic:
                n_epi += 1
                epi_tokens += kept
        n = len(rows)
        print(f"{name:10s} {n:>4d} {n_trunc:>5d}/{n} = {100*n_trunc/n:4.1f}% {tokens:>10d} "
              f"{tokens/n:>7.0f} {100*trunc_tokens/tokens:>11.1f}% "
              f"{n_epi:>4d}/{n} {100*epi_tokens/tokens:>5.1f}%")

    print("\n'no-stop tok' = share of trained tokens in examples whose <|im_end|> was truncated.")
    print("'epistemic tok' = problems from the epistemic source, then their share of tokens.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
