#!/usr/bin/env python
"""Re-harvest an existing hindsight dataset, keeping only the model's ANSWER.

Kim et al.'s generator stores the raw completion from DeepSeek-R1-Distill, which
contains the model's scratchpad AND its answer, separated by a </think> that has no
opener (the chat template emits the opener, and their script bypasses the template by
calling the completions endpoint).

DeepSeek's own chat template defines the answer as `content.split('</think>')[-1]`.
This script applies that rule, so no regeneration is needed for traces that closed
their think block.

Traces with no </think> never finished their scratchpad and contain no answer; they
are reported and, unless --drop is passed, carried over unchanged so the dataset keeps
800 rows. Regenerating those with src/antidistill/defenses/hindsight.py is preferable.

Usage:
    scripts/43_harvest_answers.py data/defended/limo_hindsight_ds32b.json \
        --output data/defended/limo_hindsight_ds32b_answer.json
"""
import argparse, json, sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--output", required=True)
    ap.add_argument("--drop", action="store_true",
                    help="Drop traces with no </think> instead of carrying them over.")
    a = ap.parse_args()

    rows = json.loads(Path(a.input).read_text(encoding="utf-8"))
    out, unclosed, empty = [], [], []
    for i, r in enumerate(rows):
        o = r["output"]
        if "</think>" not in o:
            unclosed.append(i)
            if not a.drop:
                out.append(r)
            continue
        ans = o.split("</think>")[-1].strip()
        if not ans:
            empty.append(i)
            if not a.drop:
                out.append(r)
            continue
        n = dict(r)
        n["output"] = ans
        out.append(n)

    Path(a.output).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    kept = len(rows) - len(unclosed) - len(empty)
    print(f"in {len(rows)} rows -> out {len(out)} rows")
    print(f"  harvested cleanly        : {kept}")
    print(f"  no </think> (no answer)  : {len(unclosed)}  {'dropped' if a.drop else 'carried over UNCHANGED'}")
    print(f"  </think> but empty after : {len(empty)}  {'dropped' if a.drop else 'carried over UNCHANGED'}")
    if unclosed and not a.drop:
        print(f"  indices needing regeneration: {unclosed[:20]}{' ...' if len(unclosed) > 20 else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
