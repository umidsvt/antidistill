#!/usr/bin/env python
"""Diff our generations against Kim et al.'s reference generations, per problem.

Our grader is already validated exact on their outputs (Task 0.2), so any accuracy
difference must come from the *generations*, not the grading. This isolates which
problems flipped, in which direction, and whether the traces diverge textually.

Usage:
  .venv-infer/bin/python scripts/02_diff_vs_reference.py \
      --ours outputs/base/Qwen/Qwen2.5-7B/aime/test_qwen-instruct_t0.0_k1_s0_e30.jsonl \
      --reference reference/kim_example_eval_outputs/base/Qwen2.5-7B/aime/test_qwen-instruct_t0.0_k1_s0_e30.jsonl \
      --label base
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def load(p: Path) -> dict:
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    # AIME rows carry an "id"; fall back to the question text as the join key.
    return {r.get("id", r["question"][:80]): r for r in rows}


def common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ours = load(REPO / args.ours if not Path(args.ours).is_absolute() else Path(args.ours))
    ref = load(REPO / args.reference if not Path(args.reference).is_absolute() else Path(args.reference))

    keys = sorted(set(ours) & set(ref), key=lambda k: (isinstance(k, str), k))
    gained, lost, agree = [], [], 0
    identical = 0
    prefix_fracs = []

    for k in keys:
        o, r = ours[k], ref[k]
        oc, rc = bool(o["is_correct"]), bool(r["is_correct"])
        if oc == rc:
            agree += 1
        elif oc and not rc:
            gained.append(k)
        else:
            lost.append(k)

        ot, rt = o["generated_responses"][0], r["generated_responses"][0]
        if ot == rt:
            identical += 1
        prefix_fracs.append(common_prefix_len(ot, rt) / max(len(rt), 1))

    n = len(keys)
    o_correct = sum(bool(ours[k]["is_correct"]) for k in keys)
    r_correct = sum(bool(ref[k]["is_correct"]) for k in keys)

    lines = [
        f"# Generation diff vs reference — {args.label}\n",
        f"- problems compared: **{n}**",
        f"- ours: **{o_correct}/{n} = {100*o_correct/n:.2f}%**",
        f"- Kim et al.: **{r_correct}/{n} = {100*r_correct/n:.2f}%**",
        f"- per-problem verdict agreement: **{agree}/{n}**",
        f"- we solve but they don't: **{len(gained)}** {gained}",
        f"- they solve but we don't: **{len(lost)}** {lost}",
        "",
        "## Are the traces the same text?\n",
        f"- byte-identical greedy traces: **{identical}/{n}**",
        f"- mean shared prefix (fraction of reference trace): "
        f"**{100*sum(prefix_fracs)/len(prefix_fracs):.1f}%**",
        f"- problems diverging within the first 1% of the trace: "
        f"**{sum(1 for f in prefix_fracs if f < 0.01)}/{n}**",
        "",
        "## Length / truncation\n",
        "| | ours | reference |",
        "| --- | --- | --- |",
    ]

    def stats(src):
        lens = [len(src[k]["generated_responses"][0]) for k in keys]
        boxed = sum(1 for k in keys if "\\boxed" in src[k]["generated_responses"][0])
        return lens, boxed

    ol, ob = stats(ours)
    rl, rb = stats(ref)
    lines += [
        f"| mean response chars | {int(sum(ol)/len(ol))} | {int(sum(rl)/len(rl))} |",
        f"| max response chars | {max(ol)} | {max(rl)} |",
        f"| responses containing \\boxed | {ob}/{n} | {rb}/{n} |",
    ]

    text = "\n".join(lines) + "\n"
    print(text)

    out = Path(args.out) if args.out else REPO / "results" / f"diff_vs_reference_{args.label}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
