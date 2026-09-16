#!/usr/bin/env python
"""Re-grade every stored generation both ways: boxed-only vs boxed + last-number fallback.

Why: the vendored `extract_answer` never honours its `use_last_number` parameter, so any
response answering in prose counts as "no answer produced" (see the DEVIATION note in
third_party/kim_eval/utils/parser.py). This quantifies how much of each measured effect is
that artifact rather than reasoning.

Nothing is regenerated -- every response is already on disk, so this is CPU-only.
The boxed-only column stays the Kim-comparable number; the fallback column is reported
alongside it, never instead of it.

Usage:
    scripts/33_regrade_fallback.py [--bench math amc aime aime25 gsm8k]
"""
import argparse, glob, json, re, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "third_party" / "kim_eval"))

NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def last_number(text: str) -> str | None:
    nums = NUM.findall((text or "").replace(",", ""))
    return nums[-1].rstrip(".") if nums else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", nargs="*", default=["math", "amc", "aime", "aime25", "gsm8k"])
    a = ap.parse_args()
    from utils.grader import math_equal

    rows = []
    for f in sorted(glob.glob(str(REPO / "outputs" / "**" / "*.jsonl"), recursive=True)):
        rel = Path(f).relative_to(REPO / "outputs")
        cond, bench = rel.parts[0], rel.parts[-2]
        if bench not in a.bench:
            continue
        # Greedy pass@1 only. Without this the base condition also picks up its
        # t=0.7 / k=16 avg@16 file and the pooled number silently averages two
        # different sampling regimes.
        if "_t0.0_k1_" not in Path(f).name:
            continue
        recs = [json.loads(l) for l in open(f)]
        boxed_ok = sum(bool(r["answers_correctness"][0]) for r in recs)
        unboxed = recovered = 0
        for r in recs:
            if (r["generated_answers"][0] or "").strip():
                continue
            unboxed += 1
            p = last_number(r["generated_responses"][0])
            if not p:
                continue
            try:
                recovered += bool(math_equal(p, str(r["gold_answer"])))
            except Exception:
                pass
        rows.append((cond, bench, len(recs), boxed_ok, unboxed, recovered))

    print(f"{'condition':22s} {'bench':7s} {'n':>5s} {'boxed-only':>11s} {'+fallback':>11s} "
          f"{'unboxed':>8s} {'recovered':>10s}")
    for c, b, n, ok, un, rec in rows:
        print(f"{c:22s} {b:7s} {n:5d} {100*ok/n:10.1f}% {100*(ok+rec)/n:10.1f}% "
              f"{un:8d} {rec:10d}")

    # pooled, per condition, over the four original benchmarks
    print(f"\n{'condition':22s} {'pooled boxed-only':>18s} {'pooled +fallback':>17s} {'delta':>7s}")
    by = {}
    for c, b, n, ok, un, rec in rows:
        if b == "gsm8k":
            continue
        d = by.setdefault(c, [0, 0, 0])
        d[0] += n; d[1] += ok; d[2] += ok + rec
    for c, (n, ok, fb) in sorted(by.items()):
        if n < 600:
            continue
        print(f"{c:22s} {100*ok/n:17.1f}% {100*fb/n:16.1f}% {100*(fb-ok)/n:+6.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
