#!/usr/bin/env python
"""Epistemic supplementation attack (proposal section 4.3, "naive mixing").

Threat model: the defender serves confident, epistemic-free traces (hindsight). The attacker
supplements them with epistemic traces obtained elsewhere (LIMO is public). If procedural and
epistemic information are genuinely orthogonal axes, the student should learn *what to compute*
from the defended traces and *when to doubt itself* from the supplement, recovering much of the
performance the defense removed.

Design — every condition sees the SAME 800 problems at the SAME budget, so the only variable is
what fraction of traces retain epistemic verbalization:

    M2  limo_v2                800 problems, 100% epistemic       -> 62.8% pooled
    M3  limo_hindsight_ds32b   800 problems,   0% epistemic       -> 56.5% pooled
    NEW limo_mixed_50          800 problems,  50% epistemic       -> ?

The split is INTERLEAVED (even index -> hindsight, odd -> LIMO) rather than contiguous, so the
two halves are balanced on problem difficulty and trace length by construction.

Usage:
    .venv-infer/bin/python scripts/50_make_mixed_dataset.py --epistemic-fraction 0.5
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TOK = ["wait", "hmm", "perhaps", "maybe", "actually", "alternatively", "seems", "might", "check"]
RX = [re.compile(r"\b%s\b" % t, re.I) for t in TOK]


def epistemic_count(s: str) -> int:
    return sum(len(r.findall(s)) for r in RX)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limo", default="data/raw/limo_v2.json")
    ap.add_argument("--hindsight", default="data/defended/limo_hindsight_ds32b.json")
    ap.add_argument("--output", default="data/curated/limo_mixed_50.json")
    ap.add_argument("--epistemic-fraction", type=float, default=0.5,
                    help="Fraction of problems that keep their epistemic (LIMO) trace.")
    args = ap.parse_args()

    limo = json.loads((REPO / args.limo).read_text(encoding="utf-8"))
    hind = json.loads((REPO / args.hindsight).read_text(encoding="utf-8"))
    assert len(limo) == len(hind), f"length mismatch {len(limo)} vs {len(hind)}"
    n = len(limo)

    # Interleaved assignment keeps difficulty/length balanced across the two sources.
    stride = round(1 / args.epistemic_fraction) if args.epistemic_fraction else 0
    out, src = [], []
    for i in range(n):
        use_limo = args.epistemic_fraction > 0 and (i % stride == 0)
        assert limo[i]["instruction"] == hind[i]["instruction"], f"problem mismatch at {i}"
        out.append(limo[i] if use_limo else hind[i])
        src.append("limo" if use_limo else "hindsight")

    dst = REPO / args.output
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    (dst.with_suffix(".meta.json")).write_text(
        json.dumps([{"index": i, "source": s} for i, s in enumerate(src)], indent=2),
        encoding="utf-8")

    n_limo = src.count("limo")
    ep_all = sum(epistemic_count(r["output"]) for r in out)
    words = sum(len(r["output"].split()) for r in out)
    print(f"wrote {len(out)} traces -> {dst.relative_to(REPO)}")
    print(f"  from LIMO (epistemic) : {n_limo}  ({100*n_limo/n:.0f}%)")
    print(f"  from hindsight        : {n - n_limo}")
    print(f"  epistemic tokens/trace: {ep_all/n:.1f}")
    print(f"  mean words/trace      : {words/n:.0f}")
    print("\n  for reference:")
    for name, rows in (("LIMO (M2)", limo), ("hindsight (M3)", hind)):
        e = sum(epistemic_count(r["output"]) for r in rows) / len(rows)
        w = sum(len(r["output"].split()) for r in rows) / len(rows)
        print(f"    {name:16s} epistemic/trace {e:7.1f}   mean words {w:7.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
