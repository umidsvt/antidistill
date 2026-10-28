#!/usr/bin/env python
"""Table and verdict for the OlympiadBench evaluation (results/m7_hard_eval_design.md §3).

    .venv-infer/bin/python scripts/38_olympiad_audit.py

Reads the stored per-problem verdicts written by scripts/24_olympiad_eval.sh; students not yet evaluated
are listed as such and the verdict is withheld until LIMO and both Solo students are in.
"""
from __future__ import annotations

import glob
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BENCH = "olympiadbench"
# §3 -- fixed before any result; do not edit after seeing data
ALPHA, BASE_MIN = 0.05, 10.0
STUDENTS = [("no training", "base"), ("LIMO", "limo"), ("7B Solo", "recon7b_solo"), ("32B Solo", "recon_solo"),
            ("32B Style", "recon_style"), ("hindsight v2", "hindsight_v2"), ("PART", "part"),
            ("7B Search", "recon7b_search")]


def load(cond: str) -> dict | None:
    f = glob.glob(f"{REPO}/outputs/{cond}_{BENCH}/**/{BENCH}/*_t0.0_k1_*.jsonl", recursive=True)
    if not f:
        return None
    rows = [json.loads(line) for line in open(f[0])]
    return {r["question"].strip(): (bool(r["answers_correctness"][0]), bool((r["generated_answers"][0] or "").strip()))
            for r in rows}


def paired(a: dict, b: dict) -> tuple[int, int, float]:
    """Problems only a solves, only b solves, exact two-sided p on those."""
    qs = a.keys() & b.keys()
    ab = sum(a[q][0] and not b[q][0] for q in qs)
    ba = sum(b[q][0] and not a[q][0] for q in qs)
    n = ab + ba
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(ab, ba) + 1)) / 2 ** n) if n else 1.0
    return ab, ba, p


def main() -> int:
    data = {name: load(cond) for name, cond in STUDENTS}
    print("| student | correct | accuracy | finished | accuracy among finished | vs LIMO: only this / only LIMO, p |")
    print("| --- | --- | --- | --- | --- | --- |")
    limo = data["LIMO"]
    for name, _ in STUDENTS:
        d = data[name]
        if d is None:
            print(f"| {name} | not evaluated yet | | | | |"); continue
        n, ok, fin = len(d), sum(v[0] for v in d.values()), sum(v[1] for v in d.values())
        cmp_ = ""
        if limo is not None and name != "LIMO":
            ab, ba, p = paired(d, limo)
            cmp_ = f"+{ab} / -{ba}, p = {p:.3f}"
        print(f"| {name} | {ok}/{n} | {100 * ok / n:.1f}% | {fin} | {100 * ok / max(fin, 1):.1f}% | {cmp_} |")

    base = data["no training"]
    if base is not None:
        b = 100 * sum(v[0] for v in base.values()) / len(base)
        print(f"\nValidity: untrained model {b:.1f}% ({'above' if b > BASE_MIN else 'NOT above'} the {BASE_MIN:.0f}% floor).")
    if limo is None or data["7B Solo"] is None or data["32B Solo"] is None:
        print("\nVerdict withheld: LIMO and both Solo students are needed."); return 0
    res = {s: paired(limo, data[s]) for s in ("7B Solo", "32B Solo")}      # (only LIMO, only Solo, p)
    limo_wins = all(ab > ba and p < ALPHA for ab, ba, p in res.values())
    solo_wins = any(ba > ab and p < ALPHA for ab, ba, p in res.values())
    if limo_wins:
        v = "LIMO above both Solo students: on hard problems the teacher's solutions are needed"
    elif solo_wins:
        v = "a Solo student is above LIMO: Solo's lead holds on hard problems too"
    else:
        v = "no difference shown at this size"
    print(f"\nPre-registered verdict (m7 §3): {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
