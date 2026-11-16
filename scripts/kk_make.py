#!/usr/bin/env python
"""Knights and Knaves puzzles for the puzzle testbed (results/m8_audit_design.md §17).

Made with Xie et al.'s (2024) generator, unchanged (third_party/ext/mem-kk-logic, commit 35385cf, MIT licence;
data_prep/lib_kk.py): random statement trees (depth <= 2, width <= 2), kept only when the puzzle has exactly one
solution, rendered with their templates and random common names. Fresh puzzles: our seeds (seed + N), not their 1234.

    python3 scripts/kk_make.py --people 3 4 5 6 7 8 --per 100 --seed 20261113 --out data/kk/pilot_test.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party/ext/mem-kk-logic/data_prep"))
import lib_kk  # noqa: E402


def make(n_people: int, n: int, seed: int) -> list[dict]:
    sampler = lib_kk.KKProblemSampler(seed, n_people=n_people, depth_constraint=2, width_constraint=2)
    out = []
    for i, p in enumerate(sampler.sample_valid_problems(n)):
        assert len(p["all_solutions"]) == 1
        f = lib_kk.KKProblemFormatter(seed * 1000 + i, p).format_problem(random_names=True, random_saying_template=True)
        out.append({"id": f"kk{n_people}_s{seed}_{i:04d}", "n_people": n_people, "quiz": f["quiz"], "names": f["names"],
                    "solution": [bool(x) for x in f["solution"]], "solution_text": f["solution_text"],
                    "statements": str(p["statements"])})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--people", type=int, nargs="+", required=True)
    ap.add_argument("--per", type=int, required=True, help="puzzles per number of people")
    ap.add_argument("--seed", type=int, required=True, help="base seed; N people use seed + N")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = [r for n in a.people for r in make(n, a.per, a.seed + n)]
    assert len({r["statements"] for r in rows}) == len(rows), "duplicate puzzles"
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(rows)} puzzles -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
