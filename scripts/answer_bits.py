#!/usr/bin/env python
"""Surprisal of the teacher's answers under the attacker's own models (results/m8_audit_design.md §16).

Reads outputs/setup2_answers/<dataset>_<level>_s<seed>/tries.jsonl (scripts/dg_answer_dist.py) and reports, per
attacker level, how predictable the teacher's released answers are from the models that level owns:
  p_i   fraction of the level's pooled tries (16 per owned model, equal weight) whose answer matches the teacher's
  s_i   = -log2 p_i, for reachable problems (p_i > 0)
plus the unreachable count, the histogram of p_i, and the share of problems whose majority answer is the teacher's.
The same quantities against the true answers are reported as a diagnostic.

    python3 scripts/answer_bits.py --dataset gsm8k [--root outputs/setup2_answers] [--seed 123] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

LEVELS = {"A": ["A"], "B": ["A", "B"], "C": ["A", "B", "C"], "B alone": ["B"], "C alone": ["C"]}
BINS = [("0", lambda p: p == 0), ("(0, 1/4]", lambda p: 0 < p <= 0.25), ("(1/4, 3/4)", lambda p: 0.25 < p < 0.75),
        ("[3/4, 1)", lambda p: 0.75 <= p < 1), ("1", lambda p: p == 1)]


def load(root: Path, dataset: str, level: str, seed: int) -> dict[str, dict]:
    f = root / f"{dataset}_{level}_s{seed}" / "tries.jsonl"
    return {r["example_id"]: r for r in map(json.loads, f.open())} if f.exists() else {}


def measure(rows: list[list[dict]], key: str) -> dict:
    """rows: per problem, the pooled tries; key: 'match_teacher' or 'correct'."""
    ps = [sum(t[key] for t in tries) / len(tries) for tries in rows]
    reach = [p for p in ps if p > 0]
    knows = 0
    for tries in rows:
        answered = [t for t in tries if t.get("answer")]
        if answered:
            top, _ = Counter(t["answer"] for t in answered).most_common(1)[0]
            knows += any(t[key] for t in answered if t["answer"] == top)
    n = len(ps)
    return {"problems": n, "mean_p": round(sum(ps) / n, 4),
            "mean_surprisal_bits_reachable": round(sum(-math.log2(p) for p in reach) / max(len(reach), 1), 3),
            "total_surprisal_bits_reachable": round(sum(-math.log2(p) for p in reach), 1),
            "unreachable": n - len(reach), "majority_is_it": round(knows / n, 4),
            "histogram": {name: sum(f(p) for p in ps) for name, f in BINS}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--root", default="outputs/setup2_answers")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    data = {lv: load(Path(a.root), a.dataset, lv, a.seed) for lv in "ABC"}
    report = {}
    for name, owned in LEVELS.items():
        if not all(data[m] for m in owned):
            continue
        ids = sorted(set.intersection(*(set(data[m]) for m in owned)))
        # a problem without a teacher answer counts as unreachable against the teacher; its count is reported
        rows = [[t for m in owned for t in data[m][i]["tries"]] for i in ids]
        report[name] = {"owned": owned, "tries_per_problem": len(rows[0]) if rows else 0,
                        "vs_teacher": measure(rows, "match_teacher"), "vs_truth": measure(rows, "correct"),
                        "no_teacher_answer": sum(not data[owned[0]][i]["teacher_answer"] for i in ids)}
    for name, r in report.items():
        t, g = r["vs_teacher"], r["vs_truth"]
        print(f"{a.dataset} level {name:8s} ({'+'.join(r['owned'])}, {r['tries_per_problem']} tries): "
              f"vs teacher: mean p {t['mean_p']:.3f}, surprisal {t['mean_surprisal_bits_reachable']:.3f} bits/problem, "
              f"unreachable {t['unreachable']}, majority = teacher {t['majority_is_it']:.1%}, hist {t['histogram']} | "
              f"vs truth: mean p {g['mean_p']:.3f}, majority correct {g['majority_is_it']:.1%}")
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
