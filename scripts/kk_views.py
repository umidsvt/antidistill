#!/usr/bin/env python
"""Phase 2 wave 3 (results/m8_audit_design.md §31, results/phase2_ideas.md I6): answers-only SFT sets with several
examples per training puzzle, built from the teacher's answer alone. Every target is the teacher's full answer in B1's
format (kk_phase1_sets.conclusion), with no reasoning.

  views   I6: the plain puzzle, plus --copies infill views of it: the teacher's labels of a random subset given as
          known (kk_task_sets.mask_quiz), m hidden people with m drawn uniformly from 1..N-1 per view.
  rename  I6c (control for more examples): the plain puzzle, plus --copies copies with every name replaced by other
          names from the training puzzles (same statements, answer relabeled by position).

    python3 -I scripts/kk_views.py views --copies 3 --out outputs/kk_phase2/sets/sft_views3.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kk_build import load_tries  # noqa: E402
from kk_phase1_sets import conclusion  # noqa: E402
from kk_task_sets import mask_quiz  # noqa: E402


def rename(r, new_names):
    old = r["names"]
    table = dict(zip(old, new_names))
    pat = re.compile(r"\b(" + "|".join(re.escape(x) for x in sorted(old, key=len, reverse=True)) + r")\b")
    return {**r, "quiz": pat.sub(lambda m: table[m.group(1)], r["quiz"]), "names": list(new_names)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["views", "rename"])
    ap.add_argument("--copies", type=int, default=3)
    ap.add_argument("--pool", default="data/kk/step2_train.jsonl")
    ap.add_argument("--teacher-gen", default="outputs/kk_step2/gen")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=123)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    pool = [json.loads(line) for line in open(a.pool)]
    T = load_tries(Path(a.teacher_gen), "T")
    name_pool = sorted({x for r in pool for x in r["names"]})
    rows, kinds = [], {}
    for r in pool:
        labels = T[r["id"]][0]["labels"]
        if labels is None:   # as B1: puzzles without a teacher answer cannot be trained on
            continue
        n = len(r["names"])
        base = {"id": r["id"], "n_people": r["n_people"], "correct": labels == list(r["solution"])}
        rows.append({**base, "quiz": r["quiz"], "names": r["names"], "text": conclusion(r["names"], labels),
                     "view": "plain"})
        for c in range(a.copies):
            if a.cmd == "views":
                m = rng.randint(1, n - 1)
                hidden = sorted(rng.sample(range(n), m))
                rows.append({**base, "quiz": mask_quiz(r, labels, hidden), "names": r["names"],
                             "text": conclusion(r["names"], labels), "view": f"mask m={m}", "hidden": hidden})
            else:
                new = rng.sample(name_pool, n)
                rr = rename(r, new)
                assert all(x in rr["quiz"] for x in new), r["id"]
                rows.append({**base, "quiz": rr["quiz"], "names": new, "text": conclusion(new, labels),
                             "view": "rename"})
    for x in rows:
        k = x["view"].split(" m=")[0]
        kinds[k] = kinds.get(k, 0) + 1
    rng.shuffle(rows)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(x) + "\n" for x in rows))
    print(json.dumps({"out": str(out), "rows": len(rows), "views": kinds}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
