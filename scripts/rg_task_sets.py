#!/usr/bin/env python
"""I13 (results/m8_audit_design.md §38): cryptarithm infill sets, the second task's counterpart of the knights infill
views (scripts/kk_task_sets.py mask): the puzzle with the digits of all letters but m given as known, the full
mapping asked for. The given digits come from the teacher's answer (training) or the reference (evaluation); m is
drawn per puzzle from --ms; m >= the number of letters is the plain puzzle. Rows keep rg_build.py's fields, with the
question replaced and "hidden" (the hidden letters) and "task" added, so rg_pilot.py and kk_grpo.py --rg-teacher read
them unchanged.

    python3 scripts/rg_task_sets.py --pool data/rg/crypt_test.jsonl --source truth --ms 1 --out data/rg/crypt_test_m1.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ASK = "Provide a comma separated mapping"


def mask_question(row: dict, given: dict[str, int], hidden: list[str]) -> str:
    q = row["question"]
    assert ASK in q
    head, tail = q.split(ASK, 1)
    known = ", ".join(f"{k}={given[k]}" for k in sorted(given) if k not in hidden)
    return (f"{head}Some letters are already known: {known}. Find the digits of the other letters "
            f"({', '.join(hidden)}).\n{ASK}{tail}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", choices=["teacher", "truth"], required=True)
    ap.add_argument("--teacher", default="", help="the teacher's tries (rg_pilot.py tries.jsonl), for --source teacher")
    ap.add_argument("--ms", default="1")
    ap.add_argument("--seed", type=int, default=123)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    tries = {t["id"]: t["tries"][0] for t in map(json.loads, open(a.teacher))} if a.source == "teacher" else {}
    rows, kinds, skipped = [], {}, 0
    for r in map(json.loads, open(a.pool)):
        assert r["task"] == "crypt"
        src = r["solution"] if a.source == "truth" else tries[r["id"]]["parsed"]
        if src is None:
            skipped += 1
            continue
        m = rng.choice([int(x) for x in a.ms.split(",")])
        letters = r["letters"]
        if m >= len(letters):
            hidden, q = list(letters), r["question"]
        else:
            hidden = sorted(rng.sample(letters, m))
            q = mask_question(r, src, hidden)
        task = "solve" if len(hidden) == len(letters) else f"mask m={m}"
        rows.append({**r, "question": q, "hidden": hidden, "task": task})
        kinds[task] = kinds.get(task, 0) + 1
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(x) + "\n" for x in rows))
    print(json.dumps({"out": str(out), "rows": len(rows), "skipped (no source answer)": skipped, "tasks": kinds}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
