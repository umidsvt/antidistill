#!/usr/bin/env python
"""Second task (results/m8_audit_design.md §37): answers-only SFT set with relabeled copies, the cryptarithm
counterpart of the knights puzzles' renamed copies (scripts/kk_views.py rename, I6c).

For every training puzzle with a readable teacher answer: the plain puzzle with the teacher's answer as target (B1's
example), plus --copies copies in which every letter of the puzzle is replaced by another letter (a random one-to-one
relabeling; only the puzzle lines change, not the instruction), with the teacher's answer relabeled the same way.
No reasoning in any target.

    python3 scripts/rg_views.py --pool data/rg/crypt_train.jsonl --teacher outputs/rg/crypt/T_train/tries.jsonl \
        --copies 3 --out outputs/rg/crypt/train_sets/B1_relabel3.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import string
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rg_common import correct  # noqa: E402
from rg_sets import answer_text  # noqa: E402

HEAD, SEP = "Solve this cryptarithm:\n\n", "\n\nEach letter"


def relabel(row: dict, table: dict[str, str]) -> dict:
    q = row["question"]
    assert q.startswith(HEAD) and SEP in q
    block, rest = q[len(HEAD):].split(SEP, 1)
    block = "".join(table.get(ch, ch) for ch in block)
    sol = {table[k]: v for k, v in row["solution"].items()}
    return {**row, "question": HEAD + block + SEP + rest, "letters": sorted(sol), "solution": sol,
            "words": ["".join(table[c] for c in w) for w in row["words"]],
            "result": "".join(table[c] for c in row["result"])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--copies", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=123)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    tries = {t["id"]: t["tries"][0] for t in map(json.loads, open(a.teacher))}
    rows = []
    for r in map(json.loads, open(a.pool)):
        assert r["task"] == "crypt"
        ans = tries[r["id"]]["parsed"]
        if ans is None:
            continue
        rows.append({"id": r["id"], "quiz": r["question"], "text": answer_text(ans, r), "correct": correct(ans, r),
                     "view": "plain"})
        for _ in range(a.copies):
            table = dict(zip(r["letters"], rng.sample(string.ascii_uppercase, len(r["letters"]))))
            rr = relabel(r, table)
            ans2 = {table[k]: v for k, v in ans.items()}
            rows.append({"id": r["id"], "quiz": rr["question"], "text": answer_text(ans2, rr),
                         "correct": correct(ans2, rr), "view": "relabel"})
    rng.shuffle(rows)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(x) + "\n" for x in rows))
    print(json.dumps({"out": str(out), "rows": len(rows), "right": sum(x["correct"] for x in rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
