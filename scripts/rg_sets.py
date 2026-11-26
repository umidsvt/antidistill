#!/usr/bin/env python
"""Second task (results/m8_audit_design.md §36): the students' training sets from the teacher's tries on the training
puzzles (scripts/rg_pilot.py --model T, one try per puzzle), as the knights puzzles' (scripts/kk_build.py,
scripts/kk_phase1_sets.py):

  U   every training puzzle, the teacher's first try (correct or not), with "<think>\\n" restored at the start
  B1  every training puzzle whose teacher try has a readable answer: that answer alone (crypt: "A=1,B=2,..." with the
      letters sorted; zebra: one "House i: Cat=value, ..." line per house), no reasoning

Rows: {"id", "quiz" (the question, the whole user message: kk_sft.py --plain-prompt), "text", "correct"}.

    python3 scripts/rg_sets.py --pool data/rg/crypt_train.jsonl --teacher outputs/rg/crypt/T_train/tries.jsonl \
        --out outputs/rg/crypt/train_sets
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rg_common import correct  # noqa: E402


def answer_text(parsed, row) -> str:
    if row["task"] == "crypt":
        return ",".join(f"{k}={parsed[k]}" for k in sorted(parsed))
    return "\n".join(f"House {h + 1}: " + ", ".join(f"{c}={v}" for c, v in zip(row["categories"], parsed[h]))
                     for h in range(len(parsed)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pool = {r["id"]: r for r in map(json.loads, open(a.pool))}
    tries = {t["id"]: t["tries"][0] for t in map(json.loads, open(a.teacher))}
    missing = [i for i in pool if i not in tries]
    assert not missing, f"{len(missing)} training puzzles have no teacher try"
    u, b1 = [], []
    for i, r in pool.items():
        t = tries[i]
        text = t["text"] if t["text"].lstrip().startswith("<think>") else "<think>\n" + t["text"]
        u.append({"id": i, "quiz": r["question"], "text": text, "correct": t["correct"]})
        if t["parsed"] is not None:
            b1.append({"id": i, "quiz": r["question"], "text": answer_text(t["parsed"], r),
                       "correct": correct(t["parsed"], r)})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("U", u), ("B1_answers", b1)):
        (out / f"{name}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    rep = {"pool": len(pool), "U": len(u), "U right": sum(x["correct"] for x in u), "B1": len(b1),
           "B1 right": sum(x["correct"] for x in b1), "no readable teacher answer": len(pool) - len(b1)}
    (out / "report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
