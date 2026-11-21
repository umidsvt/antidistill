#!/usr/bin/env python
"""Phase 1 on the puzzle testbed (results/m8_audit_design.md §25.2): training sets built from the teacher's released
answers only.

  answers    B1: for every training puzzle whose teacher solution has a readable answer (the teacher's first try, as
             §20), the target is that answer alone in the task's format ("CONCLUSION:" and one numbered line per
             person), no reasoning. Puzzles without a teacher answer are reported and not trained on (§20).
  reference  C2 (§25.3): the same 1,712 puzzles, with the reference answer instead of the teacher's.
  majority   C1 (§25.3): the same 1,712 puzzles, with the student's own majority answer over its tries
             (--tries, from scripts/kk_sample.py; ties: the first seen); puzzles with no readable try are reported
             and not trained on.

    python3 scripts/kk_phase1_sets.py answers --pool data/kk/step2_train.jsonl --teacher-gen outputs/kk_step2/gen \
        --out outputs/kk_phase1/train_sets/B1_answers.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kk_build import load_tries  # noqa: E402


def conclusion(names: list[str], labels: list[bool]) -> str:
    """The answer format of Xie et al.'s instruction and demonstration."""
    return "CONCLUSION:\n" + "".join(f"({i}) {n} is a {'knight' if k else 'knave'}\n"
                                     for i, (n, k) in enumerate(zip(names, labels), 1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["answers", "reference", "majority"])
    ap.add_argument("--tries", default="", help="majority: tries.jsonl of scripts/kk_sample.py")
    ap.add_argument("--pool", required=True)
    ap.add_argument("--teacher-gen", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pool = [json.loads(line) for line in open(a.pool)]
    T = load_tries(Path(a.teacher_gen), "T")
    own = {}
    if a.cmd == "majority":
        from collections import Counter
        for line in open(a.tries):
            t = json.loads(line)
            readable = [tuple(x) for x in t["labels"] if x is not None]
            own[t["id"]] = list(Counter(readable).most_common(1)[0][0]) if readable else None
    rows, missing, no_own = [], [], []
    for r in pool:
        labels = T[r["id"]][0]["labels"]
        if labels is None:
            missing.append(r["id"])
            continue
        if a.cmd == "reference":
            labels = list(r["solution"])
        elif a.cmd == "majority":
            labels = own.get(r["id"])
            if labels is None:
                no_own.append(r["id"])
                continue
        rows.append({"id": r["id"], "n_people": r["n_people"], "quiz": r["quiz"], "names": r["names"],
                     "text": conclusion(r["names"], labels), "correct": labels == list(r["solution"])})
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(x) + "\n" for x in rows))
    report = {"cmd": a.cmd, "pool": len(pool), "examples": len(rows), "answer_right": sum(x["correct"] for x in rows),
              "no_teacher_answer": len(missing), "no_readable_own_try": len(no_own)}
    out.with_suffix(".report.json").write_text(json.dumps({**report, "no_teacher_answer_ids": missing,
                                                            "no_readable_own_try_ids": no_own}, indent=1))
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
