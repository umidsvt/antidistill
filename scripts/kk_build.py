#!/usr/bin/env python
"""Puzzle testbed, step 2 (results/m8_audit_design.md §18): the training sets of the students, from the solution
writers' tries (scripts/kk_pilot.py on the training pool).

  hints  For F_D: the training puzzles where none of D's tries reaches the teacher's answer, with the teacher's
         answer as a hint (STaR's rationalization), for a second round of D tries.
  build  One training set per student, one solution per puzzle:
           U       the teacher's first try, correct or not
           Solo_B  B's first try; Solo_C: C's; Solo_D: D's first try (correct or not)
           F_D     D's first try whose answer equals the teacher's answer; else the first such try of the hint round
                   (its training text keeps the original puzzle, without the hint); puzzles still without one, and
                   puzzles where the teacher gave no readable answer, are reported and not trained on.

    python3 scripts/kk_build.py hints --pool data/kk/step2_train.jsonl --gen outputs/kk_step2/gen \
        --out data/kk/step2_hint_round.jsonl
    python3 scripts/kk_build.py build --pool data/kk/step2_train.jsonl --gen outputs/kk_step2/gen \
        --out outputs/kk_step2/train_sets
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kk_pilot import MODELS, hint_text  # noqa: E402


def load_tries(gen: Path, key: str) -> dict[str, list[dict]]:
    """All tries of one writer, merged over its shards (gen/<key>*/tries.jsonl)."""
    out: dict[str, list[dict]] = {}
    for f in sorted(glob.glob(str(gen / f"{key}_part*" / "tries.jsonl"))) + sorted(glob.glob(str(gen / key / "tries.jsonl"))):
        for line in open(f):
            r = json.loads(line)
            out.setdefault(r["id"], []).extend(r["tries"])
    return out


def as_target(text: str, key: str) -> str:
    """The R1 models' chat template opens the thinking in the prompt, so their stored text starts inside it."""
    if MODELS[key][1] and not text.lstrip().startswith("<think>"):
        return "<think>\n" + text
    return text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["hints", "build"])
    ap.add_argument("--pool", required=True)
    ap.add_argument("--gen", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pool = [json.loads(line) for line in open(a.pool)]
    gen = Path(a.gen)
    T, D = load_tries(gen, "T"), load_tries(gen, "D")
    teacher = {r["id"]: T[r["id"]][0]["labels"] for r in pool}  # None: no readable answer

    def first_match(tries, labels):
        return next((t for t in tries if labels is not None and t["labels"] == labels), None)

    if a.cmd == "hints":
        rows = [{**r, "hint": hint_text(r["names"], teacher[r["id"]])} for r in pool
                if teacher[r["id"]] is not None and first_match(D[r["id"]], teacher[r["id"]]) is None]
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(json.dumps({"pool": len(pool), "no_teacher_answer": sum(v is None for v in teacher.values()),
                          "hint_round": len(rows)}))
        return 0

    H = load_tries(gen, "D_hint")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    for student, key in (("U", "T"), ("Solo_B", "B"), ("Solo_C", "C"), ("Solo_D", "D")):
        tries = load_tries(gen, key)
        rows = [{"id": r["id"], "n_people": r["n_people"], "quiz": r["quiz"], "names": r["names"],
                 "text": as_target(tries[r["id"]][0]["text"], key), "correct": tries[r["id"]][0]["correct"]}
                for r in pool]
        (out / f"{student}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
        report[student] = {"examples": len(rows), "correct": sum(x["correct"] for x in rows)}
    rows, src = [], {"first_round": 0, "hint_round": 0, "unsolved": [], "no_teacher_answer": []}
    for r in pool:
        lab = teacher[r["id"]]
        if lab is None:
            src["no_teacher_answer"].append(r["id"]); continue
        t = first_match(D[r["id"]], lab)
        if t is not None:
            src["first_round"] += 1
        else:
            t = first_match(H.get(r["id"], []), lab)
            if t is None:
                src["unsolved"].append(r["id"]); continue
            src["hint_round"] += 1
        rows.append({"id": r["id"], "n_people": r["n_people"], "quiz": r["quiz"], "names": r["names"],
                     "text": as_target(t["text"], "D"), "correct": t["correct"]})
    (out / "F_D.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    report["F_D"] = {"examples": len(rows), "correct": sum(x["correct"] for x in rows), "first_round": src["first_round"],
                     "hint_round": src["hint_round"], "unsolved": len(src["unsolved"]),
                     "no_teacher_answer": len(src["no_teacher_answer"])}
    (out / "build_report.json").write_text(json.dumps({**report, "unsolved_ids": src["unsolved"],
                                                        "no_teacher_answer_ids": src["no_teacher_answer"]}, indent=1))
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
