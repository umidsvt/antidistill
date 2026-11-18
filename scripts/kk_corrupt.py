#!/usr/bin/env python
"""Phase 2 (results/m8_audit_design.md §22): the answer floor F_D when a share of the released answers is wrong.

  hints  The corrupted answers (all levels; nested), and the hint round for corrupted puzzles whose new answer none of
         D's step-2 tries reaches (rows for scripts/kk_pilot.py --model D --k 4).
  build  F_D^c for c in 12, 30, 50: as F_D (scripts/kk_build.py) with the released answers; unchanged answers reuse
         §18's hint round (gen/D_hint*), corrupted ones the new round (gen/D_chint*).

    python3 -I scripts/kk_corrupt.py hints --pool data/kk/step2_train.jsonl --gen outputs/kk_step2/gen \
        --out outputs/kk_step2/corrupt/hint_round.jsonl
    python3 -I scripts/kk_corrupt.py build --pool data/kk/step2_train.jsonl --gen outputs/kk_step2/gen \
        --out outputs/kk_step2/train_sets
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kk_build import as_target, load_tries  # noqa: E402
from kk_pilot import hint_text  # noqa: E402

SEED = 20261117
LEVELS = (12, 30, 50)


def corrupted_answers(pool: list[dict], teacher: dict[str, list[bool] | None]) -> tuple[list[str], dict[str, list[bool]]]:
    """The random order of the puzzles with a readable teacher answer, and each one's wrong answer (fixed per puzzle)."""
    rng = random.Random(SEED)
    ids = [r["id"] for r in pool if teacher[r["id"]] is not None]
    order = ids[:]
    rng.shuffle(order)
    wrong = {}
    for r in pool:
        if teacher[r["id"]] is None:
            continue
        others = [list(x) for x in itertools.product([True, False], repeat=r["n_people"]) if list(x) != teacher[r["id"]]]
        wrong[r["id"]] = rng.choice(others)
    return order, wrong


def chosen(order: list[str], c: int) -> set[str]:
    return set(order[: round(c / 100 * len(order))])


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
    teacher = {r["id"]: T[r["id"]][0]["labels"] for r in pool}
    order, wrong = corrupted_answers(pool, teacher)
    top = chosen(order, max(LEVELS))

    def first_match(tries, labels):
        return next((t for t in tries if t["labels"] == labels), None)

    if a.cmd == "hints":
        rows = [{**r, "hint": hint_text(r["names"], wrong[r["id"]])} for r in pool
                if r["id"] in top and first_match(D[r["id"]], wrong[r["id"]]) is None]
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(json.dumps(r) + "\n" for r in rows))
        (out.parent / "corrupted_answers.json").write_text(json.dumps(
            {"seed": SEED, "order": order, "wrong": wrong, "levels": {c: len(chosen(order, c)) for c in LEVELS}}))
        print(json.dumps({"readable": len(order), "levels": {c: len(chosen(order, c)) for c in LEVELS},
                          "reached_in_first_round": len(top) - len(rows), "hint_round": len(rows)}))
        return 0

    H, CH = load_tries(gen, "D_hint"), load_tries(gen, "D_chint")
    out = Path(a.out)
    report = {}
    truth = {r["id"]: list(r["solution"]) for r in pool}
    for c in LEVELS:
        bad = chosen(order, c)
        rows, src = [], {"first_round": 0, "hint_round": 0, "unsolved": 0, "corrupted_kept": 0,
                         "corrupted_first_round": 0, "corrupted_hint_round": 0, "no_teacher_answer": 0}
        for r in pool:
            lab = wrong[r["id"]] if r["id"] in bad else teacher[r["id"]]
            if lab is None:
                src["no_teacher_answer"] += 1
                continue
            t, rnd = first_match(D[r["id"]], lab), "first_round"
            if t is None:
                t, rnd = first_match((CH if r["id"] in bad else H).get(r["id"], []), lab), "hint_round"
            if t is None:
                src["unsolved"] += 1
                continue
            src[rnd] += 1
            if r["id"] in bad:
                src["corrupted_kept"] += 1
                src[f"corrupted_{rnd}"] += 1
            rows.append({"id": r["id"], "n_people": r["n_people"], "quiz": r["quiz"], "names": r["names"],
                         "text": as_target(t["text"], "D"), "correct": t["correct"]})
        released_wrong = sum(1 for r in pool if teacher[r["id"]] is not None
                             and (wrong[r["id"]] if r["id"] in bad else teacher[r["id"]]) != truth[r["id"]])
        (out / f"F_D_c{c}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
        report[f"F_D_c{c}"] = {"corrupted": len(bad), "examples": len(rows), **src,
                               "released_wrong": released_wrong,
                               "released_wrong_or_missing_share": round((released_wrong + src["no_teacher_answer"]) / len(pool), 4)}
    (out / "build_report_corrupt.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
