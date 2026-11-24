#!/usr/bin/env python
"""Phase 2 (results/m8_audit_design.md §29, results/phase2_ideas.md): puzzle files whose question carries a task built
from an answer, for training (from the teacher's answers) and for evaluation (from the reference answers or from a
student's stored answers). Ids, names and solutions are kept, so kk_grpo.py, kk_star.py, kk_sft.py and kk_eval.py
read these files like the plain puzzle files (their prompt is kk_sft.user_turn(quiz)).

  repair  the source answer with k people flipped (k drawn from --ks per puzzle) after "may contain errors"   (I1)
  mask    the source labels of all but m people (m from --ms) as known identities                             (I2)
  guess   a student's stored answer (--guesses results.jsonl) as the proposed answer                         (I3)
  Options: --source teacher|truth (repair, mask); --mix-solve p keeps the plain question for a share p (I5).

    python3 -I scripts/kk_task_sets.py repair --ks 1,2 --source teacher --pool data/kk/step2_train.jsonl \
        --out outputs/kk_phase2/sets/train_repair_k12.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kk_build import load_tries  # noqa: E402

REPAIR = ("Below is a proposed answer to the puzzle. It may contain errors. Check every person's statement against "
          "it and give the corrected answer.")


def answer_text(names, labels):
    return "\n".join(f"({i}) {n} is a {'knight' if k else 'knave'}" for i, (n, k) in enumerate(zip(names, labels), 1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["repair", "mask", "guess"])
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", choices=["teacher", "truth"], default="teacher")
    ap.add_argument("--ks", default="1")
    ap.add_argument("--ms", default="1")
    ap.add_argument("--guesses", default="")
    ap.add_argument("--mix-solve", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=123)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    pool = [json.loads(line) for line in open(a.pool)]
    T = load_tries(Path("outputs/kk_step2/gen"), "T") if a.source == "teacher" and a.cmd != "guess" else {}
    guesses = {r["id"]: r["labels"] for r in map(json.loads, open(a.guesses))} if a.cmd == "guess" else {}
    rows, skipped, kinds = [], 0, {}
    for r in pool:
        if a.cmd == "guess":
            src = guesses.get(r["id"])
        else:
            src = T[r["id"]][0]["labels"] if a.source == "teacher" else list(r["solution"])
        if src is None:
            skipped += 1
            continue
        if a.mix_solve and rng.random() < a.mix_solve:
            rows.append({**r, "task": "solve"})
            kinds["solve"] = kinds.get("solve", 0) + 1
            continue
        names, n = r["names"], len(r["names"])
        if a.cmd == "repair":
            k = rng.choice([int(x) for x in a.ks.split(",")])
            given = list(src)
            for j in rng.sample(range(n), k):
                given[j] = not given[j]
            quiz = f"{r['quiz']}\n\n{REPAIR}\n\n### Proposed answer\n{answer_text(names, given)}"
            task = f"repair k={k}"
        elif a.cmd == "guess":
            quiz = f"{r['quiz']}\n\n{REPAIR}\n\n### Proposed answer\n{answer_text(names, src)}"
            task = "guess"
        else:
            m = rng.choice([int(x) for x in a.ms.split(",")])
            hidden = sorted(rng.sample(range(n), m))
            known = "\n".join(f"- {names[i]} is a {'knight' if src[i] else 'knave'}" for i in range(n) if i not in hidden)
            unknown = ", ".join(names[i] for i in hidden)
            quiz = (f"{r['quiz']}\n\nSome of the identities are already known:\n{known}\nFind the identity of "
                    f"{unknown} from the statements, and give the full answer for all {n} people.")
            task = f"mask m={m}"
        rows.append({**r, "quiz": quiz, "task": task})
        kinds[task] = kinds.get(task, 0) + 1
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(x) + "\n" for x in rows))
    print(json.dumps({"out": str(out), "rows": len(rows), "skipped (no source answer)": skipped, "tasks": kinds}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
