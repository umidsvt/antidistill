#!/usr/bin/env python
"""m8 §31: score an infill evaluation by person group. For each reply (kk_eval.py results.jsonl on an infill set built
by kk_task_sets.py mask): readable, the given people copied right, every hidden person right, exact.

    python3 -I scripts/kk_hidden_acc.py outputs/kk_phase2/sets/test_mask_truth_m1.jsonl <student>/test_mask/results.jsonl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kk_pilot import parse_labels  # noqa: E402


def hidden_of(r):
    if "hidden" in r:
        return r["hidden"]
    if "Find the identity of " not in r["quiz"]:
        return list(range(len(r["names"])))
    who = r["quiz"].rsplit("Find the identity of ", 1)[1].split(" from the statements")[0].split(", ")
    return [r["names"].index(x) for x in who]


def score(puzzles: str, results: str) -> dict:
    puz = {r["id"]: r for r in map(json.loads, open(puzzles))}
    n = readable = given = hidden = exact = 0
    for res in map(json.loads, open(results)):
        p = puz[res["id"]]
        sol, h = list(p["solution"]), hidden_of(p)
        text = res["text"].rsplit("</think>", 1)[-1]
        lab = parse_labels(text, p["names"], reasoning=False)
        n += 1
        if lab is None:
            continue
        readable += 1
        given += all(lab[i] == sol[i] for i in range(len(sol)) if i not in h)
        hidden += all(lab[i] == sol[i] for i in h)
        exact += lab == sol
    return {"replies": n, "readable": round(readable / n, 4), "given right | readable": round(given / max(readable, 1), 4),
            "hidden right | readable": round(hidden / max(readable, 1), 4), "hidden right": round(hidden / n, 4),
            "exact": round(exact / n, 4)}


if __name__ == "__main__":
    print(json.dumps(score(sys.argv[1], sys.argv[2])))
