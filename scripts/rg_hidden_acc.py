#!/usr/bin/env python
"""m8 §38: score a cryptarithm infill evaluation (rg_pilot.py tries.jsonl on an rg_task_sets.py file): per try,
readable, the given letters copied right, every hidden letter right, and the share of hidden letters right.

    python3 -I scripts/rg_hidden_acc.py data/rg/crypt_probe_m1.jsonl outputs/rg/probe/m1_B/tries.jsonl
"""
from __future__ import annotations

import json
import sys

puz = {r["id"]: r for r in map(json.loads, open(sys.argv[1]))}
n = readable = given = hidden_all = 0
hidden_cells = hidden_total = 0
for t in map(json.loads, open(sys.argv[2])):
    r = puz[t["id"]]
    for x in t["tries"]:
        n += 1
        p = x["parsed"]
        hidden_total += len(r["hidden"])
        if p is None:
            continue
        readable += 1
        sol = {k: int(v) for k, v in r["solution"].items()}
        given += all(p.get(k) == sol[k] for k in sol if k not in r["hidden"])
        hidden_all += all(p.get(k) == sol[k] for k in r["hidden"])
        hidden_cells += sum(p.get(k) == sol[k] for k in r["hidden"])
print(json.dumps({"tries": n, "readable": round(readable / n, 4), "given right | readable": round(given / max(readable, 1), 4),
                  "hidden all right | readable": round(hidden_all / max(readable, 1), 4),
                  "hidden all right": round(hidden_all / n, 4), "hidden letters right": round(hidden_cells / hidden_total, 4)}))
