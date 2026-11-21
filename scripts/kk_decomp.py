#!/usr/bin/env python
"""Gap breakdown on the puzzle testbed (results/m8_audit_design.md §26), the analyses that read stored files. Seed 123
unless stated. Reference answers are used for diagnosis only.

  d1  reaching the answer: readable answers, why the others are unreadable, people labeled right, puzzles reached
  d3  the teacher's errors: B1, B5 and U_inst on the training puzzles (needs outputs/kk_phase1/decomp/train_preds/)
  d4  where U_inst beats B5 on the test puzzles
  d5  a surface-feature predictor of each person's label (logistic regression), against B1

    PYTHONPATH=scripts .venv-dg/bin/python scripts/kk_decomp.py d1 d4 d5 [d3]
"""
from __future__ import annotations

import ast
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from kk_build import load_tries  # noqa: E402

OUT = REPO / "outputs/kk_phase1/decomp"
TRAIN = REPO / "data/kk/step2_train.jsonl"
TEST = REPO / "data/kk/step2_test.jsonl"


def jl(path):
    return [json.loads(line) for line in open(path)]


def teacher_labels():
    T = load_tries(REPO / "outputs/kk_step2/gen", "T")
    return {k: v[0]["labels"] for k, v in T.items()}


def pct(a, b):
    return round(100 * a / b, 1) if b else None


def d1() -> dict:
    out = {}
    test = {r["id"]: r for r in jl(TEST)}
    for s in (123, 124, 125):
        rows = jl(REPO / f"outputs/kk_step2/students/untrained_inst_s{s}/test/results.jsonl")
        why = Counter()
        people = right = 0
        for r in rows:
            if r["labels"] is not None:
                sol = test[r["id"]]["solution"]
                people += len(sol)
                right += sum(x == y for x, y in zip(r["labels"], sol))
                continue
            plain = re.sub(r"[*_`]", "", r["text"])
            if r["finish_reason"] == "length":
                why["cut at the token limit"] += 1
            elif not re.search(r"conclusion\s*:", plain, re.I):
                why["no CONCLUSION"] += 1
            else:
                why["CONCLUSION without one label per person"] += 1
        out[f"I0 test s{s}"] = {"replies": len(rows), "readable %": pct(len(rows) - sum(why.values()), len(rows)),
                                "unreadable": dict(why), "people right in readable answers %": pct(right, people)}
    train = {r["id"]: r for r in jl(TRAIN)}
    tl = teacher_labels()
    tries = jl(REPO / "outputs/kk_phase1/own_tries_s123/tries.jsonl")
    n = read = right_t = right_ref = people = people_right = 0
    reach_teacher = reach_truth = answered = 0
    for t in tries:
        sol, lab = list(train[t["id"]]["solution"]), tl[t["id"]]
        labs = t["labels"]
        n += len(labs)
        ok = [x for x in labs if x is not None]
        read += len(ok)
        right_ref += sum(x == sol for x in ok)
        for x in ok:
            people += len(sol)
            people_right += sum(a == b for a, b in zip(x, sol))
        if lab is not None:
            answered += 1
            reach_teacher += any(x == lab for x in ok)
            reach_truth += any(x == sol for x in ok)
    out["own tries on training puzzles (8 each, s123)"] = {
        "tries": n, "readable %": pct(read, n), "right %": pct(right_ref, n),
        "people right in readable answers %": pct(people_right, people),
        "puzzles with a teacher answer": answered, "teacher's answer reached at least once": reach_teacher,
        "true answer reached at least once": reach_truth}
    return out


def d3() -> dict:
    train = {r["id"]: r for r in jl(TRAIN)}
    tl = teacher_labels()
    out = {}
    for name in ("B1", "B5", "Uinst"):
        p = OUT / f"train_preds/{name}_s123/results.jsonl"
        if not p.exists():
            out[name] = "pending"
            continue
        rows = {r["id"]: r for r in jl(p)}
        c = Counter()
        for pid, lab in tl.items():
            if pid not in rows:
                continue
            pred, sol = rows[pid]["labels"], list(train[pid]["solution"])
            if lab is None:
                c["no teacher answer: right"] += pred == sol
                c["no teacher answer: n"] += 1
            elif lab == sol:
                c["teacher right: n"] += 1
                c["teacher right: student right"] += pred == sol
            else:
                c["teacher wrong: n"] += 1
                c["teacher wrong: student = teacher"] += pred == lab
                c["teacher wrong: student = truth"] += pred == sol
                c["teacher wrong: neither"] += pred is not None and pred != lab and pred != sol
                c["teacher wrong: unreadable"] += pred is None
        out[name] = dict(c)
    return out


def d4() -> dict:
    test = {r["id"]: r for r in jl(TEST)}
    U = {r["id"]: r for r in jl(REPO / "outputs/kk_step2/students/Uinst_s123/test/results.jsonl")}
    B = {r["id"]: r for r in jl(REPO / "outputs/kk_phase1/students/B5_s123/test/results.jsonl")}
    cross, by_n = Counter(), {}
    for pid in test:
        u, b = U[pid]["correct"], B[pid]["correct"]
        key = {(1, 1): "both right", (1, 0): "U_inst only", (0, 1): "B5 only", (0, 0): "neither"}[(int(u), int(b))]
        cross[key] += 1
        by_n.setdefault(test[pid]["n_people"], Counter())[key] += 1
    fin = [p for p in test if U[p]["finish_reason"] == "stop"]
    near = [sum(x == y for x, y in zip(B[p]["labels"], test[p]["solution"])) / len(test[p]["solution"])
            for p in test if not B[p]["correct"] and B[p]["labels"] is not None]
    return {"cross-table (1,000 test puzzles)": dict(cross), "by size": {str(k): dict(v) for k, v in sorted(by_n.items())},
            "U_inst finished": len(fin), "U_inst right when finished %": pct(sum(U[p]["correct"] for p in fin), len(fin)),
            "U_inst right when cut %": pct(sum(U[p]["correct"] for p in test if p not in fin), len(test) - len(fin)),
            "B5 wrong answers: share of people right %": round(100 * sum(near) / len(near), 1) if near else None}


OPS = ["telling-truth", "lying", "and", "or", "not", "->", "<=>"]


def features(statements, n):
    """Per person: surface counts of the puzzle's statements (no logical evaluation)."""
    st = ast.literal_eval(statements)
    rows = []
    for i in range(n):
        f = Counter()
        for j, s in enumerate(st):
            who = "self" if j == i else "other"

            def walk(node, parents):
                op = node[0]
                if op in ("telling-truth", "lying"):
                    if node[1] == i:
                        f[f"{who}:{op}"] += 1
                        neg = parents.count("not") % 2
                        f[f"{who}:{op}:neg{neg}"] += 1
                        if parents:
                            f[f"{who}:{op}:under:{parents[-1]}"] += 1
                    return
                for child in node[1:]:
                    walk(child, parents + [op])
            walk(s, [])
        f[f"own top:{st[i][0]}"] += 1
        f[f"N={n}"] += 1
        rows.append(f)
    return rows


def d5() -> dict:
    import torch
    tl = teacher_labels()
    train = [r for r in jl(TRAIN) if tl[r["id"]] is not None]
    test = jl(TEST)
    keys = sorted({k for r in train + test for f in features(r["statements"], r["n_people"]) for k in f})
    idx = {k: i for i, k in enumerate(keys)}

    def mat(rows, label_of):
        X, y, groups = [], [], []
        for g, r in enumerate(rows):
            for p, f in enumerate(features(r["statements"], r["n_people"])):
                v = [0.0] * len(keys)
                for k, c in f.items():
                    v[idx[k]] = float(c)
                X.append(v)
                y.append(float(label_of(r)[p]))
                groups.append(g)
        return torch.tensor(X), torch.tensor(y), groups

    out = {"features": len(keys)}
    Xte, yte, gte = mat(test, lambda r: r["solution"])
    for name, label_of in (("teacher's answers", lambda r: tl[r["id"]]), ("reference answers", lambda r: r["solution"])):
        Xtr, ytr, _ = mat(train, label_of)
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        w = torch.zeros(len(keys), requires_grad=True)
        b = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([w, b], max_iter=500)

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(((Xtr - mu) / sd) @ w + b, ytr) + 1e-3 * (w ** 2).sum()
            loss.backward()
            return loss
        opt.step(closure)
        with torch.no_grad():
            pred = ((((Xte - mu) / sd) @ w + b) > 0).float()
        ok = (pred == yte).tolist()
        per_puzzle = Counter()
        for g, o in zip(gte, ok):
            per_puzzle[g] += 0 if o else 1
        out[f"trained on {name}"] = {"people right %": round(100 * sum(ok) / len(ok), 1),
                                     "test puzzles solved %": pct(sum(v == 0 for v in per_puzzle.values()), len(test))}
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    for cmd in sys.argv[1:]:
        res[cmd] = {"d1": d1, "d3": d3, "d4": d4, "d5": d5}[cmd]()
        print(cmd, json.dumps(res[cmd], indent=1), flush=True)
    path = OUT / "breakdown.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    path.write_text(json.dumps({**old, **res}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
