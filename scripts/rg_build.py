#!/usr/bin/env python
"""Second task (results/m8_audit_design.md §34): build puzzle files from Reasoning Gym generators. Run with .venv-rg.

  zebra  Reasoning Gym's zebra puzzles (unique solution guaranteed by the generator), K houses and M categories; the
         question asks for the full arrangement (every house, every category) instead of one cell, so the answer
         cannot be guessed (one of (K!)^M grids).
  crypt  Reasoning Gym's addition cryptarithms with W addends of 3-5 digits; only puzzles with exactly one solution
         are kept (checked by an exhaustive column-wise search), so the reference mapping is the only right answer.

    .venv-rg/bin/python -W ignore scripts/rg_build.py --task zebra --levels 4x4,5x5,6x6 --per-level 100 --seed 7 \
        --out data/rg/pilot_zebra.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from random import Random

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rg_common import ZEBRA_FORMAT  # noqa: E402


def label(cls_name: str) -> str:
    words = re.findall(r"[A-Z][a-z]*", cls_name)
    return " ".join([words[0]] + [w.lower() for w in words[1:]])


def zebra(rng: Random, k: int, m: int, idx: str) -> dict:
    from reasoning_gym.logic.contrib.logic_puzzle.generate import generate_puzzle
    _, puz = generate_puzzle(rng, k, m)
    text = str(puz).rstrip()
    classes = list(puz.element_classes)
    cats = [label(c.__name__) for c in classes]
    values = []
    for c in classes:
        head = c.description().split(":", 1)[0] + ":"   # some descriptions go on to list example values
        line = next(x for x in text.splitlines() if x.strip().startswith("- " + head))
        values.append([v.strip() for v in line.split(head, 1)[1].split(",")])
    grid = [[None] * len(classes) for _ in range(k)]
    for lit, house in puz.solution.items():
        j = classes.index(type(lit))
        shown = lit.name.replace("_", " ")
        assert shown in values[j], (shown, values[j])
        grid[house - 1][j] = shown
    assert all(x is not None for h in grid for x in h)
    example = "\n".join(f"House {h}: " + ", ".join(f"{c}=..." for c in cats) for h in range(1, k + 1))
    question = f"{text}\n\n{ZEBRA_FORMAT.format(example=example)}"
    answer = "\n".join(f"House {h + 1}: " + ", ".join(f"{c}={v}" for c, v in zip(cats, grid[h])) for h in range(k))
    return {"id": idx, "task": "zebra", "level": f"{k}x{m}", "question": question, "answer": answer,
            "categories": cats, "values": values, "solution": grid}


def count_solutions(words: list[str], result: str, limit: int = 2) -> int:
    """Number of letter -> digit maps (distinct digits, no leading zero) with sum(words) == result, up to `limit`."""
    cols = len(result)
    if max(len(w) for w in words) > cols:
        return 0
    rev = [w[::-1] for w in words]
    rres = result[::-1]
    lead = {w[0] for w in words + [result] if len(w) > 1}
    found = 0
    assign: dict[str, int] = {}
    used = [False] * 10

    def column(c: int, carry: int, pending: list[str]) -> None:
        nonlocal found
        if found >= limit:
            return
        if c == cols:
            if carry == 0:
                found += 1
            return
        if pending:   # assign the next unassigned letter of this column
            ch = pending[0]
            for d in range(10):
                if used[d] or (d == 0 and ch in lead):
                    continue
                assign[ch] = d; used[d] = True
                column(c, carry, pending[1:])
                used[d] = False; del assign[ch]
            return
        s = carry + sum(assign[w[c]] for w in rev if c < len(w))
        if s % 10 == assign[rres[c]]:
            nxt = c + 1
            if nxt < cols:
                need = []
                for ch in [w[nxt] for w in rev if nxt < len(w)] + [rres[nxt]]:
                    if ch not in assign and ch not in need:
                        need.append(ch)
                column(nxt, s // 10, need)
            else:
                column(nxt, s // 10, [])

    first = []
    for ch in [w[0] for w in rev] + [rres[0]]:
        if ch not in first:
            first.append(ch)
    column(0, 0, first)
    return found


def crypt(words_n: int, seed: int, need: int, prefix: str) -> tuple[list[dict], dict]:
    import reasoning_gym as rg
    ds = rg.create_dataset("cryptarithm", min_words=words_n, max_words=words_n, size=200 * need, seed=seed)
    rows, tried = [], 0
    for i in range(len(ds)):
        e = ds[i]
        tried += 1
        meta = e["metadata"]
        words, result = meta["words_letters"], meta["result_letters"]
        if count_solutions(words, result) != 1:
            continue
        sol = {k: int(v) for k, v in meta["letter_to_digit"].items()}
        assert sum(int("".join(str(sol[c]) for c in w)) for w in words) == int("".join(str(sol[c]) for c in result))
        rows.append({"id": f"{prefix}_{len(rows):04d}", "task": "crypt", "level": f"w{words_n}",
                     "question": e["question"], "answer": e["answer"], "letters": sorted(sol), "solution": sol,
                     "words": words, "result": result})
        if len(rows) == need:
            break
    return rows, {"level": f"w{words_n}", "generated": tried, "unique": len(rows)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["zebra", "crypt"], required=True)
    ap.add_argument("--levels", required=True, help="zebra: KxM,...; crypt: number of addends, e.g. 2,3,4")
    ap.add_argument("--per-level", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows, report = [], []
    for li, lev in enumerate(a.levels.split(",")):
        if a.task == "zebra":
            k, m = map(int, lev.split("x"))
            for i in range(a.per_level):
                rows.append(zebra(Random(a.seed * 100003 + li * 10007 + i), k, m, f"zebra{lev}_s{a.seed}_{i:04d}"))
            report.append({"level": lev, "puzzles": a.per_level})
        else:
            got, rep = crypt(int(lev), a.seed * 1000 + li, a.per_level, f"crypt_w{lev}_s{a.seed}")
            rows += got
            report.append(rep)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(json.dumps({"out": str(out), "rows": len(rows), "levels": report}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
