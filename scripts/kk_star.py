#!/usr/bin/env python
"""Puzzle testbed, level A (results/m8_audit_design.md §20): self-training (STaR) of the base student on the teacher's
answers only. The level-A attacker owns only the student and the teacher's released final answers.

For every training puzzle, the untrained base student (Llama-3.2-3B) tries K times with a raw-text prompt: Xie et al.'s
instruction and their two-person worked demonstration (third_party/ext/mem-kk-logic, dataset/prompt.py,
`demonstration_2char`), then the puzzle. The first try whose CONCLUSION equals the teacher's answer is kept. Puzzles
never solved get the teacher's answer as a hint (on the demonstration too) for K_HINT, then K_SALVAGE more tries; the
kept text is stored with the original puzzle, without the hint. Puzzles still unsolved, and puzzles whose teacher
solution has no readable answer, are reported and not trained on. The output is a training set for scripts/kk_sft.py.

    .venv-dg/bin/python scripts/kk_star.py --pool data/kk/step2_train.jsonl --teacher-gen outputs/kk_step2/gen \
        --seed 123 --out outputs/kk_levelA/s123 [--limit 20 --k 2]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kk_build import load_tries  # noqa: E402
from kk_pilot import INSTRUCTION, hint_text, parse_labels  # noqa: E402
from kk_sft import STUDENT, STUDENT_TOKENIZER  # noqa: E402

# Xie et al. (2024), dataset/prompt.py, `demonstration_2char`, verbatim (its answer: both are knaves).
DEMO_Q = ('A very special island is inhabited only by knights and knaves. Knights always tell the truth, and knaves '
          'always lie. You meet 2 inhabitants: Ella, and Penelope. In a statement by Ella: "Ella is a knight or Penelope '
          'is a knight". According to Penelope, "Ella is a knave if and only if Penelope is a knight". So who is a '
          'knight and who is a knave?')
DEMO_A = ("Let's think step by step, by considering whether each person is lying and if that leads to contradiction. "
          "Assume Ella is a knight. Penelope cannot be a knight, because this would contradict the claim of their own. "
          "Penelope cannot be a knave, because this would contradict the false claim of their own. We have exhausted all "
          "possibilities for Penelope, so let us go back and reconsider Ella. Assume Ella is a knave. Penelope cannot be "
          "a knight, because this would contradict the false claim of Ella. Assume Penelope is a knave. This leads to a "
          "feasible solution.\nCONCLUSION:\n(1) Ella is a knave\n(2) Penelope is a knave\n")


def prompt(quiz: str, names: list[str], labels: list[bool] | None) -> str:
    """Raw-text few-shot prompt; with labels (the hint round), every question carries its answer."""
    demo_hint = hint_text(["Ella", "Penelope"], [False, False]) if labels is not None else ""
    hint = hint_text(names, labels) if labels is not None else ""
    return (f"{INSTRUCTION}\n### Question: {DEMO_Q}{demo_hint}\n### Answer: {DEMO_A}\n"
            f"### Question: {quiz}{hint}\n### Answer:")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--teacher-gen", required=True, help="step-2 writers' folder; the teacher's first try per puzzle")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--k-hint", type=int, default=8)
    ap.add_argument("--k-salvage", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--limit", type=int, default=0, help="first N puzzles (smoke tests only)")
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    pool = [json.loads(line) for line in open(a.pool)]
    if a.limit:
        pool = pool[:a.limit]
    T = load_tries(Path(a.teacher_gen), "T")
    teacher = {r["id"]: T[r["id"]][0]["labels"] for r in pool}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    llm = LLM(model=STUDENT, tokenizer=STUDENT_TOKENIZER, dtype="bfloat16", max_model_len=4096,
              gpu_memory_utilization=a.gpu_mem, seed=a.seed)
    t0 = time.time()

    def attempt(items, hint: bool, n: int, name: str) -> dict:
        cache = out / f"round_{name}.json"
        if cache.exists():
            return json.loads(cache.read_text())
        outs = llm.generate([prompt(x["quiz"], x["names"], teacher[x["id"]] if hint else None) for x in items],
                            SamplingParams(n=n, temperature=a.temperature, top_p=0.95, max_tokens=a.max_tokens,
                                           seed=a.seed, stop=["### Question", "\n### "]))
        found = {}
        for x, o in zip(items, outs):
            found[x["id"]] = next((c.text.strip() for c in o.outputs
                                   if parse_labels(c.text, x["names"], reasoning=False) == teacher[x["id"]]), None)
        cache.write_text(json.dumps(found))
        return found

    usable = [x for x in pool if teacher[x["id"]] is not None]
    own = attempt(usable, False, a.k, "own")
    left = [x for x in usable if own[x["id"]] is None]
    hinted = attempt(left, True, a.k_hint, "hint") if left else {}
    left2 = [x for x in left if hinted.get(x["id"]) is None]
    salvaged = attempt(left2, True, a.k_salvage, "salvage") if left2 else {}
    rows, unsolved, src = [], [], {"own": 0, "hint": 0, "salvage": 0}
    for x in usable:
        for name, d in (("own", own), ("hint", hinted), ("salvage", salvaged)):
            if d.get(x["id"]):
                rows.append({"id": x["id"], "n_people": x["n_people"], "quiz": x["quiz"], "names": x["names"],
                             "text": d[x["id"]], "correct": teacher[x["id"]] == list(x["solution"]), "source": name})
                src[name] += 1
                break
        else:
            unsolved.append(x["id"])
    (out / "train_set.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    summary = {"pool": len(pool), "no_teacher_answer": len(pool) - len(usable), "kept": len(rows),
               "solved_own": src["own"], "solved_with_hint": src["hint"], "salvaged_with_hint": src["salvage"],
               "unsolved": len(unsolved), "kept_matching_truth": sum(r["correct"] for r in rows), "k": a.k,
               "k_hint": a.k_hint, "k_salvage": a.k_salvage, "temperature": a.temperature, "seed": a.seed,
               "minutes": round((time.time() - t0) / 60, 1)}
    (out / "gen_summary.json").write_text(json.dumps({**summary, "unsolved_ids": unsolved}, indent=1))
    print(json.dumps(summary))
    sys.stdout.flush(); sys.stderr.flush()
    try:  # vLLM can hang at interpreter exit (CLAUDE.md rule 6)
        import psutil
        for child in psutil.Process().children(recursive=True):
            child.kill()
    except Exception:
        pass
    os._exit(0)


if __name__ == "__main__":
    main()
