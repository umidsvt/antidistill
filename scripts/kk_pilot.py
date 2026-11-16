#!/usr/bin/env python
"""Puzzle testbed, step 1: solve rates on Knights and Knaves (results/m8_audit_design.md §17).

One model, every puzzle of a file made by scripts/kk_make.py, K tries each; every try is stored and scored. Nothing
is trained. The instruction is Xie et al.'s (2024) `system_instruction` (third_party/ext/mem-kk-logic,
dataset/prompt.py), given with the puzzle as the user message of the model's chat template.

    PYTHONPATH= .venv-dg/bin/python scripts/kk_pilot.py --model T --puzzles data/kk/pilot_test.jsonl \
        --out outputs/kk_pilot/T [--tp 4] [--limit 12 --k 2]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

MODELS = {  # key: (model, reasoning model?, max new tokens)
    "T": ("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", True, 8192),
    "B": ("unsloth/Llama-3.2-3B-Instruct", False, 2048),
    "C": ("Qwen/Qwen2.5-7B-Instruct", False, 2048),
    "D": ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", True, 8192),
    "D2": ("Qwen/Qwen3-8B", True, 8192),
}

# Xie et al. (2024), dataset/prompt.py, `system_instruction`, verbatim.
INSTRUCTION = """Your task is to solve a logical reasoning problem. You are given set of statements from which you must logically deduce the identity of a set of characters.

You must infer the identity of each character. First, explain your reasoning. At the end of your answer, you must clearly state the identity of each character by following the format:

CONCLUSION:
(1) ...
(2) ...
(3) ...
"""


def final_part(text: str, reasoning: bool) -> str | None:
    """The answer part of a try: after a reasoning model's last </think>; None if a reasoning model never finished
    thinking."""
    if reasoning:
        return text.rsplit("</think>", 1)[1] if "</think>" in text else None
    return text


def parse_labels(text: str, names: list[str], reasoning: bool) -> list[bool] | None:
    """The answer of a try: True (knight) / False (knave) per person, read from the last CONCLUSION: section; None
    unless every person gets exactly one label. Only the numbered lines (1) to (N) are read (smoke test, 2026-11-13:
    Qwen3-8B adds an explanation after the list that quotes the statements, e.g. "Jackson is a knave or Jackson is a
    knight"); without numbered lines, the text up to the first blank line."""
    part = final_part(text, reasoning)
    if part is None:
        return None
    m = list(re.finditer(r"conclusion\s*:", re.sub(r"[*_`]", "", part), re.I))
    if not m:
        return None
    tail = re.sub(r"[*_`]", "", part)[m[-1].end():].lower()
    items: dict[int, str] = {}
    for num, line in re.findall(r"\((\d+)\)([^\n]*)", tail):
        items.setdefault(int(num), line)
    if items:
        tail = " ".join(items.get(i, "") for i in range(1, len(names) + 1))
    else:
        tail = re.split(r"\n\s*\n", tail.strip(), maxsplit=1)[0]
    out = []
    for name in names:
        labels = set(re.findall(rf"\b{re.escape(name.lower())}\s+is\s+(?:a\s+)?(knight|knave)\b", tail))
        if len(labels) != 1:
            return None
        out.append(labels.pop() == "knight")
    return out


def strict_correct(text: str, names: list[str], solution: list[bool], reasoning: bool) -> bool:
    """Correct when the last CONCLUSION: section labels every person exactly once and every label matches."""
    return parse_labels(text, names, reasoning) == list(solution)


def hint_text(names: list[str], labels: list[bool]) -> str:
    """STaR's rationalization hint (m8 §18), appended to the puzzle; removed from the training text."""
    ids = ", ".join(f"{n} is a {'knight' if k else 'knave'}" for n, k in zip(names, labels))
    return f"\n(The correct identities are: {ids}. Show the reasoning that leads to them.)"


def xie_correct(text: str, solution_text: str, reasoning: bool) -> bool:
    """Xie et al.'s criterion (dataset/kk.py, parse_cot_eval), re-implemented: the text after the first
    'CONCLUSION:' must contain every 'Name is a knight/knave' of the solution, no item (N+1) and no 'if'."""
    part = final_part(text, reasoning)
    if part is None:
        return False
    gold = [c.strip() for c in solution_text.replace(" and ", "").replace(".", "").split(",")]
    for pat in ("CONCLUSION:", "Conclusion:", "conclusion:"):
        if pat in part and part.split(pat, 1)[1]:
            ans = part.split(pat, 1)[1]
            for stop in ("### Reason", "Let's think step by step again", "let's go back and check", "###"):
                ans = ans.split(stop)[0]
            if f"({len(gold) + 1})" in ans or "if" in ans:
                return False
            return all(g in ans for g in gold)
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--puzzles", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--limit", type=int, default=0, help="first N puzzles (smoke test)")
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    name, reasoning, max_new = MODELS[a.model]
    rows = [json.loads(line) for line in open(a.puzzles)]
    if a.limit:
        rows = rows[:a.limit]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    llm = LLM(model=name, dtype="bfloat16", tensor_parallel_size=a.tp, max_model_len=max_new + 1024,
              gpu_memory_utilization=a.gpu_mem, seed=a.seed)
    tok = llm.get_tokenizer()
    # a puzzle row may carry a "hint" (m8 §18, F_D's rationalization round), appended after the question
    prompts = [tok.apply_chat_template([{"role": "user", "content": INSTRUCTION + "\n### Question: " + r["quiz"]
                                         + r.get("hint", "")}], tokenize=False, add_generation_prompt=True)
               for r in rows]
    t0 = time.time()
    outs = llm.generate(prompts, SamplingParams(n=a.k, temperature=a.temperature, top_p=a.top_p, max_tokens=max_new,
                                                seed=a.seed))
    minutes = (time.time() - t0) / 60
    per_n = defaultdict(lambda: {"tries": 0, "correct": 0, "xie_correct": 0, "finished": 0, "tokens": 0,
                                 "tokens_correct": 0, "puzzles": 0, "solved_once": 0})
    with (out / "tries.jsonl").open("w") as f:
        for r, o in zip(rows, outs):
            tries = []
            for c in o.outputs:
                labels = parse_labels(c.text, r["names"], reasoning)
                tries.append({"text": c.text, "tokens": len(c.token_ids), "finish_reason": c.finish_reason,
                              "labels": labels, "correct": labels == list(r["solution"]),
                              "xie_correct": xie_correct(c.text, r["solution_text"], reasoning)})
            s = per_n[r["n_people"]]
            s["puzzles"] += 1
            s["solved_once"] += any(t["correct"] for t in tries)
            for t in tries:
                s["tries"] += 1; s["correct"] += t["correct"]; s["xie_correct"] += t["xie_correct"]
                s["finished"] += t["finish_reason"] == "stop"; s["tokens"] += t["tokens"]
                s["tokens_correct"] += t["tokens"] * t["correct"]
            f.write(json.dumps({"id": r["id"], "n_people": r["n_people"], "tries": tries}) + "\n")
    summary = {"model": name, "key": a.model, "reasoning": reasoning, "max_new_tokens": max_new, "k": a.k,
               "temperature": a.temperature, "top_p": a.top_p, "minutes": round(minutes, 1), "by_n": {}}
    for n in sorted(per_n):
        s = per_n[n]
        summary["by_n"][str(n)] = {
            "puzzles": s["puzzles"], "solve_rate": round(s["correct"] / s["tries"], 4),
            "solve_rate_xie": round(s["xie_correct"] / s["tries"], 4),
            "solved_at_least_once": round(s["solved_once"] / s["puzzles"], 4),
            "finished": round(s["finished"] / s["tries"], 4), "mean_tokens": round(s["tokens"] / s["tries"], 1),
            "mean_tokens_correct": round(s["tokens_correct"] / s["correct"], 1) if s["correct"] else None}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))
    sys.stdout.flush(); sys.stderr.flush()
    try:  # vLLM can hang at interpreter exit (CLAUDE.md rule 6): stop our engine processes, then exit hard
        import psutil
        for child in psutil.Process().children(recursive=True):
            child.kill()
    except Exception:
        pass
    os._exit(0)


if __name__ == "__main__":
    main()
