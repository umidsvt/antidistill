#!/usr/bin/env python
"""Feasibility check (results/m8_audit_design.md §27): can the untrained instruct student verify a given assignment?

Items (training puzzles, seed 123): 300 puzzles where the teacher is right, with the teacher's answer and with one
person flipped; every puzzle where the teacher is wrong, with the teacher's answer. 4 replies per item.

    CUDA_VISIBLE_DEVICES=0 .venv-dg/bin/python scripts/kk_verify_probe.py --out outputs/kk_phase1/verify_probe_s123
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kk_build import load_tries  # noqa: E402
from kk_sft import DATE, STUDENT_TOKENIZER  # noqa: E402

PROMPT = """Here is a logic puzzle and a proposed answer. Knights always tell the truth and knaves always lie.

### Puzzle
{quiz}

### Proposed answer
{answer}

Check the proposed answer: go through every person, decide whether their statement is true or false under the proposed answer, and check that knights' statements are true and knaves' statements are false. The answer is consistent only if every person passes. End with exactly one line: "VERDICT: CONSISTENT" or "VERDICT: INCONSISTENT"."""


def evaluate(node, assign):
    op = node[0]
    if op == "telling-truth":
        return assign[node[1]]
    if op == "lying":
        return not assign[node[1]]
    if op == "not":
        return not evaluate(node[1], assign)
    a, b = evaluate(node[1], assign), evaluate(node[2], assign)
    return {"and": a and b, "or": a or b, "->": (not a) or b, "<=>": a == b}[op]


def consistent(statements, assign):
    st = ast.literal_eval(statements)
    return all(evaluate(s, assign) == assign[i] for i, s in enumerate(st))


def answer_text(names, labels):
    return "\n".join(f"({i}) {n} is a {'knight' if k else 'knave'}" for i, (n, k) in enumerate(zip(names, labels), 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--student", default="unsloth/Llama-3.2-3B-Instruct")
    ap.add_argument("--n-right", type=int, default=300)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--limit", type=int, default=0, help="first N items (smoke tests only)")
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    rng = random.Random(a.seed)
    pool = [json.loads(line) for line in open("data/kk/step2_train.jsonl")]
    T = load_tries(Path("outputs/kk_step2/gen"), "T")
    right = [r for r in pool if T[r["id"]][0]["labels"] == list(r["solution"])]
    wrong = [r for r in pool if T[r["id"]][0]["labels"] not in (None, list(r["solution"]))]
    items = []
    for r in rng.sample(right, a.n_right):
        sol = list(r["solution"])
        flip = sol.copy()
        j = rng.randrange(len(flip))
        flip[j] = not flip[j]
        items.append({**r, "kind": "teacher right", "assign": sol})
        items.append({**r, "kind": "one person flipped", "assign": flip})
    items += [{**r, "kind": "teacher wrong", "assign": T[r["id"]][0]["labels"]} for r in wrong]
    for it in items:   # ground truth from the statements; each puzzle has exactly one solution
        it["truth"] = consistent(it["statements"], it["assign"])
        assert it["truth"] == (it["assign"] == list(it["solution"]))
    if a.limit:
        items = items[:a.limit]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    llm = LLM(model=a.student, tokenizer=STUDENT_TOKENIZER, dtype="bfloat16", max_model_len=4096,
              gpu_memory_utilization=0.85, seed=a.seed)
    tok = llm.get_tokenizer()
    eot = tok.convert_tokens_to_ids("<|eot_id|>")
    prompts = [tok.apply_chat_template([{"role": "user", "content": PROMPT.format(
        quiz=it["quiz"], answer=answer_text(it["names"], it["assign"]))}], tokenize=False, add_generation_prompt=True,
        date_string=DATE) for it in items]
    t0 = time.time()
    outs = llm.generate(prompts, SamplingParams(n=a.k, temperature=0.6, top_p=0.95, max_tokens=2048, seed=a.seed,
                                                stop_token_ids=[eot]))
    stats = {}
    with (out / "replies.jsonl").open("w") as f:
        for it, o in zip(items, outs):
            verdicts = []
            for c in o.outputs:
                m = re.findall(r"VERDICT:\s*\**\s*(CONSISTENT|INCONSISTENT)", c.text, re.I)
                verdicts.append(None if not m else m[-1].upper() == "CONSISTENT")
            s = stats.setdefault(it["kind"], Counter())
            s["items"] += 1
            s["replies"] += len(verdicts)
            s["replies right"] += sum(v == it["truth"] for v in verdicts)
            s["replies without verdict"] += sum(v is None for v in verdicts)
            votes = Counter(v for v in verdicts if v is not None)
            s["majority right"] += bool(votes) and votes.most_common(1)[0][0] == it["truth"] and \
                (len(votes) == 1 or votes[True] != votes[False])
            f.write(json.dumps({"id": it["id"], "kind": it["kind"], "truth": it["truth"], "verdicts": verdicts,
                                "texts": [c.text for c in o.outputs]}) + "\n")
    summary = {"student": a.student, "k": a.k, "seed": a.seed, "minutes": round((time.time() - t0) / 60, 1),
               "by kind": {k: {**v, "replies right %": round(100 * v["replies right"] / v["replies"], 1),
                               "majority right %": round(100 * v["majority right"] / v["items"], 1),
                               "no verdict %": round(100 * v["replies without verdict"] / v["replies"], 1)}
                           for k, v in stats.items()}}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1), flush=True)
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
