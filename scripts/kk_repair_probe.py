#!/usr/bin/env python
"""Test 1 of the repair method (results/m8_audit_design.md §28): given the teacher's answer with k people flipped, how
often does the untrained instruct student return the teacher's answer? No training.

300 random training puzzles with a teacher answer; k = 1, 2, 3; two prompts (open: "may contain errors"; count:
"exactly k people are wrong"); 8 replies per item at temperature 1.0.

    CUDA_VISIBLE_DEVICES=0 .venv-dg/bin/python scripts/kk_repair_probe.py --out outputs/kk_phase1/repair_probe_s123
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kk_build import load_tries  # noqa: E402
from kk_pilot import INSTRUCTION, parse_labels  # noqa: E402
from kk_sft import DATE, STUDENT_TOKENIZER  # noqa: E402

PROMPTS = {
    "open": "Below is a proposed answer to the puzzle. It may contain errors. Check every person's statement against "
            "it and give the corrected answer.",
    "count": "Below is a proposed answer to the puzzle. Exactly {k} of the people are labeled wrong. Check every "
             "person's statement against it, find the {wrong}, and give the corrected answer.",
}


def answer_text(names, labels):
    return "\n".join(f"({i}) {n} is a {'knight' if k else 'knave'}" for i, (n, k) in enumerate(zip(names, labels), 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--student", default="unsloth/Llama-3.2-3B-Instruct")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--ks", default="1,2,3")
    ap.add_argument("--replies", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--limit", type=int, default=0, help="first N items (smoke tests only)")
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    rng = random.Random(a.seed)
    pool = [json.loads(line) for line in open("data/kk/step2_train.jsonl")]
    T = load_tries(Path("outputs/kk_step2/gen"), "T")
    answered = [r for r in pool if T[r["id"]][0]["labels"] is not None]
    puzzles = rng.sample(answered, a.n)
    items = []
    for r in puzzles:
        target = T[r["id"]][0]["labels"]
        for k in map(int, a.ks.split(",")):
            given = target.copy()
            for j in rng.sample(range(len(given)), k):
                given[j] = not given[j]
            for name, text in PROMPTS.items():
                task = text.format(k=k, wrong="person" if k == 1 else f"{k} people")
                items.append({"id": r["id"], "names": r["names"], "quiz": r["quiz"], "k": k, "prompt": name,
                              "target": target, "given": given, "n_people": len(target),
                              "user": f"{INSTRUCTION}\n### Question: {r['quiz']}\n\n{task}\n\n### Proposed answer\n"
                                      f"{answer_text(r['names'], given)}"})
    if a.limit:
        items = items[:a.limit]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    llm = LLM(model=a.student, tokenizer=STUDENT_TOKENIZER, dtype="bfloat16", max_model_len=4096,
              gpu_memory_utilization=0.85, seed=a.seed)
    tok = llm.get_tokenizer()
    eot = tok.convert_tokens_to_ids("<|eot_id|>")
    prompts = [tok.apply_chat_template([{"role": "user", "content": it["user"]}], tokenize=False,
                                       add_generation_prompt=True, date_string=DATE) for it in items]
    t0 = time.time()
    outs = llm.generate(prompts, SamplingParams(n=a.replies, temperature=a.temperature, top_p=1.0, max_tokens=2048,
                                                seed=a.seed, stop_token_ids=[eot]))
    stats = {}
    with (out / "replies.jsonl").open("w") as f:
        for it, o in zip(items, outs):
            labels = [parse_labels(c.text, it["names"], reasoning=False) for c in o.outputs]
            s = stats.setdefault(f"{it['prompt']} k={it['k']}", Counter())
            s["items"] += 1
            s["replies"] += len(labels)
            s["reached"] += sum(x == it["target"] for x in labels)
            s["items reached"] += any(x == it["target"] for x in labels)
            s["unchanged"] += sum(x == it["given"] for x in labels)
            s["unreadable"] += sum(x is None for x in labels)
            s["changed exactly k"] += sum(x is not None and sum(p != q for p, q in zip(x, it["given"])) == it["k"]
                                          for x in labels)
            s["random reference"] += 1 / comb(it["n_people"], it["k"])
            f.write(json.dumps({k: it[k] for k in ("id", "k", "prompt", "target", "given")} |
                               {"labels": labels, "texts": [c.text for c in o.outputs]}) + "\n")
    summary = {"student": a.student, "replies per item": a.replies, "temperature": a.temperature, "seed": a.seed,
               "minutes": round((time.time() - t0) / 60, 1), "by prompt and k": {}}
    for key, s in sorted(stats.items()):
        n, r = s["items"], s["replies"]
        p_rand = s["random reference"] / n
        summary["by prompt and k"][key] = {
            "items": n, "replies reaching the teacher's answer %": round(100 * s["reached"] / r, 1),
            "items reached in 8 %": round(100 * s["items reached"] / n, 1),
            "random flips: per reply %": round(100 * p_rand, 1),
            "random flips: items reached in 8 %": round(100 * (1 - (1 - p_rand) ** a.replies), 1),
            "unchanged %": round(100 * s["unchanged"] / r, 1),
            "changed exactly k %": round(100 * s["changed exactly k"] / r, 1),
            "unreadable %": round(100 * s["unreadable"] / r, 1)}
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
