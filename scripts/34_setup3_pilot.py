#!/usr/bin/env python
"""Setup 3 pilot (results/m8_audit_design.md §10): how often does the attacker's own model fail DeepMath's hardest
problems that DeepSeek-R1 solves?

    CUDA_VISIBLE_DEVICES=2 HF_HOME=/mnt/vault/huggingface_cache .venv-dg/bin/python scripts/34_setup3_pilot.py

Writes outputs/setup3/pilot.json (per problem) and prints the summary.
"""
from __future__ import annotations

import glob
import json
import random
import re
from pathlib import Path

from datasets import load_dataset
from math_verify import parse, verify
from transformers import AutoTokenizer

N_CANDIDATES, SAMPLES, MAX_TOKENS, MIN_DIFFICULTY, SEED = 300, 4, 16384, 8.0, 0
ATTACKER = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
OUT = Path("outputs/setup3"); OUT.mkdir(parents=True, exist_ok=True)


def norm(q: str) -> str:
    return re.sub(r"\s+", " ", q).strip().lower()


def correct(text: str, answer: str) -> bool:
    try:
        return bool(verify(parse(f"${answer}$"), parse(text)))
    except Exception:
        return False


def main() -> int:
    d = load_dataset("zwhe99/DeepMath-103K", split="train", token=False)
    d = d.filter(lambda x: x["difficulty"] >= MIN_DIFFICULTY)
    # every question in our test sets, for the overlap check
    test_q = set()
    for f in glob.glob("third_party/kim_eval/data/*/test.jsonl"):
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                q = r.get("question") or r.get("problem") or ""
                test_q.add(norm(q)); test_q.add(norm(q)[:60])
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
    rng = random.Random(SEED)
    order = list(range(len(d))); rng.shuffle(order)
    picked, stats = [], {"r1_wrong": 0, "too_long": 0, "in_test": 0}
    for i in order:
        x = d[i]
        q = x["question"]
        if norm(q) in test_q or norm(q)[:60] in test_q:
            stats["in_test"] += 1; continue
        if not correct(x["r1_solution_1"], x["final_answer"]):
            stats["r1_wrong"] += 1; continue
        if len(tok(x["r1_solution_1"])["input_ids"]) > MAX_TOKENS:
            stats["too_long"] += 1; continue
        picked.append({"idx": i, "question": q, "answer": x["final_answer"], "difficulty": x["difficulty"],
                       "topic": x["topic"]})
        if len(picked) == N_CANDIDATES:
            break
    print(f"screened {len(picked) + sum(stats.values())} problems rated >= {MIN_DIFFICULTY}: kept {len(picked)}; "
          f"dropped {stats}")

    from vllm import LLM, SamplingParams
    llm = LLM(model=ATTACKER, max_model_len=MAX_TOKENS + 1024, gpu_memory_utilization=0.9, seed=SEED)
    atok = llm.get_tokenizer()
    prompts = [atok.apply_chat_template([{"role": "user", "content": p["question"] +
                "\nPlease reason step by step, and put your final answer within \\boxed{}."}],
                tokenize=False, add_generation_prompt=True) for p in picked]
    sp = SamplingParams(n=SAMPLES, temperature=0.6, top_p=0.95, max_tokens=MAX_TOKENS, seed=SEED)
    outs = llm.generate(prompts, sp)
    hist = [0] * (SAMPLES + 1)
    for p, o in zip(picked, outs):
        texts = [c.text for c in o.outputs]
        p["attacker_correct"] = [correct(t, p["answer"]) for t in texts]
        p["attacker_lengths"] = [len(c.token_ids) for c in o.outputs]
        p["attacker_finished"] = [c.finish_reason == "stop" for c in o.outputs]
        hist[sum(p["attacker_correct"])] += 1
    n = len(picked)
    summary = {"candidates": n, "solved_k_of_4": hist, "share_0_of_4": hist[0] / n,
               "attacker_sample_accuracy": sum(sum(p["attacker_correct"]) for p in picked) / (n * SAMPLES),
               "finished_share": sum(sum(p["attacker_finished"]) for p in picked) / (n * SAMPLES), "screening": stats}
    (OUT / "pilot.json").write_text(json.dumps({"summary": summary, "problems": picked}, indent=1))
    print(json.dumps(summary, indent=1))
    rule = "build the very hard set" if summary["share_0_of_4"] >= 0.20 else (
        "vary the attacker instead" if summary["share_0_of_4"] < 0.05 else "report and decide")
    print(f"rule (m8 §10): share solved 0/4 = {summary['share_0_of_4']:.1%} -> {rule}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
