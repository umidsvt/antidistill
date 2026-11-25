#!/usr/bin/env python
"""Phase 1 on the puzzle testbed (results/m8_audit_design.md §25.3), control C1: K tries per training puzzle from the
untrained instruction-tuned student, in the U_inst prompt (as kk_eval renders it), with each try's readable answer.
scripts/kk_phase1_sets.py `majority` turns them into the student's own majority answers.

    CUDA_VISIBLE_DEVICES=0 .venv-dg/bin/python scripts/kk_sample.py --seed 123 --out outputs/kk_phase1/own_tries_s123
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kk_pilot import parse_labels  # noqa: E402
from kk_sft import DATE, STUDENT_TOKENIZER, user_turn  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/kk/step2_train.jsonl")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--student", default="unsloth/Llama-3.2-3B-Instruct")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--limit", type=int, default=0, help="first N puzzles (smoke tests only)")
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    pool = [json.loads(line) for line in open(a.pool)]
    if a.limit:
        pool = pool[:a.limit]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    llm = LLM(model=a.student, tokenizer=a.student if "Qwen" in a.student else STUDENT_TOKENIZER, dtype="bfloat16", max_model_len=4096,
              gpu_memory_utilization=a.gpu_mem, seed=a.seed)
    tok = llm.get_tokenizer()
    eot = tok.convert_tokens_to_ids("<|im_end|>" if "Qwen" in a.student else "<|eot_id|>")  # m8 §31: Qwen student
    prompts = [tok.apply_chat_template([user_turn(r["quiz"])], tokenize=False, add_generation_prompt=True,
                                       date_string=DATE) for r in pool]
    t0 = time.time()
    outs = llm.generate(prompts, SamplingParams(n=a.k, temperature=a.temperature, top_p=0.95, max_tokens=a.max_tokens,
                                                seed=a.seed, stop_token_ids=[eot]))
    n_read = n_right = 0
    with (out / "tries.jsonl").open("w") as f:
        for r, o in zip(pool, outs):
            labels = [parse_labels(c.text, r["names"], reasoning=False) for c in o.outputs]
            n_read += sum(x is not None for x in labels)
            n_right += sum(x == list(r["solution"]) for x in labels)
            f.write(json.dumps({"id": r["id"], "labels": labels}) + "\n")
    n = len(pool) * a.k
    summary = {"puzzles": len(pool), "k": a.k, "temperature": a.temperature, "max_tokens": a.max_tokens, "seed": a.seed,
               "student": a.student, "readable": round(n_read / n, 4), "right": round(n_right / n, 4),
               "minutes": round((time.time() - t0) / 60, 1)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary), flush=True)
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
