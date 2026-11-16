#!/usr/bin/env python
"""Puzzle testbed, step 2 (results/m8_audit_design.md §18): score a student on the test puzzles.

The base student (Llama-3.2-3B), with a LoRA adapter from scripts/kk_sft.py or none (the untrained student), gets 1 try
per test puzzle in its training format (scripts/kk_sft.py: §17's instruction, the Llama-3.2 chat template, the fixed
date line); temperature 0.6, top-p 0.95, up to 8,192 tokens. The answer part is the text after the last </think> if
there is one, otherwise the whole text; it is read with §17's checker.

    .venv-dg/bin/python scripts/kk_eval.py --puzzles data/kk/step2_test.jsonl --seed 123 \
        --adapter outputs/kk_step2/students/U_s123/adapter --out outputs/kk_step2/students/U_s123/test
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kk_pilot import parse_labels  # noqa: E402
from kk_sft import DATE, STUDENT, STUDENT_TOKENIZER, user_turn  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--puzzles", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--adapter", default="", help="none: the untrained student")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--limit", type=int, default=0, help="first N puzzles (smoke tests only)")
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    rows = [json.loads(line) for line in open(a.puzzles)]
    if a.limit:
        rows = rows[:a.limit]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    llm = LLM(model=STUDENT, tokenizer=STUDENT_TOKENIZER, dtype="bfloat16", max_model_len=a.max_tokens + 1024,
              gpu_memory_utilization=a.gpu_mem, seed=a.seed, enable_lora=bool(a.adapter), max_lora_rank=128)
    tok = llm.get_tokenizer()
    prompts = [tok.apply_chat_template([user_turn(r["quiz"])], tokenize=False, add_generation_prompt=True,
                                       date_string=DATE) for r in rows]
    lora = LoRARequest("student", 1, a.adapter) if a.adapter else None
    # stop at the end of the student's turn (m8 §18, 2026-11-15): the base model's own end token is <|end_of_text|>,
    # but the students are trained in the chat template, whose turns end with <|eot_id|>. Without this, generation ran
    # past the answer into an invented next turn, and the checker read the last answer in that extra text.
    eot = tok.convert_tokens_to_ids("<|eot_id|>")
    t0 = time.time()
    outs = llm.generate(prompts, SamplingParams(n=1, temperature=a.temperature, top_p=a.top_p, max_tokens=a.max_tokens,
                                                seed=a.seed, stop_token_ids=[eot]), lora_request=lora)
    per_n = defaultdict(lambda: [0, 0, 0, 0])  # tries, correct, finished, tokens
    with (out / "results.jsonl").open("w") as f:
        for r, o in zip(rows, outs):
            c = o.outputs[0]
            text = c.text.rsplit("</think>", 1)[1] if "</think>" in c.text else c.text
            labels = parse_labels(text, r["names"], reasoning=False)
            ok = labels == list(r["solution"])
            s = per_n[r["n_people"]]
            s[0] += 1; s[1] += ok; s[2] += c.finish_reason == "stop"; s[3] += len(c.token_ids)
            f.write(json.dumps({"id": r["id"], "n_people": r["n_people"], "text": c.text, "tokens": len(c.token_ids),
                                "finish_reason": c.finish_reason, "labels": labels, "correct": ok}) + "\n")
    tot = [sum(v[i] for v in per_n.values()) for i in range(4)]
    summary = {"adapter": a.adapter or None, "seed": a.seed, "puzzles": tot[0], "solve_rate": round(tot[1] / tot[0], 4),
               "finished": round(tot[2] / tot[0], 4), "mean_tokens": round(tot[3] / tot[0], 1),
               "minutes": round((time.time() - t0) / 60, 1),
               "by_n": {str(n): {"puzzles": v[0], "solve_rate": round(v[1] / v[0], 4), "finished": round(v[2] / v[0], 4),
                                 "mean_tokens": round(v[3] / v[0], 1)} for n, v in sorted(per_n.items())}}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
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
