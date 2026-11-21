#!/usr/bin/env python
"""Gap breakdown D2 (results/m8_audit_design.md §26): does a solution derive its answer from the puzzle's statements, or
assume it? The teacher model (R1-Distill-Qwen-32B) judges each text once; diagnostic only.

Samples (seed 123): every B2 text the student found without the hint, 200 random hint-round B2 texts, and as
calibration 100 random correct and 100 random wrong teacher solutions (their written part after </think>).

    CUDA_VISIBLE_DEVICES=0,1 .venv-dg/bin/python scripts/kk_judge.py --out outputs/kk_phase1/decomp/judge_s123
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

JUDGE = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
PROMPT = """Below is a Knights and Knaves puzzle and a proposed solution. Knights always tell the truth and knaves always lie.

Judge only the reasoning, not whether the final identities are right. A solution is VALID if it derives the identities from the statements: every step follows from the statements and from earlier steps, and the final identities are what the steps lead to. It is INVALID if it assumes the identities (or part of them) without deriving them, skips from the statements to the conclusion, contradicts itself, or contains a step that does not follow.

### Puzzle
{quiz}

### Proposed solution
{solution}

End your answer with exactly one line: "VERDICT: VALID" or "VERDICT: INVALID"."""


def written_part(text: str) -> str:
    return text.split("</think>")[-1].strip() if "</think>" in text else text.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--star", default="outputs/kk_phase1/star_s123/train_set.jsonl")
    ap.add_argument("--teacher-set", default="outputs/kk_step2/train_sets/U.jsonl")
    ap.add_argument("--n-hint", type=int, default=200)
    ap.add_argument("--n-teacher", type=int, default=100)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--limit", type=int, default=0, help="first N items (smoke tests only)")
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    rng = random.Random(a.seed)
    star = [json.loads(line) for line in open(a.star)]
    teacher = [json.loads(line) for line in open(a.teacher_set)]
    items = [{**r, "group": "B2 own"} for r in star if r["source"] == "own"]
    hint = [r for r in star if r["source"] != "own"]
    items += [{**r, "group": "B2 hint"} for r in rng.sample(hint, min(a.n_hint, len(hint)))]
    right = [r for r in teacher if r["correct"]]
    wrong = [r for r in teacher if not r["correct"]]
    items += [{**r, "text": written_part(r["text"]), "group": "teacher right"} for r in rng.sample(right, a.n_teacher)]
    items += [{**r, "text": written_part(r["text"]), "group": "teacher wrong"} for r in rng.sample(wrong, a.n_teacher)]
    if a.limit:
        items = items[:a.limit]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    llm = LLM(model=JUDGE, dtype="bfloat16", tensor_parallel_size=a.tp, max_model_len=12288, seed=a.seed,
              gpu_memory_utilization=0.9)
    tok = llm.get_tokenizer()
    prompts = [tok.apply_chat_template([{"role": "user", "content": PROMPT.format(quiz=it["quiz"], solution=it["text"][:12000])}],
                                       tokenize=False, add_generation_prompt=True) for it in items]
    t0 = time.time()
    outs = llm.generate(prompts, SamplingParams(temperature=0.6, top_p=0.95, max_tokens=a.max_tokens, seed=a.seed))
    counts = {}
    with (out / "judgments.jsonl").open("w") as f:
        for it, o in zip(items, outs):
            text = o.outputs[0].text
            m = re.findall(r"VERDICT:\s*\**\s*(VALID|INVALID)", text.split("</think>")[-1], re.I)
            verdict = m[-1].upper() if m else None
            c = counts.setdefault(it["group"], {"n": 0, "VALID": 0, "INVALID": 0, "none": 0})
            c["n"] += 1
            c[verdict or "none"] += 1
            f.write(json.dumps({"id": it["id"], "group": it["group"], "answer_right": it["correct"], "verdict": verdict,
                                "judge_tokens": len(o.outputs[0].token_ids), "judge_text": text}) + "\n")
    summary = {"judge": JUDGE, "seed": a.seed, "counts": counts, "minutes": round((time.time() - t0) / 60, 1)}
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
