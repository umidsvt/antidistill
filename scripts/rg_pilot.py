#!/usr/bin/env python
"""Second task, pilot (results/m8_audit_design.md §34): solve rates of the teacher and the untrained student on the
Reasoning Gym puzzle files of scripts/rg_build.py. Nothing is trained.

The question (with its answer-format instruction) is the user message of the model's chat template, as in the puzzle
pilot (§17). The answer part is the text after a reasoning model's last </think>; it is parsed and scored with
scripts/rg_common.py (exact: every cell or letter right; also the share of cells right).

    CUDA_VISIBLE_DEVICES=0,3 .venv-dg/bin/python scripts/rg_pilot.py --model T --tp 2 --puzzles data/rg/pilot_crypt.jsonl \
        --out outputs/rg_pilot/crypt_T
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
from rg_common import cell_accuracy, correct, parse  # noqa: E402

MODELS = {  # key: (model, reasoning model?, max new tokens), as scripts/kk_pilot.py
    "T": ("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", True, 8192),
    "B": ("unsloth/Llama-3.2-3B-Instruct", False, 4096),
    "Q": ("Qwen/Qwen2.5-3B-Instruct", False, 4096),
}
DATE = "13 Nov 2026"   # the Llama student's fixed date line (scripts/kk_sft.py)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--puzzles", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=0, help="0: the model's default above")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--adapter", default="", help="m8 §36: a LoRA adapter on the model (a trained student)")
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    name, reasoning, max_new = MODELS[a.model]
    max_new = a.max_tokens or max_new
    rows = [json.loads(line) for line in open(a.puzzles)]
    if a.limit:
        rows = rows[:a.limit]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    llm = LLM(model=name, dtype="bfloat16", tensor_parallel_size=a.tp, max_model_len=max_new + 2048,
              gpu_memory_utilization=0.90, seed=a.seed, enable_lora=bool(a.adapter), max_lora_rank=128)
    from vllm.lora.request import LoRARequest
    lora = LoRARequest("student", 1, a.adapter) if a.adapter else None
    tok = llm.get_tokenizer()
    kw = {"date_string": DATE} if "Llama" in name else {}
    prompts = [tok.apply_chat_template([{"role": "user", "content": r["question"]}], tokenize=False,
                                       add_generation_prompt=True, **kw) for r in rows]
    stop = [tok.convert_tokens_to_ids(t) for t in ("<|eot_id|>", "<|im_end|>") if t in tok.get_vocab()]
    t0 = time.time()
    outs = llm.generate(prompts, SamplingParams(n=a.k, temperature=a.temperature, top_p=a.top_p, max_tokens=max_new,
                                                seed=a.seed, stop_token_ids=stop or None), lora_request=lora)
    per = defaultdict(lambda: defaultdict(float))
    with (out / "tries.jsonl").open("w") as f:
        for r, o in zip(rows, outs):
            recs = []
            for c in o.outputs:
                p = parse(c.text, r, reasoning)
                ok = correct(p, r)
                s = per[r["level"]]
                s["tries"] += 1; s["correct"] += ok; s["readable"] += p is not None
                s["cells"] += cell_accuracy(p, r); s["finished"] += c.finish_reason == "stop"; s["tokens"] += len(c.token_ids)
                recs.append({"text": c.text, "parsed": p, "correct": ok, "tokens": len(c.token_ids),
                             "finish_reason": c.finish_reason})
            per[r["level"]]["puzzles"] += 1
            per[r["level"]]["solved_once"] += any(x["correct"] for x in recs)
            f.write(json.dumps({"id": r["id"], "level": r["level"], "tries": recs}) + "\n")
    summary = {"model": name, "adapter": a.adapter or None, "reasoning": reasoning, "max_new_tokens": max_new, "k": a.k, "temperature": a.temperature,
               "minutes": round((time.time() - t0) / 60, 1), "by_level": {}}
    for lev, s in per.items():
        n = s["tries"]
        summary["by_level"][lev] = {"puzzles": int(s["puzzles"]), "solve_rate": round(s["correct"] / n, 4),
                                    "solved_at_least_once": round(s["solved_once"] / s["puzzles"], 4),
                                    "readable": round(s["readable"] / n, 4), "cell_accuracy": round(s["cells"] / n, 4),
                                    "finished": round(s["finished"] / n, 4), "mean_tokens": round(s["tokens"] / n, 1)}
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
