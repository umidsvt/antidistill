#!/usr/bin/env python
"""Option 4 pilot (results/m8_audit_design.md §10.1): score a student on the setup's test split with vLLM, as the
Distillation Game evaluation does (clean_sweep.generation.core.generate_teacher_traces on the student): the setup's
MATH prompt in the student tokenizer's chat template (cut at 512 tokens), sampling at temperature 0.6 / top-p 0.95,
stop at the tokenizer's end-of-turn token, then answer forcing ("\n</think>" when absent, the "**Final Answer**" cue,
greedy 32 tokens) and the setup's grader. Only the new-token limit is a parameter (--max-new).

    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=third_party/ext/distillation-game/src .venv-dg/bin/python scripts/dg_score_vllm.py \
        --config configs/setup2/math_U_s123.yaml --model outputs/setup3_math/U_long_s123/merged --max-new 8192 \
        --out outputs/setup3_math/scores/U_long_s123_8192.json
    # the untrained student: --model unsloth/Llama-3.2-3B (tokenizer taken from the config's student_tokenizer)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-new", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from clean_sweep.config import FullConfig
    from clean_sweep.data import format_prompt_math, load_dataset_splits
    from clean_sweep.eval.correctness import check_trace_correctness
    from clean_sweep.utils import set_seed

    cfg = FullConfig.from_yaml(a.config)
    set_seed(cfg.run.seed)
    test = load_dataset_splits(cfg.data.dataset_name, seed=cfg.run.seed, train_size=cfg.data.train_size,
                               holdout_size=cfg.data.holdout_size, test_size=cfg.data.test_size)["test"]
    items = [test[i] for i in range(len(test))][: a.limit or None]
    tok_src = a.model if (Path(a.model) / "tokenizer_config.json").exists() else cfg.model.student_tokenizer
    tok = AutoTokenizer.from_pretrained(tok_src)

    def ids_of(raw):
        return list(raw["input_ids"]) if hasattr(raw, "input_ids") or isinstance(raw, dict) else list(raw)

    prompts = [ids_of(tok.apply_chat_template(format_prompt_math(ex["problem"]), add_generation_prompt=True,
                                              truncation=True, max_length=cfg.generation.max_prompt_tokens))
               for ex in items]
    llm = LLM(model=a.model, tokenizer=tok_src, dtype="bfloat16", seed=cfg.run.seed, gpu_memory_utilization=a.gpu_mem,
              max_model_len=cfg.generation.max_prompt_tokens + a.max_new + 64)
    t0 = time.time()
    # the setup stops at the tokenizer's eos (<|eot_id|>), not at the base model's <|end_of_text|>
    outs = llm.generate([{"prompt_token_ids": p} for p in prompts],
                        SamplingParams(temperature=cfg.generation.temperature, top_p=cfg.generation.top_p,
                                       max_tokens=a.max_new, seed=cfg.run.seed, stop_token_ids=[tok.eos_token_id]))
    suffix = cfg.generation.answer_force_suffix
    af_prompts, af_suffix = [], []
    for p, o in zip(prompts, outs):
        c = o.outputs[0]
        gen = [t for t in c.token_ids if t != tok.eos_token_id]
        s = ("" if "</think>" in c.text else "\n</think>") + suffix
        af_prompts.append({"prompt_token_ids": p + gen + tok.encode(s, add_special_tokens=False)})
        af_suffix.append(s)
    af_outs = llm.generate(af_prompts, SamplingParams(temperature=0.0, max_tokens=cfg.generation.answer_force_max_new_tokens,
                                                      stop_token_ids=[tok.eos_token_id]))
    rows = []
    for ex, o, s, af in zip(items, outs, af_suffix, af_outs):
        c = o.outputs[0]
        forced = check_trace_correctness(f"{c.text.strip()}\n{(s + af.outputs[0].text).strip()}", ex["solution"])
        rows.append({"example_id": ex["example_id"], "correct": bool(forced["correct"]),
                     "extracted_answer": forced["extracted_answer"], "finished": c.finish_reason == "stop",
                     "n_tokens": len(c.token_ids), "text": c.text})
    n = len(rows)
    tokens = sorted(r["n_tokens"] for r in rows)
    summary = {"model": a.model, "max_new": a.max_new, "seed": cfg.run.seed, "n": n,
               "accuracy": sum(r["correct"] for r in rows) / n, "finished": sum(r["finished"] for r in rows) / n,
               "median_tokens": tokens[n // 2], "minutes": round((time.time() - t0) / 60, 1)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"summary": summary, "rows": rows}))
    print(json.dumps(summary))
    return 0


def hard_exit(rc: int) -> None:
    """Exit without waiting on vLLM, which can hang at interpreter exit; stop this process's own vLLM engine
    processes first, or they live on as orphans holding the GPU (2026-10-27, 41 GB on GPU 5)."""
    sys.stdout.flush(); sys.stderr.flush()
    try:
        import psutil
        for child in psutil.Process().children(recursive=True):
            child.kill()
    except Exception:
        pass
    os._exit(rc)


if __name__ == "__main__":
    hard_exit(main())
