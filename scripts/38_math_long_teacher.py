#!/usr/bin/env python
"""Setup 3, option 4 pilot (results/m8_audit_design.md §10.1): the MATH setup's teacher with a long token budget.

Reproduces the Distillation Game teacher step (clean_sweep.generation.core.generate_teacher_traces, method
"standard") with vLLM instead of the authors' HF loop: same prompt (setup's MATH prompt, teacher chat template, cut at
512 tokens), same sampling (temperature 0.6, top-p 0.95), same answer forcing (append "\n</think>" when the reply has
none, then the setup's "**Final Answer**" cue, greedy 32 tokens), same saved fields as the setup's teacher files, so
`run_distill` trains on them exactly as on U's. Only the new-token limit changes (default 8,192 instead of 1,024).

    # one process per GPU, each takes every n-th item of train + test
    CUDA_VISIBLE_DEVICES=5 PYTHONPATH=third_party/ext/distillation-game/src .venv-dg/bin/python \
        scripts/38_math_long_teacher.py --config configs/setup2/math_U_s123.yaml --shard 0 --nshards 3
    # then merge into train_long.json / test_long.json
    .venv-dg/bin/python scripts/38_math_long_teacher.py --config configs/setup2/math_U_s123.yaml --merge --nshards 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def out_dir(cfg, a) -> Path:
    d = Path(a.out or f"outputs/setup3_math/teacher_long{a.max_new}_s{cfg.run.seed}")
    d.mkdir(parents=True, exist_ok=True)
    return d


def generate(a, cfg) -> int:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from clean_sweep.data import format_prompt_math, load_dataset_splits
    from clean_sweep.eval.correctness import check_trace_correctness
    from clean_sweep.utils import set_seed

    set_seed(cfg.run.seed)
    splits = load_dataset_splits(cfg.data.dataset_name, seed=cfg.run.seed, train_size=cfg.data.train_size,
                                 holdout_size=cfg.data.holdout_size, test_size=cfg.data.test_size)
    items = [(s, ex) for s in ("train", "test") for ex in splits[s]][a.shard::a.nshards]
    if a.limit:  # smoke tests only
        items = items[:a.limit]
    tok = AutoTokenizer.from_pretrained(cfg.model.tokenizer or cfg.model.teacher)

    def ids_of(raw):
        return list(raw["input_ids"]) if hasattr(raw, "input_ids") or isinstance(raw, dict) else list(raw)

    prompts = [ids_of(tok.apply_chat_template(format_prompt_math(ex["problem"]), add_generation_prompt=True,
                                              truncation=True, max_length=cfg.generation.max_prompt_tokens))
               for _, ex in items]
    llm = LLM(model=cfg.model.teacher, dtype="bfloat16", seed=cfg.run.seed, gpu_memory_utilization=a.gpu_mem,
              max_model_len=cfg.generation.max_prompt_tokens + a.max_new + 64)
    t0 = time.time()
    outs = llm.generate([{"prompt_token_ids": p} for p in prompts],
                        SamplingParams(temperature=cfg.generation.temperature, top_p=cfg.generation.top_p,
                                       max_tokens=a.max_new, seed=cfg.run.seed))
    # answer forcing, as the setup: every reply gets the cue (after "\n</think>" if the reply has none)
    suffix = cfg.generation.answer_force_suffix
    af_prompts, af_suffix = [], []
    for p, o in zip(prompts, outs):
        c = o.outputs[0]
        gen = [t for t in c.token_ids if t != tok.eos_token_id]
        s = ("" if "</think>" in c.text else "\n</think>") + suffix
        af_prompts.append({"prompt_token_ids": p + gen + tok.encode(s, add_special_tokens=False)})
        af_suffix.append(s)
    af_outs = llm.generate(af_prompts, SamplingParams(temperature=0.0, max_tokens=cfg.generation.answer_force_max_new_tokens))
    rows = []
    for (split, ex), o, s, af in zip(items, outs, af_suffix, af_outs):
        c = o.outputs[0]
        reasoning = c.text.strip()
        tail = (s + af.outputs[0].text).strip()
        af_final = tail[tail.index("**Final Answer**"):] + tok.eos_token  # the setup's saved field ends in EOS
        raw = check_trace_correctness(reasoning, ex["solution"])
        forced = check_trace_correctness(f"{reasoning}\n{tail}", ex["solution"])
        rows.append({"example_id": ex["example_id"], "split": split, "method": "standard_long",
                     "prompt": ex["problem"], "trace": reasoning, "af_final_answer_only": af_final,
                     "raw_correct": bool(raw["correct"]), "af_correct": bool(forced["correct"]),
                     "correct": bool(forced["correct"]), "raw_extracted_answer": raw["extracted_answer"],
                     "af_extracted_answer": forced["extracted_answer"], "extracted_answer": forced["extracted_answer"],
                     "finished": c.finish_reason == "stop", "n_tokens": len(c.token_ids)})
    path = out_dir(cfg, a) / f"shard{a.shard}of{a.nshards}.json"
    path.write_text(json.dumps(rows))
    n = len(rows)
    print(f"shard {a.shard}/{a.nshards}: {n} items in {(time.time() - t0) / 60:.1f} min; "
          f"correct {sum(r['correct'] for r in rows) / n:.3f}, finished {sum(r['finished'] for r in rows) / n:.3f} -> {path}")
    return 0


def merge(a, cfg) -> int:
    d = out_dir(cfg, a)
    rows = [r for i in range(a.nshards) for r in json.loads((d / f"shard{i}of{a.nshards}.json").read_text())]
    summary = {}
    for split in ("train", "test"):
        part = sorted((r for r in rows if r["split"] == split), key=lambda r: int(r["example_id"].split("_")[-1]))
        (d / f"{split}_long.json").write_text(json.dumps(part, indent=1))
        tokens = sorted(r["n_tokens"] for r in part)
        summary[split] = {"n": len(part), "accuracy": sum(r["correct"] for r in part) / len(part),
                          "finished": sum(r["finished"] for r in part) / len(part),
                          "median_tokens": tokens[len(tokens) // 2], "p90_tokens": tokens[int(0.9 * len(tokens))]}
    summary["max_new_tokens"] = a.max_new
    (d / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--max-new", type=int, default=8192)
    ap.add_argument("--gpu-mem", type=float, default=0.9)
    ap.add_argument("--limit", type=int, default=0, help="first N items of this shard (smoke test)")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    from clean_sweep.config import FullConfig
    config = FullConfig.from_yaml(args.config)
    if args.merge:
        raise SystemExit(merge(args, config))
    hard_exit(generate(args, config))
