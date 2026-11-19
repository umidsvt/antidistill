#!/usr/bin/env python
"""Phase 0 of the answers-only method (results/m8_audit_design.md §24): a teacher's solutions for setup 2's training
problems (GSM8K or MATH) and its test score, with vLLM.

The Distillation Game teacher step (clean_sweep.generation.core.generate_teacher_traces, method "standard"), as in
scripts/38_math_long_teacher.py, generalized to both datasets, any teacher and tensor parallelism:
- the setup's prompt for the dataset in the teacher's chat template, cut at 512 tokens;
- sampling at temperature 0.6 / top-p 0.95, up to --max-new tokens;
- answer forcing: "\n</think>" when the reply has none, then the "**Final Answer**" cue, greedy 32 tokens;
- the setup's checker, and the setup's file fields, so `run_distill` trains on the output as on U's files.

Problems are matched by text, because the setup's example ids are positions after a per-seed shuffle. One solution
is written per distinct training problem of seeds 123-125 (GSM8K 7,252, MATH 7,221), and each seed's training file
is cut from them in that seed's order. The test split is the same set for every seed and is generated once.

    # one process per GPU pair, each takes every n-th item
    CUDA_VISIBLE_DEVICES=4,5 PYTHONPATH=third_party/ext/distillation-game/src .venv-dg/bin/python \
        scripts/dg_teacher_vllm.py --dataset math --teacher deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --tp 2 \
        --out outputs/phase0/teacher32b_math --shard 0 --nshards 2
    .venv-dg/bin/python scripts/dg_teacher_vllm.py --dataset math --out outputs/phase0/teacher32b_math --merge --nshards 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

SEEDS = (123, 124, 125)


def setup_config(dataset: str, seed: int) -> str:
    return f"configs/setup2/{dataset}_U_s{seed}.yaml"


def splits_by_seed(dataset: str) -> dict[int, dict]:
    from clean_sweep.config import FullConfig
    from clean_sweep.data import load_dataset_splits
    out = {}
    for seed in SEEDS:
        cfg = FullConfig.from_yaml(setup_config(dataset, seed))
        out[seed] = {"cfg": cfg, "splits": load_dataset_splits(cfg.data.dataset_name, seed=seed,
                                                               train_size=cfg.data.train_size,
                                                               holdout_size=cfg.data.holdout_size,
                                                               test_size=cfg.data.test_size)}
    return out


def work_items(dataset: str) -> list[dict]:
    """Distinct training problems of the three seeds (first-seen order), then the test split (seed 123's order)."""
    by_seed = splits_by_seed(dataset)
    seen, items = set(), []
    for seed in SEEDS:
        train = by_seed[seed]["splits"]["train"]
        for i in range(len(train)):
            ex = train[i]
            if ex["problem"] not in seen:
                seen.add(ex["problem"])
                items.append({"split": "train", "problem": ex["problem"], "solution": ex["solution"]})
    test = by_seed[SEEDS[0]]["splits"]["test"]
    items += [{"split": "test", "problem": test[i]["problem"], "solution": test[i]["solution"],
               "example_id": test[i]["example_id"]} for i in range(len(test))]
    return items


def generate(a) -> int:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from clean_sweep.data import format_prompt_gsm8k, format_prompt_math
    from clean_sweep.eval.correctness import check_trace_correctness
    from clean_sweep.utils import set_seed

    cfg = splits_by_seed(a.dataset)[SEEDS[0]]["cfg"]
    set_seed(a.seed)
    items = work_items(a.dataset)[a.shard::a.nshards]
    if a.limit:  # smoke tests only
        items = items[:a.limit]
    fmt = format_prompt_gsm8k if a.dataset == "gsm8k" else format_prompt_math
    tok = AutoTokenizer.from_pretrained(a.teacher)

    def ids_of(raw):
        return list(raw["input_ids"]) if hasattr(raw, "input_ids") or isinstance(raw, dict) else list(raw)

    prompts = [ids_of(tok.apply_chat_template(fmt(it["problem"]), add_generation_prompt=True, truncation=True,
                                              max_length=cfg.generation.max_prompt_tokens)) for it in items]
    llm = LLM(model=a.teacher, dtype="bfloat16", seed=a.seed, gpu_memory_utilization=a.gpu_mem,
              tensor_parallel_size=a.tp, max_model_len=cfg.generation.max_prompt_tokens + a.max_new + 64)
    t0 = time.time()
    outs = llm.generate([{"prompt_token_ids": p} for p in prompts],
                        SamplingParams(temperature=cfg.generation.temperature, top_p=cfg.generation.top_p,
                                       max_tokens=a.max_new, seed=a.seed))
    suffix = cfg.generation.answer_force_suffix
    af_prompts, af_suffix = [], []
    for p, o in zip(prompts, outs):
        c = o.outputs[0]
        gen = [t for t in c.token_ids if t != tok.eos_token_id]
        s = ("" if "</think>" in c.text else "\n</think>") + suffix
        af_prompts.append({"prompt_token_ids": p + gen + tok.encode(s, add_special_tokens=False)})
        af_suffix.append(s)
    af_outs = llm.generate(af_prompts, SamplingParams(temperature=0.0,
                                                      max_tokens=cfg.generation.answer_force_max_new_tokens))
    rows = []
    for it, o, s, af in zip(items, outs, af_suffix, af_outs):
        c = o.outputs[0]
        reasoning = c.text.strip()
        tail = (s + af.outputs[0].text).strip()
        af_final = tail[tail.index("**Final Answer**"):] + tok.eos_token  # the setup's saved field ends in EOS
        raw = check_trace_correctness(reasoning, it["solution"])
        forced = check_trace_correctness(f"{reasoning}\n{tail}", it["solution"])
        rows.append({**it, "method": "standard", "teacher": a.teacher, "max_new": a.max_new, "trace": reasoning,
                     "af_final_answer_only": af_final, "raw_correct": bool(raw["correct"]),
                     "af_correct": bool(forced["correct"]), "correct": bool(forced["correct"]),
                     "raw_extracted_answer": raw["extracted_answer"], "af_extracted_answer": forced["extracted_answer"],
                     "extracted_answer": forced["extracted_answer"], "finished": c.finish_reason == "stop",
                     "n_tokens": len(c.token_ids)})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"shard{a.shard}of{a.nshards}{'_smoke' if a.limit else ''}.json"
    path.write_text(json.dumps(rows))
    n = max(len(rows), 1)
    print(f"shard {a.shard}/{a.nshards}: {len(rows)} items in {(time.time() - t0) / 60:.1f} min; "
          f"correct {sum(r['correct'] for r in rows) / n:.3f}, finished {sum(r['finished'] for r in rows) / n:.3f} -> {path}",
          flush=True)
    return 0


def merge(a) -> int:
    out = Path(a.out)
    rows = [r for i in range(a.nshards) for r in json.loads((out / f"shard{i}of{a.nshards}.json").read_text())]
    train = {r["problem"]: r for r in rows if r["split"] == "train"}
    test = [r for r in rows if r["split"] == "test"]
    by_seed = splits_by_seed(a.dataset)
    assert len(train) == len({it["problem"] for it in work_items(a.dataset) if it["split"] == "train"}), "missing training problems"
    summary = {"teacher": rows[0]["teacher"], "max_new": rows[0]["max_new"], "distinct_train": len(train)}
    for seed in SEEDS:
        split = by_seed[seed]["splits"]["train"]
        seed_rows = [{**train[split[i]["problem"]], "example_id": split[i]["example_id"]} for i in range(len(split))]
        (out / f"train_s{seed}.json").write_text(json.dumps(seed_rows, indent=1))
        summary[f"train_s{seed}"] = {"n": len(seed_rows), "accuracy": sum(r["correct"] for r in seed_rows) / len(seed_rows),
                                     "finished": sum(r["finished"] for r in seed_rows) / len(seed_rows)}
    test.sort(key=lambda r: int(r["example_id"].split("_")[-1]))
    (out / "test.json").write_text(json.dumps(test, indent=1))
    for name, part in (("train_distinct", list(train.values())), ("test", test)):
        tokens = sorted(r["n_tokens"] for r in part)
        summary[name] = {"n": len(part), "accuracy": sum(r["correct"] for r in part) / len(part),
                         "finished": sum(r["finished"] for r in part) / len(part),
                         "median_tokens": tokens[len(tokens) // 2], "p90_tokens": tokens[int(0.9 * len(tokens))]}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


def hard_exit(rc: int) -> None:
    """Exit without waiting on vLLM, which can hang at interpreter exit; stop this process's own engine processes
    first (CLAUDE.md rule 6)."""
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
    ap.add_argument("--dataset", choices=["gsm8k", "math"], required=True)
    ap.add_argument("--teacher", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--max-new", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=123, help="sampling seed (one solution per problem)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--gpu-mem", type=float, default=0.9)
    ap.add_argument("--limit", type=int, default=0, help="first N items of this shard (smoke test)")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.merge:
        raise SystemExit(merge(args))
    hard_exit(generate(args))
