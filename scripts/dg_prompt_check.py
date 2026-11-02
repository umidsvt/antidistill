#!/usr/bin/env python
"""m8 §14, prompt check (descriptive, no threshold): how often the UNTRAINED base student answers training problems
correctly, judged against the teacher's released answers, under the setup's chat prompt (arm C) and the plain prompt
(arm P), with GRPO's sampling (temperature 1.0, top-p 1.0, 512 new tokens, 8 samples per problem). GRPO can only learn
if some samples are correct; v1 (m8 §9.3) started at a 1-2% reward under the chat prompt.

    PYTHONPATH=third_party/ext/distillation-game/src .venv-dg/bin/python scripts/dg_prompt_check.py \
        --config configs/setup2/gsm8k_U_s123.yaml --teacher-train outputs/setup2/gsm8k_U_s123_<stamp>/teacher/train_standard.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dg_rl_base import PLAIN_PROMPT, format_ok, hard_exit  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--teacher-train", required=True)
    ap.add_argument("--n", type=int, default=512, help="training problems checked")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--min-reason-words", type=int, default=20)
    ap.add_argument("--out", default="outputs/setup2/grpo_only/prompt_check.json")
    a = ap.parse_args()

    # importing dg_rl_base (TRL) initializes CUDA, so vLLM's engine must start in a fresh process, not a fork
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from clean_sweep.config import FullConfig
    from clean_sweep.data import format_prompt_gsm8k, load_dataset_splits
    from clean_sweep.eval.correctness import is_correct_gsm8k
    from clean_sweep.generation.core import ensure_chat_template

    cfg = FullConfig.from_yaml(a.config)
    splits = load_dataset_splits(cfg.data.dataset_name, seed=cfg.run.seed, train_size=cfg.data.train_size,
                                 holdout_size=cfg.data.holdout_size, test_size=cfg.data.test_size)
    rows = {r["example_id"]: r for r in json.loads(Path(a.teacher_train).read_text())}
    items = [(ex["problem"], rows[ex["example_id"]].get("extracted_answer") or "") for ex in splits["train"]][: a.n]
    tok = ensure_chat_template(AutoTokenizer.from_pretrained(cfg.model.student_tokenizer or cfg.model.student))
    # The same prompt text GRPO hands to vLLM in each arm (dg_rl_base.py).
    prompts = {
        "C_chat": [tok.apply_chat_template(format_prompt_gsm8k(p), add_generation_prompt=True, tokenize=False) for p, _ in items],
        "P_plain": [(tok.bos_token or "") + PLAIN_PROMPT.format(question=p.strip()) for p, _ in items],
    }
    llm = LLM(model=cfg.model.student, tokenizer=cfg.model.student_tokenizer or cfg.model.student, dtype="bfloat16",
              gpu_memory_utilization=0.85, max_model_len=2048, seed=cfg.run.seed)
    sp = SamplingParams(n=a.k, temperature=1.0, top_p=1.0, max_tokens=a.max_tokens, seed=cfg.run.seed)
    summary = {"n_problems": len(items), "k": a.k, "max_tokens": a.max_tokens}
    for name, ps in prompts.items():
        outs = llm.generate(ps, sp)
        correct, fmt_ok, lengths, capped, any_correct = 0, 0, [], 0, 0
        for (problem, ans), o in zip(items, outs):
            hits = 0
            for c in o.outputs:
                ok = bool(ans) and is_correct_gsm8k(c.text, f"\\boxed{{{ans}}}")
                hits += ok
                fmt_ok += format_ok(c.text, "final_reasoned", a.min_reason_words)
                lengths.append(len(c.token_ids))
                capped += c.finish_reason == "length"
            correct += hits
            any_correct += hits > 0
        total = len(items) * a.k
        summary[name] = {"correct_per_sample": round(correct / total, 4), "problems_with_a_correct_sample": round(any_correct / len(items), 4),
                         "format_reward_rate": round(fmt_ok / total, 4), "mean_tokens": round(statistics.mean(lengths), 1),
                         "hit_token_limit": round(capped / total, 4),
                         "example": outs[0].outputs[0].text[:600]}
        print(name, {k: v for k, v in summary[name].items() if k != "example"}, flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=1))
    print(f"prompt check written to {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    hard_exit(main())
