#!/usr/bin/env python
"""Setup 2, check 2 (results/m8_audit_design.md §2, §7): RL from the UNTRAINED student, rewarded by agreement
with the TEACHER'S RELEASED FINAL ANSWERS -- no teacher solution text is used.

Everything except the training method is the Distillation Game's own: the same train split and prompts
(seeded as in the U run), the same student and LoRA adapter (r=128, alpha=128, all projections, the capacity
its distilled students get), its answer checker for the reward, and its evaluation function for the score.

    PYTHONPATH=third_party/ext/distillation-game/src .venv-dg/bin/python scripts/dg_rl_base.py \
        --config configs/setup2/gsm8k_U_s123.yaml \
        --teacher-train outputs/setup2/gsm8k_U_s123_<stamp>/teacher/train_standard.json

Writes outputs/setup2/rl_<dataset>_s<seed>/: test_rl.json (per-problem), result.json, train_log.json.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from clean_sweep.config import FullConfig
from clean_sweep.data import format_prompt_gsm8k, format_prompt_math, load_dataset_splits
from clean_sweep.eval.correctness import is_correct_gsm8k
from clean_sweep.generation import generate_teacher_traces
from clean_sweep.generation.core import align_tokenizer_to_model, ensure_chat_template, get_dtype
from clean_sweep.utils import set_seed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--teacher-train", required=True, help="teacher/train_standard.json of the U run with the same seed")
    ap.add_argument("--out", default="")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--prompts-per-step", type=int, default=8)
    ap.add_argument("--epochs", type=float, default=1.0, help="passes over the train prompts (the SFT students see them 3 times)")
    ap.add_argument("--max-steps", type=int, default=-1, help="smoke tests only")
    ap.add_argument("--test-limit", type=int, default=0, help="smoke tests only")
    ap.add_argument("--vllm-mem", type=float, default=0.3)
    # v2 (m8 §9.3): a small reward for writing a boxed answer at all, and the sampling temperature
    ap.add_argument("--format-reward", type=float, default=0.0, help="weight of the boxed-answer reward (v1: 0)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--micro-batch", type=int, default=8, help="samples per forward pass (16 ran out of memory)")
    # v3 (m8 §9.3): v2's "boxed anywhere" bonus was hacked (write \\boxed{ and ramble to the limit); "final" pays only
    # for exactly one boxed answer at the very end of the text, which requires stopping
    ap.add_argument("--format-mode", choices=["anywhere", "final"], default="anywhere")
    ap.add_argument("--name", default="", help="suffix of the output folder, e.g. v2a")
    a = ap.parse_args()

    # vLLM colocate mode reads the variables a distributed launcher sets; this is a single process.
    for k, v in {"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1", "MASTER_ADDR": "127.0.0.1",
                 "MASTER_PORT": os.environ.get("DG_MASTER_PORT", "29511")}.items():
        os.environ.setdefault(k, v)
    cfg = FullConfig.from_yaml(a.config)
    set_seed(cfg.run.seed)
    out = Path(a.out or Path(cfg.run.output_dir) / (f"rl_{cfg.data.dataset_name}_s{cfg.run.seed}" + (f"_{a.name}" if a.name else "")))
    out.mkdir(parents=True, exist_ok=True)

    splits = load_dataset_splits(cfg.data.dataset_name, seed=cfg.run.seed, train_size=cfg.data.train_size,
                                 holdout_size=cfg.data.holdout_size, test_size=cfg.data.test_size)
    fmt = format_prompt_gsm8k if cfg.data.dataset_name == "gsm8k" else format_prompt_math

    # The teacher's released answers for exactly these train problems (same seed => same split).
    rows = {r["example_id"]: r for r in json.loads(Path(a.teacher_train).read_text())}
    data, mismatched, no_answer = [], 0, 0
    for ex in splits["train"]:
        r = rows.get(ex["example_id"])
        if r is None or r.get("prompt", "").strip() != ex["problem"].strip():
            mismatched += 1
            continue
        ans = r.get("extracted_answer") or ""
        no_answer += not ans
        data.append({"prompt": fmt(ex["problem"]), "teacher_answer": ans})
    if mismatched:
        raise SystemExit(f"{mismatched} train problems do not match the teacher file: wrong seed or file")
    print(f"train prompts: {len(data)} ({no_answer} without a teacher answer: their reward is always 0)")

    def reward_teacher_answer(prompts, completions, teacher_answer, **_):
        scores = []
        for c, ans in zip(completions, teacher_answer):
            text = c[0]["content"] if isinstance(c, list) else c
            scores.append(1.0 if ans and is_correct_gsm8k(text, f"\\boxed{{{ans}}}") else 0.0)
        return scores

    final_box = re.compile(r"\\boxed\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}\s*(?:\\\]|\$)?\s*\.?\s*$")

    def reward_boxed(prompts, completions, **_):
        out = []
        for c in completions:
            t = (c[0]["content"] if isinstance(c, list) else c).strip()
            if a.format_mode == "anywhere":
                out.append(1.0 if "\\boxed{" in t else 0.0)
            else:
                out.append(1.0 if t.count("\\boxed{") == 1 and final_box.search(t) else 0.0)
        return out

    funcs, weights = [reward_teacher_answer], [1.0]
    if a.format_reward > 0:
        funcs.append(reward_boxed); weights.append(a.format_reward)

    tok = ensure_chat_template(AutoTokenizer.from_pretrained(cfg.model.student_tokenizer or cfg.model.student,
                                                             padding_side="left"))
    model = AutoModelForCausalLM.from_pretrained(cfg.model.student, torch_dtype=get_dtype(cfg.model.torch_dtype),
                                                 attn_implementation=cfg.model.attn_implementation)
    tok = align_tokenizer_to_model(tok, model)
    lora = LoraConfig(r=cfg.distill.lora_r, lora_alpha=cfg.distill.lora_alpha, lora_dropout=cfg.distill.lora_dropout,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                      bias="none", task_type="CAUSAL_LM")
    gen_per_step = a.num_generations * a.prompts_per_step
    args = GRPOConfig(
        output_dir=str(out / "trainer"), seed=cfg.run.seed, bf16=True,
        learning_rate=a.lr, lr_scheduler_type="cosine", warmup_ratio=0.03, weight_decay=0.0,
        num_train_epochs=a.epochs, max_steps=a.max_steps,
        per_device_train_batch_size=a.micro_batch, gradient_accumulation_steps=gen_per_step // a.micro_batch,
        num_generations=a.num_generations, generation_batch_size=gen_per_step,
        max_prompt_length=cfg.generation.max_prompt_tokens, max_completion_length=cfg.generation.max_new_tokens,
        temperature=a.temperature, top_p=1.0, reward_weights=weights,
        use_vllm=True, vllm_mode="colocate", vllm_gpu_memory_utilization=a.vllm_mem,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=5, save_strategy="no", report_to=[], remove_unused_columns=False,
    )
    trainer = GRPOTrainer(model=model, reward_funcs=funcs, args=args,
                          train_dataset=Dataset.from_list(data), processing_class=tok, peft_config=lora)
    t0 = time.time()
    trainer.train()
    train_s = time.time() - t0
    (out / "train_log.json").write_text(json.dumps(trainer.state.log_history, indent=1))

    # Score with the Distillation Game's own evaluation, as for its distilled students.
    del trainer.llm  # free the colocated vLLM engine before HF generation
    torch.cuda.empty_cache()
    student = trainer.model
    student.eval()
    test = splits["test"].select(range(a.test_limit)) if a.test_limit else splits["test"]
    set_seed(cfg.run.seed)
    compact, _ = generate_teacher_traces(cfg=cfg, dataset=test, format_prompt=fmt, method_name="standard",
                                         device=torch.device("cuda"), model=student, tokenizer=tok)
    acc = sum(1 for r in compact if r["correct"]) / max(len(compact), 1)
    (out / "test_rl.json").write_text(json.dumps(compact, indent=1))
    res = {"student": cfg.model.student, "seed": cfg.run.seed, "dataset": cfg.data.dataset_name, "n_test": len(compact),
           "accuracy": acc, "train_prompts": len(data), "no_teacher_answer": no_answer, "train_seconds": round(train_s),
           "lr": a.lr, "num_generations": a.num_generations, "prompts_per_step": a.prompts_per_step,
           "epochs": a.epochs, "max_steps": a.max_steps, "format_reward": a.format_reward,
           "temperature": a.temperature, "name": a.name, "format_mode": a.format_mode}
    (out / "result.json").write_text(json.dumps(res, indent=1))
    print(f"RL from the untrained student, {cfg.data.dataset_name} test (seed {cfg.run.seed}): {acc:.4f} (n={len(compact)}) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
