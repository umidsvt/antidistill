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

# m8 §14, arm P: a plain-text prompt (no chat template) that a base model can follow. The answer-forcing step of the
# setup's evaluation looks for the end of the prompt; PLAIN_MARKER is that end for this prompt.
PLAIN_MARKER = "Answer: Let's think step by step."
PLAIN_PROMPT = ("Question: {question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n"
                + PLAIN_MARKER)
FINAL_BOX = re.compile(r"\\boxed\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}\s*(?:\\\]|\$)?\s*\.?\s*$")


def format_ok(text: str, mode: str, min_words: int = 0) -> bool:
    """anywhere (v2): any boxed answer; final (v3): exactly one, at the very end; final_reasoned (m8 §14): as final,
    with at least min_words words of reasoning before it (v3a's policy wrote only a bare boxed number)."""
    t = text.strip()
    if mode == "anywhere":
        return "\\boxed{" in t
    if t.count("\\boxed{") != 1 or not FINAL_BOX.search(t):
        return False
    return mode == "final" or len(t[:t.find("\\boxed{")].split()) >= min_words


class ChunkedGRPOTrainer(GRPOTrainer):
    """TRL 0.19.1 computes the reference model's log-probabilities (needed when beta > 0) without a batch size, i.e.
    over the whole generation batch at once (32 samples x 2,048 tokens x 128k vocabulary in fp32 = 31 GiB; m8 §15
    smoke test). Every other call passes per_device_train_batch_size; this applies it to all calls."""

    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep, batch_size=None):
        return super()._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep,
                                            batch_size or self.args.per_device_train_batch_size)


class PlainPromptTokenizer:
    """Lets the setup's own evaluation score a student trained on the plain prompt: same sampling and answer forcing,
    only the prompt text differs (apply_chat_template returns the plain prompt)."""

    def __init__(self, tok):
        self._tok = tok

    def __getattr__(self, name):
        return getattr(self._tok, name)

    def __call__(self, *args, **kwargs):   # answer forcing tokenizes the forced texts directly
        return self._tok(*args, **kwargs)

    def apply_chat_template(self, messages, add_generation_prompt=True, truncation=True, max_length=None, **_):
        question = messages[-1]["content"].strip()
        return self._tok(PLAIN_PROMPT.format(question=question), truncation=truncation, max_length=max_length)["input_ids"]


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
    ap.add_argument("--format-mode", choices=["anywhere", "final", "final_reasoned"], default="anywhere")
    ap.add_argument("--min-reason-words", type=int, default=20, help="final_reasoned: words of reasoning before the box")
    ap.add_argument("--prompt-style", choices=["chat", "plain"], default="chat", help="m8 §14: chat = the setup's prompt")
    ap.add_argument("--save-adapter", action="store_true", help="save the trained LoRA to <out>/adapter")
    ap.add_argument("--also-eval-chat", action="store_true", help="plain arm: also score with the setup's chat prompt")
    ap.add_argument("--eval-only", action="store_true", help="no training: score the adapter given by --init-adapter")
    ap.add_argument("--name", default="", help="suffix of the output folder, e.g. v2a")
    # m8 §13: continue an existing LoRA (e.g. the self-trained student's) instead of starting from the untrained one
    ap.add_argument("--init-adapter", default="", help="LoRA adapter folder to continue training (default: a new LoRA)")
    ap.add_argument("--max-completion", type=int, default=0, help="completion tokens during GRPO (default: the setup's)")
    # m8 §15 (option B): Javaheri et al.'s stabilizers (SimpleRL-Zoo) ported to this script
    ap.add_argument("--reward-mode", choices=["bonus", "mix"], default="bonus",
                    help="bonus: 1 if correct (+ optional format bonus); mix (SimpleRL-Zoo): +1 correct and boxed, "
                         "-0.5 boxed but wrong, -1 without a boxed answer")
    ap.add_argument("--beta", type=float, default=0.0, help="KL penalty to the starting model (SimpleRL-Zoo: 1e-4)")
    ap.add_argument("--lr-schedule", choices=["cosine", "constant"], default="cosine")
    ap.add_argument("--save-every", type=int, default=0, help="save the adapter every N steps (checkpoint selection)")
    ap.add_argument("--holdout-select", type=int, default=0,
                    help="hold out the last N training problems; pick the saved adapter that scores best on them "
                         "against the teacher's answers")
    ap.add_argument("--greedy-test", action="store_true", help="also score the kept adapter greedily (reported only)")
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

    tok = ensure_chat_template(AutoTokenizer.from_pretrained(cfg.model.student_tokenizer or cfg.model.student,
                                                             padding_side="left"))

    def train_prompt(problem: str):
        if a.prompt_style == "chat":
            return fmt(problem)
        # TRL tokenizes prompts without special tokens, so the start token is written out, as the chat template does
        return (tok.bos_token or "") + PLAIN_PROMPT.format(question=problem.strip())

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
        data.append({"prompt": train_prompt(ex["problem"]), "teacher_answer": ans,
                     "_id": ex["example_id"], "_problem": ex["problem"]})
    if mismatched:
        raise SystemExit(f"{mismatched} train problems do not match the teacher file: wrong seed or file")
    # m8 §15: training problems held out for checkpoint selection, scored against the TEACHER's answers (a level-A
    # copier has nothing else); they are not trained on
    held = data[-a.holdout_select:] if a.holdout_select else []
    data = data[:-a.holdout_select] if a.holdout_select else data
    held_ds = Dataset.from_list([{"example_id": h["_id"], "problem": h["_problem"],
                                  "solution": f"\\boxed{{{h['teacher_answer']}}}"} for h in held if h["teacher_answer"]])
    data = [{"prompt": d["prompt"], "teacher_answer": d["teacher_answer"]} for d in data]
    print(f"train prompts: {len(data)} ({no_answer} without a teacher answer: their reward is always 0); "
          f"held out for checkpoint selection: {len(held_ds)}")

    def reward_teacher_answer(prompts, completions, teacher_answer, **_):
        scores = []
        for c, ans in zip(completions, teacher_answer):
            text = c[0]["content"] if isinstance(c, list) else c
            scores.append(1.0 if ans and is_correct_gsm8k(text, f"\\boxed{{{ans}}}") else 0.0)
        return scores

    def reward_boxed(prompts, completions, **_):
        return [1.0 if format_ok(c[0]["content"] if isinstance(c, list) else c, a.format_mode, a.min_reason_words)
                else 0.0 for c in completions]

    def reward_mix(prompts, completions, teacher_answer, **_):
        out_ = []
        for c, ans in zip(completions, teacher_answer):
            t = c[0]["content"] if isinstance(c, list) else c
            if "\\boxed{" not in t:
                out_.append(-1.0)
            else:
                out_.append(1.0 if ans and is_correct_gsm8k(t, f"\\boxed{{{ans}}}") else -0.5)
        return out_

    if a.reward_mode == "mix":   # correctness is still logged, with weight 0
        funcs, weights = [reward_mix, reward_teacher_answer], [1.0, 0.0]
    else:
        funcs, weights = [reward_teacher_answer], [1.0]
        if a.format_reward > 0:
            funcs.append(reward_boxed); weights.append(a.format_reward)

    model = AutoModelForCausalLM.from_pretrained(cfg.model.student, torch_dtype=get_dtype(cfg.model.torch_dtype),
                                                 attn_implementation=cfg.model.attn_implementation)
    tok = align_tokenizer_to_model(tok, model)
    if a.init_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.init_adapter, is_trainable=True)
    lora = None if a.init_adapter else LoraConfig(r=cfg.distill.lora_r, lora_alpha=cfg.distill.lora_alpha, lora_dropout=cfg.distill.lora_dropout,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                      bias="none", task_type="CAUSAL_LM")
    gen_per_step = a.num_generations * a.prompts_per_step
    args = GRPOConfig(
        output_dir=str(out / "trainer"), seed=cfg.run.seed, bf16=True,
        learning_rate=a.lr, warmup_ratio=0.03, weight_decay=0.0, beta=a.beta,
        lr_scheduler_type="constant_with_warmup" if a.lr_schedule == "constant" else "cosine",
        num_train_epochs=a.epochs, max_steps=a.max_steps,
        per_device_train_batch_size=a.micro_batch, gradient_accumulation_steps=gen_per_step // a.micro_batch,
        num_generations=a.num_generations, generation_batch_size=gen_per_step,
        max_prompt_length=cfg.generation.max_prompt_tokens,
        max_completion_length=a.max_completion or cfg.generation.max_new_tokens,
        temperature=a.temperature, top_p=1.0, reward_weights=weights,
        use_vllm=True, vllm_mode="colocate", vllm_gpu_memory_utilization=a.vllm_mem,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=5, save_strategy="no", report_to=[], remove_unused_columns=False,
    )
    if a.eval_only:
        if not a.init_adapter:
            raise SystemExit("--eval-only needs --init-adapter")
        student, train_s = model.to("cuda"), 0.0
    else:
        from transformers import TrainerCallback

        class SaveAdapter(TrainerCallback):   # m8 §15: adapters for checkpoint selection
            def on_step_end(self, args_, state, control, model=None, **_):
                if a.save_every and model is not None and state.global_step % a.save_every == 0:
                    model.save_pretrained(str(out / f"ckpt_{state.global_step:04d}"))

        trainer = ChunkedGRPOTrainer(model=model, reward_funcs=funcs, args=args, callbacks=[SaveAdapter()],
                              train_dataset=Dataset.from_list(data), processing_class=tok, peft_config=lora)
        t0 = time.time()
        trainer.train()
        train_s = time.time() - t0
        (out / "train_log.json").write_text(json.dumps(trainer.state.log_history, indent=1))
        # Score with the Distillation Game's own evaluation, as for its distilled students.
        del trainer.llm  # free the colocated vLLM engine before HF generation
        torch.cuda.empty_cache()
        student = trainer.model
        if a.save_adapter:
            student.save_pretrained(str(out / "adapter"))
    student.eval()
    test = splits["test"].select(range(a.test_limit)) if a.test_limit else splits["test"]

    def score(eval_tok, path, dataset=None, greedy=False):
        set_seed(cfg.run.seed)
        was = cfg.generation.greedy
        cfg.generation.greedy = greedy
        rows_, _ = generate_teacher_traces(cfg=cfg, dataset=test if dataset is None else dataset, format_prompt=fmt,
                                           method_name="standard", device=torch.device("cuda"), model=student,
                                           tokenizer=eval_tok)
        cfg.generation.greedy = was
        path.write_text(json.dumps(rows_, indent=1))
        return rows_, sum(1 for r in rows_ if r["correct"]) / max(len(rows_), 1)

    if a.prompt_style == "plain":
        from clean_sweep.generation import core as gen_core
        gen_core.RESPONSE_STRINGS["plain"] = PLAIN_MARKER   # where answer forcing finds the end of the prompt
        eval_tok = PlainPromptTokenizer(tok)
    else:
        eval_tok = tok

    # m8 §15: score every saved adapter on the held-out training problems (teacher's answers) and keep the best;
    # ties go to the earlier step. No test score is used to choose.
    kept_step, select, acc_last = None, {}, None
    ckpts = sorted(out.glob("ckpt_*")) if a.save_every and len(held_ds) and not a.eval_only else []
    if ckpts:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        load = lambda d: set_peft_model_state_dict(student, load_file(str(d / "adapter_model.safetensors")))
        _, acc_last = score(eval_tok, out / "test_rl_last.json")      # the weights after the last step
        for d in ckpts:
            load(d)
            _, select[d.name] = score(eval_tok, out / f"select_{d.name}.json", dataset=held_ds)
            print(f"checkpoint {d.name}: {select[d.name]:.4f} on {len(held_ds)} held-out training problems", flush=True)
        best = max(ckpts, key=lambda d: (select[d.name], -int(d.name.split("_")[1])))
        kept_step = int(best.name.split("_")[1])
        load(best)
        student.save_pretrained(str(out / "adapter_kept"))
    acc_chat = acc_greedy = None
    compact, acc = score(eval_tok, out / "test_rl.json")
    if a.greedy_test:
        _, acc_greedy = score(eval_tok, out / "test_rl_greedy.json", greedy=True)
    if a.prompt_style == "plain" and a.also_eval_chat:
        _, acc_chat = score(tok, out / "test_rl_chatprompt.json")
    res = {"student": cfg.model.student, "seed": cfg.run.seed, "dataset": cfg.data.dataset_name, "n_test": len(compact),
           "accuracy": acc, "prompt_style": a.prompt_style, "accuracy_chat_prompt": acc_chat,
           "kept_step": kept_step, "accuracy_last": acc_last, "accuracy_greedy": acc_greedy,
           "selection_on_held_out": select, "reward_mode": a.reward_mode, "beta": a.beta, "lr_schedule": a.lr_schedule,
           "min_reason_words": a.min_reason_words,
           "train_prompts": len(data), "no_teacher_answer": no_answer, "train_seconds": round(train_s),
           "lr": a.lr, "num_generations": a.num_generations, "prompts_per_step": a.prompts_per_step,
           "epochs": a.epochs, "max_steps": a.max_steps, "format_reward": a.format_reward,
           "temperature": a.temperature, "name": a.name, "format_mode": a.format_mode,
           "init_adapter": a.init_adapter, "max_completion": a.max_completion or cfg.generation.max_new_tokens}
    (out / "result.json").write_text(json.dumps(res, indent=1))
    start = f"the adapter {a.init_adapter}" if a.init_adapter else "the untrained student"
    print(f"RL from {start}, {cfg.data.dataset_name} test (seed {cfg.run.seed}): {acc:.4f} (n={len(compact)}) -> {out}")
    return 0


def hard_exit(rc: int) -> None:
    """Exit without waiting on vLLM 0.8.5, which can hang at interpreter exit (CLAUDE.md rule 6); stop this
    process's own vLLM engine processes first so none is left holding the GPU."""
    import sys
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
