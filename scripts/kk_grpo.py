#!/usr/bin/env python
"""Phase 1 on the puzzle testbed (results/m8_audit_design.md §25.2), baselines B3 and B4: GRPO from the untrained
instruction-tuned student, rewarded by the teacher's released answers only.

- Puzzles: the 2,000 step-2 training puzzles; those with a readable teacher answer (1,712) are the training prompts.
  The teacher's answer is its first try's CONCLUSION (as §20 and B1/B2); puzzles without one are not trained on.
- Prompt: the U_inst prompt (kk_sft.user_turn in the Llama-3.2 chat template with the fixed date line), rendered as
  kk_eval renders it, so the student is trained and scored on the same text.
- Reward (--reward): `exact` (B3) 1 if the CONCLUSION labels every person as the teacher did, else 0; `partial` (B4)
  the share of people labeled as the teacher labeled them; 0 for an unreadable answer. Both, and agreement with the
  reference solution, are logged; only the chosen one has weight (the reference is never used for training).
- Model: LoRA rank 128, alpha 128 on all projections (the capacity of the SFT students), TRL 0.19 GRPO with vLLM
  colocated on the same GPU; Dr. GRPO loss (no length or std normalization).
- Output: <out>/adapter (final), <out>/ckpt_<step>/ every --save-every steps, train_log.json, train_meta.json. The
  student is scored separately with scripts/kk_eval.py, exactly as the SFT students.

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=third_party/ext/distillation-game/src .venv-dg/bin/python scripts/kk_grpo.py \
        --reward partial --seed 123 --out outputs/kk_phase1/students/B4_s123
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kk_build import load_tries  # noqa: E402
from kk_pilot import parse_labels  # noqa: E402
from kk_sft import DATE, user_turn  # noqa: E402

INSTRUCT = "unsloth/Llama-3.2-3B-Instruct"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reward", choices=["exact", "partial"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pool", default="data/kk/step2_train.jsonl")
    ap.add_argument("--teacher-gen", default="outputs/kk_step2/gen")
    ap.add_argument("--student", default=INSTRUCT)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--beta", type=float, default=1e-4, help="KL penalty toward the untrained student")
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--prompts-per-step", type=int, default=16)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--max-steps", type=int, default=-1, help="smoke tests only")
    ap.add_argument("--max-completion", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--vllm-mem", type=float, default=0.35)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--limit", type=int, default=0, help="first N prompts (smoke tests only)")
    ap.add_argument("--init-adapter", default="", help="§25.3 (B5): start from this trained LoRA adapter (e.g. B1's) "
                    "instead of a new one; the KL reference stays the untrained student (adapter off)")
    a = ap.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed
    from trl import GRPOConfig, GRPOTrainer

    class ChunkedGRPOTrainer(GRPOTrainer):
        """As scripts/dg_rl_base.py: TRL 0.19.1 computes the reference log-probabilities (beta > 0) over the whole
        generation batch at once; pass the micro-batch size to every call."""
        def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep, batch_size=None):
            return super()._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep,
                                                batch_size or self.args.per_device_train_batch_size)

    # TRL's colocated vLLM expects a (single-process) torch.distributed environment, as in scripts/dg_rl_base.py;
    # one port per GPU (KK_MASTER_PORT) so that runs on different GPUs do not clash
    for k, v in {"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1", "MASTER_ADDR": "127.0.0.1",
                 "MASTER_PORT": os.environ.get("KK_MASTER_PORT", "29711")}.items():
        os.environ.setdefault(k, v)
    set_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.student)
    if tok.pad_token is None:
        tok.pad_token = "<|finetune_right_pad_id|>"
    pool = [json.loads(line) for line in open(a.pool)]
    T = load_tries(Path(a.teacher_gen), "T")
    data = []
    for r in pool:
        labels = T[r["id"]][0]["labels"]
        if labels is None:
            continue
        prompt = tok.apply_chat_template([user_turn(r["quiz"])], tokenize=False, add_generation_prompt=True,
                                         date_string=DATE)
        data.append({"prompt": prompt, "names": r["names"], "teacher": labels, "solution": list(r["solution"])})
    if a.limit:
        data = data[:a.limit]

    def labels_of(text, names):
        return parse_labels(text, names, reasoning=False)

    def reward_exact(prompts, completions, names, teacher, **_):
        return [1.0 if labels_of(c, n) == t else 0.0 for c, n, t in zip(completions, names, teacher)]

    def reward_partial(prompts, completions, names, teacher, **_):
        out_ = []
        for c, n, t in zip(completions, names, teacher):
            lab = labels_of(c, n)
            out_.append(0.0 if lab is None else sum(x == y for x, y in zip(lab, t)) / len(t))
        return out_

    def agrees_with_reference(prompts, completions, names, solution, **_):   # logged only (weight 0)
        return [1.0 if labels_of(c, n) == s else 0.0 for c, n, s in zip(completions, names, solution)]

    funcs = [reward_exact, reward_partial, agrees_with_reference]
    weights = [1.0, 0.0, 0.0] if a.reward == "exact" else [0.0, 1.0, 0.0]

    model = AutoModelForCausalLM.from_pretrained(a.student, torch_dtype=torch.bfloat16,
                                                 attn_implementation="flash_attention_2")
    lora = LoraConfig(r=128, lora_alpha=128, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    if a.init_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.init_adapter, is_trainable=True)
        lora = None
    gen_per_step = a.num_generations * a.prompts_per_step
    args = GRPOConfig(
        output_dir=str(out / "trainer"), seed=a.seed, bf16=True,
        learning_rate=a.lr, lr_scheduler_type="constant_with_warmup", warmup_ratio=0.03, weight_decay=0.0,
        beta=a.beta, loss_type="dr_grpo", scale_rewards=False,
        num_train_epochs=a.epochs, max_steps=a.max_steps,
        per_device_train_batch_size=a.micro_batch, gradient_accumulation_steps=gen_per_step // a.micro_batch,
        num_generations=a.num_generations, generation_batch_size=gen_per_step,
        max_prompt_length=1024, max_completion_length=a.max_completion,
        temperature=a.temperature, top_p=1.0, reward_weights=weights,
        use_vllm=True, vllm_mode="colocate", vllm_gpu_memory_utilization=a.vllm_mem,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1, save_strategy="no", report_to=[], remove_unused_columns=False,
    )

    class SaveAdapter(TrainerCallback):
        def on_step_end(self, args_, state, control, model=None, **_):
            if a.save_every and model is not None and state.global_step % a.save_every == 0:
                model.save_pretrained(str(out / f"ckpt_{state.global_step:04d}"))

    trainer = ChunkedGRPOTrainer(model=model, reward_funcs=funcs, args=args, callbacks=[SaveAdapter()],
                                 train_dataset=Dataset.from_list(data), processing_class=tok, peft_config=lora)
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    trainer.train()
    trainer.model.save_pretrained(str(out / "adapter"))
    log = trainer.state.log_history
    (out / "train_log.json").write_text(json.dumps(log, indent=1))
    meta = {"method": f"GRPO, reward {a.reward}", "student": a.student, "seed": a.seed, "prompts": len(data),
            "steps": trainer.state.global_step, "lr": a.lr, "beta": a.beta, "num_generations": a.num_generations,
            "prompts_per_step": a.prompts_per_step, "epochs": a.epochs, "max_completion": a.max_completion,
            "temperature": a.temperature, "loss_type": "dr_grpo", "init_adapter": a.init_adapter, "train_minutes": round((time.time() - t0) / 60, 1),
            "peak_memory_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1)}
    (out / "train_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta), flush=True)
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
