#!/usr/bin/env python
"""Puzzle testbed, step 2 (results/m8_audit_design.md §18): fine-tune the student on one training set.

Setup 2's recipe where it applies (third_party/ext/distillation-game, clean_sweep/train/distill.py): base Llama-3.2-3B
with LoRA rank 128, alpha 128, no dropout, on q/k/v/o/gate/up/down projections; learning rate 5e-4, cosine, 3% warm-up;
3 epochs; effective batch 24; weight decay 0.1; gradient clipping 1.0; fused AdamW; bf16; gradient checkpointing; loss
on the solution only. Differences: maximum length 8,704 tokens (setup 2: 2,048) and one example per device step with
24 accumulation steps (setup 2: 6 x 4), which changes only padding. The puzzle is the user turn of the Llama-3.2 chat
template (the setup's student tokenizer), with §17's instruction and a fixed date line.

    .venv-dg/bin/python scripts/kk_sft.py --train-set outputs/kk_step2/train_sets/U.jsonl \
        --out outputs/kk_step2/students/U_s123 --seed 123 [--max-steps 3]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kk_pilot import INSTRUCTION  # noqa: E402

STUDENT, STUDENT_TOKENIZER, DATE = "unsloth/Llama-3.2-3B", "unsloth/Llama-3.2-3B-Instruct", "13 Nov 2026"


def user_turn(quiz: str) -> dict:
    return {"role": "user", "content": INSTRUCTION + "\n### Question: " + quiz}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-set", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--max-length", type=int, default=8704)
    ap.add_argument("--grad-accum", type=int, default=24)
    ap.add_argument("--max-steps", type=int, default=-1, help="smoke tests only")
    ap.add_argument("--limit", type=int, default=0, help="first N examples (smoke tests only)")
    ap.add_argument("--plain-prompt", action="store_true", help="m8 §36: the row's quiz is the whole user message "
                    "(second task), instead of Xie et al.'s instruction + quiz")
    ap.add_argument("--student", default=STUDENT, help="m8 §24: unsloth/Llama-3.2-3B-Instruct for the instruct student")
    a = ap.parse_args()
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed
    set_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.student if "Qwen" in a.student else STUDENT_TOKENIZER)  # m8 §35: Qwen student
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    rows = [json.loads(line) for line in open(a.train_set)]
    if a.limit:
        rows = rows[:a.limit]
    examples, truncated = [], 0
    turn = (lambda q: {"role": "user", "content": q}) if a.plain_prompt else user_turn
    for r in rows:
        prompt = tok.apply_chat_template([turn(r["quiz"])], add_generation_prompt=True, date_string=DATE)
        full = tok.apply_chat_template([turn(r["quiz"]), {"role": "assistant", "content": r["text"]}],
                                       add_generation_prompt=False, date_string=DATE)
        assert full[:len(prompt)] == prompt, "chat template: the prompt is not a prefix of the full conversation"
        if len(full) > a.max_length:
            full, truncated = full[:a.max_length], truncated + 1
        examples.append({"input_ids": full, "labels": [-100] * len(prompt) + full[len(prompt):]})

    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(examples)

        def __getitem__(self, i):
            return examples[i]

    def collate(batch):
        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), tok.pad_token_id)
        lab = torch.full((len(batch), n), -100)
        att = torch.zeros((len(batch), n), dtype=torch.long)
        for i, b in enumerate(batch):
            k = len(b["input_ids"])
            ids[i, :k], lab[i, :k], att[i, :k] = torch.tensor(b["input_ids"]), torch.tensor(b["labels"]), 1
        return {"input_ids": ids, "labels": lab, "attention_mask": att}

    model = AutoModelForCausalLM.from_pretrained(a.student, torch_dtype=torch.bfloat16,
                                                 attn_implementation="flash_attention_2").to("cuda")
    model.config.pad_token_id = tok.pad_token_id
    model = get_peft_model(model, LoraConfig(r=128, lora_alpha=128, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                                                             "up_proj", "down_proj"]))
    model.enable_input_require_grads()
    args = TrainingArguments(
        output_dir=str(out / "trainer"), per_device_train_batch_size=1, gradient_accumulation_steps=a.grad_accum,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        num_train_epochs=a.epochs, max_steps=a.max_steps, learning_rate=a.lr, weight_decay=0.1, max_grad_norm=1.0,
        warmup_ratio=0.03, lr_scheduler_type="cosine", optim="adamw_torch_fused", bf16=True,
        logging_steps=5, save_strategy="no", report_to=[], seed=a.seed, data_seed=a.seed,
        dataloader_num_workers=0, remove_unused_columns=False, disable_tqdm=True)
    t0 = time.time()
    trainer = Trainer(model=model, args=args, train_dataset=DS(), data_collator=collate)
    trainer.train()
    model.save_pretrained(str(out / "adapter"))
    meta = {"train_set": a.train_set, "seed": a.seed, "student": a.student, "examples": len(examples), "truncated": truncated,
            "max_length": a.max_length, "mean_tokens": round(sum(len(e["input_ids"]) for e in examples) / len(examples), 1),
            "steps": trainer.state.global_step, "minutes": round((time.time() - t0) / 60, 1),
            "loss_log": [x for x in trainer.state.log_history if "loss" in x]}
    (out / "train_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if k != "loss_log"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
