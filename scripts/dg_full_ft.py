#!/usr/bin/env python
"""Recipe check (results/m8_audit_design.md §25.1): train the setup-2 student by full fine-tuning instead of the
setup's LoRA, on one GPU.

Everything except the trained weights and the learning rate follows the setup's trainer
(clean_sweep.train.distill.run_distill, mode "naive"): the same training texts (the setup's prompt in the student's
chat template, the solution and its final line), the completion-only collator, SFTTrainer with the same schedule,
batch, clipping, weight decay, seed and data order. Changes:
- every weight is trained (no LoRA), learning rate 1e-5 by default;
- to fit one 46 GB GPU: fp32 weights with bf16 compute, paged 8-bit AdamW (bitsandbytes; its states move to CPU
  memory under pressure; embeddings keep 32-bit states, as the HF Trainer sets), Liger's fused linear cross-entropy,
  gradient checkpointing, SDPA attention. (The first full run, 2026-11-20 16:00, ran out of memory at the second
  step with non-paged 8-bit AdamW: weights, gradients and optimizer states together exceed the GPU.)
The merged weights are saved in bf16 for scripts/dg_score_vllm.py.

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=third_party/ext/distillation-game/src .venv-dg/bin/python scripts/dg_full_ft.py \
        --config configs/setup2/math_U_s123.yaml --student unsloth/Llama-3.2-3B-Instruct \
        --traces outputs/phase0/teacher32b_math/train_s123.json --out outputs/phase0/S2full/math_s123
    # memory smoke test: two optimizer steps on the 48 longest texts
    ... scripts/dg_full_ft.py ... --smoke-longest 48 --out /tmp/x --no-save
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--traces", required=True)
    ap.add_argument("--student", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=0, help="0: the config's (3)")
    ap.add_argument("--max-length", type=int, default=9216)
    ap.add_argument("--accum", type=int, default=24)
    ap.add_argument("--optim", default="paged_adamw_8bit",
                    help="8-bit AdamW whose states page to CPU under memory pressure (bitsandbytes)")
    ap.add_argument("--smoke-longest", type=int, default=0,
                    help="smoke test: train on the N longest texts, two optimizer steps (N >= 2 x accum)")
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args()

    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer
    from clean_sweep.config import FullConfig
    from clean_sweep.data import format_prompt_gsm8k
    from clean_sweep.train import distill as D
    from clean_sweep.train.collator import DataCollatorForCompletionOnlyLM
    from clean_sweep.utils import set_seed

    cfg = FullConfig.from_yaml(a.config)
    set_seed(cfg.run.seed)
    rows = json.loads(Path(a.traces).read_text())
    if not a.smoke_longest and len(rows) != cfg.data.train_size:
        raise SystemExit(f"{a.traces}: {len(rows)} solutions, expected {cfg.data.train_size}")

    # tokenizer and model as load_student_model, without LoRA and in fp32
    tok = AutoTokenizer.from_pretrained(a.student, trust_remote_code=True, padding_side="left")
    tok = D.ensure_chat_template(tok)
    model = AutoModelForCausalLM.from_pretrained(a.student, trust_remote_code=True, torch_dtype=torch.float32,
                                                 attn_implementation="sdpa")
    tok = D.align_tokenizer_to_model(tok, model)
    model = model.to("cuda")

    # training texts and collator as run_distill
    response_template = D._response_template_for_model(a.student, tok)
    collator = DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tok, mlm=False,
                                               max_length=a.max_length)

    def to_text(ex):
        prompt = ex.get("problem", ex.get("prompt", ""))
        return format_prompt_gsm8k(prompt) + [{"role": "assistant", "content": D._trace_text_for_training(ex)}]

    texts = [tok.decode(tok.apply_chat_template(to_text(ex), add_generation_prompt=False, truncation=True,
                                                max_length=a.max_length)) for ex in rows]
    if a.smoke_longest:
        texts = sorted(texts, key=lambda t: len(tok(t).input_ids), reverse=True)[: a.smoke_longest]
    n_tokens = [len(tok(t).input_ids) for t in texts]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    args = SFTConfig(
        output_dir=str(out / "trainer"),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=a.accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        num_train_epochs=a.epochs or cfg.distill.num_epochs,
        max_steps=2 if a.smoke_longest else -1,
        learning_rate=a.lr,
        weight_decay=cfg.distill.weight_decay,
        max_grad_norm=cfg.distill.max_grad_norm,
        warmup_ratio=cfg.distill.warmup_ratio,
        lr_scheduler_type=cfg.distill.lr_scheduler_type,
        optim=a.optim,
        use_liger_kernel=True,
        bf16=True,
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        logging_steps=25,
        disable_tqdm=True,
        report_to=[],
        save_strategy="no",
        do_eval=False,
        seed=cfg.run.seed,
        remove_unused_columns=False,
        dataset_text_field="text",
        max_length=a.max_length,
    )
    trainer = SFTTrainer(model=model, args=args, train_dataset=Dataset.from_dict({"text": texts}),
                         processing_class=tok, data_collator=collator)
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    result = trainer.train()
    info = {"traces": a.traces, "n": len(texts), "student": a.student, "seed": cfg.run.seed, "method": "full fine-tuning",
            "lr": a.lr, "optim": a.optim, "epochs": args.num_train_epochs, "accum": a.accum, "max_length": a.max_length,
            "over_max_length": sum(n > a.max_length for n in n_tokens), "max_tokens": max(n_tokens),
            "train_loss": result.training_loss, "train_minutes": round((time.time() - t0) / 60, 1),
            "peak_memory_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1),
            "log": [h for h in trainer.state.log_history if "loss" in h]}
    if not a.no_save:
        model = model.to(torch.bfloat16)
        model.config.torch_dtype = torch.bfloat16
        model.save_pretrained(out / "merged", safe_serialization=True)
        tok.save_pretrained(out / "merged")
    (out / ("smoke_info.json" if a.smoke_longest else "train_info.json")).write_text(json.dumps(info, indent=1))
    print(json.dumps({k: v for k, v in info.items() if k != "log"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
