#!/usr/bin/env python
"""Setup 2: score the UNTRAINED student with the Distillation Game's own evaluation (same splits, prompt,
sampling and answer forcing as its trained students). Usage:
    PYTHONPATH=third_party/ext/distillation-game/src .venv-dg/bin/python scripts/dg_eval_base.py configs/setup2/gsm8k_U_s123.yaml
Writes outputs/setup2/base_<dataset>_s<seed>.json.
"""
import json, sys
from pathlib import Path
import torch
from clean_sweep.config import FullConfig
from clean_sweep.data import format_prompt_gsm8k, format_prompt_math, load_dataset_splits
from clean_sweep.generation import generate_teacher_traces, load_model_and_tokenizer
from clean_sweep.utils import set_seed

cfg = FullConfig.from_yaml(sys.argv[1])
set_seed(cfg.run.seed)
splits = load_dataset_splits(cfg.data.dataset_name, seed=cfg.run.seed, train_size=cfg.data.train_size,
                             holdout_size=cfg.data.holdout_size, test_size=cfg.data.test_size)
fmt = format_prompt_gsm8k if cfg.data.dataset_name == "gsm8k" else format_prompt_math
device = torch.device("cuda")
model, tok = load_model_and_tokenizer(cfg.model.student, cfg.model.student_tokenizer, cfg, device)
set_seed(cfg.run.seed)
compact, _ = generate_teacher_traces(cfg=cfg, dataset=splits["test"], format_prompt=fmt, method_name="standard",
                                     device=device, model=model, tokenizer=tok)
acc = sum(1 for r in compact if r["correct"]) / max(len(compact), 1)
out = Path(cfg.run.output_dir) / f"base_{cfg.data.dataset_name}_s{cfg.run.seed}.json"
out.write_text(json.dumps({"student": cfg.model.student, "seed": cfg.run.seed, "n": len(compact), "accuracy": acc, "rows": compact}, indent=1))
print(f"untrained {cfg.model.student} on {cfg.data.dataset_name} test (seed {cfg.run.seed}): {acc:.4f} (n={len(compact)}) -> {out}")
