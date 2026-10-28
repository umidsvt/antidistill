#!/usr/bin/env python
"""Train one passive student from saved solution files with the Distillation Game recipe (`run_distill`, mode
"naive"); optionally score it with the setup's own evaluation and save the merged weights.

Used by the option 4 pilot (results/m8_audit_design.md §10.1) and the combination students of §11:
- several --traces files are concatenated into one training set (all must cover the same problems): the "mix" recipe;
- --init starts from a saved merged model instead of the untrained student: the second stage of "sequential";
- --eval scores the result on the test split exactly as the setup does (HF loop, the config's token limit).
The pilot changes two settings so that long solutions fit (cut-off 9,216, micro-batch 1 × accumulation 24, the
defaults here); the §11 runs pass the setup's own values (--max-length 2048 --micro-batch 6 --accum 4).

    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=third_party/ext/distillation-game/src .venv-dg/bin/python \
        scripts/dg_train_from_traces.py --config configs/setup2/math_U_s123.yaml \
        --traces outputs/setup3_math/teacher_long8192_s123/train_long.json --out outputs/setup3_math/U_long_s123
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--traces", required=True, nargs="+",
                    help="solution file(s) in the setup's format (prompt, trace, af_final_answer_only)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", default="", help="merged model to start from instead of the config's student")
    ap.add_argument("--max-length", type=int, default=9216)
    ap.add_argument("--micro-batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=24)
    ap.add_argument("--eval", action="store_true", help="score on the test split with the setup's evaluation")
    ap.add_argument("--no-save", action="store_true", help="do not save merged weights")
    ap.add_argument("--limit", type=int, default=0, help="first N solutions of each file (smoke test only)")
    a = ap.parse_args()

    import torch
    from clean_sweep.config import FullConfig
    from clean_sweep.train import run_distill
    from clean_sweep.utils import set_seed

    cfg = FullConfig.from_yaml(a.config)
    cfg.distill.max_length = a.max_length
    cfg.distill.per_device_train_batch_size = a.micro_batch
    cfg.distill.gradient_accumulation_steps = a.accum
    if a.init:
        cfg.model.student = a.init
    traces, ids = [], None
    for f in a.traces:
        rows = json.loads(Path(f).read_text())
        if a.limit:
            rows = rows[: a.limit]
        elif len(rows) != cfg.data.train_size:
            raise SystemExit(f"{f}: {len(rows)} solutions, expected {cfg.data.train_size}")
        these = sorted(r["example_id"] for r in rows)
        if ids is not None and these != ids:
            raise SystemExit(f"{f}: not the same problems as {a.traces[0]}")
        ids = these
        traces += rows
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    t0 = time.time()
    set_seed(cfg.run.seed)
    stats, model, tok = run_distill(cfg=cfg, train_traces=traces, holdout_traces=[], output_dir=out / "student",
                                    device=device, mode="naive", beta_s=1.0)
    info = {"traces": a.traces, "n": len(traces), "init": a.init or cfg.model.student, "seed": cfg.run.seed,
            "max_length": a.max_length, "micro_batch": a.micro_batch, "accum": a.accum,
            "train_minutes": round((time.time() - t0) / 60, 1),
            "stats": {k: float(v) for k, v in (stats or {}).items() if isinstance(v, (int, float))}}
    if a.eval:
        from clean_sweep.data import format_prompt_gsm8k, format_prompt_math, load_dataset_splits
        from clean_sweep.generation import generate_teacher_traces
        test = load_dataset_splits(cfg.data.dataset_name, seed=cfg.run.seed, train_size=cfg.data.train_size,
                                   holdout_size=cfg.data.holdout_size, test_size=cfg.data.test_size)["test"]
        fmt = format_prompt_gsm8k if cfg.data.dataset_name == "gsm8k" else format_prompt_math
        set_seed(cfg.run.seed)
        compact, _ = generate_teacher_traces(cfg=cfg, dataset=test, format_prompt=fmt, method_name="standard",
                                             device=device, model=model, tokenizer=tok)
        (out / "test_student.json").write_text(json.dumps(compact, indent=1))
        info.update({"n_test": len(compact), "accuracy": sum(1 for r in compact if r["correct"]) / max(len(compact), 1)})
    if not a.no_save:
        merged = model.merge_and_unload()
        merged.save_pretrained(out / "merged", safe_serialization=True)
        tok.save_pretrained(out / "merged")
    info["minutes"] = round((time.time() - t0) / 60, 1)
    (out / ("result.json" if a.eval else "train_info.json")).write_text(json.dumps(info, indent=1))
    print(f"trained on {len(traces)} solutions from {len(a.traces)} file(s) in {info['train_minutes']} min"
          + (f"; {cfg.data.dataset_name} test accuracy {info['accuracy']:.4f} (n={info['n_test']})" if a.eval else "")
          + f" -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
