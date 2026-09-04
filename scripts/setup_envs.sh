#!/usr/bin/env bash
# Task 0.1 — build the two virtual environments.
#
# They cannot be merged: LLaMA-Factory 0.9.2 caps transformers<=4.48.2 / datasets<=3.2.0,
# while vLLM 0.11 requires much newer. Verified working on this host
# (8x L40S sm_89, driver 580.95.05 / CUDA 13.0, CUDA toolkit 12.8, gcc 11.5).
#
# Run from the repo root:  bash scripts/setup_envs.sh
set -euo pipefail

cd "$(dirname "$0")/.."
REPO="$PWD"
mkdir -p logs

command -v uv >/dev/null || pip install uv

# ---------------------------------------------------------------- env A: training
TRAIN_PY="$REPO/.venv-train/bin/python"
uv venv .venv-train --python 3.11

# torch is pinned so the flash-attn prebuilt-wheel choice below is deterministic.
uv pip install --python "$TRAIN_PY" "torch==2.6.0" --torch-backend cu124

# `uv venv` does not seed setuptools; deepspeed and trl both import it at load time.
uv pip install --python "$TRAIN_PY" setuptools wheel packaging ninja

uv pip install --python "$TRAIN_PY" -e "third_party/LLaMA-Factory[metrics]"
uv pip install --python "$TRAIN_PY" "deepspeed>=0.10.0,<=0.16.9"

# flash-attn: use the prebuilt wheel matching (torch 2.6, cp311, cxx11abi=False).
# Building from source on 64 cores still takes ~1h; the wheel takes ~1min.
FA_WHL="flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
curl -fL -o "/tmp/$FA_WHL" \
  "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/$FA_WHL"
uv pip install --python "$TRAIN_PY" "/tmp/$FA_WHL"

"$TRAIN_PY" - <<'PY'
import torch, transformers, deepspeed, accelerate, peft, trl, datasets, llamafactory, flash_attn
from flash_attn import flash_attn_func
q = torch.randn(1, 8, 4, 64, device="cuda", dtype=torch.bfloat16)
assert flash_attn_func(q, q, q).shape == q.shape
print("env A OK |", "torch", torch.__version__, "| tf", transformers.__version__,
      "| ds", deepspeed.__version__, "| lf", llamafactory.__version__,
      "| fa", flash_attn.__version__, "| gpus", torch.cuda.device_count())
PY

# --------------------------------------------------------------- env B: inference
INFER_PY="$REPO/.venv-infer/bin/python"
uv venv .venv-infer --python 3.12

uv pip install --python "$INFER_PY" -U "vllm==0.11.0" --torch-backend auto
uv pip install --python "$INFER_PY" "sglang[all]" openai

# math-eval dependencies of third_party/kim_eval/utils/{grader,parser}.py
#
# transformers MUST be pinned <5 here. vLLM 0.11.0 declares `transformers>=4.55.2` with no
# upper bound, so a bare `transformers` in this list resolves to 5.x and breaks vLLM at
# engine init: `Qwen2Tokenizer has no attribute all_special_tokens_extended`.
uv pip install --python "$INFER_PY" \
  sympy antlr4-python3-runtime word2number Pebble timeout-decorator latex2sympy2 \
  regex jinja2 "transformers>=4.55.2,<5" datasets tqdm python_dateutil

"$INFER_PY" - <<'PY'
import vllm, torch, transformers
print("env B OK |", "vllm", vllm.__version__, "| torch", torch.__version__,
      "| tf", transformers.__version__)
PY

echo
echo "Both environments built."
echo "  training  : $TRAIN_PY"
echo "  inference : $INFER_PY   (run the eval harness from third_party/kim_eval/)"
