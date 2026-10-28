#!/usr/bin/env bash
# Env C, .venv-dg: the Distillation Game code (third_party/ext/distillation-game) and every scripts/dg_*.py and
# setup-2/3 script (results/m8_audit_design.md §9-11). Separate from .venv-train / .venv-infer (scripts/setup_envs.sh)
# because it needs vLLM 0.8.5 with transformers 4.53 and TRL 0.19. Exact versions: scripts/env_dg.lock.txt (Python 3.11).
#   bash scripts/setup_third_party.sh && bash scripts/setup_env_dg.sh
set -euo pipefail
cd "$(dirname "$0")/.."
command -v uv >/dev/null || pip install uv
uv venv .venv-dg --python 3.11
uv pip install --python .venv-dg/bin/python -r scripts/env_dg.lock.txt --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu124
.venv-dg/bin/python - <<'PY'
import torch, transformers, trl, peft, vllm, math_verify
print("env C OK | torch", torch.__version__, "| tf", transformers.__version__, "| trl", trl.__version__,
      "| peft", peft.__version__, "| vllm", vllm.__version__, "| gpus", torch.cuda.device_count())
PY
echo "Scripts run with: PYTHONPATH=third_party/ext/distillation-game/src .venv-dg/bin/python ..."
