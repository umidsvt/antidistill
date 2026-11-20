#!/usr/bin/env bash
# Recipe check (results/m8_audit_design.md §25.1): the 32B-teacher MATH student (seed 123) trained by full fine-tuning,
# then scored like every Phase-0 student (8,192 tokens). One GPU; waits until it is free of other processes.
#   G=0 setsid nohup scripts/58_recipe_check.sh > logs/phase1/recipe_check.log 2>&1 < /dev/null & disown
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/gpu_env.sh
export PYTHONPATH=third_party/ext/distillation-game/src HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_OFFLINE=1 \
       TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
G=${G:?one GPU}; NEED_MIB=${NEED_MIB:-42000}
[[ $G == 0 || $G == 1 || $G == 3 ]] || { echo "GPU $G is not ours to use now (m8 §25: GPUs 0, 1, 3)"; exit 1; }
PY=.venv-dg/bin/python; CFG=configs/setup2/math_U_s123.yaml; D=outputs/phase0/S2full/math_s123
ts() { date '+%F %T'; }
idle() { [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$G" | tr -d ' ')" ]] \
         && (( $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$G" | tr -d ' ') >= NEED_MIB )); }
mkdir -p "$D" logs/phase1
echo "[$(ts)] recipe check on GPU $G"
if [[ ! -f $D/train_info.json ]]; then
  until idle; do sleep 120; done
  CUDA_VISIBLE_DEVICES=$G $(pin "$G") $PY scripts/dg_full_ft.py --config $CFG --student unsloth/Llama-3.2-3B-Instruct \
      --traces outputs/phase0/teacher32b_math/train_s123.json --out $D >> logs/phase1/fullft_math_s123.log 2>&1
  [[ -f $D/train_info.json ]] || { echo "[$(ts)] FAILED training :: $(grep -a Error logs/phase1/fullft_math_s123.log | tail -1 | cut -c1-200)"; exit 1; }
  echo "[$(ts)] trained: $(python3 -I -c "import json,sys; d=json.load(open(sys.argv[1])); print({k: d[k] for k in ('train_loss', 'train_minutes', 'peak_memory_gib')})" $D/train_info.json)"
fi
if [[ ! -f $D/test_8192.json ]]; then
  until idle; do sleep 120; done
  CUDA_VISIBLE_DEVICES=$G $(pin "$G") $PY scripts/dg_score_vllm.py --config $CFG --model $D/merged --max-new 8192 \
      --out $D/test_8192.json >> logs/phase1/fullft_math_s123_score.log 2>&1
  [[ -f $D/test_8192.json ]] || { echo "[$(ts)] FAILED scoring"; exit 1; }
fi
echo "[$(ts)] RECIPE CHECK DONE: $(python3 -I -c "import json,sys; print(json.load(open(sys.argv[1]))['summary'])" $D/test_8192.json)"
