#!/usr/bin/env bash
# I13 stage 1 (results/m8_audit_design.md §38): cryptarithm infill RL with the hidden-letter reward, Qwen on GPU G_Q and
# Llama on GPU G_B; then scoring on the probe sets and 300 test puzzles.
#   setsid nohup scripts/80_rg_infill_rl.sh > logs/rg/infill_rl.log 2>&1 < /dev/null & disown
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/gpu_env.sh
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv-dg/bin/python; O=outputs/rg/crypt/students; G_Q=${G_Q:-2} G_B=${G_B:-7}; NEED_MIB=42000
ts() { date '+%F %T'; }
idle() { [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$1" | tr -d ' ')" ]] \
         && (( $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" | tr -d ' ') >= NEED_MIB )); }
run() {   # <gpu> <marker> <log> <command...>
  local g=$1 marker=$2 log=$3; shift 3
  [[ -f $marker ]] && return 0
  until idle $g; do sleep 60; done
  mkdir -p "$(dirname $log)"
  KK_MASTER_PORT=$((29780 + g)) CUDA_VISIBLE_DEVICES=$g $(pin $g) "$@" >> $log 2>&1
  [[ -f $marker ]] || { echo "[$(ts)] FAILED on GPU $g: $log :: $(grep -a -E 'Error' $log | tail -1 | cut -c1-200)"; return 1; }
}
lane() {   # <gpu> <model key> <model path>
  local g=$1 k=$2 mp=$3 d=$O/I13s1_${2}_s123 m
  run $g $d/train_meta.json $d/train.log $PY scripts/kk_grpo.py --reward hidden --rg-teacher outputs/rg/crypt/T_train/tries.jsonl \
      --pool data/rg/crypt_train_m12.jsonl --student $mp --seed 123 --out $d --lr 1e-5 --temperature 0.8 --epochs 1 \
      --micro-batch 2 --max-completion 3072 --save-every 25 || return 1
  for m in 1 2 3; do
    run $g $d/probe_m$m/summary.json $d/probe_m$m.log $PY scripts/rg_pilot.py --model $k --k 2 --adapter $d/adapter \
        --puzzles data/rg/crypt_probe_m$m.jsonl --out $d/probe_m$m || return 1
    echo "[$(ts)] I13 stage 1 $k m=$m: $(python3 -I scripts/rg_hidden_acc.py data/rg/crypt_probe_m$m.jsonl $d/probe_m$m/tries.jsonl)"
  done
  head -300 data/rg/crypt_test.jsonl > $d/test300.jsonl
  run $g $d/test300/summary.json $d/test300.log $PY scripts/rg_pilot.py --model $k --max-tokens 8192 --adapter $d/adapter \
      --puzzles $d/test300.jsonl --out $d/test300 || return 1
  echo "[$(ts)] I13 stage 1 $k solve (300 test): $($PY -c "import json; d=json.load(open('$d/test300/summary.json'))['by_level']; print({k: (v['solve_rate'], v['cell_accuracy']) for k, v in d.items()})")"
}
echo "[$(ts)] I13 stage 1 started: Qwen GPU $G_Q, Llama GPU $G_B"
lane $G_Q Q Qwen/Qwen2.5-3B-Instruct & lane $G_B B unsloth/Llama-3.2-3B-Instruct &
wait
echo "[$(ts)] I13 STAGE1 DONE"
