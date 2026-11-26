#!/usr/bin/env bash
# Phase 2, wave 3b (results/m8_audit_design.md §32): renamed-copy augmentation (I6c) on seeds 124-125, and the
# identical-copy control (I6d) on seed 123. One lane per GPU (never GPU 2); each job waits for a free GPU.
#   setsid nohup scripts/71_phase2_wave3b.sh > logs/phase2/wave3b.log 2>&1 < /dev/null & disown
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/gpu_env.sh
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv-dg/bin/python; B=outputs/kk_phase2; SETS=$B/sets; STUDENT=unsloth/Llama-3.2-3B-Instruct
TEST=data/kk/step2_test.jsonl; NEED_MIB=${NEED_MIB:-42000}
LANES=${LANES:-"4:I6d_dup:sft_dup3:123 6:I6c_rename:sft_rename3:124 7:I6c_rename:sft_rename3:125"}
ts() { date '+%F %T'; }
idle() { [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$1" | tr -d ' ')" ]] \
         && (( $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" | tr -d ' ') >= NEED_MIB )); }
attempt() {   # <gpu> <marker> <log> <command...>: up to two tries, each after the GPU is free
  local g=$1 marker=$2 log=$3 i; shift 3
  [[ -f "$marker" ]] && return 0
  mkdir -p "$(dirname "$log")"
  for i in 1 2; do
    until idle "$g"; do sleep 120; done
    KK_MASTER_PORT=$((29780 + g)) CUDA_VISIBLE_DEVICES=$g $(pin "$g") "$@" >> "$log" 2>&1
    [[ -f "$marker" ]] && return 0
    echo "[$(ts)] FAILED (try $i) on GPU $g: $log :: $(grep -a -E 'Error' "$log" | tail -1 | cut -c1-200)"
    sleep 300
  done
  return 1
}
rate() { $PY -c "import json; print(json.load(open('$1'))['solve_rate'])"; }
lane() {   # <gpu> <name> <set> <seed>: B1's SFT recipe on the set, then B5's RL from that adapter
  local g=$1 name=$2 set=$SETS/$3.jsonl s=$4 d r
  d=$B/students/${name}_s$s; r=$B/students/${name}rl_s$s
  attempt $g "$d/train_meta.json" "$d/train.log" $PY scripts/kk_sft.py --train-set "$set" --out "$d" --seed $s \
      --student $STUDENT || return 1
  attempt $g "$d/test/summary.json" "$d/test.log" $PY scripts/kk_eval.py --puzzles $TEST --seed $s --adapter "$d/adapter" \
      --student $STUDENT --out "$d/test" || return 1
  echo "[$(ts)] ${name} s$s SFT: solve $(rate $d/test/summary.json)"
  attempt $g "$r/train_meta.json" "$r/train.log" $PY scripts/kk_grpo.py --reward exact --seed $s --out "$r" \
      --init-adapter "$d/adapter" --micro-batch 2 --epochs 1 --save-every 25 || return 1
  attempt $g "$r/test/summary.json" "$r/test.log" $PY scripts/kk_eval.py --puzzles $TEST --seed $s --adapter "$r/adapter" \
      --student $STUDENT --out "$r/test" || return 1
  echo "[$(ts)] ${name}rl s$s: solve $(rate $r/test/summary.json)"
}
for l in $LANES; do IFS=: read -r g name set s <<< "$l"; [[ $g != 2 ]] || { echo "GPU 2 is kept"; exit 1; }; done
echo "[$(ts)] wave 3b started: $LANES"
for l in $LANES; do IFS=: read -r g name set s <<< "$l"; lane "$g" "$name" "$set" "$s" & done
wait
echo "[$(ts)] WAVE3B DONE"
