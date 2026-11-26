#!/usr/bin/env bash
# Qwen2.5-3B-Instruct as the student (results/m8_audit_design.md §35), seed 123: Q-U (U_inst's recipe), Q-B3 (RL from
# the untrained student, teacher's answers) and Q-B3c (the same with shuffled targets). One lane per GPU (never GPU 2);
# each job waits for a GPU free of other processes.
#   setsid nohup scripts/76_qwen.sh > logs/phase2/qwen.log 2>&1 < /dev/null & disown
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/gpu_env.sh
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv-dg/bin/python; B=outputs/kk_phase2/students; Q=Qwen/Qwen2.5-3B-Instruct; S=123
TEST=data/kk/step2_test.jsonl; NEED_MIB=${NEED_MIB:-42000}
G_U=${G_U:-6} G_RL=${G_RL:-7} G_RLC=${G_RLC:-1}
for g in $G_U $G_RL $G_RLC; do [[ $g != 2 ]] || { echo "GPU 2 is kept"; exit 1; }; done
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
score() {   # <gpu> <adapter> <out>
  attempt $1 "$3/summary.json" "$3.log" $PY scripts/kk_eval.py --puzzles $TEST --seed $S --adapter "$2" --student $Q \
      --out "$3" || return 1
  echo "[$(ts)] $3: $($PY -c "import json; d=json.load(open('$3/summary.json')); print(d['solve_rate'], d['mean_tokens'])")"
}
lane_u() {
  local d=$B/QU_s$S
  attempt $G_U "$d/train_meta.json" "$d/train.log" $PY scripts/kk_sft.py --train-set outputs/kk_step2/train_sets/U.jsonl \
      --out "$d" --seed $S --student $Q || return 1
  score $G_U "$d/adapter" "$d/test"
}
lane_rl() {   # <gpu> <name> [extra]
  local g=$1 d=$B/$2_s$S; shift 2
  attempt $g "$d/train_meta.json" "$d/train.log" $PY scripts/kk_grpo.py --reward exact --student $Q --seed $S --out "$d" \
      --temperature 0.8 --lr 1e-5 --epochs 2 --micro-batch 2 --save-every 25 "$@" || return 1
  score $g "$d/ckpt_0100" "$d/test_ckpt0100"; score $g "$d/adapter" "$d/test"
}
echo "[$(ts)] Qwen runs started: Q-U GPU $G_U, Q-B3 GPU $G_RL, Q-B3c GPU $G_RLC"
lane_u & lane_rl $G_RL QB3 & lane_rl $G_RLC QB3c --shuffle-targets &
wait
echo "[$(ts)] QWEN DONE"
