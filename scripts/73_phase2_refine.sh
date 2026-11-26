#!/usr/bin/env bash
# Phase 2, I9-I11 (results/m8_audit_design.md §33): answer-space refinement, test 2. One lane per GPU (never GPU 2);
# each job waits for a GPU free of other processes.
#   setsid nohup scripts/73_phase2_refine.sh > logs/phase2/refine.log 2>&1 < /dev/null & disown
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/gpu_env.sh
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv-dg/bin/python; B=outputs/kk_phase2; SETS=$B/sets; R=$B/refine; STUDENT=unsloth/Llama-3.2-3B-Instruct; S=123
TEST=data/kk/step2_test.jsonl; NEED_MIB=${NEED_MIB:-42000}; SCHED=1,1,1,2,3
G_BLK=${G_BLK:-3} G_RL=${G_RL:-7} G_DENSE=${G_DENSE:-6}
for g in $G_BLK $G_RL $G_DENSE; do [[ $g != 2 ]] || { echo "GPU 2 is kept"; exit 1; }; done
mkdir -p $R
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
refine() {   # <gpu> <adapter> <name> <schedule> [extra]
  local g=$1 ad=$2 name=$3 sch=$4; shift 4
  attempt $g "$R/$name/summary.json" "$R/$name.log" $PY scripts/kk_refine.py --adapter "$ad" --out "$R/$name" \
      --schedule "$sch" "$@" || return 1
  echo "[$(ts)] refine $name: $($PY -c "import json; d=json.load(open('$R/$name/summary.json')); print(d['accuracy by step'], d['mean generated tokens per puzzle'], d.get('truth check: reference answer is a fixed point'))")"
}
solve() {   # <gpu> <student dir>
  attempt $1 "$2/test/summary.json" "$2/test.log" $PY scripts/kk_eval.py --puzzles $TEST --seed $S --adapter "$2/adapter" \
      --student $STUDENT --out "$2/test" || return 1
  echo "[$(ts)] solve $(basename $2): $($PY -c "import json; print(json.load(open('$2/test/summary.json'))['solve_rate'])")"
}
views_rl() {   # <gpu> <init adapter> <out dir>: RL on plain + infill views, hidden-person reward, 2 epochs
  attempt $1 "$3/train_meta.json" "$3/train.log" $PY scripts/kk_grpo.py --reward hidden --seed $S --out "$3" \
      --pool $SETS/train_views_rl.jsonl --init-adapter "$2" --micro-batch 2 --epochs 2 --save-every 50
}
lane_blk() { refine $G_BLK $B/students/I6_viewsrl_s123/adapter I6_viewsrl_s123_blocks 1,1,2,2,3,3; }
lane_rl() {
  local d=$B/students/I10_viewsrl_s$S
  views_rl $G_RL $B/students/I6_views_s123/adapter $d || return 1
  solve $G_RL $d; refine $G_RL $d/adapter I10_viewsrl_s$S $SCHED --truth-check
}
lane_dense() {
  local d=$B/students/I11_dense_s$S r=$B/students/I11_denserl_s$S
  attempt $G_DENSE "$d/train_meta.json" "$d/train.log" $PY scripts/kk_sft.py --train-set $SETS/sft_dense.jsonl --out "$d" \
      --seed $S --student $STUDENT || return 1
  solve $G_DENSE $d; refine $G_DENSE $d/adapter I11_dense_s$S $SCHED --truth-check
  views_rl $G_DENSE $d/adapter $r || return 1
  solve $G_DENSE $r; refine $G_DENSE $r/adapter I11_denserl_s$S $SCHED --truth-check
}
echo "[$(ts)] refine test 2 started: blocks GPU $G_BLK, views RL GPU $G_RL, dense views GPU $G_DENSE"
lane_blk & lane_rl & lane_dense &
wait
echo "[$(ts)] REFINE2 DONE"
