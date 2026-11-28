#!/usr/bin/env bash
# Qwen's answers-only students on the knights puzzles (results/m8_audit_design.md §39 item 3), seed 123.
#   setsid nohup scripts/82_qwen_kk_answers.sh > logs/qwen_kk_answers.log 2>&1 < /dev/null & disown
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/gpu_env.sh
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv-dg/bin/python; NEED_MIB=42000; KS=outputs/kk_phase2/students; RF=outputs/kk_phase2/refine; Q=Qwen/Qwen2.5-3B-Instruct
GA=${GA:-6} GB=${GB:-7}
ts() { date '+%F %T'; }
idle() { [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$1" | tr -d ' ')" ]] \
         && (( $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" | tr -d ' ') >= NEED_MIB )); }
attempt() {
  local g=$1 marker=$2 log=$3 i; shift 3
  [[ -f "$marker" ]] && return 0
  mkdir -p "$(dirname "$log")"
  for i in 1 2; do
    until idle "$g"; do sleep 60; done
    KK_MASTER_PORT=$((29780 + g)) CUDA_VISIBLE_DEVICES=$g $(pin "$g") "$@" >> "$log" 2>&1
    [[ -f "$marker" ]] && return 0
    echo "[$(ts)] FAILED (try $i) on GPU $g: $log :: $(grep -a -E 'Error' "$log" | tail -1 | cut -c1-200)"; sleep 120
  done
  return 1
}
rate() { $PY -c "import json; print(json.load(open('$1'))['solve_rate'])"; }
ev() { attempt $1 $3/summary.json $3.log $PY scripts/kk_eval.py --puzzles data/kk/step2_test.jsonl --seed 123 --adapter $2 \
         --student $Q --out $3 && echo "[$(ts)] $3: $(rate $3/summary.json)"; }
rl() { attempt $1 $3/train_meta.json $3/train.log $PY scripts/kk_grpo.py --reward exact --student $Q --seed 123 --out $3 \
         --init-adapter $2 --micro-batch 2 --epochs 1 --save-every 25; }
lane_b1() {
  local d=$KS/QB1_s123 r=$KS/QB5_s123
  attempt $GA $d/train_meta.json $d/train.log $PY scripts/kk_sft.py --train-set outputs/kk_phase1/train_sets/B1_answers.jsonl \
      --out $d --seed 123 --student $Q || return 1
  ev $GA $d/adapter $d/test || return 1
  rl $GA $d/adapter $r && ev $GA $r/adapter $r/test
}
lane_i6c() {
  local d=$KS/QI6c2_rename_s123 r=$KS/QI6c2_renamerl_s123 t
  attempt $GB $d/train_meta.json $d/train.log $PY scripts/kk_sft.py --train-set outputs/kk_phase2/sets/sft_rename3.jsonl \
      --out $d --seed 123 --student $Q --lr 2e-4 || return 1
  ev $GB $d/adapter $d/test || return 1
  rl $GB $d/adapter $r && ev $GB $r/adapter $r/test || return 1
  until grep -q "chosen T" logs/rerun_lr.log; do grep -q "lane 123 failed" logs/rerun_lr.log && return 1; sleep 120; done
  t=$(grep -o "chosen T = [0-9]" logs/rerun_lr.log | tail -1 | grep -o "[0-9]$")
  attempt $GB $RF/test_QI6c2_renamerl_s123_tta$t/summary.json $RF/test_QI6c2_renamerl_s123_tta$t.log $PY scripts/kk_refine.py \
      --adapter $r/adapter --student $Q --puzzles data/kk/step2_test.jsonl --out $RF/test_QI6c2_renamerl_s123_tta$t --tta $t \
      --sweeps 0 --seed 123 && echo "[$(ts)] Qwen I6c2 + RL, votes T=$t: $($PY -c "import json; print(json.load(open('$RF/test_QI6c2_renamerl_s123_tta$t/summary.json'))['accuracy by step']['solve'])")"
}
echo "[$(ts)] Qwen knights answers-only started: B1/B5 GPU $GA, renamed copies GPU $GB"
{ lane_b1 || echo "[$(ts)] lane B1 failed"; } & { lane_i6c || echo "[$(ts)] lane I6c failed"; } &
wait
echo "[$(ts)] QWEN KK DONE"
