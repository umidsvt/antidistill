#!/usr/bin/env bash
# Phase 2 wave 1: the evaluations that failed or never ran when another user's jobs took the GPUs between our runs
# (m8 §29, 2026-11-25). One worker per GPU in GPUS (never GPU 2); each evaluation waits until its GPU is free.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/gpu_env.sh
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn
PY=.venv-dg/bin/python; B=outputs/kk_phase2; SETS=$B/sets; GPUS=${GPUS:-"1 3 6 7"}
ts() { date '+%F %T'; }
idle() { [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$1" | tr -d ' ')" ]] \
         && (( $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" | tr -d ' ') >= 42000 )); }
Q=$B/fixup_jobs.txt; QN=$B/fixup_jobs.next
if [[ ! -f "$Q" ]]; then
  for d in I1_repair I1_repair_lr2e5 I2_mask I2b_mask_m1234; do
    for x in "test_repair test_repair_truth_k1" "test_guess test_guessB5" "test_mask test_mask_truth_m1"; do
      set -- $x; [[ -f $B/students/${d}_s123/$1/summary.json ]] || echo "$d $1 $2"; done; done > "$Q"
  echo 0 > "$QN"
fi
next_job() { ( flock 9; local n; n=$(cat "$QN"); echo $((n + 1)) > "$QN"; sed -n "$((n + 1))p" "$Q" ) 9> "$B/fixup_jobs.lock"; }
worker() {
  local g=$1 line d t f i
  while line=$(next_job) && [[ -n "$line" ]]; do
    set -- $line; d=$B/students/$1_s123; t=$2; f=$SETS/$3.jsonl
    for i in 1 2 3; do
      [[ -f $d/$t/summary.json ]] && break
      until idle "$g"; do sleep 60; done
      CUDA_VISIBLE_DEVICES=$g $(pin "$g") $PY scripts/kk_eval.py --puzzles "$f" --seed 123 --adapter "$d/adapter" \
          --student unsloth/Llama-3.2-3B-Instruct --out "$d/$t" >> "$d/$t.log" 2>&1
    done
    [[ -f $d/$t/summary.json ]] && echo "[$(ts)] GPU $g: done $line" || echo "[$(ts)] GPU $g: FAILED $line"
  done
}
echo "[$(ts)] fix-up evaluations on GPUs $GPUS"
for g in $GPUS; do worker "$g" & done
wait
echo "[$(ts)] FIXUP DONE"
