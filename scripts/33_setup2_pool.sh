#!/usr/bin/env bash
# Setup-2 job pool (results/m8_audit_design.md §9.4): each GPU worker takes the next free job from a list,
# waits until its GPU has enough free memory, and retries a failed job twice (10 min apart) instead of skipping.
#   JOBS=outputs/setup2/math_jobs.txt GPUS=4,5,6,7 setsid nohup scripts/33_setup2_pool.sh > logs/setup2_math_pool.log 2>&1 < /dev/null & disown
# Job lines: "base <config stem>" (score the untrained student) or "distill <config stem>" (the full pipeline).
# A GPU listed in LATE_GPUS joins only once it has enough free memory (e.g. after another job on it ends).
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=third_party/ext/distillation-game/src HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
       HF_HOME=/mnt/vault/huggingface_cache TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv-dg/bin/python; OUT=outputs/setup2; NEED_MIB=${NEED_MIB:-36000}
JOBS="${JOBS:?JOBS=<job list>}"; LOCKS="${JOBS%.txt}_locks"
G="${GPUS:?GPUS=...}"; LATE="${LATE_GPUS:-}"
for g in ${G//,/ } ${LATE//,/ }; do [[ $g == 0 || $g == 1 ]] && { echo "GPUs 0 and 1 are reserved"; exit 1; }; done
ts() { date '+%F %T'; }
latest() { ls -dt "$OUT"/"$1"_[0-9]* 2>/dev/null | head -1; }
free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" | tr -d ' '; }
wait_free() { while (( $(free_mib "$1") < NEED_MIB )); do sleep 300; done; }
done_job() {   # $1 kind, $2 stem
  local seed="${2##*_s}" ds="${2%%_*}"
  if [[ $1 == base ]]; then [[ -f "$OUT/base_${ds}_s${seed}.json" ]]; else [[ -f "$(latest "$2")/RESULTS.md" ]]; fi
}
run_job() {    # $1 gpu, $2 kind, $3 stem
  if [[ $2 == base ]]; then
    CUDA_VISIBLE_DEVICES=$1 $PY scripts/dg_eval_base.py "configs/setup2/$3.yaml" > "logs/setup2_base_$3.log" 2>&1
  else
    CUDA_VISIBLE_DEVICES=$1 $PY third_party/ext/distillation-game/scripts/run_pipeline.py --config "configs/setup2/$3.yaml" \
      > "logs/setup2_$3.log" 2>&1
  fi
}
result() {     # $1 kind, $2 stem
  if [[ $1 == base ]]; then tail -n 1 "logs/setup2_base_$2.log" | cut -c1-120
  else grep -a -A4 'Comparison Matrix' "$(latest "$2")/RESULTS.md" 2>/dev/null | tail -1; fi
}
worker() {     # $1 gpu
  local kind stem i
  while read -r kind stem; do
    [[ -z "$kind" ]] && continue
    mkdir "$LOCKS/$kind.$stem" 2> /dev/null || continue
    done_job "$kind" "$stem" && { echo "[$(ts)] SKIP  $kind $stem (done)"; continue; }
    for i in 1 2 3; do
      wait_free "$1"
      echo "[$(ts)] START $kind $stem on GPU $1 (attempt $i)"
      run_job "$1" "$kind" "$stem"
      if done_job "$kind" "$stem"; then echo "[$(ts)] DONE  $kind $stem: $(result "$kind" "$stem")"; break; fi
      echo "[$(ts)] FAILED $kind $stem on GPU $1 (attempt $i)"; sleep 600
    done
  done < "$JOBS"
  echo "[$(ts)] worker GPU $1: no jobs left"
}
rm -rf "$LOCKS"; mkdir -p "$LOCKS"
echo "[$(ts)] pool on GPUs $G${LATE:+ (+ $LATE when free)}: $JOBS"
pids=()
for g in ${G//,/ }; do worker "$g" & pids+=($!); done
for g in ${LATE//,/ }; do ( wait_free "$g"; echo "[$(ts)] GPU $g joins"; worker "$g" ) & pids+=($!); done
wait "${pids[@]}"
echo "[$(ts)] POOL DONE: $JOBS"
