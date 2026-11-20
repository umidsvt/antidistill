#!/usr/bin/env bash
# Phase 1 on the puzzle testbed (results/m8_audit_design.md §25.2): RL baselines from the untrained instruct student,
# rewarded by the teacher's released answers (scripts/kk_grpo.py), then scored like every student (kk_eval.py).
#   b4  partial credit (share of people labeled as the teacher did)
#   b3  exact agreement with the teacher's answer
# A GPU is used only when the earlier launcher on it is finished (GPU 1, 3: "GPU g: queue empty" in
# logs/phase1/kk.log; GPU 0: the recipe check done) and it is free of other processes. Only GPUs 0, 1, 3 are ours now
# (m8 §25).
#   GPUS="1 3 0" setsid nohup scripts/60_phase1_rl.sh > logs/phase1/rl.log 2>&1 < /dev/null & disown
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/gpu_env.sh
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv-dg/bin/python; B=outputs/kk_phase1; STUDENT=unsloth/Llama-3.2-3B-Instruct; TEST=data/kk/step2_test.jsonl
GPUS=${GPUS:-"1 3 0"}; NEED_MIB=${NEED_MIB:-42000}
for g in $GPUS; do [[ $g == 0 || $g == 1 || $g == 3 ]] || { echo "GPU $g is not ours to use now (m8 §25)"; exit 1; }; done
mkdir -p "$B/students" logs/phase1
ts() { date '+%F %T'; }
released() {   # the earlier launcher on GPU $1 is done with it
  case $1 in
    0) grep -q -E "RECIPE CHECK DONE|FAILED" logs/phase1/recipe_check.log ;;
    *) grep -q "GPU $1: queue empty" logs/phase1/kk.log ;;
  esac; }
idle() { [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$1" | tr -d ' ')" ]] \
         && (( $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" | tr -d ' ') >= NEED_MIB )); }
attempt() {   # <gpu> <marker> <log> <command...>
  local g=$1 marker=$2 log=$3; shift 3
  [[ -f "$marker" ]] && return 0
  until idle "$g"; do sleep 120; done
  mkdir -p "$(dirname "$log")"
  KK_MASTER_PORT=$((29700 + g)) CUDA_VISIBLE_DEVICES=$g $(pin "$g") "$@" >> "$log" 2>&1
  [[ -f "$marker" ]] || { echo "[$(ts)] FAILED on GPU $g: ${*: -6} :: $(grep -a -E 'Error' "$log" | tail -1 | cut -c1-200)"; return 1; }
}
Q=$B/rl_jobs.txt; QN=$B/rl_jobs.next
[[ -f "$Q" ]] || printf 'b4 123\nb3 123\n' > "$Q"
[[ -f "$QN" ]] || echo 0 > "$QN"
next_job() { ( flock 9; local n; n=$(cat "$QN"); echo $((n + 1)) > "$QN"; sed -n "$((n + 1))p" "$Q" ) 9> "$B/rl_jobs.lock"; }
job() {   # <gpu> <kind> <seed>
  local g=$1 kind=$2 s=$3 d reward
  case $kind in b4) reward=partial ;; b3) reward=exact ;; *) return 1 ;; esac
  d=$B/students/${kind^^}_s$s
  attempt "$g" "$d/train_meta.json" "$d/train.log" $PY scripts/kk_grpo.py --reward $reward --seed "$s" --out "$d" \
      --micro-batch 2 --epochs 1 --save-every 25 \
    && attempt "$g" "$d/test/summary.json" "$d/test.log" $PY scripts/kk_eval.py --puzzles $TEST --seed "$s" \
      --adapter "$d/adapter" --student $STUDENT --out "$d/test"
}
worker() {   # <gpu>
  local g=$1 line i ok
  until released "$g"; do sleep 300; done
  while line=$(next_job) && [[ -n "$line" ]]; do
    echo "[$(ts)] GPU $g: start $line"
    ok=1; for i in 1 2; do job "$g" $line && { ok=0; break; }; sleep 120; done
    (( ok == 0 )) && echo "[$(ts)] GPU $g: done  $line" || echo "[$(ts)] GPU $g: FAILED $line"
  done
  echo "[$(ts)] GPU $g: queue empty"
}
echo "[$(ts)] Phase 1 RL baselines (m8 §25.2) on GPUs $GPUS"
for g in $GPUS; do worker "$g" & done
wait
echo "[$(ts)] PHASE 1 RL DONE"
