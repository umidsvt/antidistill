#!/usr/bin/env bash
# Phase 1 on the puzzle testbed (results/m8_audit_design.md §25.2): answers-only baselines with the instruction-tuned
# student, seed 123.
#   b1  SFT on the teacher's answers alone (outputs/kk_phase1/train_sets/B1_answers.jsonl), then the test puzzles
#   b2  self-training with hints (scripts/kk_star.py, instruct mode), SFT on the kept texts, then the test puzzles
# One worker per GPU in GPUS (only 0, 1, 3 are ours now: m8 §25); each job waits until its GPU is free of other
# processes and is tried up to twice.
#   GPUS="3 1" setsid nohup scripts/59_phase1_kk.sh > logs/phase1/kk.log 2>&1 < /dev/null & disown
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/gpu_env.sh
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn
PY=.venv-dg/bin/python; B=outputs/kk_phase1; STUDENT=unsloth/Llama-3.2-3B-Instruct; TEST=data/kk/step2_test.jsonl
GPUS=${GPUS:-"3 1"}; NEED_MIB=${NEED_MIB:-42000}
for g in $GPUS; do [[ $g == 0 || $g == 1 || $g == 3 ]] || { echo "GPU $g is not ours to use now (m8 §25)"; exit 1; }; done
mkdir -p "$B/students" logs/phase1
ts() { date '+%F %T'; }
idle() { [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$1" | tr -d ' ')" ]] \
         && (( $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" | tr -d ' ') >= NEED_MIB )); }
attempt() {   # <gpu> <marker> <log> <command...>: one run after the GPU is free; 0 if the marker exists afterwards
  local g=$1 marker=$2 log=$3; shift 3
  [[ -f "$marker" ]] && return 0
  until idle "$g"; do sleep 120; done
  mkdir -p "$(dirname "$log")"
  CUDA_VISIBLE_DEVICES=$g $(pin "$g") "$@" >> "$log" 2>&1
  [[ -f "$marker" ]] || { echo "[$(ts)] FAILED on GPU $g: ${*: -6} :: $(grep -a -E 'Error' "$log" | tail -1 | cut -c1-200)"; return 1; }
}
sft_eval() {   # <gpu> <train set> <student dir> <seed>
  local g=$1 set=$2 d=$3 s=$4
  attempt "$g" "$d/train_meta.json" "$d/train.log" $PY scripts/kk_sft.py --train-set "$set" --out "$d" --seed "$s" \
      --student $STUDENT \
    && attempt "$g" "$d/test/summary.json" "$d/test.log" $PY scripts/kk_eval.py --puzzles $TEST --seed "$s" \
      --adapter "$d/adapter" --student $STUDENT --out "$d/test"
}
Q=$B/jobs.txt; QN=$B/jobs.next
[[ -f "$Q" ]] || printf 'b1 123\nb2 123\n' > "$Q"
[[ -f "$QN" ]] || echo 0 > "$QN"
next_job() { ( flock 9; local n; n=$(cat "$QN"); echo $((n + 1)) > "$QN"; sed -n "$((n + 1))p" "$Q" ) 9> "$B/jobs.lock"; }
job() {   # <gpu> <kind> <seed>
  local g=$1 kind=$2 s=$3
  case $kind in
    b1) sft_eval "$g" "$B/train_sets/B1_answers.jsonl" "$B/students/B1_s$s" "$s" ;;
    b2) attempt "$g" "$B/star_s$s/train_set.jsonl" "$B/star_s$s.log" $PY scripts/kk_star.py --pool data/kk/step2_train.jsonl \
            --teacher-gen outputs/kk_step2/gen --seed "$s" --out "$B/star_s$s" --student $STUDENT \
          && sft_eval "$g" "$B/star_s$s/train_set.jsonl" "$B/students/B2_s$s" "$s" ;;
  esac
}
worker() {   # <gpu>
  local g=$1 line i ok
  while line=$(next_job) && [[ -n "$line" ]]; do
    echo "[$(ts)] GPU $g: start $line"
    ok=1; for i in 1 2; do job "$g" $line && { ok=0; break; }; sleep 120; done
    (( ok == 0 )) && echo "[$(ts)] GPU $g: done  $line" || echo "[$(ts)] GPU $g: FAILED $line"
  done
  echo "[$(ts)] GPU $g: queue empty"
}
echo "[$(ts)] Phase 1 on the puzzles (m8 §25.2) on GPUs $GPUS"
for g in $GPUS; do worker "$g" & done
wait
echo "[$(ts)] PHASE 1 KK DONE"
