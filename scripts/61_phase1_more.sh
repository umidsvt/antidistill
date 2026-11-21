#!/usr/bin/env bash
# Phase 1 on the puzzle testbed, continued (results/m8_audit_design.md §25.3):
#   b5 123      RL (GRPO, exact agreement with the teacher's answer) starting from B1 s123's adapter
#   sample 123  8 tries per training puzzle from the untrained instruct student (for C1)
#   b1 124/125  B1 (SFT on the teacher's answers alone) on two more seeds
#   c2 123      B1's recipe on the reference answers of the same 1,712 puzzles
#   c1 123      B1's recipe on the student's own majority answers (waits for sample 123)
# Each finished student is scored with kk_eval.py. One worker per GPU in GPUS (only 0, 1, 3 are ours now, m8 §25);
# every job waits until its GPU is free of other processes.
#   GPUS="1 3 0" setsid nohup scripts/61_phase1_more.sh > logs/phase1/more.log 2>&1 < /dev/null & disown
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/gpu_env.sh
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv-dg/bin/python; B=outputs/kk_phase1; STUDENT=unsloth/Llama-3.2-3B-Instruct; TEST=data/kk/step2_test.jsonl
POOL=data/kk/step2_train.jsonl; GEN=outputs/kk_step2/gen
GPUS=${GPUS:-"1 3 0"}; NEED_MIB=${NEED_MIB:-42000}
for g in $GPUS; do [[ $g == 0 || $g == 1 || $g == 3 ]] || { echo "GPU $g is not ours to use now (m8 §25)"; exit 1; }; done
mkdir -p "$B/students" logs/phase1
ts() { date '+%F %T'; }
idle() { [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$1" | tr -d ' ')" ]] \
         && (( $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" | tr -d ' ') >= NEED_MIB )); }
attempt() {   # <gpu> <marker> <log> <command...>
  local g=$1 marker=$2 log=$3; shift 3
  [[ -f "$marker" ]] && return 0
  until idle "$g"; do sleep 120; done
  mkdir -p "$(dirname "$log")"
  KK_MASTER_PORT=$((29740 + g)) CUDA_VISIBLE_DEVICES=$g $(pin "$g") "$@" >> "$log" 2>&1
  [[ -f "$marker" ]] || { echo "[$(ts)] FAILED on GPU $g: ${*: -6} :: $(grep -a -E 'Error' "$log" | tail -1 | cut -c1-200)"; return 1; }
}
sft_eval() {   # <gpu> <train set> <student dir> <seed>
  local g=$1 set=$2 d=$3 s=$4
  attempt "$g" "$d/train_meta.json" "$d/train.log" $PY scripts/kk_sft.py --train-set "$set" --out "$d" --seed "$s" \
      --student $STUDENT \
    && attempt "$g" "$d/test/summary.json" "$d/test.log" $PY scripts/kk_eval.py --puzzles $TEST --seed "$s" \
      --adapter "$d/adapter" --student $STUDENT --out "$d/test"
}
Q=$B/more_jobs.txt; QN=$B/more_jobs.next
[[ -f "$Q" ]] || printf 'b5 123\nsample 123\nb1 124\nb1 125\nc2 123\nc1 123\n' > "$Q"
[[ -f "$QN" ]] || echo 0 > "$QN"
next_job() { ( flock 9; local n; n=$(cat "$QN"); echo $((n + 1)) > "$QN"; sed -n "$((n + 1))p" "$Q" ) 9> "$B/more_jobs.lock"; }
job() {   # <gpu> <kind> <seed>
  local g=$1 kind=$2 s=$3 d
  case $kind in
    b5) d=$B/students/B5_s$s
        attempt "$g" "$d/train_meta.json" "$d/train.log" $PY scripts/kk_grpo.py --reward exact --seed "$s" --out "$d" \
            --init-adapter "$B/students/B1_s$s/adapter" --micro-batch 2 --epochs 1 --save-every 25 \
          && attempt "$g" "$d/test/summary.json" "$d/test.log" $PY scripts/kk_eval.py --puzzles $TEST --seed "$s" \
            --adapter "$d/adapter" --student $STUDENT --out "$d/test" ;;
    sample) attempt "$g" "$B/own_tries_s$s/summary.json" "$B/own_tries_s$s.log" $PY scripts/kk_sample.py --seed "$s" \
            --out "$B/own_tries_s$s" ;;
    b1) sft_eval "$g" "$B/train_sets/B1_answers.jsonl" "$B/students/B1_s$s" "$s" ;;
    c2) python3 -I scripts/kk_phase1_sets.py reference --pool $POOL --teacher-gen $GEN \
            --out "$B/train_sets/C2_reference.jsonl" > /dev/null \
          && sft_eval "$g" "$B/train_sets/C2_reference.jsonl" "$B/students/C2_s$s" "$s" ;;
    c1) until [[ -f "$B/own_tries_s$s/summary.json" ]] || grep -q "FAILED sample $s" logs/phase1/more.log; do sleep 120; done
        [[ -f "$B/own_tries_s$s/summary.json" ]] || return 1
        python3 -I scripts/kk_phase1_sets.py majority --tries "$B/own_tries_s$s/tries.jsonl" --pool $POOL --teacher-gen $GEN \
            --out "$B/train_sets/C1_majority_s$s.jsonl" > /dev/null \
          && sft_eval "$g" "$B/train_sets/C1_majority_s$s.jsonl" "$B/students/C1_s$s" "$s" ;;
    *) return 1 ;;
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
echo "[$(ts)] Phase 1, continued (m8 §25.3) on GPUs $GPUS"
for g in $GPUS; do worker "$g" & done
wait
echo "[$(ts)] PHASE 1 MORE DONE"
