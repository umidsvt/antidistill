#!/usr/bin/env bash
# Phase 1 on the puzzle testbed (results/m8_audit_design.md §25.4):
#   b5 s       RL (GRPO, exact agreement with the teacher's answer) from the B1 adapter of seed s, one epoch
#   c2 s       B1's recipe on the reference answers (C2_reference.jsonl), training seed s
#   c1 s       B1's recipe on the seed-123 majority answers (C1_majority_s123.jsonl), training seed s
#   b5e3 123   as b5, three epochs (development run, seed 123)
# Each finished student is scored with kk_eval.py. One worker per GPU in GPUS (only 0, 1, 3 are ours now, m8 §25);
# every job waits until its GPU is free of other processes.
#   GPUS="1 3 0" setsid nohup scripts/62_phase1_seeds.sh > logs/phase1/seeds.log 2>&1 < /dev/null & disown
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
idle() { [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$1" | tr -d ' ')" ]] \
         && (( $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" | tr -d ' ') >= NEED_MIB )); }
attempt() {   # <gpu> <marker> <log> <command...>
  local g=$1 marker=$2 log=$3; shift 3
  [[ -f "$marker" ]] && return 0
  until idle "$g"; do sleep 120; done
  mkdir -p "$(dirname "$log")"
  KK_MASTER_PORT=$((29760 + g)) CUDA_VISIBLE_DEVICES=$g $(pin "$g") "$@" >> "$log" 2>&1
  [[ -f "$marker" ]] || { echo "[$(ts)] FAILED on GPU $g: ${*: -6} :: $(grep -a -E 'Error' "$log" | tail -1 | cut -c1-200)"; return 1; }
}
score() {   # <gpu> <student dir> <seed>
  attempt "$1" "$2/test/summary.json" "$2/test.log" $PY scripts/kk_eval.py --puzzles $TEST --seed "$3" \
      --adapter "$2/adapter" --student $STUDENT --out "$2/test"
}
sft_eval() {   # <gpu> <train set> <student dir> <seed>
  attempt "$1" "$3/train_meta.json" "$3/train.log" $PY scripts/kk_sft.py --train-set "$2" --out "$3" --seed "$4" \
      --student $STUDENT && score "$1" "$3" "$4"
}
rl_eval() {   # <gpu> <student dir> <seed> <epochs>
  attempt "$1" "$2/train_meta.json" "$2/train.log" $PY scripts/kk_grpo.py --reward exact --seed "$3" --out "$2" \
      --init-adapter "$B/students/B1_s$3/adapter" --micro-batch 2 --epochs "$4" --save-every 25 && score "$1" "$2" "$3"
}
Q=$B/seeds_jobs.txt; QN=$B/seeds_jobs.next
[[ -f "$Q" ]] || printf 'b5e3 123\nb5 124\nb5 125\nc2 124\nc2 125\nc1 124\nc1 125\n' > "$Q"
[[ -f "$QN" ]] || echo 0 > "$QN"
next_job() { ( flock 9; local n; n=$(cat "$QN"); echo $((n + 1)) > "$QN"; sed -n "$((n + 1))p" "$Q" ) 9> "$B/seeds_jobs.lock"; }
job() {   # <gpu> <kind> <seed>
  local g=$1 kind=$2 s=$3
  case $kind in
    b5)   rl_eval "$g" "$B/students/B5_s$s" "$s" 1 ;;
    b5e3) rl_eval "$g" "$B/students/B5e3_s$s" "$s" 3 ;;
    c2)   sft_eval "$g" "$B/train_sets/C2_reference.jsonl" "$B/students/C2_s$s" "$s" ;;
    c1)   sft_eval "$g" "$B/train_sets/C1_majority_s123.jsonl" "$B/students/C1_s$s" "$s" ;;
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
echo "[$(ts)] Phase 1, seeds (m8 §25.4) on GPUs $GPUS"
for g in $GPUS; do worker "$g" & done
wait
echo "[$(ts)] PHASE 1 SEEDS DONE"
