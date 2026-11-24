#!/usr/bin/env bash
# Phase 2, wave 1 (results/m8_audit_design.md §29, results/phase2_ideas.md): answer-space curricula on the puzzle
# testbed, seed 123. One worker per GPU in GPUS (never GPU 2: kept for the user's other project); each job waits until
# its GPU is free of other processes.
#   GPUS="0 1 3 4 5 6 7" setsid nohup scripts/64_phase2_wave1.sh > logs/phase2/wave1.log 2>&1 < /dev/null & disown
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/gpu_env.sh
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv-dg/bin/python; B=outputs/kk_phase2; SETS=$B/sets; STUDENT=unsloth/Llama-3.2-3B-Instruct; S=123
GPUS=${GPUS:-"0 1 3 4 5 6 7"}; NEED_MIB=${NEED_MIB:-42000}
for g in $GPUS; do [[ $g != 2 ]] || { echo "GPU 2 is kept for the user's other project"; exit 1; }; done
mkdir -p "$B/students" logs/phase2
ts() { date '+%F %T'; }
idle() { [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$1" | tr -d ' ')" ]] \
         && (( $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" | tr -d ' ') >= NEED_MIB )); }
attempt() {   # <gpu> <marker> <log> <command...>
  local g=$1 marker=$2 log=$3; shift 3
  [[ -f "$marker" ]] && return 0
  until idle "$g"; do sleep 120; done
  mkdir -p "$(dirname "$log")"
  KK_MASTER_PORT=$((29780 + g)) CUDA_VISIBLE_DEVICES=$g $(pin "$g") "$@" >> "$log" 2>&1
  [[ -f "$marker" ]] || { echo "[$(ts)] FAILED on GPU $g: ${*: -6} :: $(grep -a -E 'Error' "$log" | tail -1 | cut -c1-200)"; return 1; }
}
evals() {   # <gpu> <student dir> [adapter]; the standard test, repair, guess -> repair, and infill for mask students
  local g=$1 d=$2 ad=${3:-$2/adapter} name file
  local adapter_args=(--adapter "$ad"); [[ "$ad" == none ]] && adapter_args=()
  for x in "test data/kk/step2_test.jsonl" "test_repair $SETS/test_repair_truth_k1.jsonl" \
           "test_guess $SETS/test_guessB5.jsonl" "test_mask $SETS/test_mask_truth_m1.jsonl"; do
    set -- $x; name=$1; file=$2
    attempt "$g" "$d/$name/summary.json" "$d/$name.log" $PY scripts/kk_eval.py --puzzles "$file" --seed $S \
        "${adapter_args[@]}" --student $STUDENT --out "$d/$name" || return 1
  done
}
rl() {   # <gpu> <name> <pool>
  local d=$B/students/$2_s$S
  attempt "$1" "$d/train_meta.json" "$d/train.log" $PY scripts/kk_grpo.py --reward exact --seed $S --out "$d" \
      --pool "$3" --temperature 0.6 --micro-batch 2 --epochs 1 --save-every 25 && evals "$1" "$d"
}
Q=$B/wave1_jobs.txt; QN=$B/wave1_jobs.next
[[ -f "$Q" ]] || printf 'rl I1_repair train_repair_k12\nrl I2_mask train_mask_m12\nrl I3_guess train_guessB1\nrl I5_mix train_mix\nstar I4_repairstar train_repair_k1\nbase I0 -\n' > "$Q"
[[ -f "$QN" ]] || echo 0 > "$QN"
next_job() { ( flock 9; local n; n=$(cat "$QN"); echo $((n + 1)) > "$QN"; sed -n "$((n + 1))p" "$Q" ) 9> "$B/wave1_jobs.lock"; }
job() {   # <gpu> <kind> <name> <set>
  local g=$1 kind=$2 name=$3 set=$4 d
  case $kind in
    rl)   rl "$g" "$name" "$SETS/$set.jsonl" ;;
    star) d=$B/students/${name}_s$S
          attempt "$g" "$B/star_repair_s$S/train_set.jsonl" "$B/star_repair_s$S.log" $PY scripts/kk_star.py \
              --pool "$SETS/$set.jsonl" --teacher-gen outputs/kk_step2/gen --seed $S --out "$B/star_repair_s$S" \
              --student $STUDENT --no-hint --temperature 0.6 --k 8 --max-tokens 2048 \
            && attempt "$g" "$d/train_meta.json" "$d/train.log" $PY scripts/kk_sft.py \
              --train-set "$B/star_repair_s$S/train_set.jsonl" --out "$d" --seed $S --student $STUDENT \
            && evals "$g" "$d" ;;
    base) evals "$g" "$B/students/I0_s$S" none ;;
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
echo "[$(ts)] Phase 2 wave 1 (m8 §29) on GPUs $GPUS"
for g in $GPUS; do worker "$g" & done
wait
echo "[$(ts)] PHASE 2 WAVE 1 DONE"
