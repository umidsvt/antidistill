#!/usr/bin/env bash
# Phase 0 on the puzzle testbed (results/m8_audit_design.md §24, addendum of 2026-11-19): the instruction-tuned student
# Llama-3.2-3B-Instruct, untrained (I0) and trained on the teacher's 2,000 step-2 solutions (U_inst, §18 recipe), seeds
# 123-125, scored on the 1,000 test puzzles; and the teacher (R1-Distill-Qwen-32B) on 6 test puzzles, for the examples
# file only (results/phase0_examples.md).
# GPUs: only GPUs that Phase 0's queue (scripts/56_phase0.sh) has released ("GPU g: queue empty" in logs/phase0.log),
# and GPUs 1-3, which Phase 0 does not use; every job waits until its GPU is free of other processes. Per-GPU locks
# keep this launcher's own jobs apart.
#   setsid nohup scripts/57_kk_instruct.sh > logs/kk_instruct.log 2>&1 < /dev/null & disown
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/gpu_env.sh
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn
PY=.venv-dg/bin/python; BASE=outputs/kk_step2; STUDENT=unsloth/Llama-3.2-3B-Instruct; TEST=data/kk/step2_test.jsonl
NEED_MIB=${NEED_MIB:-42000}; L=$BASE/locks; mkdir -p "$L" "$BASE/examples"
ts() { date '+%F %T'; }
allowed() { [[ $1 == 1 || $1 == 2 || $1 == 3 ]] || grep -q "GPU $1: queue empty" logs/phase0.log; }
idle() { [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$1" | tr -d ' ')" ]] \
         && (( $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" | tr -d ' ') >= NEED_MIB )); }
grab() { allowed "$1" && idle "$1" && mkdir "$L/gpu$1" 2> /dev/null; }   # 0 when GPU $1 is now ours
release() { rmdir "$L/gpu$1" 2> /dev/null; }
attempt() {   # <gpus> <marker> <log> <command...>: one run; 0 if the marker exists afterwards
  local gl=$1 marker=$2 log=$3; shift 3
  [[ -f "$marker" ]] && return 0
  mkdir -p "$(dirname "$log")"
  CUDA_VISIBLE_DEVICES=$gl $(pin "$gl") "$@" >> "$log" 2>&1
  [[ -f "$marker" ]] || { echo "[$(ts)] FAILED on GPUs $gl: ${*: -8} :: $(grep -a -E 'Error' "$log" | tail -1 | cut -c1-200)"; return 1; }
}

Q=$BASE/instruct_jobs.txt; QN=$BASE/instruct_jobs.next
[[ -f "$Q" ]] || { for s in 123 124 125; do echo "u $s"; done; for s in 123 124 125; do echo "i0 $s"; done; } > "$Q"
[[ -f "$QN" ]] || echo 0 > "$QN"
next_job() { ( flock 9; local n; n=$(cat "$QN"); echo $((n + 1)) > "$QN"; sed -n "$((n + 1))p" "$Q" ) 9> "$BASE/instruct_jobs.lock"; }
job() {   # <gpu> <kind> <seed>
  local g=$1 kind=$2 s=$3 d
  case $kind in
    u)  d=$BASE/students/Uinst_s$s
        attempt "$g" "$d/train_meta.json" "$d/train.log" $PY scripts/kk_sft.py --train-set "$BASE/train_sets/U.jsonl" \
            --out "$d" --seed "$s" --student $STUDENT \
          && attempt "$g" "$d/test/summary.json" "$d/test.log" $PY scripts/kk_eval.py --puzzles "$TEST" --seed "$s" \
            --adapter "$d/adapter" --student $STUDENT --out "$d/test" ;;
    i0) d=$BASE/students/untrained_inst_s$s
        attempt "$g" "$d/test/summary.json" "$d/test.log" $PY scripts/kk_eval.py --puzzles "$TEST" --seed "$s" \
            --student $STUDENT --out "$d/test" ;;
  esac
}
worker() {   # <gpu>: one job at a time on GPU $1, each tried up to twice
  local g=$1 line i ok
  while :; do
    until grab "$g"; do sleep 120; done
    line=$(next_job)
    [[ -z "$line" ]] && { release "$g"; break; }
    echo "[$(ts)] GPU $g: start $line"
    # (until 2026-11-20 the loop's status was that of its last `sleep`, so a job that failed twice was logged "done")
    ok=1; for i in 1 2; do job "$g" $line && { ok=0; break; }; sleep 120; done
    (( ok == 0 )) && echo "[$(ts)] GPU $g: done  $line" || echo "[$(ts)] GPU $g: FAILED $line"
    release "$g"
  done
}
teacher_examples() {   # the teacher on 6 test puzzles (3 at N = 7, 3 at N = 8) on a free pair of one group
  local d=$BASE/examples/T_test p a b
  { grep '"n_people": 7' "$TEST" | head -3; grep '"n_people": 8' "$TEST" | head -3; } > "$BASE/examples/test6.jsonl"
  while [[ ! -f "$d/summary.json" ]]; do
    for p in "0 1" "1 2" "2 3" "0 2" "0 3" "1 3" "4 5" "6 7" "4 6" "5 7" "4 7" "5 6"; do
      set -- $p; a=$1; b=$2
      if grab "$a"; then
        if grab "$b"; then
          echo "[$(ts)] teacher examples on GPUs $a,$b"
          attempt "$a,$b" "$d/summary.json" "$BASE/examples/T_test.log" $PY scripts/kk_pilot.py --model T --tp 2 --k 1 \
              --puzzles "$BASE/examples/test6.jsonl" --out "$d"
          release "$b"; release "$a"; break
        fi
        release "$a"
      fi
    done
    [[ -f "$d/summary.json" ]] || sleep 180
  done
  echo "[$(ts)] teacher examples done"
}

echo "[$(ts)] puzzle testbed, instruct student (m8 §24 addendum)"
teacher_examples &
for g in 0 1 2 3 4 5 6 7; do worker "$g" & done
wait
echo "[$(ts)] KK INSTRUCT DONE"
