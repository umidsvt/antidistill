#!/usr/bin/env bash
# Phase 0 of the answers-only method (results/m8_audit_design.md §24): an instruction-tuned student (Llama-3.2-3B-Instruct)
# on setup 2, with the field's teacher (S1) and a bigger teacher (S2, R1-Distill-Qwen-32B, 8,192 tokens).
#   teacher  R1-Distill-Qwen-32B on one GPU group (tensor parallel over its GPUs): a 16-item smoke test, then GSM8K and
#            MATH (distinct training problems of seeds 123-125, and the test split), then the per-seed files.
#   queue    single-GPU jobs, one worker per GPU, each job on a GPU free of other processes:
#              i0  the untrained student, scored at 1,024 and 8,192 tokens (12 jobs)
#              s1  U_inst on the field teacher's stored solutions (setup recipe), scored at 1,024 (6 jobs)
#              s2  U_inst on the 32B teacher's solutions (cut-off 9,216, micro-batch 1 x 24), scored at 8,192 (6 jobs)
#   WORKERS: GPUs whose workers start now; LATE_WORKERS: GPUs that join when the teacher is done.
#   TEACHER_GPUS=4,5,6,7 WORKERS="0" LATE_WORKERS="4 5 6 7" setsid nohup scripts/56_phase0.sh > logs/phase0.log 2>&1 < /dev/null & disown
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/gpu_env.sh
export PYTHONPATH=third_party/ext/distillation-game/src HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_OFFLINE=1 \
       TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn
PY=.venv-dg/bin/python; OUT=outputs/phase0; STUDENT=unsloth/Llama-3.2-3B-Instruct
TEACHER=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
TEACHER_GPUS=${TEACHER_GPUS:?one GPU group for the teacher}; WORKERS=${WORKERS:-""}; LATE_WORKERS=${LATE_WORKERS:-""}
NEED_MIB=${NEED_MIB:-42000}
require_one_group "$TEACHER_GPUS"
mkdir -p "$OUT" logs/phase0
ts() { date '+%F %T'; }
idle() {   # GPU list free of other processes, with NEED_MIB free
  local g; for g in ${1//,/ }; do
    [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$g" | tr -d ' ')" ]] && return 1
    (( $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$g" | tr -d ' ') < NEED_MIB )) && return 1
  done; return 0; }
wait_idle() { until idle "$1"; do sleep 120; done; }
run() {   # <gpus> <marker> <log> <command...>: up to 2 attempts on idle GPUs
  local gl=$1 marker=$2 log=$3 i; shift 3
  for i in 1 2; do
    [[ -f "$marker" ]] && return 0
    wait_idle "$gl"; mkdir -p "$(dirname "$log")"
    CUDA_VISIBLE_DEVICES=$gl $(pin "$gl") "$@" >> "$log" 2>&1
    [[ -f "$marker" ]] && return 0
    echo "[$(ts)] FAILED (attempt $i, GPUs $gl): ${*: -6} :: $(grep -a -E 'Error' "$log" | tail -1 | cut -c1-200)"; sleep 120
  done
  return 1
}

# ---------------- the 32B teacher ----------------
teacher() {
  local tp; tp=$(echo "${TEACHER_GPUS//,/ }" | wc -w)
  local sm=$OUT/smoke/teacher32b_math
  run "$TEACHER_GPUS" "$sm/shard0of1_smoke.json" logs/phase0/teacher_smoke.log $PY scripts/dg_teacher_vllm.py \
      --dataset math --teacher $TEACHER --tp "$tp" --limit 16 --out "$sm" \
    && echo "[$(ts)] TEACHER SMOKE: $(tail -n 1 logs/phase0/teacher_smoke.log | cut -c1-200)" \
    || { echo "[$(ts)] teacher smoke test failed; stopping the teacher"; return 1; }
  for ds in gsm8k math; do
    local d=$OUT/teacher32b_$ds
    run "$TEACHER_GPUS" "$d/shard0of1.json" logs/phase0/teacher_$ds.log $PY scripts/dg_teacher_vllm.py \
        --dataset "$ds" --teacher $TEACHER --tp "$tp" --out "$d" \
      || { echo "[$(ts)] teacher $ds failed; stopping the teacher"; return 1; }
    $PY scripts/dg_teacher_vllm.py --dataset "$ds" --out "$d" --merge --nshards 1 > logs/phase0/teacher_${ds}_merge.log 2>&1 \
      && echo "[$(ts)] TEACHER $ds: $(python3 -I -c "import json,sys; d=json.load(open(sys.argv[1])); print({k: d[k] for k in ('train_distinct', 'test')})" "$d/summary.json")" \
      || { echo "[$(ts)] merge $ds failed"; return 1; }
  done
}

# ---------------- the single-GPU queue ----------------
Q=$OUT/jobs.txt; QN=$OUT/jobs.next
if [[ ! -f "$Q" ]]; then
  { for ds in gsm8k math; do for s in 123 124 125; do for b in 1024 8192; do echo "i0 $ds $s $b"; done; done; done
    for ds in math gsm8k; do for s in 123 124 125; do echo "s1 $ds $s"; done; done
    for ds in gsm8k math; do for s in 123 124 125; do echo "s2 $ds $s"; done; done; } > "$Q"
  echo 0 > "$QN"
fi
next_job() { ( flock 9; local n; n=$(cat "$QN"); echo $((n + 1)) > "$QN"; sed -n "$((n + 1))p" "$Q" ) 9> "$OUT/jobs.lock"; }
job() {   # <gpu> <kind> <dataset> <seed> [budget]
  local g=$1 kind=$2 ds=$3 s=$4 b=${5:-} cfg="configs/setup2/${3}_U_s${4}.yaml" d
  case $kind in
    i0) d=$OUT/I0/${ds}_s${s}_$b.json
        run "$g" "$d" logs/phase0/i0_${ds}_s${s}_$b.log $PY scripts/dg_score_vllm.py --config "$cfg" --model $STUDENT \
            --max-new "$b" --out "$d" ;;
    s1) d=$OUT/S1/${ds}_s$s
        local t; t=$(ls outputs/setup2/${ds}_U_s${s}_*/teacher/train_standard.json | head -1)
        run "$g" "$d/train_info.json" logs/phase0/s1_${ds}_s$s.log $PY scripts/dg_train_from_traces.py --config "$cfg" \
            --student $STUDENT --traces "$t" --max-length 2048 --micro-batch 6 --accum 4 --out "$d" \
          && run "$g" "$d/test_1024.json" logs/phase0/s1_${ds}_s${s}_score.log $PY scripts/dg_score_vllm.py --config "$cfg" \
            --model "$d/merged" --max-new 1024 --out "$d/test_1024.json" ;;
    s2) d=$OUT/S2/${ds}_s$s
        local t=$OUT/teacher32b_$ds/train_s$s.json
        until [[ -f "$t" ]] || [[ -f "$OUT/teacher_failed" ]]; do sleep 300; done
        [[ -f "$t" ]] || return 1
        run "$g" "$d/train_info.json" logs/phase0/s2_${ds}_s$s.log $PY scripts/dg_train_from_traces.py --config "$cfg" \
            --student $STUDENT --traces "$t" --max-length 9216 --micro-batch 1 --accum 24 --out "$d" \
          && run "$g" "$d/test_8192.json" logs/phase0/s2_${ds}_s${s}_score.log $PY scripts/dg_score_vllm.py --config "$cfg" \
            --model "$d/merged" --max-new 8192 --out "$d/test_8192.json" ;;
  esac
}
worker() {   # <gpu>
  local g=$1 line
  while line=$(next_job) && [[ -n "$line" ]]; do
    echo "[$(ts)] GPU $g: start $line"
    if job "$g" $line; then echo "[$(ts)] GPU $g: done  $line"; else echo "[$(ts)] GPU $g: FAILED $line"; fi
  done
  echo "[$(ts)] GPU $g: queue empty"
}

echo "[$(ts)] Phase 0 (m8 §24): teacher on GPUs $TEACHER_GPUS; workers now: ${WORKERS:-none}; after the teacher: ${LATE_WORKERS:-none}"
pids=()
for g in $WORKERS; do worker "$g" & pids+=($!); done
{ teacher || touch "$OUT/teacher_failed"; for g in $LATE_WORKERS; do worker "$g" & done; wait; } & pids+=($!)
wait "${pids[@]}"
echo "[$(ts)] PHASE 0 DONE"
