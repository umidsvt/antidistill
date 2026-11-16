#!/usr/bin/env bash
# Level A in the cells still empty (results/m8_audit_design.md §19-20): self-training (STaR) of the base student on the
# teacher's answers, on MATH (§19: scripts/dg_star.py, 4-shot MATH prompt) and on the puzzles (§20: scripts/kk_star.py,
# then kk_sft.py and kk_eval.py). Jobs "math <seed>" and "kk <seed>" are pulled from one queue by one worker per GPU.
#   GPUS="2 3 4 6 7" setsid nohup scripts/52_levelA_fill.sh > logs/levelA_fill.log 2>&1 < /dev/null & disown
#   SMOKE=1 GPUS="2 6" scripts/52_levelA_fill.sh     # generation on a few problems, 2 training steps; into *_smoke dirs
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/gpu_env.sh
export PYTHONPATH=third_party/ext/distillation-game/src HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_OFFLINE=1 \
       TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv-dg/bin/python; SEEDS=${SEEDS:-"123 124 125"}; JOBS_KIND=${KINDS:-"math kk"}; NEED_MIB=${NEED_MIB:-40000}
SFX=""; GEN_EXTRA=(); KK_TRAIN_EXTRA=(); KK_EVAL_EXTRA=()
if [[ -n "${SMOKE:-}" ]]; then
  SFX="_smoke"; SEEDS=123; GEN_EXTRA=(--limit 16 --k 2 --k-hint 2 --k-salvage 2)
  KK_TRAIN_EXTRA=(--max-steps 2 --grad-accum 2); KK_EVAL_EXTRA=(--limit 10)
fi
mkdir -p logs outputs/kk_levelA
ts() { date '+%F %T'; }
latest() { ls -dt outputs/setup2/"$1"_[0-9]* 2>/dev/null | head -1; }   # the regenerated U runs (m8 §12)
wait_free() { local g; for g in ${1//,/ }; do
  while (( $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$g" | tr -d ' ') < NEED_MIB )); do sleep 120; done; done; }

# run <gpu> <marker> <log> <command...>: up to 2 attempts; stops a process still alive 2 minutes after its marker exists
run() {
  local g=$1 marker=$2 log=$3 i pid; shift 3
  for i in 1 2; do
    [[ -f "$marker" ]] && return 0
    wait_free "$g"; mkdir -p "$(dirname "$log")"
    CUDA_VISIBLE_DEVICES=$g $(pin "$g") "$@" > "$log" 2>&1 &
    pid=$!
    while kill -0 "$pid" 2>/dev/null; do
      if [[ -f "$marker" ]]; then
        sleep 120
        kill -0 "$pid" 2>/dev/null && { echo "[$(ts)] PID $pid did not exit after writing $marker; stopping it"; pkill -9 -P "$pid"; kill -9 "$pid"; }
        break
      fi
      sleep 20
    done
    wait "$pid" 2>/dev/null
    [[ -f "$marker" ]] && return 0
    echo "[$(ts)] FAILED (attempt $i): $* :: $(grep -a -m1 -E 'Error' "$log" | cut -c1-160)"; sleep 120
  done
  return 1
}

job_math() {   # <gpu> <seed>
  local g=$1 s=$2 d="outputs/setup2/star_fewshot_math_s$2$SFX"
  local args=(--fewshot --config "configs/setup2/math_U_s$s.yaml" --teacher-train "$(latest "math_U_s$s")/teacher/train_standard.json" --out "$d")
  mkdir -p "$d"
  run "$g" "$d/gen_summary.json" "$d/gen.log" $PY scripts/dg_star.py --phase gen "${args[@]}" "${GEN_EXTRA[@]}" || return 1
  echo "[$(ts)] MATH s$s generation: $(python3 -I -c "import json,sys; d=json.load(open(sys.argv[1])); d.pop('unsolved_ids',None); print(d)" "$d/gen_summary.json")"
  [[ -n "${SMOKE:-}" ]] && return 0      # the training phase is unchanged from GSM8K STaR (m8 §13)
  run "$g" "$d/result.json" "$d/train.log" $PY scripts/dg_star.py --phase train "${args[@]}" || return 1
  echo "[$(ts)] DONE  MATH s$s: $(tail -1 "$d/train.log" | cut -c1-200)"
}

job_kk() {   # <gpu> <seed>
  local g=$1 s=$2 d="outputs/kk_levelA/s$2$SFX"
  mkdir -p "$d"
  run "$g" "$d/gen_summary.json" "$d/gen.log" $PY scripts/kk_star.py --pool data/kk/step2_train.jsonl \
      --teacher-gen outputs/kk_step2/gen --seed "$s" --out "$d" "${GEN_EXTRA[@]}" || return 1
  echo "[$(ts)] KK s$s generation: $(python3 -I -c "import json,sys; d=json.load(open(sys.argv[1])); d.pop('unsolved_ids',None); print(d)" "$d/gen_summary.json")"
  NEED_MIB=30000 run "$g" "$d/train_meta.json" "$d/train.log" $PY scripts/kk_sft.py --train-set "$d/train_set.jsonl" \
      --out "$d" --seed "$s" "${KK_TRAIN_EXTRA[@]}" || return 1
  run "$g" "$d/test_eot/summary.json" "$d/eval.log" $PY scripts/kk_eval.py --puzzles data/kk/step2_test.jsonl --seed "$s" \
      --adapter "$d/adapter" --out "$d/test_eot" "${KK_EVAL_EXTRA[@]}" || return 1
  echo "[$(ts)] DONE  KK s$s: $(python3 -I -c "import json,sys; d=json.load(open(sys.argv[1])); print('solve (last answer)', d['solve_rate'], 'finished', d['finished'])" "$d/test_eot/summary.json")"
}

q=logs/levelA_fill${SFX}.queue; lock=$q.lock; : > "$q"
for s in $SEEDS; do for k in $JOBS_KIND; do echo "$k $s" >> "$q"; done; done
echo "[$(ts)] level A fill (m8 §19-20)${SMOKE:+, smoke test}: $(wc -l < "$q") jobs on GPUs ${GPUS:?set GPUS}"
for g in $GPUS; do
  ( while true; do
      line=$(flock "$lock" sh -c "head -1 '$q'; sed -i '1d' '$q'")
      [[ -z $line ]] && break
      set -- $line; "job_$1" "$g" "$2"
    done ) &
done
wait
echo "[$(ts)] LEVEL A FILL DONE"
