#!/usr/bin/env bash
# Puzzle testbed, step 2 (results/m8_audit_design.md §18): is there something to protect at N = 7-8?
#   PHASE=data   solution writers on the 2,000 training puzzles (teacher on GPUs 0-3; B then C on GPU 4; D's 4 tries
#                split by N over GPUs 6 and 7), F_D's hint round (GPUs 6, 7), then the students' training sets
#   PHASE=train  students U, Solo_B, Solo_C, Solo_D, F_D x seeds (train, then score) and the untrained student,
#                pulled from one queue by one worker per GPU in TRAIN_GPUS
#   PHASE=all    both
#   setsid nohup env PHASE=all scripts/51_kk_step2.sh > logs/kk_step2.log 2>&1 < /dev/null & disown
#   SMOKE=1 PHASE=all scripts/51_kk_step2.sh    # 20 training and 10 test puzzles, seed 123, 3 steps; outputs/kk_step2/smoke
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/gpu_env.sh
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn
PY=.venv-dg/bin/python; BASE=outputs/kk_step2; POOL=data/kk/step2_train.jsonl; TEST=data/kk/step2_test.jsonl
# free memory a job waits for (MiB): vLLM solution writers 42,000 (memory setting 0.90); student scoring 40,000
# (0.85); training 30,000 (2026-11-13 smoke test: 42,000 kept a training job waiting behind a 3.5 GB job for 34 minutes)
NEED_MIB=${NEED_MIB:-42000}; EVAL_NEED_MIB=${EVAL_NEED_MIB:-40000}; TRAIN_NEED_MIB=${TRAIN_NEED_MIB:-30000}; SEEDS=${SEEDS:-"123 124 125"}; TRAIN_GPUS=${TRAIN_GPUS:-"0 1 2 4 6 7"}
TRAIN_EXTRA=()
if [[ -n "${SMOKE:-}" ]]; then
  BASE=outputs/kk_step2/smoke; mkdir -p "$BASE"; SEEDS=123; TRAIN_EXTRA=(--max-steps 3 --grad-accum 4)
  { grep '"n_people": 7' "$POOL" | head -10; grep '"n_people": 8' "$POOL" | head -10; } > "$BASE/pool.jsonl"
  { grep '"n_people": 7' "$TEST" | head -5; grep '"n_people": 8' "$TEST" | head -5; } > "$BASE/test.jsonl"
  POOL=$BASE/pool.jsonl; TEST=$BASE/test.jsonl
fi
mkdir -p "$BASE/gen" "$BASE/students"
ts() { date '+%F %T'; }
wait_free() { local g; for g in ${1//,/ }; do
  while (( $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$g" | tr -d ' ') < ${NEED_NOW:-$NEED_MIB} )); do sleep 120; done; done; }

# run <gpus> <marker file> <log> <command...>: up to 2 attempts; a watchdog stops a process still alive 2 minutes after
# its marker is written (vLLM can hang at exit, CLAUDE.md rule 6). Returns 0 if the marker exists.
run() {
  local gl=$1 marker=$2 log=$3 i pid; shift 3
  for i in 1 2; do
    [[ -f "$marker" ]] && return 0
    wait_free "$gl"; mkdir -p "$(dirname "$log")"
    CUDA_VISIBLE_DEVICES=$gl $(pin "$gl") "$@" > "$log" 2>&1 &
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

writer() {   # <gpus> <tp> <model key> <k> <puzzles> <name>
  local d="$BASE/gen/$6"
  echo "[$(ts)] GEN   $6 ($3, k=$4) on GPUs $1"
  run "$1" "$d/summary.json" "$d.log" $PY scripts/kk_pilot.py --model "$3" --tp "$2" --k "$4" --puzzles "$5" --out "$d" \
    && echo "[$(ts)] GEN   $6 done: $(python3 -I -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['minutes'], 'min;', {n: v['solve_rate'] for n, v in d['by_n'].items()})" "$d/summary.json")"
}

# GPU layout (2026-11-13: GPUs 3 and 5 held by labmates, so neither group has 4 free GPUs): the teacher as two
# tensor-parallel-2 copies, one per N; D's tries split by N over two GPUs; B and C on one GPU. Change with T_GPUS,
# D_GPUS, BC_GPU (T_GPUS="0,1,2,3" runs a single tensor-parallel-4 copy, as in the pilot).
read -r -a TG <<< "${T_GPUS:-0,1 6,7}"; read -r -a DG <<< "${D_GPUS:-2 4}"; BC=${BC_GPU:-4}
ntp() { local x=${1//[^,]/}; echo $(( ${#x} + 1 )); }

phase_data() {
  grep '"n_people": 7' "$POOL" > "$BASE/pool_n7.jsonl"; grep '"n_people": 8' "$POOL" > "$BASE/pool_n8.jsonl"
  if (( ${#TG[@]} == 1 )); then writer "${TG[0]}" "$(ntp "${TG[0]}")" T 1 "$POOL" T &
  else writer "${TG[0]}" "$(ntp "${TG[0]}")" T 1 "$BASE/pool_n7.jsonl" T_part1 &
       writer "${TG[1]}" "$(ntp "${TG[1]}")" T 1 "$BASE/pool_n8.jsonl" T_part2 &
  fi
  writer "${DG[0]}" 1 D 4 "$BASE/pool_n7.jsonl" D_part1 &
  if [[ $BC == "${DG[1]}" ]]; then
    { writer "$BC" 1 B 1 "$POOL" B; writer "$BC" 1 C 1 "$POOL" C; writer "${DG[1]}" 1 D 4 "$BASE/pool_n8.jsonl" D_part2; } &
  else
    { writer "$BC" 1 B 1 "$POOL" B; writer "$BC" 1 C 1 "$POOL" C; } &
    writer "${DG[1]}" 1 D 4 "$BASE/pool_n8.jsonl" D_part2 &
  fi
  wait
  python3 -I scripts/kk_build.py hints --pool "$POOL" --gen "$BASE/gen" --out "$BASE/hint_round.jsonl" | sed "s/^/[$(ts)] HINTS /"
  local n; n=$(wc -l < "$BASE/hint_round.jsonl")
  if (( n > 0 )); then
    head -n $(( (n + 1) / 2 )) "$BASE/hint_round.jsonl" > "$BASE/hint_part1.jsonl"
    tail -n +$(( (n + 1) / 2 + 1 )) "$BASE/hint_round.jsonl" > "$BASE/hint_part2.jsonl"
    writer "${DG[0]}" 1 D 4 "$BASE/hint_part1.jsonl" D_hint_part1 &
    [[ -s "$BASE/hint_part2.jsonl" ]] && writer "${DG[1]}" 1 D 4 "$BASE/hint_part2.jsonl" D_hint_part2 &
    wait
  fi
  python3 -I scripts/kk_build.py build --pool "$POOL" --gen "$BASE/gen" --out "$BASE/train_sets" | sed "s/^/[$(ts)] BUILD /"
}

job() {   # <gpu> <student|untrained> <seed>
  local g=$1 s=$2 seed=$3 d="$BASE/students/$2_s$3"
  if [[ $s == untrained ]]; then
    NEED_NOW=$EVAL_NEED_MIB run "$g" "$d/${EVAL_DIR:-test}/summary.json" "$d/${EVAL_DIR:-test}.log" $PY scripts/kk_eval.py --puzzles "$TEST" --seed "$seed" --out "$d/${EVAL_DIR:-test}"
  else
    NEED_NOW=$TRAIN_NEED_MIB run "$g" "$d/train_meta.json" "$d/train.log" $PY scripts/kk_sft.py --train-set "$BASE/train_sets/$s.jsonl" \
        --out "$d" --seed "$seed" "${TRAIN_EXTRA[@]}" || return 1
    NEED_NOW=$EVAL_NEED_MIB run "$g" "$d/${EVAL_DIR:-test}/summary.json" "$d/${EVAL_DIR:-test}.log" $PY scripts/kk_eval.py --puzzles "$TEST" --seed "$seed" \
        --adapter "$d/adapter" --out "$d/${EVAL_DIR:-test}"
  fi
  [[ -f "$d/${EVAL_DIR:-test}/summary.json" ]] && echo "[$(ts)] DONE  $s s$seed on GPU $g: $(python3 -I -c "import json,sys; d=json.load(open(sys.argv[1])); print('solve', d['solve_rate'], {n: v['solve_rate'] for n, v in d['by_n'].items()}, 'finished', d['finished'])" "$d/${EVAL_DIR:-test}/summary.json")"
}

phase_train() {
  local q="$BASE/queue.txt" lock="$BASE/queue.lock" g
  : > "$q"
  # STUDENTS / WITH_UNTRAINED=0 restrict the queue (m8 §21: only the defended student D_rewrite)
  for seed in $SEEDS; do
    [[ ${WITH_UNTRAINED:-1} == 1 ]] && echo "untrained $seed" >> "$q"
    for s in ${STUDENTS:-U Solo_D F_D Solo_C Solo_B}; do echo "$s $seed" >> "$q"; done
  done
  echo "[$(ts)] TRAIN queue: $(wc -l < "$q") jobs on GPUs $TRAIN_GPUS"
  for g in $TRAIN_GPUS; do
    ( while true; do
        line=$(flock "$lock" sh -c "head -1 '$q'; sed -i '1d' '$q'")
        [[ -z $line ]] && break
        job "$g" $line
      done ) &
  done
  wait
}

echo "[$(ts)] Knights and Knaves step 2 (m8 §18)${SMOKE:+, smoke test}: PHASE=${PHASE:?set PHASE}"
case $PHASE in
  data) phase_data ;;
  train) phase_train ;;
  all) phase_data && phase_train ;;
  *) echo "PHASE must be data, train or all"; exit 1 ;;
esac
echo "[$(ts)] KK STEP 2 ${PHASE^^} DONE"
