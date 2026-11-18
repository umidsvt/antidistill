#!/usr/bin/env bash
# Phase 2 (results/m8_audit_design.md §22): the floor F_D when 12, 30 or 50% of the teacher's answers are wrong.
# The corrupted answers and the hint round for them (D, 4 tries; two single-GPU copies), the three training sets, then
# students F_D_c12, F_D_c30, F_D_c50 x seeds 123-125 (scripts/51_kk_step2.sh, first-answer test set test_eot).
#   HINT_GPUS="6 7" TRAIN_GPUS="4 6 7" setsid nohup scripts/54_kk_corrupt.sh > logs/kk_corrupt.log 2>&1 < /dev/null & disown
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/gpu_env.sh
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn
PY=.venv-dg/bin/python; BASE=outputs/kk_step2; POOL=data/kk/step2_train.jsonl; NEED_MIB=${NEED_MIB:-42000}
read -r -a HG <<< "${HINT_GPUS:?set HINT_GPUS (two GPUs)}"
ts() { date '+%F %T'; }
wait_free() { while (( $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" | tr -d ' ') < NEED_MIB )); do sleep 120; done; }
run() {   # <gpu> <marker> <log> <command...>: up to 2 attempts, with the exit watchdog of the other launchers
  local g=$1 marker=$2 log=$3 i pid; shift 3
  for i in 1 2; do
    [[ -f "$marker" ]] && return 0
    wait_free "$g"; mkdir -p "$(dirname "$log")"
    CUDA_VISIBLE_DEVICES=$g $(pin "$g") "$@" > "$log" 2>&1 &
    pid=$!
    while kill -0 "$pid" 2>/dev/null; do
      if [[ -f "$marker" ]]; then sleep 120; kill -0 "$pid" 2>/dev/null && { pkill -9 -P "$pid"; kill -9 "$pid"; }; break; fi
      sleep 20
    done
    wait "$pid" 2>/dev/null
    [[ -f "$marker" ]] && return 0
    echo "[$(ts)] FAILED (attempt $i): $* :: $(grep -a -m1 -E 'Error' "$log" | cut -c1-160)"; sleep 120
  done
  return 1
}
echo "[$(ts)] floor under corrupted answers (m8 §22): hint round on GPUs ${HG[*]}"
mkdir -p "$BASE/corrupt"
python3 -I scripts/kk_corrupt.py hints --pool "$POOL" --gen "$BASE/gen" --out "$BASE/corrupt/hint_round.jsonl" | sed "s/^/[$(ts)] HINTS /"
n=$(wc -l < "$BASE/corrupt/hint_round.jsonl")
head -n $(( (n + 1) / 2 )) "$BASE/corrupt/hint_round.jsonl" > "$BASE/corrupt/hint_part1.jsonl"
tail -n +$(( (n + 1) / 2 + 1 )) "$BASE/corrupt/hint_round.jsonl" > "$BASE/corrupt/hint_part2.jsonl"
for k in 1 2; do
  d="$BASE/gen/D_chint_part$k"
  { run "${HG[$((k-1))]}" "$d/summary.json" "$d.log" $PY scripts/kk_pilot.py --model D --tp 1 --k 4 \
      --puzzles "$BASE/corrupt/hint_part$k.jsonl" --out "$d" && echo "[$(ts)] GEN D_chint_part$k done"; } &
done
wait
for k in 1 2; do [[ -f "$BASE/gen/D_chint_part$k/summary.json" ]] || { echo "[$(ts)] missing D_chint_part$k; stopping"; exit 1; }; done
python3 -I scripts/kk_corrupt.py build --pool "$POOL" --gen "$BASE/gen" --out "$BASE/train_sets" | sed "s/^/[$(ts)] BUILD /"
EVAL_DIR=test_eot PHASE=train STUDENTS="F_D_c12 F_D_c30 F_D_c50" WITH_UNTRAINED=0 TRAIN_GPUS="${TRAIN_GPUS:?set TRAIN_GPUS}" \
  scripts/51_kk_step2.sh
echo "[$(ts)] CORRUPT DONE"
