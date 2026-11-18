#!/usr/bin/env bash
# Phase 1, ADS and PoE on the puzzle testbed (results/m8_audit_design.md §23), with the Distillation Game's code
# (scripts/kk_defend.py). One GPU group: the teacher on all its GPUs but the last, the proxies on the last.
#   PHASE=timing  the proxy gradient (first GPU), then the timing pilot: 16 calibration puzzles (8 per N) under
#                 standard sampling, ADS (lambda 0.1) and PoE (gamma 0.4); time and memory only
#   GPUS=0,1,2,3 PHASE=timing setsid nohup scripts/55_kk_defense.sh > logs/kk_defense_timing.log 2>&1 < /dev/null & disown
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/gpu_env.sh
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv-dg/bin/python; BASE=outputs/kk_defense; GPUS=${GPUS:?set GPUS (one group)}; NEED_MIB=${NEED_MIB:-42000}
require_one_group "$GPUS"
mkdir -p "$BASE"
ts() { date '+%F %T'; }
wait_free() { local g; for g in ${1//,/ }; do
  while (( $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$g" | tr -d ' ') < NEED_MIB )); do sleep 120; done; done; }
step() {   # <gpus> <marker> <log> <command...>: one attempt; the script exits hard after writing its output
  local gl=$1 marker=$2 log=$3; shift 3
  [[ -f "$marker" ]] && return 0
  wait_free "$gl"; mkdir -p "$(dirname "$log")"
  CUDA_VISIBLE_DEVICES=$gl $(pin "$gl") "$@" > "$log" 2>&1
  [[ -f "$marker" ]] || { echo "[$(ts)] FAILED: $* :: $(grep -a -E 'Error|error' "$log" | tail -1 | cut -c1-200)"; return 1; }
}
case ${PHASE:?set PHASE} in
  timing)
    echo "[$(ts)] ADS/PoE (m8 §23) timing pilot on GPUs $GPUS"
    { grep '"n_people": 7' data/kk/calib.jsonl | head -8; grep '"n_people": 8' data/kk/calib.jsonl | head -8; } > "$BASE/timing_puzzles.jsonl"
    step "${GPUS%%,*}" "$BASE/proxy_grad.pt.json" "$BASE/proxy_grad.log" $PY scripts/kk_defend.py grad --out "$BASE/proxy_grad.pt" \
      && echo "[$(ts)] GRAD $(cat "$BASE/proxy_grad.pt.json" | python3 -I -c "import json,sys; d=json.load(sys.stdin); print({k: d[k] for k in ('held_out', 'grad_norm', 'minutes')})")" || exit 1
    for m in "standard 0" "ads 0.1" "poe 0.4"; do
      set -- $m; d="$BASE/timing/$1_$2"
      step "$GPUS" "$d/summary.json" "$d.log" $PY scripts/kk_defend.py gen --method "$1" --strength "$2" --timing \
        --puzzles "$BASE/timing_puzzles.jsonl" --grad "$BASE/proxy_grad.pt" --out "$d" --batch 16 \
        && echo "[$(ts)] TIMING $1 $2: $(tr -d '\n ' < "$d/summary.json")"
    done
    echo "[$(ts)] TIMING DONE" ;;
  *) echo "PHASE must be timing (calibration and the full run are added after the pilot)"; exit 1 ;;
esac
