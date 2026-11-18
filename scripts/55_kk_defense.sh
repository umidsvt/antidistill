#!/usr/bin/env bash
# Phase 1, ADS and PoE on the puzzle testbed (results/m8_audit_design.md §23), with the Distillation Game's code
# (scripts/kk_defend.py). One GPU group: the teacher on all its GPUs but the last, the proxies on the last.
#   PHASE=timing  the proxy gradient (first GPU), then the timing pilot: 16 calibration puzzles (8 per N) under
#                 standard sampling, ADS (lambda 0.1) and PoE (gamma 0.4); time and memory only
#   PHASE=ads     the authors' smaller plan (§23, 2026-11-18): calibration (baseline, ADS lambda 0.05-0.4 on the 100
#                 calibration puzzles), the choice by §23's rule, ADS solutions for all 2,000 training puzzles, D_ADS,
#                 students D_ADS_s123-125. Batch 25; 20 for every run if the first runs out of memory.
#   GPUS=0,1,2,3 PHASE=timing setsid nohup scripts/55_kk_defense.sh > logs/kk_defense_timing.log 2>&1 < /dev/null & disown
#   GPUS=0,1,2,3 PHASE=ads setsid nohup scripts/55_kk_defense.sh > logs/kk_ads.log 2>&1 < /dev/null & disown
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
wait_idle() {   # every GPU in the list free of other processes and with NEED_MIB free, checked together
  local g busy; while :; do busy=0
    for g in ${1//,/ }; do
      [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$g" | tr -d ' ')" ]] && busy=1
      (( $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$g" | tr -d ' ') < NEED_MIB )) && busy=1
    done
    (( busy )) || return 0; sleep 120; done; }
gen_step() {   # <marker> <log> <command...>: up to 2 attempts on idle GPUs; a retry resumes from the stored tries
  local marker=$1 log=$2 i; shift 2
  for i in 1 2; do
    [[ -f "$marker" ]] && return 0
    wait_idle "$GPUS"; mkdir -p "$(dirname "$log")"
    echo "[$(ts)] start (attempt $i): ${*: -12}"
    CUDA_VISIBLE_DEVICES=$GPUS $(pin "$GPUS") "$@" >> "$log" 2>&1
    [[ -f "$marker" ]] && return 0
    grep -a -q -E "OutOfMemoryError|CUDA out of memory" "$log" && return 2
    echo "[$(ts)] FAILED (attempt $i): $(grep -a -E 'Error' "$log" | tail -1 | cut -c1-200)"; sleep 120
  done
  return 1
}
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
  ads)
    CAL=$BASE/calib; FULL=$BASE/full; POOL=data/kk/step2_train.jsonl
    [[ -f "$BASE/ads_batch" ]] || echo 25 > "$BASE/ads_batch"
    echo "[$(ts)] ADS (m8 §23, authors' decision 2026-11-18) on GPUs $GPUS, batch $(cat "$BASE/ads_batch")"
    for m in "standard 0" "ads 0.05" "ads 0.1" "ads 0.2" "ads 0.4"; do
      set -- $m; d="$CAL/$1_$2"
      while :; do
        gen_step "$d/summary.json" "$d.log" $PY scripts/kk_defend.py gen --method "$1" --strength "$2" \
          --puzzles data/kk/calib.jsonl --grad "$BASE/proxy_grad.pt" --out "$d" --batch "$(cat "$BASE/ads_batch")" --resume
        rc=$?
        (( rc == 0 )) && break
        if (( rc == 2 )) && [[ $(cat "$BASE/ads_batch") == 25 && "$1_$2" == standard_0 ]]; then
          echo "[$(ts)] out of memory at batch 25 in the first run: batch 20 for every run (m8 §23)"
          echo 20 > "$BASE/ads_batch"; mv "$d" "$d.oom_b25"; mv "$d.log" "$d.oom_b25.log"; continue
        fi
        echo "[$(ts)] stopping: $1 $2 failed (rc $rc)"; exit 1
      done
      echo "[$(ts)] CALIB $1 $2: $(python3 -I -c "import json,sys; d=json.load(open(sys.argv[1])); print({k: d[k] for k in ('accuracy', 'finished', 'minutes', 'sec_per_step', 'peak_gib')})" "$d/summary.json")"
    done
    $PY scripts/kk_defend.py choose --calib "$CAL" > /dev/null || exit 1
    lam=$(python3 -I -c "import json,sys; print(json.load(open(sys.argv[1]))['chosen'])" "$CAL/choice.json")
    echo "[$(ts)] CHOICE lambda $lam: $(python3 -I -c "import json,sys; d=json.load(open(sys.argv[1])); print('baseline', d['baseline']['correct'], {s: g['correct'] for s, g in d['grid'].items()}, 'none qualified' if d['none_qualified'] else '')" "$CAL/choice.json")"
    d="$FULL/ads_$lam"
    gen_step "$d/summary.json" "$d.log" $PY scripts/kk_defend.py gen --method ads --strength "$lam" --puzzles "$POOL" \
      --grad "$BASE/proxy_grad.pt" --out "$d" --batch "$(cat "$BASE/ads_batch")" --length-from outputs/kk_step2/gen --resume \
      || { echo "[$(ts)] stopping: the full run failed"; exit 1; }
    echo "[$(ts)] FULL ads $lam: $(python3 -I -c "import json,sys; d=json.load(open(sys.argv[1])); print({k: d[k] for k in ('accuracy', 'finished', 'minutes', 'sec_per_step', 'resumed_from')})" "$d/summary.json")"
    $PY scripts/kk_defend.py build --pool "$POOL" --gen "$d" --out outputs/kk_step2/train_sets/D_ADS.jsonl \
      | sed "s/^/[$(ts)] BUILD D_ADS /" || exit 1
    tg=""; while [[ -z "$tg" ]]; do   # training GPUs: the group's GPUs free of other processes
      for g in ${GPUS//,/ }; do [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$g" | tr -d ' ')" ]] && tg="$tg $g"; done
      [[ -z "$tg" ]] && sleep 300; done
    echo "[$(ts)] students D_ADS on GPUs$tg"
    EVAL_DIR=test_eot PHASE=train STUDENTS=D_ADS WITH_UNTRAINED=0 TRAIN_GPUS="${tg# }" scripts/51_kk_step2.sh
    echo "[$(ts)] ADS DONE" ;;
  *) echo "PHASE must be timing or ads"; exit 1 ;;
esac
