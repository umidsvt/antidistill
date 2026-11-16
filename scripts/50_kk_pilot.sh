#!/usr/bin/env bash
# Puzzle testbed, step 1: Knights and Knaves solve-rate pilot (results/m8_audit_design.md §17). Each model tries the 600
# pilot puzzles (N = 3..8, 100 each) 4 times; every try is stored and scored. Nothing is trained.
#   JOBS="0,1,2,3:T:4 4:B:1 4:C:1 6:D:1 7:D2:1" setsid nohup scripts/50_kk_pilot.sh > logs/kk_pilot.log 2>&1 < /dev/null & disown
#   SMOKE=1 JOBS="4:B:1" scripts/50_kk_pilot.sh      # 12 puzzles x 2 tries, into outputs/kk_pilot/smoke/
# JOBS: gpus:model:tensor-parallel entries; entries with the same GPU list run in order, different lists in parallel.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/gpu_env.sh
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn
PY=.venv-dg/bin/python; OUT=outputs/kk_pilot; PUZ=data/kk/pilot_test.jsonl; NEED_MIB=${NEED_MIB:-42000}
EXTRA=(); [[ -n "${SMOKE:-}" ]] && { EXTRA=(--limit 12 --k 2); OUT="$OUT/smoke"; }
[[ -f $PUZ ]] || { echo "missing $PUZ (scripts/kk_make.py)"; exit 1; }
mkdir -p "$OUT"
ts() { date '+%F %T'; }
wait_free() { local g; for g in ${1//,/ }; do
  while (( $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$g" | tr -d ' ') < NEED_MIB )); do sleep 120; done; done; }

one() {   # $1 gpu list, $2 model key, $3 tensor parallel
  local gl=$1 m=$2 tp=$3 dir="$OUT/$2" i pid
  mkdir -p "$dir"
  for i in 1 2; do
    [[ -f "$dir/summary.json" ]] && break
    wait_free "$gl"; echo "[$(ts)] START $m on GPUs $gl (tp $tp, attempt $i)"
    CUDA_VISIBLE_DEVICES=$gl $(pin "$gl") $PY scripts/kk_pilot.py --model "$m" --tp "$tp" --puzzles "$PUZ" \
      --out "$dir" "${EXTRA[@]}" > "$dir/run.log" 2>&1 &
    pid=$!
    while kill -0 "$pid" 2>/dev/null; do   # watchdog: vLLM can hang at exit after the output is written
      if [[ -f "$dir/summary.json" ]]; then
        sleep 120
        if kill -0 "$pid" 2>/dev/null; then echo "[$(ts)] $m: output written but PID $pid did not exit; stopping it"
          pkill -9 -P "$pid"; kill -9 "$pid"; fi
        break
      fi
      sleep 20
    done
    wait "$pid" 2>/dev/null
    [[ -f "$dir/summary.json" ]] || { echo "[$(ts)] FAILED $m (attempt $i): $(grep -a -m1 -E 'Error' "$dir/run.log" | cut -c1-160)"; sleep 300; }
  done
  [[ -f "$dir/summary.json" ]] && echo "[$(ts)] DONE  $m: $(python3 -I -c "import json,sys; d=json.load(open(sys.argv[1]))['by_n']; print(' '.join(f'N{n}:{v[\"solve_rate\"]:.2f}' for n,v in d.items()))" "$dir/summary.json")"
}

declare -A Q
for j in ${JOBS:?set JOBS}; do IFS=: read -r gl m tp <<< "$j"; require_one_group "$gl"; Q[$gl]+="$m:$tp "; done
echo "[$(ts)] Knights and Knaves pilot (m8 §17)${SMOKE:+, smoke test}: $JOBS"
pids=()
for gl in "${!Q[@]}"; do
  ( for e in ${Q[$gl]}; do IFS=: read -r m tp <<< "$e"; one "$gl" "$m" "$tp"; done ) & pids+=($!)
done
wait "${pids[@]}"
echo "[$(ts)] KK PILOT DONE"
