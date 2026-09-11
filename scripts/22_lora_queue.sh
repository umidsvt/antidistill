#!/usr/bin/env bash
# Run the six LoRA conditions sequentially: {limo, hindsight, mixed50} x {16k, 32k}.
#
# Why a queue rather than parallel: all four GPUs are used by every run (global batch 8 =
# 4 x bs1 x ga2), so the runs cannot overlap on this node.
#
# One run failing does not stop the queue -- a 32k OOM should not cost you the 16k results.
# Exit status per run is recorded in logs/lora_queue_status.tsv and summarised at the end.
#
# Usage:
#   GPUS=0,1,2,3 scripts/22_lora_queue.sh                    # all six, 16k triple first
#   GPUS=0,1,2,3 scripts/22_lora_queue.sh limo_lora32k ...   # explicit subset
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
mkdir -p logs

# 16k first so a complete, mutually comparable triple lands before the longer 32k runs start.
DEFAULT_ORDER=(limo_lora16k hindsight_lora16k mixed50_lora16k
               limo_lora32k hindsight_lora32k mixed50_lora32k)
RUNS=("${@:-}")
[[ -z "${RUNS[0]:-}" ]] && RUNS=("${DEFAULT_ORDER[@]}")

STATUS="$REPO/logs/lora_queue_status.tsv"
[[ -f "$STATUS" ]] || printf 'run\tstatus\tstarted\telapsed_s\n' > "$STATUS"

echo "queue: ${RUNS[*]}"
echo "gpus : ${GPUS:-0,1,2,3}"
echo

for name in "${RUNS[@]}"; do
  cfg="configs/train/qwen2.5-7b_${name}.yaml"
  if [[ ! -f "$cfg" ]]; then
    echo "SKIP $name — no such config: $cfg" >&2
    printf '%s\tno-config\t%s\t0\n' "$name" "$(date -Is)" >> "$STATUS"
    continue
  fi
  echo "=== $(date -Is)  starting $name ==="
  t0=$(date +%s)
  GPUS="${GPUS:-0,1,2,3}" scripts/20_train.sh "$cfg"
  rc=$?
  el=$(( $(date +%s) - t0 ))
  [[ $rc -eq 0 ]] && st=ok || st="FAILED(rc=$rc)"
  printf '%s\t%s\t%s\t%d\n' "$name" "$st" "$(date -Is)" "$el" >> "$STATUS"
  printf '=== %s  %s: %s after %02d:%02d:%02d ===\n\n' \
    "$(date -Is)" "$name" "$st" $((el/3600)) $((el%3600/60)) $((el%60))
done

echo "queue finished."
column -t "$STATUS"
