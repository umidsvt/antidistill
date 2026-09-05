#!/usr/bin/env bash
# M3 evaluation. Waits for training to finish, then runs the same sweep M2 got so the two
# conditions are directly comparable, plus the epoch sweep.
set -uo pipefail
cd "$(dirname "$0")/.."
REPO="$PWD"

echo "[$(date +%H:%M:%S)] waiting for M3 training to finish ..."
while pgrep -f "llamafactory/launcher.py" >/dev/null 2>&1; do sleep 60; done
echo "[$(date +%H:%M:%S)] training done"

sleep 120
while pgrep -f "21_prune_checkpoints.sh saves/Qwen2.5-7B_hindsight" >/dev/null 2>&1; do sleep 30; done
echo "[$(date +%H:%M:%S)] pruner finished; checkpoints:"
du -sh saves/Qwen2.5-7B_hindsight/checkpoint-* 2>/dev/null

CK=saves/Qwen2.5-7B_hindsight/checkpoint-1500
if [ ! -d "$CK" ]; then echo "ERROR: $CK missing"; exit 1; fi

QUEUE=(
  "hindsight_ep15 $CK aime"
  "hindsight_ep15 $CK amc"
  "hindsight_ep15 $CK aime25"
  "hindsight_ep15 $CK math"
  "hindsight_ep5 saves/Qwen2.5-7B_hindsight/checkpoint-500 aime"
  "hindsight_ep10 saves/Qwen2.5-7B_hindsight/checkpoint-1000 aime"
)

for entry in "${QUEUE[@]}"; do
  read -r label ckpt bench <<<"$entry"
  echo "=============================================================="
  echo "[$(date +%H:%M:%S)] START $label / $bench"
  bash scripts/10_eval.sh "$ckpt" "$label" "$bench" 0.0 1 2>&1 | grep -E "correct cnt|^Acc:|pass = " | tail -2
  echo "[$(date +%H:%M:%S)] DONE  $label / $bench"
done

echo "M3 EVAL COMPLETE"
"$REPO/.venv-infer/bin/python" scripts/30_collect_results.py
