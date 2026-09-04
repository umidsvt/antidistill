#!/usr/bin/env bash
# Run a series of evals back-to-back on the shared GPUs, waiting for any in-flight eval first.
# Ordered by priority so the headline numbers land before the diagnostic ones.
#
# Usage: scripts/11_eval_queue.sh
set -uo pipefail
cd "$(dirname "$0")/.."
REPO="$PWD"

# Wait for any currently running eval to release the GPUs.
while pgrep -f "kim_eval/eval(_acc)?\.py" >/dev/null 2>&1; do sleep 30; done

CKPT_DIR="saves/Qwen2.5-7B_limo"

# condition-label  checkpoint  temp  n_sampling
QUEUE=(
  "limo_ep15 $CKPT_DIR/checkpoint-1500 0.7 16"   # headline low-variance metric
  "limo_ep5  $CKPT_DIR/checkpoint-500  0.0 1"    # did it peak before epoch 15?
  "limo_ep10 $CKPT_DIR/checkpoint-1000 0.0 1"
  "limo_ep5  $CKPT_DIR/checkpoint-500  0.7 16"
  "limo_ep10 $CKPT_DIR/checkpoint-1000 0.7 16"
)

for entry in "${QUEUE[@]}"; do
  read -r label ckpt temp nsamp <<<"$entry"
  echo "=============================================================="
  echo "[$(date +%H:%M:%S)] queue: $label  temp=$temp  n=$nsamp"
  echo "=============================================================="
  bash scripts/10_eval.sh "$ckpt" "$label" aime "$temp" "$nsamp" \
    2>&1 | grep -viE "it/s\]|s/it\]|toks/s|^INFO|^WARNING|^\[transformers\]|Gloo" | tail -6
  echo "[$(date +%H:%M:%S)] done: $label temp=$temp n=$nsamp"
done

echo "QUEUE COMPLETE"
"$REPO/.venv-infer/bin/python" scripts/30_collect_results.py
