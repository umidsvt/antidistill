#!/usr/bin/env bash
# Evaluate the epistemic-fraction sweep (25% and 10%) on the same 600-problem set.
# Waits for BOTH training runs to finish, then evaluates sequentially so the two jobs
# do not contend for GPUs.
set -uo pipefail
cd "$(dirname "$0")/.."
REPO="$PWD"
echo "[$(date +%H:%M:%S)] waiting for both training runs ..."
while pgrep -f "llamafactory/launcher.py" >/dev/null 2>&1; do sleep 60; done
echo "[$(date +%H:%M:%S)] training done"
sleep 120
while pgrep -f "prune_checkpoints.sh saves/Qwen2.5-7B_mixed" >/dev/null 2>&1; do sleep 30; done
echo "[$(date +%H:%M:%S)] pruners finished"
du -sh saves/Qwen2.5-7B_mixed25/checkpoint-* saves/Qwen2.5-7B_mixed10/checkpoint-* 2>/dev/null

for pct in 25 10; do
  CK=saves/Qwen2.5-7B_mixed${pct}/checkpoint-1500
  [ -d "$CK" ] || { echo "ERROR: $CK missing"; continue; }
  for bench in aime amc aime25 math; do
    echo "=============================================================="
    echo "[$(date +%H:%M:%S)] START mixed${pct}_ep15 / $bench"
    bash scripts/10_eval.sh "$CK" mixed${pct}_ep15 "$bench" 0.0 1 2>&1 | grep -E "correct cnt|^Acc:" | tail -2
    echo "[$(date +%H:%M:%S)] DONE  mixed${pct}_ep15 / $bench"
  done
done
echo "MIXSWEEP EVAL COMPLETE"
"$REPO/.venv-infer/bin/python" scripts/30_collect_results.py
