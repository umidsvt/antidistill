#!/usr/bin/env bash
# Evaluate the epistemic-supplementation attack on the same 600-problem sweep M2 and M3 got.
set -uo pipefail
cd "$(dirname "$0")/.."
REPO="$PWD"
CK=saves/Qwen2.5-7B_mixed50/checkpoint-1500
[ -d "$CK" ] || { echo "ERROR: $CK missing"; exit 1; }
for bench in aime amc aime25 math; do
  echo "=============================================================="
  echo "[$(date +%H:%M:%S)] START mixed50_ep15 / $bench"
  bash scripts/10_eval.sh "$CK" mixed50_ep15 "$bench" 0.0 1 2>&1 | grep -E "correct cnt|^Acc:" | tail -2
  echo "[$(date +%H:%M:%S)] DONE  mixed50_ep15 / $bench"
done
echo "MIXED50 EVAL COMPLETE"
"$REPO/.venv-infer/bin/python" scripts/30_collect_results.py
