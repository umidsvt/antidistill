#!/usr/bin/env bash
# Evaluate M6b reconstruction checkpoints on the same four benchmarks as every other condition.
# Condition labels are recon_<mode>_ep15: scripts/30_collect_results.py keys on the first path
# component under outputs/, so distinct labels are what keep these as separate rows.
set -uo pipefail
cd "$(dirname "$0")/.."
for m in "$@"; do
  CK=saves/Qwen2.5-7B_recon_${m}/checkpoint-1500
  [ -d "$CK" ] || { echo "SKIP $m: $CK missing"; continue; }
  ls "$CK"/model-0000?-of-00004.safetensors >/dev/null 2>&1 || { echo "SKIP $m: weights missing"; continue; }
  for bench in math amc aime aime25; do
    echo "[$(date +%H:%M:%S)] === recon_$m / $bench ==="
    bash scripts/10_eval.sh "$CK" "recon_${m}_ep15" "$bench" 0.0 1 2>&1 \
      | grep -E "correct cnt|^Acc:|pass = " | tail -2
  done
done
.venv-infer/bin/python scripts/30_collect_results.py
