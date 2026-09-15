#!/usr/bin/env bash
# Evaluate the 7B-attacker reconstruction checkpoints. Labels are recon7b_<mode>_ep15 so
# scripts/30_collect_results.py keeps them distinct from the 32B-attacker rows.
# Run SERIALLY, never alongside training: GPUs 0-3 and 4-7 are on separate NUMA nodes and a
# concurrent eval ran ~10x slower (CLAUDE.md §7.3).
set -uo pipefail
cd "$(dirname "$0")/.."
for m in "$@"; do
  CK=saves/Qwen2.5-7B_recon7b_${m}/checkpoint-1500
  [ -d "$CK" ] || { echo "SKIP $m"; continue; }
  ls "$CK"/model-0000?-of-00004.safetensors >/dev/null 2>&1 || { echo "SKIP $m: weights missing"; continue; }
  for bench in math amc aime aime25; do
    echo "[$(date +%H:%M:%S)] === recon7b_$m / $bench ==="
    bash scripts/10_eval.sh "$CK" "recon7b_${m}_ep15" "$bench" 0.0 1 2>&1 \
      | grep -E "correct cnt|^Acc:|pass = " | tail -2
  done
done
.venv-infer/bin/python scripts/30_collect_results.py
