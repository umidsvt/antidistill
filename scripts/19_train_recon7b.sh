#!/usr/bin/env bash
# Train the three M6c (7B-attacker) reconstruction conditions back to back.
# Pruner launched WITH each trainer — a ZeRO-3 checkpoint is ~100 GB (CLAUDE.md §5).
set -uo pipefail
cd "$(dirname "$0")/.."
GPUS="${GPUS:-0,1,2,3}"
for m in search solo style; do
  echo "[$(date +%H:%M:%S)] === training recon7b_$m ==="
  GPUS="$GPUS" scripts/20_train.sh configs/train/qwen2.5-7b_recon7b_${m}.yaml &
  TP=$!
  sleep 150
  scripts/21_prune_checkpoints.sh saves/Qwen2.5-7B_recon7b_${m} 500 1500 &
  PP=$!
  wait $TP
  kill $PP 2>/dev/null || true
  echo "[$(date +%H:%M:%S)] === recon7b_$m done ==="
done
echo "ALL RECON7B TRAINING COMPLETE"
