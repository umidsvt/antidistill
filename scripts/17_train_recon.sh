#!/usr/bin/env bash
# Train the three M6b reconstruction conditions back to back on one 4-GPU group.
# The pruner is launched WITH each trainer, never as an afterthought: a ZeRO-3 checkpoint
# is ~100 GB and save_strategy:epoch x 15 epochs needs ~1.5 TB (see CLAUDE.md §5).
set -uo pipefail
cd "$(dirname "$0")/.."
GPUS="${GPUS:-0,1,2,3}"
for m in search solo style; do
  echo "[$(date +%H:%M:%S)] === training recon_$m ==="
  GPUS="$GPUS" scripts/20_train.sh configs/train/qwen2.5-7b_recon_${m}.yaml &
  TP=$!
  sleep 150
  scripts/21_prune_checkpoints.sh saves/Qwen2.5-7B_recon_${m} 500 1500 &
  PP=$!
  wait $TP
  kill $PP 2>/dev/null || true
  echo "[$(date +%H:%M:%S)] === recon_$m done ==="
done
echo "ALL RECON TRAINING COMPLETE"
