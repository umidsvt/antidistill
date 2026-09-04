#!/usr/bin/env bash
# Keep only selected epoch checkpoints while a run is in flight.
#
# HF Trainer's `save_total_limit` keeps the *most recent* N, which for a 15-epoch run means
# epochs 13/14/15 — not the spread we want for a sensitivity check. This watcher instead keeps
# checkpoints at a fixed step interval (default every 500 steps = epochs 5/10/15 at the LIMO
# config's 100 steps/epoch) and deletes the rest as they appear.
#
# Safety: a checkpoint is only deleted once a *newer* checkpoint directory exists, which means
# the Trainer has finished writing it and moved on. Never deletes the newest directory.
#
# Usage:
#   scripts/21_prune_checkpoints.sh saves/Qwen2.5-7B_limo [keep_every] [total_steps]
set -euo pipefail

OUTDIR="${1:?usage: 21_prune_checkpoints.sh <output_dir> [keep_every] [total_steps]}"
KEEP_EVERY="${2:-500}"
TOTAL_STEPS="${3:-1500}"
POLL="${POLL:-60}"

echo "pruner: watching $OUTDIR — keeping every ${KEEP_EVERY} steps (through ${TOTAL_STEPS}), poll ${POLL}s"

freed=0
while true; do
  # Sorted numerically by step; the last one is the newest and is never touched.
  mapfile -t ckpts < <(find "$OUTDIR" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null \
                        | sed 's/.*checkpoint-//' | sort -n)
  n=${#ckpts[@]}
  if (( n > 1 )); then
    for (( i = 0; i < n - 1; i++ )); do
      step=${ckpts[$i]}
      d="$OUTDIR/checkpoint-$step"
      if (( step % KEEP_EVERY != 0 )); then
        sz=$(du -sm "$d" 2>/dev/null | cut -f1 || echo 0)
        rm -rf "$d" && freed=$(( freed + sz ))
        echo "pruned checkpoint-$step (${sz}MB, cumulative freed ${freed}MB)"
      else
        # A kept checkpoint: drop the ZeRO optimizer shards but keep the model weights.
        # A DeepSpeed ZeRO-3 checkpoint here is ~100GB, of which global_step*/ is ~86GB of
        # fp32 master weights + Adam moments. That is only needed to *resume* training; the
        # 14.4GB of model-*.safetensors is all evaluation needs. Stripping it takes the
        # three kept checkpoints from ~300GB to ~43GB on a shared disk.
        # Only ever applied to a checkpoint that is not the newest, so the most recent one
        # always remains fully resumable.
        for g in "$d"/global_step*; do
          [[ -d "$g" ]] || continue
          sz=$(du -sm "$g" 2>/dev/null | cut -f1 || echo 0)
          rm -rf "$g" && freed=$(( freed + sz ))
          echo "stripped optimizer state from checkpoint-$step (${sz}MB, weights kept, cumulative freed ${freed}MB)"
        done
      fi
    done
  fi

  # Stop once the final checkpoint has been written and settled.
  if [[ -d "$OUTDIR/checkpoint-$TOTAL_STEPS" ]]; then
    sleep "$POLL"
    mapfile -t ckpts < <(find "$OUTDIR" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null \
                          | sed 's/.*checkpoint-//' | sort -n)
    for step in "${ckpts[@]}"; do
      if (( step % KEEP_EVERY != 0 )); then
        rm -rf "$OUTDIR/checkpoint-$step" && echo "pruned checkpoint-$step (final sweep)"
      elif (( step != TOTAL_STEPS )); then
        # Final checkpoint keeps its optimizer state; earlier kept ones keep weights only.
        rm -rf "$OUTDIR"/checkpoint-$step/global_step* 2>/dev/null \
          && echo "stripped optimizer state from checkpoint-$step (final sweep)"
      fi
    done
    echo "pruner: done. kept: $(ls -d "$OUTDIR"/checkpoint-* 2>/dev/null | tr '\n' ' ')"
    echo "pruner: total freed ${freed}MB"
    exit 0
  fi
  sleep "$POLL"
done
