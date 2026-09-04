#!/usr/bin/env bash
# SFT driver. Thin wrapper around llamafactory-cli so every run is logged and timed
# identically across conditions.
#
# Must run from the repo root: the training yamls reference `configs/deepspeed/...` and
# `dataset_dir: data` relatively.
#
# Usage:
#   scripts/20_train.sh configs/train/qwen2.5-7b_limo.yaml
#   GPUS=0,1,2,3 scripts/20_train.sh configs/train/qwen2.5-7b_limo.yaml   # partial node
#
# NOTE on GPUS: the LIMO config is per_device_train_batch_size=1, gradient_accumulation_steps=1,
# so the *global batch size equals the GPU count*. Changing GPUS silently changes the effective
# hyperparameters (8 GPUs -> global batch 8, 100 steps/epoch). Use all 8 for the replication;
# if you must use fewer, raise gradient_accumulation_steps to compensate and record it.
set -euo pipefail

CONFIG="${1:?usage: 20_train.sh <config.yaml>}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
NGPU=$(awk -F',' '{print NF}' <<<"$GPUS")
NAME="$(basename "${CONFIG%.yaml}")"
LOG="$REPO/logs/train_${NAME}.log"
mkdir -p "$REPO/logs"

OUTDIR=$(awk '/^output_dir:/{print $2}' "$CONFIG")
echo "config      : $CONFIG"
echo "output_dir  : $OUTDIR"
echo "gpus        : $GPUS  (global batch = $NGPU x bs x grad_accum)"
echo "log         : ${LOG#$REPO/}"

# Fail early on the two things that silently waste hours.
[[ -d "$OUTDIR" ]] && echo "WARNING: $OUTDIR exists and overwrite_output_dir will clobber it."
AVAIL=$(df -BG --output=avail "$REPO" | tail -1 | tr -dc '0-9')
echo "disk free   : ${AVAIL}G  (MEASURED: each ZeRO-3 checkpoint is ~100GB - 14.4GB weights + ~86GB optimizer state. 15 epochs unpruned = ~1.5TB, so scripts/21_prune_checkpoints.sh is REQUIRED, not optional.)"
if (( AVAIL < 250 )); then
  echo "ERROR: under 250G free; prune old checkpoints before starting." >&2
  exit 1
fi

# llamafactory-cli shells out to `torchrun` *by name* under FORCE_TORCHRUN, so the venv's bin
# must be on PATH — invoking the CLI by absolute path alone is not enough.
export PATH="$REPO/.venv-train/bin:$PATH"

# HOST WORKAROUND — mandatory on this node.
# GPU-to-GPU P2P is advertised by CUDA (`can_device_access_peer` is true for every pair) but is
# non-functional: any NCCL collective using it hangs forever. A bare 1-element all_reduce times
# out after 90s on all 8 ranks (reproduce with `scripts/diag_nccl.py`). This is the classic
# symptom of ACS being enabled on the PCIe bridges (or IOMMU translation) — peer DMA is silently
# black-holed. Fixing it properly needs root (BIOS ACS / IOMMU), so we route collectives through
# host shared memory instead. Costs bandwidth (~1.5 GB/s bus bw) but works.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"

# Fragmentation control. The first M2 attempt OOMed at step 2 with 33.5GB allocated, a 9.28GB
# request failing, and 7-10GB sitting "reserved but unallocated" — i.e. lost to fragmentation.
# The 9.28GB request is the fp32 cross-entropy upcast over Qwen's 152k vocab at seq len 16384;
# it is allocated and freed every step, which is exactly the pattern that fragments the caching
# allocator. expandable_segments lets the allocator grow/shrink segments instead of stranding
# them. Purely an allocator change — no effect on numerics or results.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

START=$(date +%s)
CUDA_VISIBLE_DEVICES="$GPUS" FORCE_TORCHRUN=1 \
  "$REPO/.venv-train/bin/llamafactory-cli" train "$CONFIG" 2>&1 | tee "$LOG"

ELAPSED=$(( $(date +%s) - START ))
printf 'elapsed: %02d:%02d:%02d\n' $((ELAPSED/3600)) $((ELAPSED%3600/60)) $((ELAPSED%60)) | tee -a "$LOG"

echo
echo "checkpoints written:"
du -sh "$OUTDIR"/checkpoint-* 2>/dev/null || true
echo
echo "Prune to the final checkpoint plus epochs 5 and 10 before the next run (see plan §5.3)."
