#!/usr/bin/env bash
# SFT driver. Thin wrapper around llamafactory-cli so every run is logged and timed
# identically across conditions.
#
# Must run from the repo root: the training yamls reference `configs/deepspeed/...` and
# `dataset_dir: data` relatively.
#
# Usage:
#   GPUS=4,5,6,7 scripts/20_train.sh configs/train/qwen2.5-7b_limo_s43.yaml
#
# NOTE on GPUS: LIMO's recipe has global batch 8 = GPUs x per_device_train_batch_size x
# gradient_accumulation_steps. One job's GPUs must come from one group (0-3 or 4-7), so the 7B configs
# use 4 GPUs x 1 x 2 and the 1.5B/3B configs 2 x 1 x 4; the seed-42 qwen2.5-7b_limo.yaml (8 x 1 x 1)
# no longer fits one group. The script refuses any other global batch unless GLOBAL_BATCH_OK=1.
set -euo pipefail

CONFIG="${1:?usage: 20_train.sh <config.yaml>}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

GPUS="${GPUS:-0,1,2,3}"
source scripts/gpu_env.sh; require_one_group "$GPUS"
NGPU=$(awk -F',' '{print NF}' <<<"$GPUS")
BS=$(awk '/^per_device_train_batch_size:/{print $2}' "$CONFIG")
GA=$(awk '/^gradient_accumulation_steps:/{print $2}' "$CONFIG")
if (( NGPU * ${BS:-1} * ${GA:-1} != 8 )) && [[ "${GLOBAL_BATCH_OK:-0}" != 1 ]]; then
  echo "ERROR: global batch $NGPU x ${BS:-1} x ${GA:-1} != 8 (LIMO's recipe); set GLOBAL_BATCH_OK=1 to override." >&2
  exit 1
fi
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

# HOST-SPECIFIC settings come from .host_profile, written by scripts/check_host.sh.
#
# Nothing about the interconnect is hardcoded here on purpose. The machine this was developed on
# has GPU P2P advertised but non-functional, and needs NCCL_P2P_DISABLE=1 — but that setting is
# HARMFUL on a healthy host, where it disables NVLink and forces collectives through host memory.
# So we measure per host rather than assume. Run `bash scripts/check_host.sh` once per machine.
if [[ -f "$REPO/.host_profile" ]]; then
  # shellcheck disable=SC1091
  source "$REPO/.host_profile"
else
  echo "WARNING: no .host_profile — run 'bash scripts/check_host.sh' first." >&2
  echo "         If training hangs with GPUs at 100% util but low power draw (~90W), that is" >&2
  echo "         NCCL busy-wait spin, not compute: your host likely needs NCCL_P2P_DISABLE=1." >&2
fi

# Fragmentation control (host-independent). The first M2 attempt OOMed at step 2 with 33.5GB
# allocated, a 9.28GB request failing, and 7-10GB "reserved but unallocated" — lost to
# fragmentation. That 9.28GB is the fp32 cross-entropy upcast over a large vocabulary at seq
# 16384, allocated and freed every step, which is exactly what fragments the caching allocator.
# Purely an allocator change — no effect on numerics or results.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

START=$(date +%s)
CUDA_VISIBLE_DEVICES="$GPUS" FORCE_TORCHRUN=1 $(pin "$GPUS") \
  "$REPO/.venv-train/bin/llamafactory-cli" train "$CONFIG" 2>&1 | tee "$LOG"

ELAPSED=$(( $(date +%s) - START ))
printf 'elapsed: %02d:%02d:%02d\n' $((ELAPSED/3600)) $((ELAPSED%3600/60)) $((ELAPSED%60)) | tee -a "$LOG"

echo
echo "checkpoints written:"
du -sh "$OUTDIR"/checkpoint-* 2>/dev/null || true
echo
echo "Prune to the final checkpoint plus epochs 5 and 10 before the next run (see plan §5.3)."
