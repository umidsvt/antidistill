#!/usr/bin/env bash
# Merge a LoRA adapter checkpoint into the base weights, producing a standalone model
# directory that scripts/10_eval.sh (and vLLM underneath it) can load.
#
# Why this is needed: the eval harness hands its --model_name_or_path straight to vLLM as a
# full model. A checkpoint written by the LoRA configs contains only adapter tensors
# (~161MB), so it must be merged first.
#
# Merged output is ~15GB per checkpoint, so merge only the checkpoints you intend to
# evaluate -- not all fifteen epochs.
#
# Usage:
#   scripts/23_merge_lora.sh saves/Qwen2.5-7B_limo_lora32k/checkpoint-1500
#   scripts/23_merge_lora.sh <ckpt> <export_dir>
set -euo pipefail

CKPT="${1:?usage: 23_merge_lora.sh <adapter_checkpoint_dir> [export_dir]}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
[[ -d "$CKPT" ]] || { echo "no such checkpoint: $CKPT" >&2; exit 1; }
[[ -f "$CKPT/adapter_config.json" ]] || {
  echo "$CKPT has no adapter_config.json — is it a LoRA checkpoint?" >&2; exit 1; }

CKPT="$(cd "$CKPT" && pwd)"
EXPORT_DIR="${2:-merged/$(basename "$(dirname "$CKPT")")_$(basename "$CKPT")}"
mkdir -p "$(dirname "$EXPORT_DIR")"

CFG="$(mktemp -t merge_lora_XXXXXX.yaml)"
trap 'rm -f "$CFG"' EXIT
cat > "$CFG" <<YAML
model_name_or_path: Qwen/Qwen2.5-7B
adapter_name_or_path: $CKPT
template: qwen
finetuning_type: lora
trust_remote_code: true
export_dir: $EXPORT_DIR
export_size: 5
export_device: cpu
export_legacy_format: false
YAML

echo "adapter : $CKPT"
echo "export  : $EXPORT_DIR"
export PATH="$REPO/.venv-train/bin:$PATH"
# Merging is a CPU operation; keep it off the GPUs so it can run alongside a training job.
CUDA_VISIBLE_DEVICES="" "$REPO/.venv-train/bin/llamafactory-cli" export "$CFG"

echo
echo "merged model written to $EXPORT_DIR"
du -sh "$EXPORT_DIR"
echo "evaluate with: scripts/10_eval.sh $EXPORT_DIR <condition> aime 0.0 1"
