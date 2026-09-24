#!/usr/bin/env bash
# S1 marker-stripping: train, then evaluate. GPUs 4-7 ONLY (0-1 are reserved by the user).
# Eval runs strictly AFTER training on the same GPUs: concurrent train+eval on this host ran
# ~10x slower (CLAUDE.md §7.3). Then: .venv-infer/bin/python scripts/35_strip_s1_analysis.py
set -uo pipefail
cd "$(dirname "$0")/.."
G=4,5,6,7
echo "[$(date +%H:%M:%S)] === train strip_s1 on $G ==="
GPUS=$G scripts/20_train.sh configs/train/qwen2.5-7b_strip_s1.yaml &
TP=$!
sleep 150
scripts/21_prune_checkpoints.sh saves/Qwen2.5-7B_strip_s1 500 1500 &     # REQUIRED
PP=$!
wait $TP; kill $PP 2>/dev/null || true
CK=saves/Qwen2.5-7B_strip_s1/checkpoint-1500
[ -d "$CK/global_step1500" ] && ls $CK/model-0000?-of-00004.safetensors >/dev/null 2>&1 && rm -rf $CK/global_step1500
for b in math amc aime aime25; do
  echo "[$(date +%H:%M:%S)] === eval strip_s1 / $b (TP=4) ==="
  GPUS=$G bash scripts/10_eval.sh "$CK" strip_s1_ep15 "$b" 0.0 1 2>&1 | grep -E "^Acc:" | tail -1
done
if [ "${WITH_GSM8K:-0}" = "1" ]; then
  echo "[$(date +%H:%M:%S)] === eval strip_s1 / gsm8k (TP=1, GPU 7) ==="
  GPUS=7 bash scripts/10_eval.sh "$CK" strip_s1_gsm8k gsm8k 0.0 1 2>&1 | grep -E "^Acc:" | tail -1
fi
echo "[$(date +%H:%M:%S)] DONE -> .venv-infer/bin/python scripts/35_strip_s1_analysis.py"
