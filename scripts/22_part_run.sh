#!/usr/bin/env bash
# PART defense: train, then evaluate. GPUs 4-7 ONLY (0-1 are reserved by the user).
#
#   scripts/22_part_run.sh                         # the PART condition
#   CONDITION=part_step1 scripts/22_part_run.sh    # ablation: self-talk removed, no reordering
#   WITH_GSM8K=1 ...                               # also GSM8K (TP=1, GPU 7, ~7 h)
#
# Prerequisites: data/defended/limo_part.json from scripts/45_generate_part.sh, and every gate of
# scripts/36_part_audit.py passing -- the runner refuses to train otherwise.
# Eval runs strictly AFTER training on the same GPUs: concurrent train+eval on this host ran ~10x
# slower (CLAUDE.md §7.3). Then: .venv-infer/bin/python scripts/36_part_audit.py --verdict
set -uo pipefail
cd "$(dirname "$0")/.."
G=4,5,6,7
C="${CONDITION:-part}"
CFG="configs/train/qwen2.5-7b_${C}.yaml"
SAVE="saves/Qwen2.5-7B_${C}"
mkdir -p logs
[[ -f "$CFG" ]] || { echo "no config $CFG"; exit 1; }
[[ -f "data/defended/limo_${C}.json" ]] || { echo "no dataset data/defended/limo_${C}.json -- run scripts/45_generate_part.sh"; exit 1; }

echo "[$(date +%H:%M:%S)] === audit gates (design §6) ==="
if ! .venv-infer/bin/python scripts/36_part_audit.py > "logs/part_audit_${C}.log" 2>&1; then
  tail -25 "logs/part_audit_${C}.log"; echo "GATES FAILED -- not training (logs/part_audit_${C}.log)"; exit 1
fi
grep -E "ALL GATES PASS" "logs/part_audit_${C}.log"

echo "[$(date +%H:%M:%S)] === train ${C} on $G ==="
GPUS=$G scripts/20_train.sh "$CFG" &
TP=$!
sleep 150
scripts/21_prune_checkpoints.sh "$SAVE" 500 1500 &     # REQUIRED (CLAUDE.md §5)
PP=$!
wait $TP; kill $PP 2>/dev/null || true
CK="$SAVE/checkpoint-1500"
[ -d "$CK/global_step1500" ] && ls $CK/model-0000?-of-00004.safetensors >/dev/null 2>&1 && rm -rf "$CK/global_step1500"
for b in math amc aime aime25; do
  echo "[$(date +%H:%M:%S)] === eval ${C} / $b (TP=4) ==="
  GPUS=$G bash scripts/10_eval.sh "$CK" "${C}_ep15" "$b" 0.0 1 2>&1 | grep -E "^Acc:" | tail -1
done
if [ "${WITH_GSM8K:-0}" = "1" ]; then
  echo "[$(date +%H:%M:%S)] === eval ${C} / gsm8k (TP=1, GPU 7) ==="
  GPUS=7 bash scripts/10_eval.sh "$CK" "${C}_gsm8k" gsm8k 0.0 1 2>&1 | grep -E "^Acc:" | tail -1
fi
echo "[$(date +%H:%M:%S)] DONE -> .venv-infer/bin/python scripts/36_part_audit.py --verdict"
