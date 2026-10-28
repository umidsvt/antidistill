#!/usr/bin/env bash
# Attacks on PART, end to end: ATTACK=revoice_part (m6e) or ATTACK=style_part (m6f): generate the data, apply the
# §5 checks, train, evaluate. GPUs 4-7 ONLY (0-1 are reserved). Stops before training if a check fails.
# Then: .venv-infer/bin/python scripts/37_revoice_audit.py --verdict
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
G=4,5,6,7
C="${ATTACK:-revoice_part}"          # revoice_part | style_part
MODULE="antidistill.attacks.${C%_part}"
[[ "$C" == revoice_part ]] && MODULE=antidistill.attacks.revoice
[[ "$C" == style_part ]] && MODULE=antidistill.attacks.style_part
CFG="configs/train/qwen2.5-7b_${C}.yaml"
SAVE="saves/Qwen2.5-7B_${C}"
# `saves/` links to the shared vault, which was at 98% (190 GB free) on 2026-10-12; a run needs ~200 GB
# at peak and a disk-full crash is not resumable (CLAUDE.md §5). So a NEW run's folder is created on the
# root disk and linked into saves/, leaving the config's output_dir and all other paths unchanged.
LOCAL_SAVES="${LOCAL_SAVES:-$(cd .. && pwd)/saves_local}"
if [[ ! -e "$SAVE" ]]; then mkdir -p "$LOCAL_SAVES/Qwen2.5-7B_${C}" && ln -s "$LOCAL_SAVES/Qwen2.5-7B_${C}" "$SAVE"; fi
SAVE_REAL="$(readlink -f "$SAVE")"     # the pruner's `find` does not descend into a symlinked start path
echo "checkpoints -> $SAVE_REAL ($(df -h "$SAVE_REAL" | tail -1 | awk '{print $4}') free)"

# SKIP_GEN=1: the data was already generated (the run stopped at the checks); do not start the rewriter again.
if [[ -n "${SKIP_GEN:-}" && -s "data/curated/${C}.json" && -s "data/curated/${C}.meta.json" ]]; then
  echo "[$(date +%H:%M:%S)] === data exists, generation skipped: data/curated/${C}.json ==="
else
  echo "[$(date +%H:%M:%S)] === generate re-voiced data on $G ==="
  GEN_MODULE=$MODULE INPUT=data/defended/limo_part.json OUTPUT=data/curated/${C}.json \
    GPUS=$G GPUS2=${GPUS2:-} scripts/45_generate_part.sh || { echo "generation failed"; exit 1; }
fi

echo "[$(date +%H:%M:%S)] === checks (design §5) ==="
# WAIVE=1 adds --waive-self-talk: the restoration check is reported but does not block (used for the
# re-voice run by decision 2026-10-06, m6e §5); every other check still blocks.
# WAIVE_ANSWER=<indices>: the final-answer flag on exactly those problems does not block (style_part, problem
# 355, decision 2026-10-13, m6f "Full data"); a flag on any other problem still does.
if ! .venv-infer/bin/python scripts/37_revoice_audit.py --attack "$C" ${WAIVE:+--waive-self-talk} \
     ${WAIVE_ANSWER:+--waive-answer "$WAIVE_ANSWER"} > "logs/${C}_audit.log" 2>&1; then
  tail -15 "logs/${C}_audit.log"; echo "CHECKS FAILED -- not training"; exit 1
fi
grep -E "ALL CHECKS PASS|WAIVED" "logs/${C}_audit.log"

echo "[$(date +%H:%M:%S)] === train ${C} on $G ==="
GPUS=$G scripts/20_train.sh "$CFG" &
TP=$!
sleep 150
scripts/21_prune_checkpoints.sh "$SAVE_REAL" 500 1500 &     # REQUIRED (CLAUDE.md §5)
PP=$!
wait $TP; kill $PP 2>/dev/null || true
CK="$SAVE/checkpoint-1500"
[ -d "$CK/global_step1500" ] && ls $CK/model-0000?-of-00004.safetensors >/dev/null 2>&1 && rm -rf "$CK/global_step1500"
# the top-level final model is a byte-identical copy of checkpoint-1500: hard-link it (saves ~15 GB)
for f in "$SAVE"/model-0000?-of-00004.safetensors; do c="$CK/$(basename "$f")"; [ -f "$c" ] && cmp -s "$f" "$c" && ln -f "$c" "$f"; done
for b in math amc aime aime25; do
  echo "[$(date +%H:%M:%S)] === eval ${C} / $b (TP=4) ==="
  GPUS=$G bash scripts/10_eval.sh "$CK" "${C}_ep15" "$b" 0.0 1 2>&1 | grep -E "^Acc:" | tail -1
done
if [[ -n "${SKIP_GSM8K:-}" ]]; then      # run later: GPUS=7 bash scripts/10_eval.sh <ckpt> ${C}_gsm8k gsm8k 0.0 1
  echo "[$(date +%H:%M:%S)] === gsm8k skipped (SKIP_GSM8K) ==="
else
  echo "[$(date +%H:%M:%S)] === eval ${C} / gsm8k (TP=1, GPU 7) ==="
  GPUS=7 bash scripts/10_eval.sh "$CK" "${C}_gsm8k" gsm8k 0.0 1 2>&1 | grep -E "^Acc:" | tail -1
fi
echo "[$(date +%H:%M:%S)] DONE -> .venv-infer/bin/python scripts/37_revoice_audit.py --attack $C --verdict"
