#!/usr/bin/env bash
# Resume after GPUs 4-7 were lent to a labmate (our runs stopped 2026-10-15 15:57): the OlympiadBench queue
# (finished students are skipped, so PART and 7B Search remain), then GSM8K for Style-on-PART on GPU 7.
# Starts its own shutdown-hang watchdog (scripts/26_eval_exit_watchdog.sh). Refuses to start if GPUs 4-7
# are in use.
#
#   setsid nohup scripts/27_resume_after_pause.sh > logs/resume_after_pause.log 2>&1 < /dev/null & disown
set -uo pipefail
cd "$(dirname "$0")/.."
ts() { date '+%F %T'; }
used="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$1 >= 4 {s += $2} END {print s + 0}')"
(( used < 2000 )) || { echo "[$(ts)] GPUs 4-7 are in use ($used MiB) -- not starting"; exit 1; }

WATCH_WHILE="scripts/27_resume_after_pause.sh" setsid nohup scripts/26_eval_exit_watchdog.sh \
  >> logs/eval_exit_watchdog.log 2>&1 < /dev/null &

echo "[$(ts)] === OlympiadBench: remaining students (GPUs 4-7) ==="
scripts/24_olympiad_eval.sh >> logs/olympiad_eval.log 2>&1

CK=saves/Qwen2.5-7B_style_part/checkpoint-1500
if [[ -z "$(find outputs/style_part_gsm8k -name '*_t0.0_k1_*.jsonl' 2>/dev/null | head -1)" ]]; then
  echo "[$(ts)] === Style-on-PART: GSM8K (TP=1, GPU 7) ==="
  GPUS=7 bash scripts/10_eval.sh "$CK" style_part_gsm8k gsm8k 0.0 1 2>&1 | grep -E "^Acc:" | tail -1
fi
echo "[$(ts)] ALL DONE"
