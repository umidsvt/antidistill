#!/usr/bin/env bash
# Sequencing on GPUs 4-7 after the decision of 2026-10-13 to train Style-on-PART on the data as generated
# (results/m6f_style_part_design.md, "Full data"). Training and the OlympiadBench queue need the same four
# GPUs, and training and evaluation must not overlap on this host, so:
#   1. wait until the OlympiadBench queue has finished the four students its rule needs
#      (no training, LIMO, 7B Solo, 32B Solo), then stop the queue;
#   2. train Style-on-PART and evaluate it on the pooled 600;
#   3. restart the OlympiadBench queue for the remaining students (finished ones are skipped);
#   4. GSM8K for Style-on-PART on GPU 7 (TP=1, as for every other student).
#
#   setsid nohup scripts/25_style_part_after_solo.sh > logs/style_part_after_solo.log 2>&1 < /dev/null & disown
set -uo pipefail
cd "$(dirname "$0")/.."
QLOG=logs/olympiad_eval.log
ts() { date '+%F %T'; }

echo "[$(ts)] waiting for the OlympiadBench queue to finish 32B Solo"
until grep -q "DONE  recon_solo_olympiadbench" "$QLOG"; do
  pgrep -f "scripts/24_olympiad_eval.sh" > /dev/null \
    || { echo "[$(ts)] the queue is not running and 32B Solo is not done -- stopping, nothing started"; exit 1; }
  sleep 30
done

# Stop the queue (it has just started the next student; no result file exists for it yet, so the restart
# in step 3 runs it from the beginning). The queue was started with setsid, so its session holds the
# wrapper, the harness and the vLLM workers.
QPID="$(pgrep -f 'scripts/24_olympiad_eval.sh' | head -1)"
if [[ -n "$QPID" ]]; then
  SID="$(ps -o sid= -p "$QPID" | tr -d ' ')"
  echo "[$(ts)] stopping the queue (session $SID)"
  pkill -TERM -s "$SID"; sleep 20; pkill -KILL -s "$SID" 2>/dev/null
fi
used=99999
for _ in $(seq 60); do
  used="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$1 >= 4 {s += $2} END {print s + 0}')"
  (( used < 2000 )) && break
  sleep 10
done
(( used < 2000 )) || { echo "[$(ts)] GPUs 4-7 are not free ($used MiB in use) -- stopping, training not started"; exit 1; }

echo "[$(ts)] === Style-on-PART: checks, training, pooled-600 evaluation -> logs/style_part_run2.log ==="
ATTACK=style_part SKIP_GEN=1 WAIVE_ANSWER=355 SKIP_GSM8K=1 scripts/23_revoice_run.sh > logs/style_part_run2.log 2>&1
echo "[$(ts)] Style-on-PART runner exit code $?"
.venv-infer/bin/python scripts/37_revoice_audit.py --attack style_part --verdict > logs/style_part_verdict.log 2>&1
tail -n 3 logs/style_part_verdict.log

echo "[$(ts)] === OlympiadBench: remaining students ==="
scripts/24_olympiad_eval.sh >> "$QLOG" 2>&1

CK=saves/Qwen2.5-7B_style_part/checkpoint-1500
if [[ -d "$CK" ]]; then
  echo "[$(ts)] === Style-on-PART: GSM8K (TP=1, GPU 7) ==="
  GPUS=7 bash scripts/10_eval.sh "$CK" style_part_gsm8k gsm8k 0.0 1 2>&1 | grep -E "^Acc:" | tail -1
fi
echo "[$(ts)] ALL DONE"
