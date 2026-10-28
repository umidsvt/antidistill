#!/usr/bin/env bash
# Watchdog for a vLLM shutdown hang seen twice (2026-10-13 and 2026-10-14): eval.py writes its result file
# and prints its score, then the engine process never exits and keeps the GPUs (27 h lost on 2026-10-14).
# When a running eval.py's result file exists and is older than GRACE seconds, its engine is stopped;
# eval.py then exits and the queue moves on. The result file is not touched. Exits when the script it
# serves has finished (WATCH_WHILE, default scripts/25_style_part_after_solo.sh).
#
#   setsid nohup scripts/26_eval_exit_watchdog.sh > logs/eval_exit_watchdog.log 2>&1 < /dev/null & disown
set -uo pipefail
cd "$(dirname "$0")/.."
GRACE="${GRACE:-300}"          # a normal exit takes seconds after the file is written
WATCH_WHILE="${WATCH_WHILE:-scripts/25_style_part_after_solo.sh}"
ts() { date '+%F %T'; }

echo "[$(ts)] watching eval.py processes (grace ${GRACE}s)"
while pgrep -f "$WATCH_WHILE" > /dev/null; do
  for pid in $(pgrep -u "$(id -un)" -f "eval.py --model_name_or_path"); do
    args="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null)" || continue
    out="$(sed -n 's/.*--output_dir \([^ ]*\).*/\1/p' <<<"$args")"
    data="$(sed -n 's/.*--data_name \([^ ]*\).*/\1/p' <<<"$args")"
    [[ -n "$out" && -n "$data" ]] || continue
    # the queue skips students whose result exists, so a file found here was written by this run
    f="$(find "$out" -path "*/$data/*_t0.0_k1_*.jsonl" 2>/dev/null | head -1)"
    [[ -n "$f" ]] || continue
    age=$(( $(date +%s) - $(stat -c %Y "$f") ))
    (( age > GRACE )) || continue
    eng="$(pgrep -P "$pid" -f "VLLM::EngineCore" | tr '\n' ' ')"
    echo "[$(ts)] eval.py $pid: $f written ${age}s ago ($(wc -l < "$f") rows), still running -> stopping engine ${eng:-none}"
    [[ -n "$eng" ]] && kill -TERM $eng
    sleep 30
    if kill -0 "$pid" 2>/dev/null; then
      for c in $(pgrep -P "$pid"); do pkill -KILL -P "$c"; kill -KILL "$c" 2>/dev/null; done
      echo "[$(ts)] eval.py $pid: engine and workers killed"
    else
      echo "[$(ts)] eval.py $pid exited"
    fi
  done
  sleep 60
done
echo "[$(ts)] $WATCH_WHILE finished; watchdog exits"
