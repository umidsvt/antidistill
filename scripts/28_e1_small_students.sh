#!/usr/bin/env bash
# E1 (results/m8_audit_design.md §6-7): do Qwen2.5-1.5B and 3B students gain from LIMO's 800 long solutions?
# Untrained, LIMO (U) and 7B Solo for both sizes, on the pooled 600 problems (MATH500, AMC23, AIME24, AIME25).
#
# Two GPUs (default 6,7): training one run at a time on both (global batch 8 = 2 x 1 x 4); evaluation at
# TP=1, one student per GPU, the same for all six E1 students. Finished results are skipped, so the script
# can be restarted. New model folders go to the root disk (the shared vault is at 98%).
#
#   setsid nohup scripts/28_e1_small_students.sh > logs/e1_small_students.log 2>&1 < /dev/null & disown
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
G="${GPUS:-6,7}"
IFS=, read -r GA GB <<<"$G"
[[ ",$G," == *",0,"* || ",$G," == *",1,"* ]] && { echo "GPUs 0 and 1 are reserved"; exit 1; }
ts() { date '+%F %T'; }
BENCH=(math amc aime aime25)
LOCAL_SAVES="$(cd .. && pwd)/saves_local"

# The watchdog is a child of this script, so every `wait` below names its PIDs (a bare `wait` would wait for
# the watchdog, which only exits after this script: deadlock, 2026-10-20 05:20-07:55).
WATCH_WHILE="scripts/28_e1_small_students.sh" setsid nohup scripts/26_eval_exit_watchdog.sh \
  >> logs/eval_exit_watchdog.log 2>&1 < /dev/null &

eval_one() {   # $1 gpu, $2 model (path or hub id), $3 condition label
  local b
  for b in "${BENCH[@]}"; do
    [[ -n "$(find "outputs/$3" -path "*/$b/*_t0.0_k1_*.jsonl" 2>/dev/null | head -1)" ]] && continue
    echo "[$(ts)] eval $3 / $b on GPU $1"
    GPUS=$1 bash scripts/10_eval.sh "$2" "$3" "$b" 0.0 1 2>&1 | grep -E "^Acc:" | tail -1 | sed "s|^|[$(ts)] $3 / $b: |"
  done
}

echo "[$(ts)] === E1 on GPUs $G: untrained students ==="
eval_one "$GA" Qwen/Qwen2.5-1.5B q15b_base & p1=$!
eval_one "$GB" Qwen/Qwen2.5-3B q3b_base & p2=$!
wait $p1 $p2

for c in qwen2.5-1.5b_limo qwen2.5-1.5b_recon7b_solo qwen2.5-3b_limo qwen2.5-3b_recon7b_solo; do
  out="$(awk '/^output_dir:/ {print $2}' "configs/train/$c.yaml")"
  if ls "$out"/model*.safetensors > /dev/null 2>&1; then echo "[$(ts)] SKIP train $c (model exists)"; continue; fi
  [[ -e "$out" ]] || { mkdir -p "$LOCAL_SAVES/$(basename "$out")" && ln -s "$LOCAL_SAVES/$(basename "$out")" "$out"; }
  echo "[$(ts)] === train $c on $G -> logs/train_$c.log ==="
  # 20_train.sh writes logs/train_$c.log itself; its exit code is tee's, so success = the model was saved
  GPUS=$G scripts/20_train.sh "configs/train/$c.yaml" > "logs/train_$c.wrapper.log" 2>&1
  if ls "$out"/model*.safetensors > /dev/null 2>&1; then
    echo "[$(ts)] trained $c ($(grep -a -E '^elapsed:' "logs/train_$c.log" | tail -1))"
  else
    echo "[$(ts)] train $c FAILED: no model saved (see logs/train_$c.log)"
  fi
done

echo "[$(ts)] === E1: trained students ==="
eval_one "$GA" saves/Qwen2.5-1.5B_limo q15b_limo & p1=$!
eval_one "$GB" saves/Qwen2.5-3B_limo q3b_limo & p2=$!
wait $p1 $p2
eval_one "$GA" saves/Qwen2.5-1.5B_recon7b_solo q15b_solo7b & p1=$!
eval_one "$GB" saves/Qwen2.5-3B_recon7b_solo q3b_solo7b & p2=$!
wait $p1 $p2
echo "[$(ts)] E1 DONE"
