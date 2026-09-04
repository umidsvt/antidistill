#!/usr/bin/env bash
# Multi-benchmark sweep: is the LIMO effect visible at all on this hardware?
#
# AIME24 alone is 30 problems at 3.33 pp granularity, which M2 showed cannot resolve a
# +13.4 pp effect (4 argmax tie-breaks span it). AMC23 (40) + AIME25 (30) + MATH500 (499)
# add 569 more problems using checkpoints we already have.
#
# Ordered cheapest-first: the base model terminates normally (~2 s/prompt) so its three
# benchmarks finish quickly; the LIMO model runs to the token cap on hard problems and is
# far slower. MATH500 is last because it is 499 problems, but it is also where the
# statistical power is — and, being easy, the LIMO model may terminate normally there.
set -uo pipefail
cd "$(dirname "$0")/.."

while pgrep -f "kim_eval/eval(_acc)?\.py" >/dev/null 2>&1; do sleep 30; done

CK15="saves/Qwen2.5-7B_limo/checkpoint-1500"

QUEUE=(
  "base       Qwen/Qwen2.5-7B amc"
  "base       Qwen/Qwen2.5-7B aime25"
  "base       Qwen/Qwen2.5-7B math"
  "limo_ep15  $CK15           amc"
  "limo_ep15  $CK15           aime25"
  "limo_ep15  $CK15           math"
)

for entry in "${QUEUE[@]}"; do
  read -r label model bench <<<"$entry"
  echo "=============================================================="
  echo "[$(date +%H:%M:%S)] START $label / $bench"
  bash scripts/10_eval.sh "$model" "$label" "$bench" 0.0 1 \
    2>&1 | grep -E "correct cnt|^Acc:|pass = " | tail -2
  echo "[$(date +%H:%M:%S)] DONE  $label / $bench"
done

echo "MULTIBENCH COMPLETE"
