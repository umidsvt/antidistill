#!/usr/bin/env bash
# GSM8K across every condition, two streams in parallel on GPUs 6 and 7.
#
# TP=1 on purpose. The base GSM8K row was produced at TP=1 (only 2 GPUs were free), and greedy
# decoding is not bitwise-stable across tensor-parallel sizes (CLAUDE.md §8). Every GSM8K row
# must therefore use TP=1 so the benchmark is internally consistent; it is NOT comparable
# token-for-token with the MATH500/AMC/AIME rows, which were produced at TP=4.
#
# Grading: generation is scored boxed-only by the harness, exactly as every other benchmark.
# GSM8K additionally needs the last-number fallback (base is 37.2% boxed-only vs 72.2% with it,
# because word problems do not elicit \boxed{}). Both columns come from
# scripts/33_regrade_fallback.py afterwards -- no decision is baked into the run.
set -uo pipefail
cd "$(dirname "$0")/.."

run_one () {   # $1=label  $2=checkpoint-or-model  $3=gpu
  echo "[$(date +%H:%M:%S)] === $1 on GPU $3 ==="
  GPUS="$3" bash scripts/10_eval.sh "$2" "$1" gsm8k 0.0 1 2>&1 \
    | grep -E "correct cnt|^Acc:" | tail -1
}

S=saves
QUEUE_A=( "limo_gsm8k:$S/Qwen2.5-7B_limo/checkpoint-1500"
          "hindsight_v2_gsm8k:$S/Qwen2.5-7B_hindsight_v2/checkpoint-1500"
          "recon_search_gsm8k:$S/Qwen2.5-7B_recon_search/checkpoint-1500"
          "recon_solo_gsm8k:$S/Qwen2.5-7B_recon_solo/checkpoint-1500" )
QUEUE_B=( "recon_style_gsm8k:$S/Qwen2.5-7B_recon_style/checkpoint-1500"
          "recon7b_search_gsm8k:$S/Qwen2.5-7B_recon7b_search/checkpoint-1500"
          "recon7b_solo_gsm8k:$S/Qwen2.5-7B_recon7b_solo/checkpoint-1500"
          "recon7b_style_gsm8k:$S/Qwen2.5-7B_recon7b_style/checkpoint-1500" )

( for e in "${QUEUE_A[@]}"; do run_one "${e%%:*}" "${e##*:}" 6; done ) &
A=$!
( for e in "${QUEUE_B[@]}"; do run_one "${e%%:*}" "${e##*:}" 7; done ) &
B=$!
wait $A $B
echo "ALL GSM8K EVALS COMPLETE"
.venv-infer/bin/python scripts/33_regrade_fallback.py --bench gsm8k
