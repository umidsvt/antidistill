#!/usr/bin/env bash
# Parameterized wrapper around Kim et al.'s eval harness.
#
# The harness resolves ./prompts and ./data relative to the working directory, so it
# MUST run from third_party/kim_eval/ — this script handles that and writes results
# back into the repo's outputs/ tree.
#
# tensor_parallel_size is fixed at 4 (CUDA_VISIBLE_DEVICES=0,1,2,3) for every run:
# greedy decoding is not bitwise-stable across TP sizes, so this keeps all conditions
# comparable. Override with GPUS=... only if you also re-run the baselines.
#
# Usage:
#   scripts/10_eval.sh <model_path> <condition> [data_name] [temperature] [n_sampling]
#
#   # M1 headline number (greedy pass@1, the metric in the paper's table)
#   scripts/10_eval.sh Qwen/Qwen2.5-7B base aime 0.0 1
#
#   # low-variance companion metric (avg@16), reported alongside every greedy number
#   scripts/10_eval.sh Qwen/Qwen2.5-7B base aime 0.7 16
set -euo pipefail

MODEL="${1:?usage: 10_eval.sh <model_path> <condition> [data] [temp] [n_sampling]}"
CONDITION="${2:?missing condition label, e.g. base|limo|hindsight}"
DATA="${3:-aime}"
TEMP="${4:-0.0}"
NSAMP="${5:-1}"

REPO="$(cd "$(dirname "$0")/.." && pwd)"

# This script cd's into third_party/kim_eval, so a repo-relative checkpoint path (e.g.
# saves/.../checkpoint-1500) would not resolve there and transformers would treat it as a
# HuggingFace repo id. Absolutise anything that exists on disk; leave hub ids (Qwen/...) alone.
if [[ -d "$MODEL" ]]; then MODEL="$(cd "$MODEL" && pwd)"
elif [[ -d "$REPO/$MODEL" ]]; then MODEL="$(cd "$REPO/$MODEL" && pwd)"
fi
GPUS="${GPUS:-0,1,2,3}"
PY="$REPO/.venv-infer/bin/python"
OUT="$REPO/outputs/$CONDITION"

# n_sampling==1 => greedy pass@1; otherwise eval_acc.py also reports avg@n, which is a
# much lower-variance estimator of the same quantity on a 30-problem benchmark.
if [[ "$NSAMP" == "1" ]]; then SCRIPT=eval.py; EXTRA=(--k 1); else SCRIPT=eval_acc.py; EXTRA=(); fi

# top_p follows the paper: 1.0 everywhere for Qwen models (eval.py forces 1 when temp==0).
TOP_P="${TOP_P:-1.0}"

mkdir -p "$OUT" "$REPO/logs"
LOG="$REPO/logs/eval_${CONDITION}_${DATA}_t${TEMP}_k${NSAMP}.log"

echo "model=$MODEL condition=$CONDITION data=$DATA temp=$TEMP n=$NSAMP gpus=$GPUS"
echo "log -> ${LOG#$REPO/}"

cd "$REPO/third_party/kim_eval"
CUDA_VISIBLE_DEVICES="$GPUS" VLLM_ATTENTION_BACKEND=FLASH_ATTN \
"$PY" "$SCRIPT" \
  --model_name_or_path "$MODEL" \
  --data_name "$DATA" \
  --prompt_type "qwen-instruct" \
  --temperature "$TEMP" \
  --n_sampling "$NSAMP" \
  "${EXTRA[@]}" \
  --top_p "$TOP_P" \
  --max_tokens 32768 \
  --split "test" \
  --start_idx 0 --end_idx -1 \
  --seed 0 \
  --surround_with_messages \
  --gpu_memory_utilization "${GPU_MEM_UTIL:-0.90}" \
  --output_dir "$OUT" \
  --completions_save_dir "$OUT/completions" \
  2>&1 | tee "$LOG"

echo
echo "--- accuracy from written jsonl ---"
"$PY" - "$OUT" "$DATA" <<'PY'
import json, sys, pathlib
out, data = pathlib.Path(sys.argv[1]), sys.argv[2]
for f in sorted(out.rglob(f"{data}/*.jsonl")):
    rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    if not rows:
        continue
    c = sum(bool(r["is_correct"]) for r in rows)
    line = f"{f.relative_to(out)}: pass = {c}/{len(rows)} = {100*c/len(rows):.2f}%"
    if "avg_at_n" in rows[0]:
        a = sum(r["avg_at_n"] for r in rows) / len(rows)
        line += f" | avg@n = {100*a:.2f}%"
    print(line)
PY
