#!/usr/bin/env bash
# Evaluate the CORRECTED hindsight dataset (v2) on the same four benchmarks the other
# conditions got, so it drops straight into the existing comparison.
#
# Condition label is `hindsight_v2_ep15`, NEVER `hindsight_ep15`:
# scripts/30_collect_results.py takes the condition from the first path component under
# outputs/ (`condition = rel.parts[0]`), and results/eval_table.md is regenerated wholesale.
# Reusing the v1 label would merge two different models into one row.
# See results/hindsight_versions.md.
set -uo pipefail
cd "$(dirname "$0")/.."

CK=saves/Qwen2.5-7B_hindsight_v2/checkpoint-1500
[ -d "$CK" ] || { echo "ERROR: $CK missing"; exit 1; }
ls "$CK"/model-0000?-of-00004.safetensors >/dev/null 2>&1 || { echo "ERROR: weights missing"; exit 1; }

for bench in math amc aime aime25; do
  echo "[$(date +%H:%M:%S)] === $bench ==="
  bash scripts/10_eval.sh "$CK" hindsight_v2_ep15 "$bench" 0.0 1 2>&1 \
    | grep -E "correct cnt|^Acc:|pass = " | tail -2
done
echo "[$(date +%H:%M:%S)] done; refreshing the results table"
.venv-infer/bin/python scripts/30_collect_results.py
