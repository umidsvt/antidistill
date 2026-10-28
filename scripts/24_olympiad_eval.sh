#!/usr/bin/env bash
# Harder evaluation: the students we already have, on OlympiadBench (675 problems, shipped with Kim et
# al.'s harness, never run here before). No training. Design and the rule fixed before any result:
# results/m7_hard_eval_design.md.
#
# One student at a time at tensor_parallel_size 4, the setting of every pooled-600 number (CLAUDE.md §8).
# Ordered so the comparison the question is about (LIMO against Solo) finishes first. A student whose
# result file exists is skipped, so the queue can be stopped and restarted.
#
#   setsid nohup scripts/24_olympiad_eval.sh > logs/olympiad_eval.log 2>&1 < /dev/null & disown
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
G="${GPUS:-4,5,6,7}"
[[ ",$G," == *",0,"* || ",$G," == *",1,"* ]] && { echo "GPUs 0 and 1 are reserved"; exit 1; }
BENCH=olympiadbench

QUEUE=(
  "base           Qwen/Qwen2.5-7B"
  "limo           saves/Qwen2.5-7B_limo/checkpoint-1500"
  "recon7b_solo   saves/Qwen2.5-7B_recon7b_solo/checkpoint-1500"
  "recon_solo     saves/Qwen2.5-7B_recon_solo/checkpoint-1500"
  "recon_style    saves/Qwen2.5-7B_recon_style/checkpoint-1500"
  "hindsight_v2   saves/Qwen2.5-7B_hindsight_v2/checkpoint-1500"
  "part           saves/Qwen2.5-7B_part/checkpoint-1500"
  "recon7b_search saves/Qwen2.5-7B_recon7b_search/checkpoint-1500"
)
[[ -n "${ONLY:-}" ]] && QUEUE=("${QUEUE[@]:0:$ONLY}")

for entry in "${QUEUE[@]}"; do
  read -r cond model <<<"$entry"
  label="${cond}_${BENCH}"
  if [[ -n "$(find "outputs/$label" -path "*/$BENCH/*_t0.0_k1_*.jsonl" 2>/dev/null | head -1)" ]]; then
    echo "[$(date '+%F %T')] SKIP  $label (result exists)"; continue
  fi
  [[ "$model" == saves/* && ! -d "$model" ]] && { echo "[$(date '+%F %T')] MISSING $model"; continue; }
  echo "[$(date '+%F %T')] START $label on GPUs $G"
  GPUS=$G bash scripts/10_eval.sh "$model" "$label" "$BENCH" 0.0 1 2>&1 | grep -E "^Acc:|pass = " | tail -2
  echo "[$(date '+%F %T')] DONE  $label"
done
echo "[$(date '+%F %T')] OLYMPIAD QUEUE COMPLETE"
