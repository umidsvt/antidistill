#!/usr/bin/env bash
# Setup 2 (results/m8_audit_design.md §8.5): the GSM8K setup shared by ADS, PoE, trace rewriting, DOGe and TARS.
# For seeds 123-125: the untrained student (base), U (distilled from the paper's teacher), Solo (distilled from
# Qwen2.5-7B-Instruct) and RL from the untrained student rewarded by the teacher's answers (needs U's run).
# Four lanes, one per GPU; with fewer GPUs, lanes share GPUs in turn. Finished parts are skipped.
#
#   GPUS=2,3,4,5 setsid nohup scripts/29_setup2_run.sh > logs/setup2_run.log 2>&1 < /dev/null & disown
set -uo pipefail
cd "$(dirname "$0")/.."
G="${GPUS:-2,3,4,5}"
[[ ",$G," == *",0,"* || ",$G," == *",1,"* ]] && { echo "GPUs 0 and 1 are reserved"; exit 1; }
IFS=, read -ra GPU <<<"$G"
export PYTHONPATH=third_party/ext/distillation-game/src HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
       HF_HOME=/mnt/vault/huggingface_cache TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv-dg/bin/python
OUT=outputs/setup2
ts() { date '+%F %T'; }
latest() { ls -dt "$OUT"/"$1"_[0-9]* 2>/dev/null | head -1; }      # newest finished-or-not run folder

run_distill() {   # $1 gpu, $2 config stem (e.g. gsm8k_U_s123)
  local d; d="$(latest "$2")"
  if [[ -n "$d" && -f "$d/RESULTS.md" ]]; then echo "[$(ts)] SKIP $2 (done: $d)"; return; fi
  echo "[$(ts)] START $2 on GPU $1"
  CUDA_VISIBLE_DEVICES=$1 $PY third_party/ext/distillation-game/scripts/run_pipeline.py \
    --config "configs/setup2/$2.yaml" > "logs/setup2_$2.log" 2>&1
  d="$(latest "$2")"
  echo "[$(ts)] DONE  $2: $(grep -a -A4 'Comparison Matrix' "$d/RESULTS.md" 2>/dev/null | tail -1)"
}
run_base() {      # $1 gpu, $2 seed
  [[ -f "$OUT/base_gsm8k_s$2.json" ]] && { echo "[$(ts)] SKIP base s$2"; return; }
  echo "[$(ts)] START base s$2 on GPU $1"
  CUDA_VISIBLE_DEVICES=$1 $PY scripts/dg_eval_base.py "configs/setup2/gsm8k_U_s$2.yaml" > "logs/setup2_base_s$2.log" 2>&1
  echo "[$(ts)] DONE  base s$2: $(tail -1 "logs/setup2_base_s$2.log")"
}
run_rl() {        # $1 gpu, $2 seed
  [[ -f "$OUT/rl_gsm8k_s$2/result.json" ]] && { echo "[$(ts)] SKIP rl s$2"; return; }
  local d; d="$(latest "gsm8k_U_s$2")"
  [[ -f "$d/teacher/train_standard.json" ]] || { echo "[$(ts)] rl s$2: no teacher answers (U run missing)"; return; }
  echo "[$(ts)] START rl s$2 on GPU $1"
  CUDA_VISIBLE_DEVICES=$1 DG_MASTER_PORT=$((29500 + $1)) $PY scripts/dg_rl_base.py --config "configs/setup2/gsm8k_U_s$2.yaml" \
    --teacher-train "$d/teacher/train_standard.json" --vllm-mem 0.5 > "logs/setup2_rl_s$2.log" 2>&1
  echo "[$(ts)] DONE  rl s$2: $(tail -1 "logs/setup2_rl_s$2.log")"
}

lane() {          # $1 lane number, $2 gpu
  case $1 in
    0) run_distill "$2" gsm8k_U_s123; run_rl "$2" 123 ;;
    1) run_distill "$2" gsm8k_U_s124; run_rl "$2" 124 ;;
    2) run_distill "$2" gsm8k_U_s125; run_rl "$2" 125 ;;
    3) run_base "$2" 123; run_base "$2" 124; run_base "$2" 125
       run_distill "$2" gsm8k_solo_q7i_s123; run_distill "$2" gsm8k_solo_q7i_s124; run_distill "$2" gsm8k_solo_q7i_s125 ;;
  esac
}

echo "[$(ts)] setup 2 on GPUs $G"
n=${#GPU[@]}
for ((g = 0; g < n; g++)); do
  ( for ((l = g; l < 4; l += n)); do lane "$l" "${GPU[$g]}"; done ) &
done
wait
echo "[$(ts)] SETUP 2 DONE"
