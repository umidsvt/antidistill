#!/usr/bin/env bash
# Setup 2, level A by self-training (m8 §9.3): STaR for seeds 123-125 on one GPU (default 2), each job waiting for
# free memory and retried twice. Phase gen (vLLM) and phase train run as separate processes so the GPU memory of the
# first is fully released. FEWSHOT=1 runs the few-shot recipe (approved 2026-10-27) into star_fewshot_<ds>_s<seed>/.
#   GPU=2 SEEDS="123 125" setsid nohup scripts/35_star_pool.sh > logs/setup2_star.log 2>&1 < /dev/null & disown
#   GPU=3 SEEDS=123 FEWSHOT=1 setsid nohup scripts/35_star_pool.sh > logs/setup2_star_fewshot_s123.log 2>&1 < /dev/null & disown
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=third_party/ext/distillation-game/src HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
       HF_HOME=/mnt/vault/huggingface_cache TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv-dg/bin/python; OUT=outputs/setup2; G="${GPU:-2}"; DS="${DS:-gsm8k}"
TAG=""; EXTRA=(); [[ -n "${FEWSHOT:-}" ]] && { TAG="fewshot_"; EXTRA=(--fewshot); }
[[ $G == 0 || $G == 1 ]] && { echo "GPUs 0 and 1 are reserved"; exit 1; }
ts() { date '+%F %T'; }
latest() { ls -dt "$OUT"/"$1"_[0-9]* 2>/dev/null | head -1; }
wait_free() { while (( $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$G" | tr -d ' ') < 36000 )); do sleep 300; done; }
for s in ${SEEDS:-123 124 125}; do
  dir="$OUT/star_${TAG}${DS}_s$s"; res="$dir/result.json"
  glog="logs/setup2_star_${TAG}gen_${DS}_s$s.log"; tlog="logs/setup2_star_${TAG}train_${DS}_s$s.log"
  teacher="$(latest "${DS}_U_s$s")/teacher/train_standard.json"
  [[ -f "$teacher" ]] || { echo "[$(ts)] s$s: no teacher answers ($teacher)"; continue; }
  for i in 1 2 3; do
    [[ -f "$res" ]] && break
    wait_free
    if [[ ! -f "$dir/traces.json" ]]; then
      echo "[$(ts)] START star ${TAG}gen s$s on GPU $G (attempt $i)"
      CUDA_VISIBLE_DEVICES=$G $PY scripts/dg_star.py --phase gen "${EXTRA[@]}" --config "configs/setup2/${DS}_U_s$s.yaml" \
        --teacher-train "$teacher" > "$glog" 2>&1
      echo "[$(ts)] gen s$s: $(tail -n 1 "$glog" | cut -c1-220)"
    fi
    if [[ -f "$dir/traces.json" ]]; then
      echo "[$(ts)] START star ${TAG}train s$s on GPU $G"
      CUDA_VISIBLE_DEVICES=$G $PY scripts/dg_star.py --phase train "${EXTRA[@]}" --config "configs/setup2/${DS}_U_s$s.yaml" \
        --teacher-train "$teacher" > "$tlog" 2>&1
    fi
    if [[ -f "$res" ]]; then echo "[$(ts)] DONE  star ${TAG}s$s: $(grep -a 'STaR' "$tlog" | tail -1 | cut -c1-160)"
    else echo "[$(ts)] FAILED star ${TAG}s$s (attempt $i)"; sleep 600; fi
  done
done
echo "[$(ts)] STAR POOL DONE"
