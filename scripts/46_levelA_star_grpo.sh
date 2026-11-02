#!/usr/bin/env bash
# Setup 2, level A with RL after self-training (results/m8_audit_design.md §13): for each seed, on its own GPU,
# few-shot STaR (gen, then train, which now saves its LoRA), then GRPO continuing that LoRA with the labmate's step-3
# settings (8 problems x 8 samples, temperature 1.0, 512 completion tokens, 300 steps, lr 2e-5, no KL). Each step is
# skipped when its output exists; a failed seed is retried twice, 10 minutes apart; every job waits for free memory.
#   GPUS="0 1 2" SEEDS="123 124 125" setsid nohup scripts/46_levelA_star_grpo.sh > logs/setup2_levelA.log 2>&1 < /dev/null & disown
#   SMOKE=1 GPUS=0 SEEDS=123 scripts/46_levelA_star_grpo.sh    # 32 problems, 3 GRPO steps; writes to outputs/setup2/smoke46/
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/gpu_env.sh
export PYTHONPATH=third_party/ext/distillation-game/src HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
       TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv-dg/bin/python; OUT=outputs/setup2; DS=gsm8k; NEED_MIB=${NEED_MIB:-36000}
read -r -a GL <<< "${GPUS:-0 1 2}"; read -r -a SL <<< "${SEEDS:-123 124 125}"
(( ${#GL[@]} == ${#SL[@]} )) || { echo "GPUS and SEEDS need the same number of entries"; exit 1; }
SM=(); RLX=(); BASE="$OUT"; SFX=""
if [[ -n "${SMOKE:-}" ]]; then SM=(--limit 32); RLX=(--max-steps 3 --test-limit 16); BASE="$OUT/smoke46"; SFX="_smoke"; fi
ts() { date '+%F %T'; }
latest() { ls -dt "$OUT"/"$1"_[0-9]* 2>/dev/null | head -1; }   # the regenerated U runs (m8 §12)
wait_free() { while (( $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" | tr -d ' ') < NEED_MIB )); do sleep 300; done; }

worker() {   # $1 gpu, $2 seed
  local g=$1 s=$2 i
  local star="$BASE/star_fewshot_${DS}_s$s" rl="$BASE/rl_${DS}_s${s}_star_grpo"
  local cfg="configs/setup2/${DS}_U_s$s.yaml" teacher log="logs/setup2_levelA_s$s$SFX"
  teacher="$(latest "${DS}_U_s$s")/teacher/train_standard.json"
  [[ -f "$teacher" ]] || { echo "[$(ts)] s$s: no teacher answers ($teacher)"; return; }
  for i in 1 2 3; do
    [[ -f "$rl/result.json" ]] && break
    wait_free "$g"
    if [[ ! -f "$star/traces.json" ]]; then
      echo "[$(ts)] START star gen s$s on GPU $g (attempt $i)"
      CUDA_VISIBLE_DEVICES=$g $(pin $g) $PY scripts/dg_star.py --phase gen --fewshot "${SM[@]}" --config "$cfg" \
        --teacher-train "$teacher" --out "$star" > "${log}_gen.log" 2>&1
      echo "[$(ts)] gen s$s: $(tail -n 1 "${log}_gen.log" | cut -c1-200)"
    fi
    if [[ -f "$star/traces.json" && ! -f "$star/adapter/adapter_config.json" ]]; then
      echo "[$(ts)] START star train s$s on GPU $g"
      CUDA_VISIBLE_DEVICES=$g $(pin $g) $PY scripts/dg_star.py --phase train --fewshot "${SM[@]}" --config "$cfg" \
        --teacher-train "$teacher" --out "$star" > "${log}_train.log" 2>&1
      [[ -f "$star/result.json" ]] && echo "[$(ts)] DONE  star s$s: $(grep -a 'STaR' "${log}_train.log" | tail -1 | cut -c1-160)"
    fi
    if [[ -f "$star/adapter/adapter_config.json" ]]; then
      echo "[$(ts)] START grpo s$s on GPU $g"
      DG_MASTER_PORT=$((29600 + g)) CUDA_VISIBLE_DEVICES=$g $(pin $g) $PY scripts/dg_rl_base.py --config "$cfg" \
        --teacher-train "$teacher" --init-adapter "$star/adapter" --max-completion 512 --max-steps 300 --vllm-mem 0.5 "${RLX[@]}" \
        --name star_grpo --out "$rl" > "${log}_grpo.log" 2>&1
    fi
    if [[ -f "$rl/result.json" ]]; then echo "[$(ts)] DONE  grpo s$s: $(grep -a 'test (seed' "${log}_grpo.log" | tail -1 | cut -c1-160)"; break; fi
    echo "[$(ts)] FAILED s$s (attempt $i)"; sleep 600
  done
}

echo "[$(ts)] level A, STaR then GRPO (m8 §13): seeds ${SL[*]} on GPUs ${GL[*]}${SMOKE:+ (smoke test)}"
pids=()
for k in "${!GL[@]}"; do worker "${GL[$k]}" "${SL[$k]}" & pids+=($!); done
wait "${pids[@]}"
echo "[$(ts)] LEVEL A DONE"
