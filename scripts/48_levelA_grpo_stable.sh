#!/usr/bin/env bash
# Setup 2, level A by GRPO alone from the base student, with Javaheri et al.'s stabilizers ported (results/m8_audit_design.md
# §15, option B): plain prompt; reward +1 / -0.5 / -1 (SimpleRL-Zoo "mix") against the teacher's answers; KL penalty 1e-4;
# learning rate 5e-6 constant; 32 problems x 8 samples per step; answers up to 2,048 tokens; 300 steps; an adapter saved
# every 25 steps and the best one kept by its score on 256 held-out training problems (teacher's answers).
#   GPUS="4 5 6" SEEDS="123 124 125" setsid nohup scripts/48_levelA_grpo_stable.sh > logs/setup2_grpo_stable.log 2>&1 < /dev/null & disown
#   SMOKE=1 GPUS=4 SEEDS=123 scripts/48_levelA_grpo_stable.sh     # 4 steps, tiny held-out and test sets
#   CONT=1 GPUS="4 6 7" ... scripts/48_levelA_grpo_stable.sh      # m8 §15 continuation: 300 more steps from each seed's
#                                                                  # step-300 adapter, into s<seed>_cont/
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/gpu_env.sh
export PYTHONPATH=third_party/ext/distillation-game/src HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_OFFLINE=1 \
       TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv-dg/bin/python; OUT=outputs/setup2; BASE="$OUT/grpo_stable"; NEED_MIB=${NEED_MIB:-36000}
read -r -a GL <<< "${GPUS:-4 5 6}"; read -r -a SL <<< "${SEEDS:-123 124 125}"
(( ${#GL[@]} == ${#SL[@]} )) || { echo "GPUS and SEEDS need the same number of entries"; exit 1; }
RUN=(--max-steps 300 --save-every 25 --holdout-select 256 --prompts-per-step 32); SFX=""
[[ -n "${SMOKE:-}" ]] && { RUN=(--max-steps 4 --save-every 2 --holdout-select 16 --prompts-per-step 4 --test-limit 16); SFX="_smoke"; }
[[ -n "${CONT:-}" ]] && SFX="_cont"
mkdir -p "$BASE"
ts() { date '+%F %T'; }
latest() { ls -dt "$OUT"/"$1"_[0-9]* 2>/dev/null | head -1; }   # the regenerated U runs (m8 §12)
wait_free() { while (( $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" | tr -d ' ') < NEED_MIB )); do sleep 300; done; }

worker() {   # $1 gpu, $2 seed
  local g=$1 s=$2 i dir="$BASE/s$2$SFX" init=()
  if [[ -n "${CONT:-}" ]]; then
    init=(--init-adapter "$BASE/s$s/ckpt_0300")
    [[ -f "$BASE/s$s/ckpt_0300/adapter_config.json" ]] || { echo "[$(ts)] s$s: no step-300 adapter to continue"; return 1; }
  fi
  for i in 1 2 3; do
    [[ -f "$dir/result.json" ]] && return 0
    rm -rf "$dir"; wait_free "$g"
    echo "[$(ts)] START s$s on GPU $g (attempt $i)"
    DG_MASTER_PORT=$((29800 + g)) CUDA_VISIBLE_DEVICES=$g $(pin $g) $PY scripts/dg_rl_base.py \
      --config "configs/setup2/gsm8k_U_s$s.yaml" --teacher-train "$(latest "gsm8k_U_s$s")/teacher/train_standard.json" \
      --prompt-style plain --reward-mode mix --beta 1e-4 --lr 5e-6 --lr-schedule constant --num-generations 8 \
      --micro-batch 2 --max-completion 2048 --vllm-mem 0.45 --greedy-test --name grpo_stable "${RUN[@]}" "${init[@]}" \
      --out "$dir" > "$dir.log" 2>&1
    if [[ -f "$dir/result.json" ]]; then echo "[$(ts)] DONE  s$s: $(grep -a 'test (seed' "$dir.log" | tail -1 | cut -c1-120)"; return 0; fi
    echo "[$(ts)] FAILED s$s (attempt $i): $(grep -a -m1 -E 'Error' "$dir.log" | cut -c1-160)"; sleep 600
  done
}

echo "[$(ts)] level A, stable GRPO (m8 §15): seeds ${SL[*]} on GPUs ${GL[*]}${SMOKE:+ (smoke test)}"
pids=(); for k in "${!GL[@]}"; do worker "${GL[$k]}" "${SL[$k]}" & pids+=($!); done
wait "${pids[@]}"
echo "[$(ts)] STABLE GRPO DONE"
