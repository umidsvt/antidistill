#!/usr/bin/env bash
# Setup 2, level A by GRPO alone from the base student (results/m8_audit_design.md §14). Two arms differ only in the
# prompt: C = the setup's chat prompt, P = a plain-text prompt. Reward: 1.0 for matching the teacher's released answer,
# plus 0.2 for exactly one boxed answer at the end after at least 20 words of reasoning.
#   PHASE=check setsid nohup scripts/47_levelA_grpo_only.sh > logs/setup2_grpo_only_check.log 2>&1 < /dev/null & disown
#   PHASE=pilot setsid nohup scripts/47_levelA_grpo_only.sh > logs/setup2_grpo_only_pilot.log 2>&1 < /dev/null & disown
#   PHASE=full ARMS="C P" setsid nohup scripts/47_levelA_grpo_only.sh > logs/setup2_grpo_only.log 2>&1 < /dev/null & disown
# Every job waits for free memory on its GPU and is retried twice; finished jobs (result.json) are skipped.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/gpu_env.sh
export PYTHONPATH=third_party/ext/distillation-game/src HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
       TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv-dg/bin/python; OUT=outputs/setup2; BASE="$OUT/grpo_only"; NEED_MIB=${NEED_MIB:-36000}
PHASE="${PHASE:?PHASE=check|pilot|full}"; read -r -a GL <<< "${GPUS:-4 5 6 7}"
mkdir -p "$BASE"
ts() { date '+%F %T'; }
latest() { ls -dt "$OUT"/"$1"_[0-9]* 2>/dev/null | head -1; }   # the regenerated U runs (m8 §12)
wait_free() { while (( $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" | tr -d ' ') < NEED_MIB )); do sleep 300; done; }
teacher() { echo "$(latest "gsm8k_U_s$1")/teacher/train_standard.json"; }
COMMON=(--format-reward 0.2 --format-mode final_reasoned --min-reason-words 20 --max-completion 512 --vllm-mem 0.5)

run_arm() {   # $1 gpu, $2 arm (C|P), $3 seed, $4 steps, $5 test limit (0 = all), $6 output dir
  local g=$1 arm=$2 s=$3 steps=$4 lim=$5 dir=$6 style=chat extra=() i
  [[ $arm == P ]] && { style=plain; extra=(--also-eval-chat); }
  (( lim > 0 )) && extra+=(--test-limit "$lim")
  for i in 1 2 3; do
    [[ -f "$dir/result.json" ]] && return 0
    wait_free "$g"
    echo "[$(ts)] START arm $arm s$s ($steps steps) on GPU $g (attempt $i)"
    DG_MASTER_PORT=$((29700 + g)) CUDA_VISIBLE_DEVICES=$g $(pin $g) $PY scripts/dg_rl_base.py \
      --config "configs/setup2/gsm8k_U_s$s.yaml" --teacher-train "$(teacher "$s")" --prompt-style "$style" "${COMMON[@]}" \
      --max-steps "$steps" --save-adapter "${extra[@]}" --name "grpo_only_$arm" --out "$dir" > "$dir.log" 2>&1
    if [[ -f "$dir/result.json" ]]; then echo "[$(ts)] DONE  arm $arm s$s: $(grep -a 'test (seed' "$dir.log" | tail -1 | cut -c1-120)"; return 0; fi
    echo "[$(ts)] FAILED arm $arm s$s (attempt $i): $(grep -a -m1 -E 'Error|error' "$dir.log" | cut -c1-160)"; sleep 600
  done
}

case "$PHASE" in
  check)
    g=${GL[0]}; wait_free "$g"; echo "[$(ts)] START prompt check on GPU $g"
    CUDA_VISIBLE_DEVICES=$g $(pin $g) $PY scripts/dg_prompt_check.py --config configs/setup2/gsm8k_U_s123.yaml \
      --teacher-train "$(teacher 123)" --out "$BASE/prompt_check.json" > "$BASE/prompt_check.log" 2>&1
    grep -a -E "^(C_chat|P_plain)|written" "$BASE/prompt_check.log" ;;
  pilot)   # m8 §14: seed 123, 60 steps per arm, 300 test problems; both arms at once
    run_arm "${GL[0]}" C 123 60 300 "$BASE/pilot_C_s123" & p1=$!
    run_arm "${GL[1]}" P 123 60 300 "$BASE/pilot_P_s123" & p2=$!
    wait $p1 $p2 ;;
  full)    # passing arms, 600 steps, seeds 123-125; one worker per GPU takes the next job
    read -r -a AL <<< "${ARMS:?ARMS=\"C P\"}"; jobs="$BASE/full_jobs.txt"; locks="$BASE/full_locks"
    : > "$jobs"; for s in 123 124 125; do for arm in "${AL[@]}"; do echo "$arm $s" >> "$jobs"; done; done
    rm -rf "$locks"; mkdir -p "$locks"
    worker() { local g=$1 arm s; while read -r arm s; do mkdir "$locks/$arm.$s" 2>/dev/null || continue
        run_arm "$g" "$arm" "$s" 600 0 "$BASE/full_${arm}_s$s"; done < "$jobs"; }
    pids=(); for g in "${GL[@]}"; do worker "$g" & pids+=($!); done; wait "${pids[@]}" ;;
esac
echo "[$(ts)] PHASE $PHASE DONE"
