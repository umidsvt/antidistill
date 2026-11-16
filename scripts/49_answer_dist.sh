#!/usr/bin/env bash
# Answer distributions of the attacker's own models (results/m8_audit_design.md §16): 16 tries per training problem from
# each owned model (A base student, B Llama-3.2-3B-Instruct, C Qwen2.5-7B-Instruct) on the seed-123 GSM8K and MATH pools,
# every try stored and graded. Nothing is trained.
#   JOBS="7:math:C 2:gsm8k:C 2:gsm8k:A 3:math:B 3:gsm8k:B 3:math:A" setsid nohup scripts/49_answer_dist.sh \
#       > logs/setup2_answer_dist.log 2>&1 < /dev/null & disown
#   PILOT=1 JOBS=... scripts/49_answer_dist.sh     # first 200 problems, into outputs/setup2_answers/pilot/
# JOBS: gpu:dataset:level entries; each GPU runs its entries in order, the GPUs in parallel. Grading (CPU) runs in the
# background so the GPU moves on to its next entry.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/gpu_env.sh
export PYTHONPATH=third_party/ext/distillation-game/src HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_OFFLINE=1 \
       TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn
PY=.venv-dg/bin/python; OUT=outputs/setup2_answers; NEED_MIB=${NEED_MIB:-40000}; SEED=123
LIM=(); [[ -n "${PILOT:-}" ]] && { LIM=(--limit 200); OUT="$OUT/pilot"; }
mkdir -p "$OUT"
ts() { date '+%F %T'; }
latest() { ls -dt outputs/setup2/"$1"_[0-9]* 2>/dev/null | head -1; }   # the regenerated U runs (m8 §12)
wait_free() { while (( $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" | tr -d ' ') < NEED_MIB )); do sleep 120; done; }

one() {   # $1 gpu, $2 dataset, $3 level
  local g=$1 d=$2 l=$3 i dir="$OUT/$2_$3_s$SEED"
  local args=(--level "$l" --config "configs/setup2/${d}_U_s$SEED.yaml"
              --teacher-train "$(latest "${d}_U_s$SEED")/teacher/train_standard.json" --out "$dir" "${LIM[@]}")
  mkdir -p "$dir"
  for i in 1 2; do
    if [[ ! -f "$dir/gen_meta.json" ]]; then   # gen_meta.json is written last, after raw.jsonl
      rm -f "$dir/raw.jsonl"; wait_free "$g"; echo "[$(ts)] GEN   $d $l on GPU $g (attempt $i)"
      CUDA_VISIBLE_DEVICES=$g $(pin $g) $PY scripts/dg_answer_dist.py --phase gen "${args[@]}" > "$dir/gen.log" 2>&1 &
      local pid=$!
      # watchdog: vLLM can hang at exit after the output is written (pilot, 2026-11-08: two jobs, ~50 min each)
      while kill -0 "$pid" 2>/dev/null; do
        if [[ -f "$dir/gen_meta.json" ]]; then
          sleep 120
          if kill -0 "$pid" 2>/dev/null; then
            echo "[$(ts)] GEN   $d $l: output written but PID $pid did not exit; stopping it"
            pkill -9 -P "$pid"; kill -9 "$pid"
          fi
          break
        fi
        sleep 20
      done
      wait "$pid" 2>/dev/null
    fi
    if [[ -f "$dir/gen_meta.json" ]]; then
      echo "[$(ts)] GEN   $d $l done: $(tail -1 "$dir/gen.log" | cut -c1-200)"
      ( $(pin $g) $PY scripts/dg_answer_dist.py --phase grade "${args[@]}" > "$dir/grade.log" 2>&1
        echo "[$(ts)] GRADE $d $l: $(tail -1 "$dir/grade.log" | cut -c1-240)" ) &
      return 0
    fi
    echo "[$(ts)] FAILED $d $l (attempt $i): $(grep -a -m1 -E 'Error' "$dir/gen.log" | cut -c1-160)"; sleep 300
  done
}

declare -A Q
for j in ${JOBS:?set JOBS}; do IFS=: read -r g d l <<< "$j"; require_one_group "$g"; Q[$g]+="$d:$l "; done
echo "[$(ts)] answer distributions (m8 §16)${PILOT:+, pilot}: ${JOBS}"
pids=()
for g in "${!Q[@]}"; do
  ( for e in ${Q[$g]}; do IFS=: read -r d l <<< "$e"; one "$g" "$d" "$l"; done; wait ) & pids+=($!)
done
wait "${pids[@]}"
echo "[$(ts)] ANSWER DISTRIBUTIONS DONE"
