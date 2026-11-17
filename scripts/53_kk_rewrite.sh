#!/usr/bin/env bash
# Phase 1, first defense (results/m8_audit_design.md §21): the teacher rewrites its 2,000 step-2 training solutions with
# Ma et al.'s optimized instruction (scripts/kk_rewrite.py; two tensor-parallel-2 copies, half the puzzles each), then
# the defended student D_rewrite is trained and scored, seeds 123-125 (scripts/51_kk_step2.sh, STUDENTS=D_rewrite).
#   REWRITE_GPUS="2,3 6,7" TRAIN_GPUS="2 3 6 7" setsid nohup scripts/53_kk_rewrite.sh > logs/kk_rewrite.log 2>&1 < /dev/null & disown
#   SMOKE=1 REWRITE_GPUS="2,3" scripts/53_kk_rewrite.sh     # 8 rewrites, no training; into outputs/kk_step2/rewrite_smoke/
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/gpu_env.sh
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn
PY=.venv-dg/bin/python; NEED_MIB=${NEED_MIB:-42000}; OUT=outputs/kk_step2/rewrite; LIM=()
[[ -n "${SMOKE:-}" ]] && { OUT=outputs/kk_step2/rewrite_smoke; LIM=(--limit 8); }
read -r -a RG <<< "${REWRITE_GPUS:?set REWRITE_GPUS}"; N=${#RG[@]}
mkdir -p "$OUT"
ts() { date '+%F %T'; }
wait_free() { local g; for g in ${1//,/ }; do
  while (( $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$g" | tr -d ' ') < NEED_MIB )); do sleep 120; done; done; }
copy() {   # <gpus> <part>: one rewriting copy, with the exit watchdog of the other launchers
  local gl=$1 k=$2 f="$OUT/part$2.jsonl" pid i
  require_one_group "$gl"
  for i in 1 2; do
    [[ -f "$f.summary.json" ]] && break
    wait_free "$gl"; echo "[$(ts)] REWRITE part $k/$N on GPUs $gl (attempt $i)"
    CUDA_VISIBLE_DEVICES=$gl $(pin "$gl") $PY scripts/kk_rewrite.py --pool data/kk/step2_train.jsonl \
      --gen outputs/kk_step2/gen --out "$f" --tp 2 --part "$k" --parts "$N" "${LIM[@]}" > "$OUT/part$k.log" 2>&1 &
    pid=$!
    while kill -0 "$pid" 2>/dev/null; do
      if [[ -f "$f.summary.json" ]]; then sleep 120; kill -0 "$pid" 2>/dev/null && { pkill -9 -P "$pid"; kill -9 "$pid"; }; break; fi
      sleep 30
    done
    wait "$pid" 2>/dev/null
  done
  [[ -f "$f.summary.json" ]] && echo "[$(ts)] REWRITE part $k done: $(cat "$f.summary.json" | tr -d '\n ')" || echo "[$(ts)] REWRITE part $k FAILED: $(grep -a -m1 Error "$OUT/part$k.log" | cut -c1-160)"
}
echo "[$(ts)] trace rewriting (m8 §21)${SMOKE:+, smoke test}: copies on ${RG[*]}"
for k in $(seq 1 "$N"); do copy "${RG[$((k-1))]}" "$k" & done
wait
for k in $(seq 1 "$N"); do [[ -f "$OUT/part$k.jsonl.summary.json" ]] || { echo "[$(ts)] missing part $k; stopping"; exit 1; }; done
[[ -n "${SMOKE:-}" ]] && { echo "[$(ts)] SMOKE DONE"; exit 0; }
cat "$OUT"/part*.jsonl > outputs/kk_step2/train_sets/D_rewrite.jsonl
echo "[$(ts)] defended training set: $(wc -l < outputs/kk_step2/train_sets/D_rewrite.jsonl) solutions"
EVAL_DIR=test_eot PHASE=train STUDENTS=D_rewrite WITH_UNTRAINED=0 TRAIN_GPUS="${TRAIN_GPUS:?set TRAIN_GPUS}" scripts/51_kk_step2.sh
echo "[$(ts)] REWRITE DEFENSE DONE"
