#!/usr/bin/env bash
# Gap breakdown (results/m8_audit_design.md §26), the parts that need GPUs (seed 123):
#   D2  the teacher judges B2's kept texts and calibration teacher solutions (scripts/kk_judge.py), GPUs 0,1
#   D3  B1, B5 and U_inst answer every training puzzle once (kk_eval.py on the training pool), GPU 3
# Each job waits until its GPUs are free of other processes. Only GPUs 0, 1, 3 are ours now (m8 §25).
#   setsid nohup scripts/63_decomp.sh > logs/phase1/decomp.log 2>&1 < /dev/null & disown
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/gpu_env.sh
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn
PY=.venv-dg/bin/python; D=outputs/kk_phase1/decomp; STUDENT=unsloth/Llama-3.2-3B-Instruct; NEED_MIB=42000
mkdir -p "$D" logs/phase1
ts() { date '+%F %T'; }
idle() { local g; for g in ${1//,/ }; do
  [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$g" | tr -d ' ')" ]] && return 1
  (( $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$g" | tr -d ' ') < NEED_MIB )) && return 1
  done; return 0; }
run() {   # <gpus> <marker> <log> <command...>
  local gl=$1 marker=$2 log=$3; shift 3
  [[ -f "$marker" ]] && return 0
  until idle "$gl"; do sleep 120; done
  CUDA_VISIBLE_DEVICES=$gl $(pin "$gl") "$@" >> "$log" 2>&1
  [[ -f "$marker" ]] && echo "[$(ts)] done $marker" || echo "[$(ts)] FAILED $marker :: $(grep -a Error "$log" | tail -1 | cut -c1-200)"
}
echo "[$(ts)] gap breakdown (m8 §26)"
run 0,1 $D/judge_s123/summary.json logs/phase1/judge_s123.log $PY scripts/kk_judge.py --out $D/judge_s123 &
{ for x in "B1 outputs/kk_phase1/students/B1_s123/adapter" "B5 outputs/kk_phase1/students/B5_s123/adapter" \
           "Uinst outputs/kk_step2/students/Uinst_s123/adapter"; do
    set -- $x
    run 3 $D/train_preds/$1_s123/summary.json logs/phase1/train_preds_$1.log $PY scripts/kk_eval.py \
        --puzzles data/kk/step2_train.jsonl --seed 123 --adapter "$2" --student $STUDENT --out $D/train_preds/$1_s123
  done; } &
wait
echo "[$(ts)] DECOMP GPU PARTS DONE"
