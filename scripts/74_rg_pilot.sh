#!/usr/bin/env bash
# Second task, pilot (results/m8_audit_design.md §34): the untrained student on GPU G_STU, then the teacher on the two
# GPUs G_T (one group, never GPU 2). Each job waits until its GPUs are free of other processes.
#   setsid nohup scripts/74_rg_pilot.sh > logs/rg/pilot.log 2>&1 < /dev/null & disown
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/gpu_env.sh
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn
PY=.venv-dg/bin/python; O=outputs/rg_pilot; G_STU=${G_STU:-0}; G_T=${G_T:-0,3}; NEED_MIB=42000
require_one_group "$G_T"
for g in $G_STU ${G_T//,/ }; do [[ $g != 2 ]] || { echo "GPU 2 is kept"; exit 1; }; done
mkdir -p $O logs/rg
ts() { date '+%F %T'; }
idle() { local g; for g in ${1//,/ }; do
           [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$g" | tr -d ' ')" ]] || return 1
           (( $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$g" | tr -d ' ') >= NEED_MIB )) || return 1
         done; }
run() {   # <gpus> <model> <task> [extra]
  local g=$1 m=$2 t=$3 d=$O/${3}_$2 i; shift 3
  until [[ -f data/rg/pilot_$t.jsonl ]]; do sleep 60; done
  for i in 1 2; do
    [[ -f $d/summary.json ]] && break
    until idle "$g"; do sleep 60; done
    CUDA_VISIBLE_DEVICES=$g $(pin "$g") $PY scripts/rg_pilot.py --model $m --puzzles data/rg/pilot_$t.jsonl --out $d "$@" \
        >> $d.log 2>&1
  done
  echo "[$(ts)] $t $m: $($PY -c "import json; d=json.load(open('$d/summary.json'))['by_level']; print({k: (v['solve_rate'], v['finished'], v['mean_tokens']) for k, v in sorted(d.items())})" 2>/dev/null || echo FAILED)"
}
echo "[$(ts)] pilot started: student GPU $G_STU, teacher GPUs $G_T"
run $G_STU B crypt; run $G_STU B zebra
run $G_T T crypt --tp 2; run $G_T T zebra --tp 2
echo "[$(ts)] PILOT DONE"
