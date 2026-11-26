#!/usr/bin/env bash
# Second task, pilot round 3 (results/m8_audit_design.md §34): easier levels, same settings as scripts/74_rg_pilot.sh.
# Teacher on two GPU pairs in parallel, the student on one GPU (never GPU 2); each job waits for free GPUs.
#   setsid nohup scripts/75_rg_pilot2.sh > logs/rg/pilot2.log 2>&1 < /dev/null & disown
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/gpu_env.sh
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn
PY=.venv-dg/bin/python; O=outputs/rg_pilot3; NEED_MIB=42000
G_TC=${G_TC:-0,3} G_TZ=${G_TZ:-4,5} G_STU=${G_STU:-4}
require_one_group "$G_TC"; require_one_group "$G_TZ"
for g in ${G_TC//,/ } $G_STU; do [[ $g != 2 ]] || { echo "GPU 2 is kept"; exit 1; }; done
mkdir -p $O logs/rg
ts() { date '+%F %T'; }
idle() { local g; for g in ${1//,/ }; do
           [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$g" | tr -d ' ')" ]] || return 1
           (( $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$g" | tr -d ' ') >= NEED_MIB )) || return 1
         done; }
run() {   # <gpus> <model> <task> [extra]
  local g=$1 m=$2 t=$3 d=$O/${3}_$2 i; shift 3
  for i in 1 2; do
    [[ -f $d/summary.json ]] && break
    until idle "$g"; do sleep 60; done
    CUDA_VISIBLE_DEVICES=$g $(pin "$g") $PY scripts/rg_pilot.py --model $m --puzzles data/rg/pilot3_$t.jsonl --out $d "$@" \
        >> $d.log 2>&1
  done
  echo "[$(ts)] $t $m: $($PY -c "import json; d=json.load(open('$d/summary.json'))['by_level']; print({k: (v['solve_rate'], v['finished'], v['mean_tokens']) for k, v in sorted(d.items())})" 2>/dev/null || echo FAILED)"
}
echo "[$(ts)] pilot round 3 started: teacher crypt $G_TC, teacher zebra $G_TZ, student $G_STU"
run $G_TC T crypt --tp 2 & run $G_STU B crypt &
wait
echo "[$(ts)] PILOT3 DONE"
