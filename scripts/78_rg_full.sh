#!/usr/bin/env bash
# Second task, full build (results/m8_audit_design.md §36), seed 123:
#   1. 2,000 training + 1,000 test puzzles of one level (distinct across both), CPU;
#   2. the teacher's try on every training puzzle (GPUs T_TRAIN) and every test puzzle (GPUs T_TEST), in parallel;
#   3. the training sets U and B1 (scripts/rg_sets.py);
#   4. a queue of student jobs on WORKERS (one GPU each): untrained (I0), U and B1 for Llama-3.2-3B-Instruct and
#      Qwen2.5-3B-Instruct, each scored on the test puzzles with 8,192 tokens.
# Never GPU 2; every job waits until its GPUs are free of other processes.
#   TASK=crypt LEVEL=2x4 setsid nohup scripts/78_rg_full.sh > logs/rg/full_crypt.log 2>&1 < /dev/null & disown
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/gpu_env.sh
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv-dg/bin/python; TASK=${TASK:?}; LEVEL=${LEVEL:?}; S=123; SEED_DATA=${SEED_DATA:-11}
T_TRAIN=${T_TRAIN:-0,3} T_TEST=${T_TEST:-4,5} WORKERS=${WORKERS:-"0 3 4 5"}; NEED_MIB=42000
D=data/rg/$TASK; O=outputs/rg/$TASK
require_one_group "$T_TRAIN"; require_one_group "$T_TEST"
for g in ${T_TRAIN//,/ } ${T_TEST//,/ } $WORKERS; do [[ $g != 2 ]] || { echo "GPU 2 is kept"; exit 1; }; done
mkdir -p $O logs/rg
ts() { date '+%F %T'; }
idle() { local g; for g in ${1//,/ }; do
           [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$g" | tr -d ' ')" ]] || return 1
           (( $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$g" | tr -d ' ') >= NEED_MIB )) || return 1
         done; }
attempt() {   # <gpus> <marker> <log> <command...>: up to two tries, each after the GPUs are free
  local g=$1 marker=$2 log=$3 i; shift 3
  [[ -f "$marker" ]] && return 0
  mkdir -p "$(dirname "$log")"
  for i in 1 2; do
    until idle "$g"; do sleep 60; done
    KK_MASTER_PORT=$((29780 + ${g%%,*})) CUDA_VISIBLE_DEVICES=$g $(pin "$g") "$@" >> "$log" 2>&1
    [[ -f "$marker" ]] && return 0
    echo "[$(ts)] FAILED (try $i) on GPU $g: $log :: $(grep -a -E 'Error' "$log" | tail -1 | cut -c1-200)"
    sleep 120
  done
  return 1
}
rate() { $PY -c "import json; d=json.load(open('$1'))['by_level']; print({k: (v['solve_rate'], v['cell_accuracy'], v['mean_tokens']) for k, v in d.items()})"; }

# 1. puzzles
if [[ ! -f ${D}_test.jsonl ]]; then
  .venv-rg/bin/python -W ignore scripts/rg_build.py --task $TASK --levels $LEVEL --split train:2000,test:1000 \
      --seed $SEED_DATA --out $D 2>&1 | grep -v Warning | tail -1
fi
[[ -f ${D}_train.jsonl && -f ${D}_test.jsonl ]] || { echo "[$(ts)] puzzle build failed"; exit 1; }
echo "[$(ts)] puzzles: $(wc -l < ${D}_train.jsonl) train, $(wc -l < ${D}_test.jsonl) test"

# 2. teacher, training and test puzzles in parallel
attempt $T_TRAIN $O/T_train/summary.json $O/T_train.log $PY scripts/rg_pilot.py --model T --tp 2 \
    --puzzles ${D}_train.jsonl --out $O/T_train &
attempt $T_TEST $O/T_test/summary.json $O/T_test.log $PY scripts/rg_pilot.py --model T --tp 2 \
    --puzzles ${D}_test.jsonl --out $O/T_test &
wait
[[ -f $O/T_train/summary.json ]] || { echo "[$(ts)] teacher on the training puzzles failed"; exit 1; }
echo "[$(ts)] teacher, training puzzles: $(rate $O/T_train/summary.json)"
[[ -f $O/T_test/summary.json ]] && echo "[$(ts)] teacher, test puzzles: $(rate $O/T_test/summary.json)"

# 3. training sets
python3 -I scripts/rg_sets.py --pool ${D}_train.jsonl --teacher $O/T_train/tries.jsonl --out $O/train_sets || exit 1

# 4. students
Q=$O/jobs.txt; QN=$O/jobs.next
[[ -f $Q ]] || printf '%s\n' "u B" "u Q" "i0 B" "i0 Q" "b1 B" "b1 Q" > $Q
[[ -f $QN ]] || echo 0 > $QN
next_job() { ( flock 9; local n; n=$(cat $QN); echo $((n + 1)) > $QN; sed -n "$((n + 1))p" $Q ) 9> $O/jobs.lock; }
model_path() { [[ $1 == B ]] && echo unsloth/Llama-3.2-3B-Instruct || echo Qwen/Qwen2.5-3B-Instruct; }
job() {   # <gpu> <kind> <model key>
  local g=$1 kind=$2 m=$3 d
  case $kind in
    i0) d=$O/students/I0_${m}_s$S
        attempt $g $d/test/summary.json $d/test.log $PY scripts/rg_pilot.py --model $m --max-tokens 8192 --seed $S \
            --puzzles ${D}_test.jsonl --out $d/test || return 1 ;;
    u|b1) local set=$O/train_sets/$([[ $kind == u ]] && echo U || echo B1_answers).jsonl
        d=$O/students/${kind^^}_${m}_s$S
        attempt $g $d/train_meta.json $d/train.log $PY scripts/kk_sft.py --train-set $set --out $d --seed $S \
            --student $(model_path $m) --plain-prompt || return 1
        attempt $g $d/test/summary.json $d/test.log $PY scripts/rg_pilot.py --model $m --max-tokens 8192 --seed $S \
            --adapter $d/adapter --puzzles ${D}_test.jsonl --out $d/test || return 1 ;;
  esac
  echo "[$(ts)] $kind $m: $(rate $d/test/summary.json)"
}
worker() { local g=$1 line; while line=$(next_job) && [[ -n $line ]]; do job $g $line; done; }
for g in $WORKERS; do worker $g & done
wait
echo "[$(ts)] FULL $TASK DONE"
