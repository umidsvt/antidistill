#!/usr/bin/env bash
# Option 4 pilot step 1 (m8 §10.1): long-budget teacher on the seed-123 MATH split, 3 shards on GPUs 5-7, then merge.
#   setsid nohup scripts/39_math_long_teacher.sh > logs/setup3_math_teacher.log 2>&1 < /dev/null & disown
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=third_party/ext/distillation-game/src HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
       HF_HOME=/mnt/vault/huggingface_cache TOKENIZERS_PARALLELISM=false
CFG=configs/setup2/math_U_s123.yaml; GPUS=(${GPUS:-5 6 7}); N=${#GPUS[@]}
ts() { date '+%F %T'; }
pids=()
for i in "${!GPUS[@]}"; do
  echo "[$(ts)] START shard $i/$N on GPU ${GPUS[$i]}"
  CUDA_VISIBLE_DEVICES=${GPUS[$i]} .venv-dg/bin/python scripts/38_math_long_teacher.py --config $CFG --shard $i --nshards $N \
    > logs/setup3_math_teacher_shard$i.log 2>&1 & pids+=($!)
done
for i in "${!pids[@]}"; do wait ${pids[$i]}; echo "[$(ts)] shard $i exit $?: $(tail -n 1 logs/setup3_math_teacher_shard$i.log | cut -c1-200)"; done
.venv-dg/bin/python scripts/38_math_long_teacher.py --config $CFG --merge --nshards $N && echo "[$(ts)] TEACHER DONE" || echo "[$(ts)] MERGE FAILED"
