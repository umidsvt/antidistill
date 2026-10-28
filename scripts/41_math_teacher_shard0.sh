#!/usr/bin/env bash
# Option 4 pilot step 1, repair (m8 §10.1): shard 0 of the long-budget teacher failed at start (GPU 5 held by an
# orphaned vLLM engine of an earlier check). Re-run it once GPU 5 has >= 40 GB free, then merge all 3 shards.
#   setsid nohup scripts/41_math_teacher_shard0.sh > logs/setup3_math_teacher_shard0_retry.log 2>&1 < /dev/null & disown
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=third_party/ext/distillation-game/src HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
       HF_HOME=/mnt/vault/huggingface_cache TOKENIZERS_PARALLELISM=false
CFG=configs/setup2/math_U_s123.yaml; D=outputs/setup3_math/teacher_long8192_s123; G=${GPU:-5}
ts() { date '+%F %T'; }
echo "[$(ts)] waiting for GPU $G"
while (( $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $G | tr -d ' ') < 40000 )); do sleep 120; done
echo "[$(ts)] START shard 0/3 on GPU $G"
CUDA_VISIBLE_DEVICES=$G .venv-dg/bin/python scripts/38_math_long_teacher.py --config $CFG --shard 0 --nshards 3 > logs/setup3_math_teacher_shard0.log 2>&1
echo "[$(ts)] shard 0: $(tail -n 1 logs/setup3_math_teacher_shard0.log | cut -c1-200)"
until [[ -f $D/shard0of3.json && -f $D/shard1of3.json && -f $D/shard2of3.json ]]; do sleep 120; done
.venv-dg/bin/python scripts/38_math_long_teacher.py --config $CFG --merge --nshards 3 && echo "[$(ts)] TEACHER DONE" || echo "[$(ts)] MERGE FAILED"
