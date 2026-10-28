#!/usr/bin/env bash
# Option 4 pilot, combination students with the long-budget teacher (m8 §10.1 and §11): U_long alone did not pass,
# so Solo + long-teacher students are trained before the pilot is read as a failure. Seed 123, pilot settings
# (cut-off 9,216, micro-batch 1 x 24), scored at 8,192 tokens. Sequential reuses the pilot's Solo students as stage 1.
#   GPU 4: B mix, then B seq     GPU 5: C mix, then C seq
#   setsid nohup scripts/43_math_pilot_combo.sh > logs/setup3_math_combo.log 2>&1 < /dev/null & disown
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=third_party/ext/distillation-game/src HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
       HF_HOME=/mnt/vault/huggingface_cache TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv-dg/bin/python; CFG=configs/setup2/math_U_s123.yaml; O=outputs/setup3_math
LONG=$O/teacher_long8192_s123/train_long.json
SOLO_B=$(ls -dt outputs/setup2/math_solo_l3i_s123_[0-9]* | head -1)/teacher/train_standard.json
SOLO_C=$(ls -dt outputs/setup2/math_solo_q7i_s123_[0-9]* | head -1)/teacher/train_standard.json
ts() { date '+%F %T'; }
wait_free() { while (( $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" | tr -d ' ') < 36000 )); do sleep 120; done; }
job() {  # gpu name train-args...
  local g=$1 n=$2; shift 2
  local f=$O/scores/${n}_8192.json
  [[ -f $f ]] && return 0
  if [[ ! -f $O/$n/merged/config.json ]]; then
    wait_free $g; echo "[$(ts)] TRAIN $n on GPU $g"
    CUDA_VISIBLE_DEVICES=$g $PY scripts/dg_train_from_traces.py --config $CFG --out $O/$n "$@" > logs/setup3_math_train_$n.log 2>&1
    [[ -f $O/$n/merged/config.json ]] || { echo "[$(ts)] FAILED train $n: $(tail -n 1 logs/setup3_math_train_$n.log | cut -c1-200)"; return 1; }
  fi
  wait_free $g; echo "[$(ts)] SCORE $n at 8192 on GPU $g"
  CUDA_VISIBLE_DEVICES=$g $PY scripts/dg_score_vllm.py --config $CFG --model $O/$n/merged --max-new 8192 --out $f > logs/setup3_math_score_${n}_8192.log 2>&1
  [[ -f $f ]] && echo "[$(ts)] SCORED $n: $(tail -n 1 logs/setup3_math_score_${n}_8192.log | cut -c1-200)" \
    || { echo "[$(ts)] FAILED score $n: $(tail -n 1 logs/setup3_math_score_${n}_8192.log | cut -c1-200)"; return 1; }
}
( for i in 1 2; do job 4 longmix_B --traces "$SOLO_B" "$LONG" && break; sleep 600; done
  for i in 1 2; do job 4 longseq_B --traces "$LONG" --init $O/solo_B/merged && break; sleep 600; done ) &
( for i in 1 2; do job 5 longmix_C --traces "$SOLO_C" "$LONG" && break; sleep 600; done
  for i in 1 2; do job 5 longseq_C --traces "$LONG" --init $O/solo_C/merged && break; sleep 600; done ) &
wait
echo "[$(ts)] PILOT COMBO DONE"
