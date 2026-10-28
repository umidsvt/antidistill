#!/usr/bin/env bash
# Combination students for U_ℓ (results/m8_audit_design.md §11): the attacker's own Solo solutions combined with the
# teacher's, in the setup-2 GSM8K and MATH setups. One worker per GPU takes the next unclaimed job (lock directory),
# waits for >= 36 GB free, retries a failed job once. Job lines: <dataset> <seed> <level B|C> <recipe mix|seq>.
#   GPUS="6 7" JOBS=outputs/setup2_combo/jobs_s123.txt setsid nohup scripts/42_combo_pool.sh > logs/setup2_combo.log 2>&1 < /dev/null & disown
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=third_party/ext/distillation-game/src HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
       HF_HOME=/mnt/vault/huggingface_cache TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv-dg/bin/python; O=outputs/setup2_combo; JOBS=${JOBS:?job file}; LOCKS=$O/locks; mkdir -p $LOCKS
SETUP=(--max-length 2048 --micro-batch 6 --accum 4)   # the setup's own training settings
ts() { date '+%F %T'; }
latest() { ls -dt outputs/setup2/"$1"_[0-9]* 2>/dev/null | head -1; }
wait_free() { while (( $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" | tr -d ' ') < 36000 )); do sleep 120; done; }

run_job() {  # gpu ds seed level recipe
  local g=$1 ds=$2 s=$3 lv=$4 rc=$5
  local solo=$([[ $lv == B ]] && echo l3i || echo q7i) name="${ds}_${lv}_${rc}_s$s"
  local cfg=configs/setup2/${ds}_U_s$s.yaml d=$O/$name log=logs/setup2_combo_$name.log
  local teacher="$(latest ${ds}_U_s$s)/teacher/train_standard.json" own="$(latest ${ds}_solo_${solo}_s$s)/teacher/train_standard.json"
  [[ -f $teacher && -f $own ]] || { echo "[$(ts)] $name: missing $teacher or $own"; return 1; }
  wait_free $g; echo "[$(ts)] START $name on GPU $g"
  if [[ $rc == mix ]]; then
    CUDA_VISIBLE_DEVICES=$g $PY scripts/dg_train_from_traces.py --config $cfg --traces "$own" "$teacher" --out $d \
      "${SETUP[@]}" --eval --no-save > $log 2>&1
  else
    [[ -f $d/stage1/merged/config.json ]] || CUDA_VISIBLE_DEVICES=$g $PY scripts/dg_train_from_traces.py --config $cfg \
      --traces "$own" --out $d/stage1 "${SETUP[@]}" > $log 2>&1
    [[ -f $d/stage1/merged/config.json ]] && CUDA_VISIBLE_DEVICES=$g $PY scripts/dg_train_from_traces.py --config $cfg \
      --traces "$teacher" --init $d/stage1/merged --out $d "${SETUP[@]}" --eval --no-save >> $log 2>&1
  fi
  if [[ -f $d/result.json ]]; then
    rm -rf $d/stage1/merged
    echo "[$(ts)] DONE  $name: $(tail -n 1 $log | cut -c1-200)"
  else echo "[$(ts)] FAILED $name: $(tail -n 1 $log | cut -c1-200)"; return 1; fi
}

worker() {  # gpu
  local g=$1
  while read -r ds s lv rc; do
    [[ -z ${ds:-} || $ds == \#* ]] && continue
    local name="${ds}_${lv}_${rc}_s$s"
    [[ -f $O/$name/result.json ]] && continue
    mkdir $LOCKS/$name 2>/dev/null || continue   # another worker has it
    run_job $g $ds $s $lv $rc || { sleep 600; run_job $g $ds $s $lv $rc; }
  done < "$JOBS"
  echo "[$(ts)] worker GPU $g: no jobs left"
}

for g in ${GPUS:?gpus}; do [[ $g == 0 || $g == 1 ]] && { echo "GPUs 0 and 1 are reserved"; exit 1; }; done
for g in $GPUS; do worker $g & done
wait
echo "[$(ts)] COMBO POOL DONE: $JOBS"
