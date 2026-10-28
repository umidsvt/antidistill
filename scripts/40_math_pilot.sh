#!/usr/bin/env bash
# Option 4 pilot, steps 2-3 (results/m8_audit_design.md §10.1), seed 123. GPUs 2-4 are first used by few-shot
# self-training (seeds 123-125), so each chain there waits for that seed's queue to finish. A smoke test (train on 48
# solutions, score 50 test problems) runs first on GPU 2; nothing else starts unless it passes. Then one chain per GPU,
# each step waiting for >= 36 GB free on its GPU:
#   GPU 2: U_1024 (train, score 1,024, score 8,192), then the untrained student (score 1,024 and 8,192)
#   GPU 3: Solo C (Qwen2.5-7B-Instruct solutions)   GPU 4: Solo B (Llama-3.2-3B-Instruct solutions)
#   GPU 5: U_long, after the long-budget teacher is merged (scripts/39_math_long_teacher.sh)
#   setsid nohup scripts/40_math_pilot.sh > logs/setup3_math_pilot.log 2>&1 < /dev/null & disown
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=third_party/ext/distillation-game/src HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
       HF_HOME=/mnt/vault/huggingface_cache TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv-dg/bin/python; CFG=configs/setup2/math_U_s123.yaml; O=outputs/setup3_math; mkdir -p $O/scores
LONG=$O/teacher_long8192_s123/train_long.json
U1024=$(ls -dt outputs/setup2/math_U_s123_[0-9]* | head -1)/teacher/train_standard.json
SOLOC=$(ls -dt outputs/setup2/math_solo_q7i_s123_[0-9]* | head -1)/teacher/train_standard.json
SOLOB=$(ls -dt outputs/setup2/math_solo_l3i_s123_[0-9]* | head -1)/teacher/train_standard.json
ts() { date '+%F %T'; }
after_star() { until grep -q "STAR POOL DONE" logs/setup2_star_fewshot_s$1.log 2>/dev/null; do sleep 120; done; }
wait_free() { while (( $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" | tr -d ' ') < 36000 )); do sleep 120; done; }

train() {  # gpu name traces
  local g=$1 n=$2 t=$3
  [[ -f $O/$n/merged/config.json ]] && return 0
  wait_free $g; echo "[$(ts)] TRAIN $n on GPU $g"
  CUDA_VISIBLE_DEVICES=$g $PY scripts/dg_train_from_traces.py --config $CFG --traces "$t" --out $O/$n > logs/setup3_math_train_$n.log 2>&1
  [[ -f $O/$n/merged/config.json ]] && echo "[$(ts)] TRAINED $n: $(tail -n 1 logs/setup3_math_train_$n.log | cut -c1-150)" \
    || { echo "[$(ts)] FAILED train $n: $(tail -n 1 logs/setup3_math_train_$n.log | cut -c1-200)"; return 1; }
}
score() {  # gpu name model max_new
  local g=$1 n=$2 m=$3 k=$4 f=$O/scores/${2}_$4.json
  [[ -f $f ]] && return 0
  wait_free $g; echo "[$(ts)] SCORE $n at $k on GPU $g"
  CUDA_VISIBLE_DEVICES=$g $PY scripts/dg_score_vllm.py --config $CFG --model "$m" --max-new $k --out $f > logs/setup3_math_score_${n}_$k.log 2>&1
  [[ -f $f ]] && echo "[$(ts)] SCORED $n at $k: $(tail -n 1 logs/setup3_math_score_${n}_$k.log | cut -c1-200)" \
    || { echo "[$(ts)] FAILED score $n at $k: $(tail -n 1 logs/setup3_math_score_${n}_$k.log | cut -c1-200)"; return 1; }
}
chain() {  # gpu name traces k1 k2
  local g=$1 n=$2
  for i in 1 2; do train $g $n "$3" && break; sleep 600; done || return 1
  for k in $4 $5; do for i in 1 2; do score $g $n $O/$n/merged $k && break; sleep 600; done; done
}

# smoke test: the whole chain on a few items, outside the pilot's output folders
S=$O/smoke
if [[ ! -f $S/ok ]]; then
  after_star 123; wait_free 2; echo "[$(ts)] SMOKE on GPU 2"
  CUDA_VISIBLE_DEVICES=2 $PY scripts/dg_train_from_traces.py --config $CFG --traces "$U1024" --out $S/U_1024 --limit 48 > logs/setup3_math_smoke.log 2>&1 &&
  CUDA_VISIBLE_DEVICES=2 $PY scripts/dg_score_vllm.py --config $CFG --model $S/U_1024/merged --max-new 1024 --limit 50 --out $S/score.json >> logs/setup3_math_smoke.log 2>&1 &&
  touch $S/ok
  [[ -f $S/ok ]] && echo "[$(ts)] SMOKE OK: $(tail -n 1 logs/setup3_math_smoke.log | cut -c1-200)" \
    || { echo "[$(ts)] SMOKE FAILED, nothing else started: $(tail -n 3 logs/setup3_math_smoke.log | tr '\n' ' ' | cut -c1-300)"; exit 1; }
  rm -rf $S/U_1024/merged
fi

( chain 2 U_1024 "$U1024" 1024 8192
  for k in 1024 8192; do for i in 1 2; do score 2 untrained unsloth/Llama-3.2-3B $k && break; sleep 600; done; done ) &
( after_star 124; chain 3 solo_C "$SOLOC" 1024 8192 ) &
( after_star 125; chain 4 solo_B "$SOLOB" 8192 1024 ) &
( until [[ -f $LONG ]]; do sleep 300; done; echo "[$(ts)] long-budget teacher ready"; chain 5 U_long "$LONG" 8192 1024 ) &
wait
echo "[$(ts)] PILOT DONE"
