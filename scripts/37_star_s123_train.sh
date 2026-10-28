#!/usr/bin/env bash
# Level A, self-training as fixed (m8 §9.3): finish seed 123 only. Waits for its running gen phase (PID given) to
# exit, then runs the train phase on GPU 2. Seeds 124-125 of this recipe are held (smoke test: junk data).
#   setsid nohup scripts/37_star_s123_train.sh <gen_pid> > logs/setup2_star_g2.log 2>&1 < /dev/null & disown
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=third_party/ext/distillation-game/src HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
       HF_HOME=/mnt/vault/huggingface_cache TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ts() { date '+%F %T'; }
while kill -0 "$1" 2>/dev/null; do sleep 60; done
out=outputs/setup2/star_gsm8k_s123
teacher="$(ls -dt outputs/setup2/gsm8k_U_s123_[0-9]* | head -1)/teacher/train_standard.json"
[[ -f $out/traces.json ]] || { echo "[$(ts)] gen s123 ended without traces.json: $(tail -n 1 logs/setup2_star_gen_gsm8k_s123.log | cut -c1-220)"; exit 1; }
echo "[$(ts)] gen s123 done: $(tail -n 1 logs/setup2_star_gen_gsm8k_s123.log | cut -c1-220)"
echo "[$(ts)] START star train s123 on GPU 2"
CUDA_VISIBLE_DEVICES=2 .venv-dg/bin/python scripts/dg_star.py --phase train --config configs/setup2/gsm8k_U_s123.yaml \
  --teacher-train "$teacher" > logs/setup2_star_train_gsm8k_s123.log 2>&1
if [[ -f $out/result.json ]]; then echo "[$(ts)] DONE  star s123: $(grep -a 'STaR' logs/setup2_star_train_gsm8k_s123.log | tail -1 | cut -c1-160)"
else echo "[$(ts)] FAILED star train s123: $(tail -n 1 logs/setup2_star_train_gsm8k_s123.log | cut -c1-220)"; fi
