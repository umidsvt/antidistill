#!/usr/bin/env bash
cd /home/umid/antidistill/antidistill
source scripts/gpu_env.sh
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv-dg/bin/python; B=outputs/kk_phase2; D=$B/students/I1_repair_lr2e5_s123; G=3
run() { local m=$1; shift; [[ -f $m ]] && return 0; until [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $G | tr -d ' ')" ]]; do sleep 120; done
        KK_MASTER_PORT=29793 CUDA_VISIBLE_DEVICES=$G $(pin $G) "$@"; [[ -f $m ]]; }
mkdir -p $D
run $D/train_meta.json $PY scripts/kk_grpo.py --reward exact --seed 123 --out $D --pool $B/sets/train_repair_k12.jsonl \
    --temperature 0.6 --lr 2e-5 --micro-batch 2 --epochs 1 --save-every 25 >> $D/train.log 2>&1 || { echo "FAILED train" >> logs/phase2/i1lr.log; exit 1; }
for x in "test data/kk/step2_test.jsonl" "test_repair $B/sets/test_repair_truth_k1.jsonl" "test_guess $B/sets/test_guessB5.jsonl"; do
  set -- $x
  run $D/$1/summary.json $PY scripts/kk_eval.py --puzzles $2 --seed 123 --adapter $D/adapter --student unsloth/Llama-3.2-3B-Instruct --out $D/$1 >> $D/$1.log 2>&1 || echo "FAILED $1" >> logs/phase2/i1lr.log
done
echo "I1 lr2e-5 DONE $(date '+%F %T')" >> logs/phase2/i1lr.log
