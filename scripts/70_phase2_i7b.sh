#!/usr/bin/env bash
# Phase 2, wave 3, run 5 (results/m8_audit_design.md §31): I7b, RL on one-hidden-person infill only (hidden-person
# reward, 2 epochs), seed 123, one GPU (never GPU 2).
#   G=5 setsid nohup scripts/70_phase2_i7b.sh > logs/phase2/i7b.log 2>&1 < /dev/null & disown
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/gpu_env.sh
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv-dg/bin/python; B=outputs/kk_phase2; SETS=$B/sets; STUDENT=unsloth/Llama-3.2-3B-Instruct; S=123
G=${G:-5}; [[ $G != 2 ]] || exit 1; NEED_MIB=42000; d=$B/students/I7b_m1_s$S
ts() { date '+%F %T'; }
idle() { [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$G" | tr -d ' ')" ]] \
         && (( $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$G" | tr -d ' ') >= NEED_MIB )); }
attempt() {   # <marker> <log> <command...>
  local marker=$1 log=$2 i; shift 2
  [[ -f "$marker" ]] && return 0
  for i in 1 2; do
    until idle; do sleep 120; done
    KK_MASTER_PORT=$((29780 + G)) CUDA_VISIBLE_DEVICES=$G $(pin "$G") "$@" >> "$log" 2>&1
    [[ -f "$marker" ]] && return 0
    echo "[$(ts)] FAILED (try $i): $log :: $(grep -a -E 'Error' "$log" | tail -1 | cut -c1-200)"; sleep 300
  done
  return 1
}
mkdir -p "$d"
attempt "$d/train_meta.json" "$d/train.log" $PY scripts/kk_grpo.py --reward hidden --seed $S --out "$d" \
    --pool $SETS/train_mask_hid_m1.jsonl --temperature 0.6 --lr 2e-5 --micro-batch 2 --epochs 2 --save-every 25 || exit 1
for ck in ckpt_0100 adapter; do
  attempt "$d/test_mask_$ck/summary.json" "$d/test_mask_$ck.log" $PY scripts/kk_eval.py \
      --puzzles $SETS/test_mask_truth_m1.jsonl --seed $S --adapter "$d/$ck" --student $STUDENT --out "$d/test_mask_$ck" || exit 1
  echo "[$(ts)] I7b $ck m=1: $($PY -I scripts/kk_hidden_acc.py $SETS/test_mask_truth_m1.jsonl "$d/test_mask_$ck/results.jsonl")"
done
attempt "$d/test/summary.json" "$d/test.log" $PY scripts/kk_eval.py --puzzles data/kk/step2_test.jsonl --seed $S \
    --adapter "$d/adapter" --student $STUDENT --out "$d/test" || exit 1
echo "[$(ts)] I7b solve: $(grep -o '"solve_rate": [0-9.]*' "$d/test/summary.json")"
echo "I7B DONE"
