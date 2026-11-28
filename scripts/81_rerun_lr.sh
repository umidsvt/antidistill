#!/usr/bin/env bash
# Rerun of the unstable recipes at lr 2e-4 (results/m8_audit_design.md §39). One lane per GPU; each job waits for a free GPU.
#   setsid nohup scripts/81_rerun_lr.sh > logs/rerun_lr.log 2>&1 < /dev/null & disown
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/gpu_env.sh
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv-dg/bin/python; NEED_MIB=42000; KS=outputs/kk_phase2/students; SETS=outputs/kk_phase2/sets; RF=outputs/kk_phase2/refine
RG=outputs/rg/crypt; LLAMA=unsloth/Llama-3.2-3B-Instruct; QWEN=Qwen/Qwen2.5-3B-Instruct
G123=${G123:-1} G124=${G124:-3} G125=${G125:-4} GQ=${GQ:-5}
ts() { date '+%F %T'; }
idle() { [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$1" | tr -d ' ')" ]] \
         && (( $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" | tr -d ' ') >= NEED_MIB )); }
attempt() {   # <gpu> <marker> <log> <command...>: up to two tries, each after the GPU is free
  local g=$1 marker=$2 log=$3 i; shift 3
  [[ -f "$marker" ]] && return 0
  mkdir -p "$(dirname "$log")"
  for i in 1 2; do
    until idle "$g"; do sleep 60; done
    KK_MASTER_PORT=$((29780 + g)) CUDA_VISIBLE_DEVICES=$g $(pin "$g") "$@" >> "$log" 2>&1
    [[ -f "$marker" ]] && return 0
    echo "[$(ts)] FAILED (try $i) on GPU $g: $log :: $(grep -a -E 'Error' "$log" | tail -1 | cut -c1-200)"
    sleep 120
  done
  return 1
}
rate() { $PY -c "import json; print(json.load(open('$1'))['solve_rate'])"; }
votes() {   # <gpu> <adapter> <puzzles> <T> <seed> <out>
  attempt $1 $6/summary.json $6.log $PY scripts/kk_refine.py --adapter $2 --puzzles $3 --out $6 --tta $4 --sweeps 0 --seed $5 \
    || return 1
  echo "[$(ts)] $6: $($PY -c "import json; print(json.load(open('$6/summary.json'))['accuracy by step']['solve'])")"
}
chosen() { $PY -c "
import json
acc = {t: json.load(open(f'$RF/val_I6c2_renamerl_s123_tta{t}/summary.json'))['accuracy by step']['solve'] for t in (1, 3, 5)}
print(max(sorted(acc), key=lambda t: (acc[t], -t)))"; }
lane_kk() {   # <gpu> <seed>
  local g=$1 s=$2 d=$KS/I6c2_rename_s$2 r=$KS/I6c2_renamerl_s$2 t
  attempt $g $d/train_meta.json $d/train.log $PY scripts/kk_sft.py --train-set $SETS/sft_rename3.jsonl --out $d --seed $s \
      --student $LLAMA --lr 2e-4 || return 1
  attempt $g $d/test/summary.json $d/test.log $PY scripts/kk_eval.py --puzzles data/kk/step2_test.jsonl --seed $s \
      --adapter $d/adapter --student $LLAMA --out $d/test || return 1
  echo "[$(ts)] I6c2 s$s SFT: $(rate $d/test/summary.json)"
  attempt $g $r/train_meta.json $r/train.log $PY scripts/kk_grpo.py --reward exact --seed $s --out $r --init-adapter $d/adapter \
      --micro-batch 2 --epochs 1 --save-every 25 || return 1
  attempt $g $r/test/summary.json $r/test.log $PY scripts/kk_eval.py --puzzles data/kk/step2_test.jsonl --seed $s \
      --adapter $r/adapter --student $LLAMA --out $r/test || return 1
  echo "[$(ts)] I6c2 + RL s$s: $(rate $r/test/summary.json)"
  if [[ $s == 123 ]]; then
    for t in 1 3 5; do votes $g $r/adapter data/kk/step2_val.jsonl $t 123 $RF/val_I6c2_renamerl_s123_tta$t || return 1; done
    echo "[$(ts)] validation (I6c2 + RL s123): chosen T = $(chosen)"
  fi
  until [[ -f $RF/val_I6c2_renamerl_s123_tta5/summary.json ]]; do grep -q "I6c2.*s123.*FAILED\|lane 123 failed" logs/rerun_lr.log && return 1; sleep 120; done
  t=$(chosen)
  votes $g $r/adapter data/kk/step2_test.jsonl $t $s $RF/test_I6c2_renamerl_s${s}_tta$t || return 1
  [[ $s == 125 ]] && votes $g outputs/kk_phase1/students/B5_s125/adapter data/kk/step2_test.jsonl $t 125 $RF/test_B5_s125_tta$t
  return 0
}
lane_q() {
  local g=$GQ d=$RG/students/B1r2_Q_s123 r=$RG/students/B5r2_Q_s123
  attempt $g $d/train_meta.json $d/train.log $PY scripts/kk_sft.py --train-set $RG/train_sets/B1_relabel3.jsonl --out $d \
      --seed 123 --student $QWEN --plain-prompt --lr 2e-4 || return 1
  attempt $g $d/test/summary.json $d/test.log $PY scripts/rg_pilot.py --model Q --max-tokens 8192 --seed 123 --adapter $d/adapter \
      --puzzles data/rg/crypt_test.jsonl --out $d/test || return 1
  echo "[$(ts)] B1r2 Q: $($PY -c "import json; d=json.load(open('$d/test/summary.json'))['by_level']['w2x4']; print(d['solve_rate'], d['cell_accuracy'], d['readable'])")"
  attempt $g $r/train_meta.json $r/train.log $PY scripts/kk_grpo.py --reward exact --rg-teacher $RG/T_train/tries.jsonl \
      --pool data/rg/crypt_train.jsonl --student $QWEN --init-adapter $d/adapter --seed 123 --out $r --micro-batch 2 --epochs 1 \
      --save-every 25 || return 1
  attempt $g $r/test/summary.json $r/test.log $PY scripts/rg_pilot.py --model Q --max-tokens 8192 --seed 123 --adapter $r/adapter \
      --puzzles data/rg/crypt_test.jsonl --out $r/test || return 1
  echo "[$(ts)] B5r2 Q: $($PY -c "import json; d=json.load(open('$r/test/summary.json'))['by_level']['w2x4']; print(d['solve_rate'], d['cell_accuracy'], d['readable'])")"
}
echo "[$(ts)] rerun started: I6c2 s123/124/125 on GPUs $G123/$G124/$G125, Qwen B1r2 on GPU $GQ"
{ lane_kk $G123 123 || echo "[$(ts)] lane 123 failed"; } & { lane_kk $G124 124 || echo "[$(ts)] lane 124 failed"; } &
{ lane_kk $G125 125 || echo "[$(ts)] lane 125 failed"; } & { lane_q || echo "[$(ts)] lane Q failed"; } &
wait
echo "[$(ts)] RERUN DONE"
