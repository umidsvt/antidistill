#!/usr/bin/env bash
# Seeds for the main numbers and the strongest answers-only attacker on cryptarithms (results/m8_audit_design.md §37).
# A job queue on WORKERS (one GPU per job; never GPU 2); each job waits until its GPU is free of other processes.
#   setsid nohup scripts/79_seeds_attackers.sh > logs/seeds_attackers.log 2>&1 < /dev/null & disown
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/gpu_env.sh
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv-dg/bin/python; NEED_MIB=42000; WORKERS=${WORKERS:-"0 1 3 4 5 6"}
for g in $WORKERS; do [[ $g != 2 ]] || { echo "GPU 2 is kept"; exit 1; }; done
RG=outputs/rg/crypt; RD=data/rg/crypt; KS=outputs/kk_phase2/students; KSETS=outputs/kk_phase2/sets; RF=outputs/kk_phase2/refine
LLAMA=unsloth/Llama-3.2-3B-Instruct; QWEN=Qwen/Qwen2.5-3B-Instruct
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
mp() { [[ $1 == B ]] && echo $LLAMA || echo $QWEN; }
rgrate() { $PY -c "import json; d=json.load(open('$1'))['by_level']; print({k: (v['solve_rate'], v['cell_accuracy'], v['mean_tokens']) for k, v in d.items()})"; }
rg_score() {   # <gpu> <model key> <seed> <adapter|none> <out>
  local ad=(); [[ $4 == none ]] || ad=(--adapter $4)
  attempt $1 $5/summary.json $5.log $PY scripts/rg_pilot.py --model $2 --max-tokens 8192 --seed $3 "${ad[@]}" \
      --puzzles ${RD}_test.jsonl --out $5 || return 1
  echo "[$(ts)] $5: $(rgrate $5/summary.json)"
}
rg_sft() {   # <gpu> <model key> <seed> <set> <dir>
  attempt $1 $5/train_meta.json $5/train.log $PY scripts/kk_sft.py --train-set $4 --out $5 --seed $3 --student $(mp $2) \
      --plain-prompt && rg_score $1 $2 $3 $5/adapter $5/test
}
rg_rl() {   # <gpu> <model key> <init dir> <dir>
  attempt $1 $4/train_meta.json $4/train.log $PY scripts/kk_grpo.py --reward exact --rg-teacher $RG/T_train/tries.jsonl \
      --pool ${RD}_train.jsonl --student $(mp $2) --init-adapter $3/adapter --seed 123 --out $4 --micro-batch 2 \
      --epochs 1 --save-every 25 && rg_score $1 $2 123 $4/adapter $4/test
}
kk_rate() { $PY -c "import json; print(json.load(open('$1'))['solve_rate'])"; }
votes() {   # <gpu> <adapter> <puzzles> <T> <seed> <out>
  attempt $1 $6/summary.json $6.log $PY scripts/kk_refine.py --adapter $2 --puzzles $3 --out $6 --tta $4 --sweeps 0 \
      --seed $5 || return 1
  echo "[$(ts)] $6: $($PY -c "import json; print(json.load(open('$6/summary.json'))['accuracy by step']['solve'])")"
}
chosen_T() {   # the rule of §37: best validation T for I6c + RL s123 (ties: smaller T)
  $PY -c "
import json
acc = {t: json.load(open(f'$RF/val_I6c_renamerl_s123_tta{t}/summary.json'))['accuracy by step']['solve'] for t in (1, 3, 5)}
print(max(sorted(acc), key=lambda t: (acc[t], -t)))"
}
job() {   # <gpu> <kind> <args...>
  local g=$1 kind=$2 m=${3:-} s=${4:-} d t
  case $kind in
    rg_u)  rg_sft $g $m $s $RG/train_sets/U.jsonl $RG/students/U_${m}_s$s ;;
    rg_b1) rg_sft $g $m $s $RG/train_sets/B1_answers.jsonl $RG/students/B1_${m}_s$s ;;
    rg_i0) rg_score $g $m $s none $RG/students/I0_${m}_s$s/test ;;
    rg_b5) rg_rl $g $m $RG/students/B1_${m}_s123 $RG/students/B5_${m}_s123 ;;
    rg_b1r) rg_sft $g $m 123 $RG/train_sets/B1_relabel3.jsonl $RG/students/B1r_${m}_s123 \
              && rg_rl $g $m $RG/students/B1r_${m}_s123 $RG/students/B5r_${m}_s123 ;;
    kk_i6c) s=$m; d=$KS/I6c_rename_s$s
        attempt $g $d/train_meta.json $d/train.log $PY scripts/kk_sft.py --train-set $KSETS/sft_rename3.jsonl --out $d \
            --seed $s --student $LLAMA || return 1
        attempt $g $d/test/summary.json $d/test.log $PY scripts/kk_eval.py --puzzles data/kk/step2_test.jsonl --seed $s \
            --adapter $d/adapter --student $LLAMA --out $d/test || return 1
        echo "[$(ts)] I6c s$s SFT: $(kk_rate $d/test/summary.json)"
        local r=$KS/I6c_renamerl_s$s
        attempt $g $r/train_meta.json $r/train.log $PY scripts/kk_grpo.py --reward exact --seed $s --out $r \
            --init-adapter $d/adapter --micro-batch 2 --epochs 1 --save-every 25 || return 1
        attempt $g $r/test/summary.json $r/test.log $PY scripts/kk_eval.py --puzzles data/kk/step2_test.jsonl --seed $s \
            --adapter $r/adapter --student $LLAMA --out $r/test || return 1
        echo "[$(ts)] I6c + RL s$s: $(kk_rate $r/test/summary.json)" ;;
    kk_val) for t in 1 3 5; do
              votes $g $KS/I6c_renamerl_s123/adapter data/kk/step2_val.jsonl $t 123 $RF/val_I6c_renamerl_s123_tta$t || return 1
              votes $g outputs/kk_phase1/students/B5_s123/adapter data/kk/step2_val.jsonl $t 123 $RF/val_B5_s123_tta$t || return 1
            done
            echo "[$(ts)] validation: chosen T = $(chosen_T)" ;;
    kk_test) s=$m
        until [[ -f $RF/val_I6c_renamerl_s123_tta5/summary.json && -f $KS/I6c_renamerl_s$s/adapter/adapter_config.json ]]; do
          grep -q -E "I6c.*s$s.*failed|validation failed" logs/seeds_attackers.log 2>/dev/null && return 1; sleep 120; done
        t=$(chosen_T)
        votes $g $KS/I6c_renamerl_s$s/adapter data/kk/step2_test.jsonl $t $s $RF/test_I6c_renamerl_s${s}_tta$t || return 1
        votes $g outputs/kk_phase1/students/B5_s$s/adapter data/kk/step2_test.jsonl $t $s $RF/test_B5_s${s}_tta$t ;;
  esac
}
Q=logs/seeds_attackers_jobs.txt; QN=logs/seeds_attackers_jobs.next
[[ -f $Q ]] || printf '%s\n' "rg_u B 124" "rg_u B 125" "rg_u Q 124" "rg_u Q 125" "kk_i6c 124" "kk_i6c 125" \
    "rg_b1r B" "rg_b1r Q" "rg_b5 B" "rg_b5 Q" "kk_val" "rg_b1 B 124" "rg_b1 B 125" "rg_b1 Q 124" "rg_b1 Q 125" \
    "rg_i0 B 124" "rg_i0 B 125" "rg_i0 Q 124" "rg_i0 Q 125" "kk_test 123" "kk_test 124" "kk_test 125" > $Q
[[ -f $QN ]] || echo 0 > $QN
next_job() { ( flock 9; local n; n=$(cat $QN); echo $((n + 1)) > $QN; sed -n "$((n + 1))p" $Q ) 9> logs/seeds_attackers_jobs.lock; }
worker() { local g=$1 line; while line=$(next_job) && [[ -n $line ]]; do job $g $line || echo "[$(ts)] JOB FAILED: $line"; done; }
echo "[$(ts)] started on GPUs $WORKERS"
for g in $WORKERS; do worker $g & done
wait
echo "[$(ts)] SEEDS+ATTACKERS DONE"
