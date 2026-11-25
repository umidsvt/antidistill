#!/usr/bin/env bash
# Phase 2, wave 3 (results/m8_audit_design.md §31, results/phase2_ideas.md I6-I8), seed 123. One lane per GPU
# (never GPU 2: kept for the user's other project); each job waits until its GPU is free of other processes.
#   setsid nohup scripts/69_phase2_wave3.sh > logs/phase2/wave3.log 2>&1 < /dev/null & disown
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/gpu_env.sh
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv-dg/bin/python; B=outputs/kk_phase2; SETS=$B/sets; STUDENT=unsloth/Llama-3.2-3B-Instruct; S=123
QWEN=Qwen/Qwen2.5-3B-Instruct; TEST=data/kk/step2_test.jsonl; NEED_MIB=${NEED_MIB:-42000}
G_I7=${G_I7:-3} G_I6=${G_I6:-4} G_I6C=${G_I6C:-6} G_I8=${G_I8:-7}
for g in $G_I7 $G_I6 $G_I6C $G_I8; do [[ $g != 2 ]] || { echo "GPU 2 is kept for the user's other project"; exit 1; }; done
mkdir -p "$B/students" logs/phase2
ts() { date '+%F %T'; }
idle() { [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$1" | tr -d ' ')" ]] \
         && (( $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" | tr -d ' ') >= NEED_MIB )); }
attempt() {   # <gpu> <marker> <log> <command...>: up to two tries, each after the GPU is free
  local g=$1 marker=$2 log=$3 i; shift 3
  [[ -f "$marker" ]] && return 0
  mkdir -p "$(dirname "$log")"
  for i in 1 2; do
    until idle "$g"; do sleep 120; done
    KK_MASTER_PORT=$((29780 + g)) CUDA_VISIBLE_DEVICES=$g $(pin "$g") "$@" >> "$log" 2>&1
    [[ -f "$marker" ]] && return 0
    echo "[$(ts)] FAILED (try $i) on GPU $g: $log :: $(grep -a -E 'Error' "$log" | tail -1 | cut -c1-200)"
    sleep 300
  done
  return 1
}
ev() {   # <gpu> <out dir> <puzzles> <adapter|none> [student]
  local g=$1 out=$2 puz=$3 ad=$4 st=${5:-$STUDENT} args=()
  [[ $ad == none ]] || args=(--adapter "$ad")
  attempt "$g" "$out/summary.json" "$out.log" $PY scripts/kk_eval.py --puzzles "$puz" --seed $S "${args[@]}" \
      --student "$st" --out "$out"
}
hid() { $PY -I scripts/kk_hidden_acc.py "$1" "$2/results.jsonl" 2>/dev/null; }

lane_i7() {   # infill curriculum with the hidden-person reward
  local g=$G_I7 prev="" k d init
  for k in 1 2 3 4; do
    d=$B/students/I7_st${k}_s$S; init=(); [[ -n $prev ]] && init=(--init-adapter "$prev")
    attempt $g "$d/train_meta.json" "$d/train.log" $PY scripts/kk_grpo.py --reward hidden --seed $S --out "$d" \
        --pool $SETS/train_mask_hid_st$k.jsonl --temperature 0.6 --lr 2e-5 --micro-batch 2 --epochs 1 \
        --save-every 25 "${init[@]}" || { echo "[$(ts)] I7 stage $k training failed"; return 1; }
    ev $g "$d/test_mask" $SETS/test_mask_truth_m1.jsonl "$d/adapter" || return 1
    ev $g "$d/test_mask3" $SETS/test_mask_truth_m3.jsonl "$d/adapter" || return 1
    ev $g "$d/test" $TEST "$d/adapter" || return 1
    local m1 m3; m1=$(hid $SETS/test_mask_truth_m1.jsonl "$d/test_mask"); m3=$(hid $SETS/test_mask_truth_m3.jsonl "$d/test_mask3")
    echo "[$(ts)] I7 stage $k: solve $(grep -o '"solve_rate": [0-9.]*' "$d/test/summary.json") | m=1 $m1 | m=3 $m3"
    if [[ $k == 1 ]]; then
      local h; h=$(echo "$m1" | $PY -c 'import json,sys; print(json.load(sys.stdin)["hidden right | readable"])')
      $PY -c "import sys; sys.exit(0 if float('$h') >= 0.75 else 1)" \
        || { echo "[$(ts)] I7 GATE: hidden right | readable $h < 0.75 at one hidden person; chain stopped (m8 §31)"; return 0; }
    fi
    prev=$d/adapter
  done
  echo "[$(ts)] I7 done"
}

lane_sft_rl() {   # <gpu> <name> <sft set>: B1's SFT recipe on the set, then B5's RL from that adapter
  local g=$1 name=$2 set=$3 d r
  d=$B/students/${name}_s$S; r=$B/students/${name}rl_s$S
  attempt $g "$d/train_meta.json" "$d/train.log" $PY scripts/kk_sft.py --train-set "$set" --out "$d" --seed $S \
      --student $STUDENT || { echo "[$(ts)] $name SFT failed"; return 1; }
  ev $g "$d/test" $TEST "$d/adapter" || return 1
  ev $g "$d/test_mask" $SETS/test_mask_truth_m1.jsonl "$d/adapter" || return 1
  echo "[$(ts)] $name SFT: solve $(grep -o '"solve_rate": [0-9.]*' "$d/test/summary.json") | m=1 $(hid $SETS/test_mask_truth_m1.jsonl "$d/test_mask")"
  attempt $g "$r/train_meta.json" "$r/train.log" $PY scripts/kk_grpo.py --reward exact --seed $S --out "$r" \
      --init-adapter "$d/adapter" --micro-batch 2 --epochs 1 --save-every 25 || { echo "[$(ts)] $name RL failed"; return 1; }
  ev $g "$r/test" $TEST "$r/adapter" || return 1
  echo "[$(ts)] ${name}rl: solve $(grep -o '"solve_rate": [0-9.]*' "$r/test/summary.json")"
}

lane_i8() {   # Qwen2.5-3B-Instruct, untrained
  local g=$G_I8 d=$B/students/Q0_s$S
  until grep -q "^/" logs/phase2/qwen_download.log 2>/dev/null; do
    grep -q -E "Error|Traceback" logs/phase2/qwen_download.log && { echo "[$(ts)] I8: download failed"; return 1; }
    sleep 60
  done
  ev $g "$d/test" $TEST none $QWEN || return 1
  ev $g "$d/test_mask" $SETS/test_mask_truth_m1.jsonl none $QWEN || return 1
  echo "[$(ts)] I8 Qwen untrained: solve $(grep -o '"solve_rate": [0-9.]*' "$d/test/summary.json") | m=1 $(hid $SETS/test_mask_truth_m1.jsonl "$d/test_mask")"
  attempt $g "$d/tries/summary.json" "$d/tries.log" $PY scripts/kk_sample.py --seed $S --student $QWEN --out "$d/tries" \
    || return 1
  $PY - "$d/tries/tries.jsonl" <<'PY'
import json, sys
sys.path.insert(0, "scripts")
from kk_build import load_tries
from pathlib import Path
T = {k: v[0]["labels"] for k, v in load_tries(Path("outputs/kk_step2/gen"), "T").items()}
rows = [json.loads(l) for l in open(sys.argv[1])]
ans = [r for r in rows if T.get(r["id"]) is not None]
reach = sum(any(x == T[r["id"]] for x in r["labels"] if x is not None) for r in ans)
print(f"I8 coverage: teacher's answer reached at least once on {reach} of {len(ans)} puzzles ({100*reach/len(ans):.1f}%)")
PY
}

echo "[$(ts)] wave 3 started: I7 GPU $G_I7, I6 GPU $G_I6, I6c GPU $G_I6C, I8 GPU $G_I8"
lane_i7 & lane_sft_rl $G_I6 I6_views $SETS/sft_views3.jsonl & lane_sft_rl $G_I6C I6c_rename $SETS/sft_rename3.jsonl &
lane_i8 &
wait
echo "[$(ts)] WAVE3 DONE"
