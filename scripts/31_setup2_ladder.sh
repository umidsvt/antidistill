#!/usr/bin/env bash
# Setup 2, attacker ladder (results/m8_audit_design.md §9.3):
#   level B: Solo from Llama-3.2-3B-Instruct, seeds 123-125 (GPUs 5, 6, 7);
#   level A: RL recipe v2 -- two 60-step pilots on seed 123 (GPUs 2, 4), the rule fixed in §9.3 picks a
#            variant, then that variant runs 655 steps for seeds 123 (GPU 2), 124 (GPU 4), 125 (GPU 5, after B).
# Every job first waits until its GPU has enough free memory (other users share these GPUs), and a failed
# job is retried twice after 10 minutes instead of being skipped.
#
#   setsid nohup scripts/31_setup2_ladder.sh > logs/setup2_ladder.log 2>&1 < /dev/null & disown
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=third_party/ext/distillation-game/src HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
       HF_HOME=/mnt/vault/huggingface_cache TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv-dg/bin/python
OUT=outputs/setup2
NEED_MIB=36000
ts() { date '+%F %T'; }
latest() { ls -dt "$OUT"/"$1"_[0-9]* 2>/dev/null | head -1; }

wait_free() {   # $1 gpu: wait until it has NEED_MIB free
  local f
  while true; do
    f="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" | tr -d ' ')"
    (( f >= NEED_MIB )) && return
    echo "[$(ts)] GPU $1 has ${f} MiB free; waiting"; sleep 600
  done
}
try3() {        # $1 gpu, $2 done-test command, rest: the job command
  local g=$1 done=$2 i; shift 2
  for i in 1 2 3; do
    eval "$done" && return 0
    wait_free "$g"
    CUDA_VISIBLE_DEVICES=$g DG_MASTER_PORT=$((29500 + g)) "$@"
    eval "$done" && return 0
    echo "[$(ts)] attempt $i failed on GPU $g: $*" | cut -c1-200; sleep 600
  done
  return 1
}
solo_b() {      # $1 gpu, $2 seed
  local stem="gsm8k_solo_l3i_s$2"
  echo "[$(ts)] START $stem on GPU $1"
  try3 "$1" "[[ -f \"\$(latest $stem)/RESULTS.md\" ]]" \
    $PY third_party/ext/distillation-game/scripts/run_pipeline.py --config "configs/setup2/$stem.yaml" \
    > "logs/setup2_$stem.log" 2>&1
  echo "[$(ts)] DONE  $stem: $(grep -a -A4 'Comparison Matrix' "$(latest $stem)/RESULTS.md" 2>/dev/null | tail -1)"
}
rl_v2() {       # $1 gpu, $2 seed, $3 name, $4 max steps, $5 test limit, rest: variant flags
  local g=$1 s=$2 name=$3 steps=$4 lim=$5; shift 5
  local d; d="$(latest "gsm8k_U_s$s")"
  local res="$OUT/rl_gsm8k_s${s}_${name}/result.json"
  echo "[$(ts)] START rl s$s $name on GPU $g"
  try3 "$g" "[[ -f $res ]]" $PY scripts/dg_rl_base.py --config "configs/setup2/gsm8k_U_s$s.yaml" \
    --teacher-train "$d/teacher/train_standard.json" --vllm-mem 0.5 --name "$name" --max-steps "$steps" \
    --test-limit "$lim" "$@" > "logs/setup2_rl_s${s}_${name}.log" 2>&1
  echo "[$(ts)] DONE  rl s$s $name: $(tail -n 3 "logs/setup2_rl_s${s}_${name}.log" | grep -a 'RL from' | cut -c1-120)"
}
VA=(--format-reward 0.2 --lr 1e-4 --temperature 0.8 --num-generations 8 --prompts-per-step 8)
VB=(--format-reward 0.2 --lr 1e-4 --temperature 0.8 --num-generations 16 --prompts-per-step 4)

echo "[$(ts)] setup 2 ladder: level B Solo on GPUs 5-7, level A pilots on GPUs 2 and 4"
( solo_b 5 123 ) & b1=$!
( solo_b 6 124 ) & b2=$!
( solo_b 7 125 ) & b3=$!
( rl_v2 2 123 pilot_v2a 60 300 "${VA[@]}" ) & pa=$!
( rl_v2 4 123 pilot_v2b 60 300 "${VB[@]}" ) & pb=$!
wait $pa $pb

# Pilot rule (m8 §9.3, fixed before the pilots): correctness reward averaged over steps 41-60 >= 5%.
choice="$($PY - <<'PY'
import json, pathlib
best, best_v = "none", 0.05
for v in ("a", "b"):
    f = pathlib.Path(f"outputs/setup2/rl_gsm8k_s123_pilot_v2{v}/train_log.json")
    if not f.exists():
        print(f"pilot v2{v}: no log", file=__import__("sys").stderr); continue
    rows = [r for r in json.loads(f.read_text()) if "rewards/reward_teacher_answer/mean" in r and 41 <= r.get("step", 0) <= 60]
    m = sum(r["rewards/reward_teacher_answer/mean"] for r in rows) / max(len(rows), 1)
    print(f"pilot v2{v}: correctness reward over steps 41-60 = {m:.4f} ({len(rows)} log points)", file=__import__("sys").stderr)
    if m >= best_v:
        best, best_v = v, m
print(best)
PY
)" 2>> logs/setup2_ladder.log
echo "$choice" > "$OUT/rl_v2_choice.txt"
echo "[$(ts)] pilot rule -> variant: $choice"
if [[ "$choice" == none ]]; then
  echo "[$(ts)] no RL variant learns: level A reported as 'RL from the student does not work here'"
  wait $b1 $b2 $b3
else
  [[ "$choice" == a ]] && V=("${VA[@]}") || V=("${VB[@]}")
  ( rl_v2 2 123 v2 655 0 "${V[@]}" ) & r1=$!
  ( rl_v2 4 124 v2 655 0 "${V[@]}" ) & r2=$!
  ( wait $b1 2>/dev/null; while kill -0 $b1 2>/dev/null; do sleep 60; done; rl_v2 5 125 v2 655 0 "${V[@]}" ) & r3=$!
  wait $b1 $b2 $b3 $r1 $r2 $r3
fi
echo "[$(ts)] SETUP 2 LADDER DONE"
