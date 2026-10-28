#!/usr/bin/env bash
# Setup 2, level A, RL recipe v3 (results/m8_audit_design.md §9.3): two 60-step pilots on seed 123 (GPUs 2, 4),
# the fixed rule picks a variant, which then runs 655 steps for seeds 123 (GPU 2), 124 (GPU 4) and 125 (first of
# GPUs 5-7 with enough free memory). Jobs wait for free memory and retry twice.
#   setsid nohup scripts/32_setup2_rl_v3.sh > logs/setup2_rl_v3.log 2>&1 < /dev/null & disown
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=third_party/ext/distillation-game/src HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
       HF_HOME=/mnt/vault/huggingface_cache TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv-dg/bin/python; OUT=outputs/setup2; NEED_MIB=36000
ts() { date '+%F %T'; }
latest() { ls -dt "$OUT"/"$1"_[0-9]* 2>/dev/null | head -1; }
free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" | tr -d ' '; }
wait_free() { while (( $(free_mib "$1") < NEED_MIB )); do echo "[$(ts)] GPU $1: $(free_mib "$1") MiB free; waiting"; sleep 600; done; }
rl() {   # $1 gpu, $2 seed, $3 name, $4 steps, $5 test limit, rest: flags
  local g=$1 s=$2 name=$3 steps=$4 lim=$5 i; shift 5
  local res="$OUT/rl_gsm8k_s${s}_${name}/result.json" d; d="$(latest "gsm8k_U_s$s")"
  echo "[$(ts)] START rl s$s $name on GPU $g"
  for i in 1 2 3; do
    [[ -f $res ]] && break
    wait_free "$g"
    CUDA_VISIBLE_DEVICES=$g DG_MASTER_PORT=$((29500 + g)) $PY scripts/dg_rl_base.py --config "configs/setup2/gsm8k_U_s$s.yaml" \
      --teacher-train "$d/teacher/train_standard.json" --vllm-mem 0.5 --name "$name" --max-steps "$steps" \
      --test-limit "$lim" "$@" > "logs/setup2_rl_s${s}_${name}.log" 2>&1
    [[ -f $res ]] || { echo "[$(ts)] attempt $i failed: rl s$s $name"; sleep 600; }
  done
  echo "[$(ts)] DONE  rl s$s $name: $(grep -a 'RL from' "logs/setup2_rl_s${s}_${name}.log" | tail -1 | cut -c1-110)"
}
COMMON=(--format-reward 0.2 --format-mode final --temperature 0.8 --num-generations 8 --prompts-per-step 8)
( rl 2 123 pilot_v3a 60 300 "${COMMON[@]}" --lr 5e-5 ) & pa=$!
( rl 4 123 pilot_v3b 60 300 "${COMMON[@]}" --lr 2e-5 ) & pb=$!
wait $pa $pb
choice="$($PY - <<'PY'
import json, pathlib, sys
best, best_v = "none", 0.05
for v in ("a", "b"):
    f = pathlib.Path(f"outputs/setup2/rl_gsm8k_s123_pilot_v3{v}/train_log.json")
    if not f.exists():
        print(f"pilot v3{v}: no log", file=sys.stderr); continue
    rows = [r for r in json.loads(f.read_text()) if "rewards/reward_teacher_answer/mean" in r and 41 <= r.get("step", 0) <= 60]
    m = sum(r["rewards/reward_teacher_answer/mean"] for r in rows) / max(len(rows), 1)
    fb = [r.get("rewards/reward_boxed/mean", 0) for r in rows]
    print(f"pilot v3{v}: correctness over steps 41-60 = {m:.4f}; format reward {sum(fb)/max(len(fb),1):.3f}", file=sys.stderr)
    if m >= best_v:
        best, best_v = v, m
print(best)
PY
)" 2>> logs/setup2_rl_v3.log
echo "$choice" > "$OUT/rl_v3_choice.txt"; echo "[$(ts)] pilot rule -> variant: $choice"
[[ "$choice" == none ]] && { echo "[$(ts)] v3 does not learn either; level A stays open"; exit 0; }
[[ "$choice" == a ]] && LR=5e-5 || LR=2e-5
( rl 2 123 v3 655 0 "${COMMON[@]}" --lr $LR ) & r1=$!
( rl 4 124 v3 655 0 "${COMMON[@]}" --lr $LR ) & r2=$!
( g=""; until [[ -n "$g" ]]; do for c in 5 6 7; do (( $(free_mib $c) >= NEED_MIB )) && { g=$c; break; }; done; [[ -n "$g" ]] || sleep 600; done
  rl "$g" 125 v3 655 0 "${COMMON[@]}" --lr $LR ) & r3=$!
wait $r1 $r2 $r3
echo "[$(ts)] RL v3 DONE"
