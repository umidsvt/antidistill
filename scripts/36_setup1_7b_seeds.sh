#!/usr/bin/env bash
# Setup 1, seeds 2-3 at 7B (m8 §8.6, decided 2026-10-22): U (LIMO) and Solo (7B Solo data), seeds 43 and 44 (the
# existing runs are seed 42). Each: train on GPUs 4-7 (global batch 8 = 4 x 1 x 2, as the original), then the pooled
# 600 at TP=4 like every 7B result. Waits until GPUs 4-7 are all free; retries a failed training twice.
#   setsid nohup scripts/36_setup1_7b_seeds.sh > logs/setup1_7b_seeds.log 2>&1 < /dev/null & disown
set -uo pipefail
cd "$(dirname "$0")/.."
G=4,5,6,7; LOCAL_SAVES="$(cd .. && pwd)/saves_local"
ts() { date '+%F %T'; }
all_free() { local g; for g in 4 5 6 7; do (( $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $g | tr -d ' ') >= 40000 )) || return 1; done; }
WATCH_WHILE="scripts/36_setup1_7b_seeds.sh" setsid nohup scripts/26_eval_exit_watchdog.sh >> logs/eval_exit_watchdog.log 2>&1 < /dev/null &
for job in "limo 43" "recon7b_solo 43" "limo 44" "recon7b_solo 44"; do
  read -r c s <<<"$job"
  cfg="configs/train/qwen2.5-7b_${c}_s$s.yaml"; out="saves/Qwen2.5-7B_${c}_s$s"; label="s1_7b_${c}_s$s"
  for i in 1 2 3; do
    ls "$out"/model*.safetensors > /dev/null 2>&1 && break
    until all_free; do sleep 600; done
    [[ -e "$out" ]] || { mkdir -p "$LOCAL_SAVES/$(basename "$out")" && ln -s "$LOCAL_SAVES/$(basename "$out")" "$out"; }
    echo "[$(ts)] TRAIN $c seed $s on $G (attempt $i)"
    GPUS=$G scripts/20_train.sh "$cfg" > "logs/train_$(basename "${cfg%.yaml}").wrapper.log" 2>&1
    ls "$out"/model*.safetensors > /dev/null 2>&1 || { echo "[$(ts)] FAILED training $c seed $s"; sleep 600; }
  done
  ls "$out"/model*.safetensors > /dev/null 2>&1 || continue
  for b in math amc aime aime25; do
    [[ -n "$(find "outputs/$label" -path "*/$b/*_t0.0_k1_*.jsonl" 2>/dev/null | head -1)" ]] && continue
    until all_free; do sleep 300; done
    GPUS=$G bash scripts/10_eval.sh "$out" "$label" "$b" 0.0 1 > /dev/null 2>&1
  done
  echo "[$(ts)] EVAL $label: $(.venv-infer/bin/python - "$label" <<'PY'
import glob, json, sys
c, ok, n = sys.argv[1], 0, 0
for b in ["math", "amc", "aime", "aime25"]:
    f = glob.glob(f"outputs/{c}/**/{b}/*_t0.0_k1_*.jsonl", recursive=True)
    if f:
        r = [json.loads(l) for l in open(f[0])]; ok += sum(bool(x["answers_correctness"][0]) for x in r); n += len(r)
print(f"pooled {100 * ok / max(n, 1):.1f}% of {n}")
PY
)"
done
echo "[$(ts)] SETUP 1 7B SEEDS DONE"
