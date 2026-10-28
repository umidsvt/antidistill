#!/usr/bin/env bash
# Setup 1 with seeds (results/m8_audit_design.md §8.6 step 3): LIMO long solutions with the small student E1
# selects. 13 jobs, each = train on a GPU pair, then evaluate on the pooled 600 (TP=1, MATH500 on one GPU, the
# three small sets on the other), the same protocol as E1:
#   U (LIMO) and Solo (7B Solo data): seeds 43, 44 (E1's runs are seed 42, LLaMA-Factory's default);
#   hindsight v2, PART, Style on PART: seeds 42, 43, 44.
#
# Student choice = the E1 rule fixed before E1 (m8 §7): a size is usable if LIMO raises its pooled score by at
# least 5.0 points over the untrained student; prefer 3B over 1.5B (fixed 2026-10-20, before E1's results). If
# neither is usable the queue stops: the 7B costs ~3x more and needs a decision.
#
# Worker pairs: 6,7 start when E1 has finished; 2,3 and 4,5 start when setup 2 has finished. Each worker takes
# the next free job (lock directories), so the pairs share the list. Finished jobs are skipped on restart.
#
#   setsid nohup scripts/30_setup1_queue.sh > logs/setup1_queue.log 2>&1 < /dev/null & disown
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs outputs/setup1/locks
ts() { date '+%F %T'; }
LOCAL_SAVES="$(cd .. && pwd)/saves_local"
BENCH=(math amc aime aime25)

pooled() {   # $1 condition label -> pooled accuracy in % (empty if any benchmark is missing)
  .venv-infer/bin/python - "$1" <<'PY'
import glob, json, sys
c, ok, n = sys.argv[1], 0, 0
for b in ["math", "amc", "aime", "aime25"]:
    f = glob.glob(f"outputs/{c}/**/{b}/*_t0.0_k1_*.jsonl", recursive=True)
    if not f:
        sys.exit(0)
    rows = [json.loads(l) for l in open(f[0])]
    ok += sum(bool(r["answers_correctness"][0]) for r in rows); n += len(rows)
print(f"{100 * ok / n:.2f}")
PY
}

choose_size() {   # writes outputs/setup1/size (3b, 1.5b or none) once
  [[ -f outputs/setup1/size ]] && return
  local z b l g pick=none
  for z in 3b 1.5b; do
    local tag; [[ $z == 3b ]] && tag=q3b || tag=q15b
    b="$(pooled ${tag}_base)"; l="$(pooled ${tag}_limo)"
    g="$(.venv-infer/bin/python -c "print(round(float('${l:-nan}') - float('${b:-nan}'), 2))")"
    echo "[$(ts)] E1 rule: ${z}: untrained ${b:-?}%, LIMO ${l:-?}%, gain ${g} points (usable if >= 5.0)"
    if [[ $pick == none ]] && .venv-infer/bin/python -c "import sys; sys.exit(0 if float('$g') >= 5.0 else 1)" 2>/dev/null; then pick=$z; fi
  done
  echo "$pick" > outputs/setup1/size
  echo "[$(ts)] E1 rule -> setup 1 student: $pick"
}

make_config() {   # $1 size, $2 cond, $3 dataset, $4 seed -> path of the config
  local src="configs/train/qwen2.5-$1_limo.yaml" dst="configs/train/qwen2.5-$1_$2_s$4.yaml"
  local z; [[ $1 == 3b ]] && z=3B || z=1.5B
  if [[ ! -f $dst ]]; then
    sed -e "s|^dataset: limo_v2|dataset: $3|" \
        -e "s|^output_dir: saves/Qwen2.5-${z}_limo|output_dir: saves/Qwen2.5-${z}_$2_s$4|" \
        -e "1s|^|# Setup 1 seed run (scripts/30_setup1_queue.sh): $2, seed $4; otherwise identical to qwen2.5-$1_limo.yaml.\n|" \
        "$src" > "$dst"
    printf 'seed: %s\n' "$4" >> "$dst"
  fi
  echo "$dst"
}

eval_student() {   # $1 gpuA, $2 gpuB, $3 model dir, $4 label
  local b
  ( for b in math; do
      [[ -n "$(find "outputs/$4" -path "*/$b/*_t0.0_k1_*.jsonl" 2>/dev/null | head -1)" ]] && continue
      GPUS=$1 bash scripts/10_eval.sh "$3" "$4" "$b" 0.0 1 > /dev/null 2>&1
    done ) & local p1=$!
  ( for b in amc aime aime25; do
      [[ -n "$(find "outputs/$4" -path "*/$b/*_t0.0_k1_*.jsonl" 2>/dev/null | head -1)" ]] && continue
      GPUS=$2 bash scripts/10_eval.sh "$3" "$4" "$b" 0.0 1 > /dev/null 2>&1
    done ) & local p2=$!
  wait $p1 $p2
  echo "[$(ts)] EVAL $4: pooled $(pooled "$4")%"
}

worker() {   # $1 gpuA, $2 gpuB
  local size job cond ds seed cfg out label z
  size="$(cat outputs/setup1/size)"
  [[ $size == 3b ]] && z=3B || z=1.5B
  while read -r cond ds seed; do
    job="${cond}_s${seed}"
    mkdir "outputs/setup1/locks/$job" 2> /dev/null || continue          # another worker has it
    cfg="$(make_config "$size" "$cond" "$ds" "$seed")"
    out="saves/Qwen2.5-${z}_${cond}_s${seed}"
    label="s1_${size//./}_${cond}_s${seed}"
    if ! ls "$out"/model*.safetensors > /dev/null 2>&1; then
      [[ -e "$out" ]] || { mkdir -p "$LOCAL_SAVES/$(basename "$out")" && ln -s "$LOCAL_SAVES/$(basename "$out")" "$out"; }
      echo "[$(ts)] TRAIN $job on $1,$2"
      GPUS="$1,$2" scripts/20_train.sh "$cfg" > "logs/train_$(basename "${cfg%.yaml}").wrapper.log" 2>&1
      if ! ls "$out"/model*.safetensors > /dev/null 2>&1; then
        echo "[$(ts)] FAILED $job: no model saved (logs/train_$(basename "${cfg%.yaml}").log)"; continue
      fi
    fi
    eval_student "$1" "$2" "$out" "$label"
  done < outputs/setup1/jobs.txt
  echo "[$(ts)] worker $1,$2: no jobs left"
}

# one queue instance at a time: stale locks from a stopped run are cleared (finished work is skipped anyway)
rm -rf outputs/setup1/locks && mkdir -p outputs/setup1/locks
cat > outputs/setup1/jobs.txt <<'EOF'
limo limo_v2 43
recon7b_solo recon7b_solo 43
hindsight_v2 limo_hindsight_chat 42
part limo_part 42
style_part style_part 42
limo limo_v2 44
recon7b_solo recon7b_solo 44
hindsight_v2 limo_hindsight_chat 43
part limo_part 43
style_part style_part 43
hindsight_v2 limo_hindsight_chat 44
part limo_part 44
style_part style_part 44
EOF

WATCH_WHILE="scripts/30_setup1_queue.sh" setsid nohup scripts/26_eval_exit_watchdog.sh \
  >> logs/eval_exit_watchdog.log 2>&1 < /dev/null &

# Start logic (revised 2026-10-20 21:45: GPUs 2-5 sat idle from 19:26 while the queue waited for all of E1).
# GPUs 2-5 first run E1's evaluations that are possible now (E1's own script skips finished ones); the size
# rule is applied as soon as its inputs exist; GPUs 6-7 join when E1 has finished.
e1_running() { pgrep -f "^bash scripts/28_e1_small_students.sh" > /dev/null; }
wait_model() {   # $1 model dir: wait until saved (or E1 has stopped)
  until ls "$1"/model*.safetensors > /dev/null 2>&1; do e1_running || return 1; sleep 120; done
}
echo "[$(ts)] setup 1 queue: E1's pending evaluations on GPUs 2-5"
( eval_student 2 3 saves/Qwen2.5-1.5B_limo q15b_limo
  eval_student 2 3 saves/Qwen2.5-1.5B_recon7b_solo q15b_solo7b ) & la=$!
( wait_model saves/Qwen2.5-3B_limo && eval_student 4 5 saves/Qwen2.5-3B_limo q3b_limo ) & lb=$!
wait $lb
g3="$(.venv-infer/bin/python -c "print(float('$(pooled q3b_limo)' or 'nan') - float('$(pooled q3b_base)' or 'nan'))" 2>/dev/null)"
if .venv-infer/bin/python -c "import sys; sys.exit(0 if float('$g3') >= 5.0 else 1)" 2>/dev/null; then
  choose_size                                    # 3B qualifies; the rule prefers it, 1.5B results not needed
else
  wait $la; choose_size
fi
[[ "$(cat outputs/setup1/size)" == none ]] && { echo "[$(ts)] no small student is usable: stopping for a decision"; exit 0; }

worker 4 5 & w1=$!
( wait $la 2>/dev/null; while kill -0 $la 2>/dev/null; do sleep 60; done; worker 2 3 ) & w2=$!
( until grep -q "E1 DONE" logs/e1_small_students_2.log 2>/dev/null || ! e1_running; do sleep 300; done
  echo "[$(ts)] E1 has ended: GPUs 6-7 join"; worker 6 7 ) & w3=$!
wait $w1 $w2 $w3
echo "[$(ts)] SETUP 1 QUEUE DONE"
