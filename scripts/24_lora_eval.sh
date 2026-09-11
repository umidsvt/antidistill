#!/usr/bin/env bash
# Evaluate the M7 LoRA arm with EXACTLY the protocol every other condition got:
# the epoch-15 checkpoint on all four benchmarks, greedy (t=0.0, k=1), TP=4.
#
#   aime (30) + amc (40) + aime25 (30) + math (500) = 600 problems per condition
#
# Two things this does that 12_multibench.sh / 13_m3_eval.sh did not have to:
#
# 1. MERGE FIRST. The harness hands --model_name_or_path straight to vLLM as a full model,
#    and a LoRA checkpoint holds only adapter tensors. Each merge is ~15GB, so we merge ->
#    evaluate -> delete rather than materialising six models at once. Merging (not vLLM's
#    --enable-lora) is deliberate: it keeps the eval path byte-identical to how M1/M2/M3/M6
#    were evaluated, so the comparison is about the checkpoints and not about the harness.
#
# 2. RE-MEASURE `base` ON THIS HOST. The committed numbers in results/eval_table.md were
#    produced on dspl1.cs.vt.edu (8x L40S, sm_89). This is 4x A40 (sm_86). Greedy decoding
#    is not reproducible across GPU architectures (results/m1_base.md), so the committed
#    rows are NOT a valid baseline for these runs. `base` here is the anchor that makes the
#    LoRA rows interpretable; everything else is compared against it, not against L40S.
#
# Usage:
#   nohup scripts/24_lora_eval.sh > logs/lora_eval.log 2>&1 &
#   SKIP_BASE=1 scripts/24_lora_eval.sh          # base already measured on this host
#   CONDITIONS="limo_lora32k" scripts/24_lora_eval.sh   # explicit subset
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
mkdir -p logs merged

STATUS="$REPO/logs/lora_eval_status.tsv"
[[ -f "$STATUS" ]] || printf 'condition\tbenchmark\tstatus\tfinished\telapsed_s\n' > "$STATUS"

BENCHES=(aime amc aime25 math)
read -r -a CONDS <<<"${CONDITIONS:-hindsight_lora16k limo_lora16k mixed50_lora16k hindsight_lora32k limo_lora32k mixed50_lora32k}"

# ---------------------------------------------------------------- wait for training
# Six rows in the queue status file means all six runs have reported, and no llamafactory
# process left means the last one has actually torn down (checkpoints flushed).
echo "[$(date +%H:%M:%S)] waiting for the training queue to finish ..."
while :; do
  done_runs=$(( $(wc -l < "$REPO/logs/lora_queue_status.tsv" 2>/dev/null || echo 1) - 1 ))
  running=$(pgrep -fc "llamafactory/launcher.py" 2>/dev/null || true)
  [[ "$done_runs" -ge 6 && "${running:-0}" -eq 0 ]] && break
  sleep 60
done
echo "[$(date +%H:%M:%S)] training complete; starting evaluation"
column -t "$REPO/logs/lora_queue_status.tsv"
echo

run_bench() {  # $1 = model path, $2 = label, $3 = benchmark
  local model="$1" label="$2" bench="$3" t0 rc el
  echo "=============================================================="
  echo "[$(date +%H:%M:%S)] START $label / $bench"
  t0=$(date +%s)
  bash scripts/10_eval.sh "$model" "$label" "$bench" 0.0 1 2>&1 \
    | grep -E "correct cnt|^Acc:|pass = " | tail -2
  rc=${PIPESTATUS[0]}
  el=$(( $(date +%s) - t0 ))
  [[ $rc -eq 0 ]] && st=ok || st="FAILED(rc=$rc)"
  printf '%s\t%s\t%s\t%s\t%d\n' "$label" "$bench" "$st" "$(date -Is)" "$el" >> "$STATUS"
  printf '[%s] DONE  %s / %s  (%s, %02d:%02d:%02d)\n\n' \
    "$(date +%H:%M:%S)" "$label" "$bench" "$st" $((el/3600)) $((el%3600/60)) $((el%60))
}

# ---------------------------------------------------------------- 1. base anchor
if [[ -z "${SKIP_BASE:-}" ]]; then
  echo "### base (Qwen/Qwen2.5-7B) — re-measured on THIS host, see header"
  for b in "${BENCHES[@]}"; do run_bench "Qwen/Qwen2.5-7B" base "$b"; done
fi

# ---------------------------------------------------------------- 2. the six LoRA runs
for c in "${CONDS[@]}"; do
  CK="saves/Qwen2.5-7B_${c}/checkpoint-1500"
  MERGED="$REPO/merged/${c}_ep15"

  if [[ ! -d "$CK" ]]; then
    echo "SKIP $c — missing $CK" >&2
    printf '%s\t-\tno-checkpoint\t%s\t0\n' "${c}_ep15" "$(date -Is)" >> "$STATUS"
    continue
  fi

  AVAIL=$(df -BG --output=avail "$REPO" | tail -1 | tr -dc '0-9')
  if (( AVAIL < 40 )); then
    echo "ERROR: only ${AVAIL}G free; a merge needs ~15G. Stopping." >&2
    exit 1
  fi

  echo "[$(date +%H:%M:%S)] merging $c ..."
  if ! bash scripts/23_merge_lora.sh "$CK" "$MERGED" > "logs/merge_${c}.log" 2>&1; then
    echo "MERGE FAILED for $c — see logs/merge_${c}.log" >&2
    printf '%s\t-\tmerge-failed\t%s\t0\n' "${c}_ep15" "$(date -Is)" >> "$STATUS"
    rm -rf "$MERGED"
    continue
  fi

  for b in "${BENCHES[@]}"; do run_bench "$MERGED" "${c}_ep15" "$b"; done

  # ~15GB per merged model; the adapter in saves/ is the durable artifact, this is derived.
  echo "[$(date +%H:%M:%S)] removing merged copy $MERGED"
  rm -rf "$MERGED"
done

echo "LORA EVAL COMPLETE"
column -t "$STATUS"
echo
"$REPO/.venv-infer/bin/python" scripts/30_collect_results.py --out results/eval_table_lora.md
