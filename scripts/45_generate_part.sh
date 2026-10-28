#!/usr/bin/env bash
# Generate the PART-defended dataset (results/m4_part_design.md): serve the rewriter, rewrite the
# 800 LIMO traces segment by segment, stop the server(s). GPUs 4-7, plus 2-3 via GPUS2; never 0-1.
#
#   PILOT=1 scripts/45_generate_part.sh      # the 20-trace pilot (design §7) -> limo_part_pilot.*
#   GPUS2=2,3 scripts/45_generate_part.sh    # add a TP=2 replica on GPUs 2-3 (caps set from each KV cache)
#   PART_ARGS="..."                          # extra arguments for antidistill.defenses.part
#   scripts/45_generate_part.sh              # all 800 -> data/defended/limo_part.{json,meta.json}
#   then: .venv-infer/bin/python scripts/36_part_audit.py
#
# The rewriter must already be on local disk (REWRITER_DIR). This script deliberately does not
# download it: an HF model id here would make vLLM fetch ~65 GB into HF_HOME, which on this host
# is the shared, nearly full /mnt/vault. Download it to the root disk first (design §8).
#
# Resumable: finished segments are cached in <output>.segments.jsonl, so rerunning after an
# interruption only rewrites what is missing.
set -uo pipefail
cd "$(dirname "$0")/.."
REPO="$PWD"
PY="$REPO/.venv-infer/bin/python"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"

REWRITER_DIR="${REWRITER_DIR:-$REPO/../models/Qwen2.5-32B-Instruct}"
SERVED_NAME="${SERVED_NAME:-Qwen/Qwen2.5-32B-Instruct}"
GPUS="${GPUS:-4,5,6,7}"
PORT="${PORT:-8003}"
# GEN_MODULE / INPUT / OUTPUT let other rewriting jobs reuse this wrapper (e.g. the re-voice attack:
# GEN_MODULE=antidistill.attacks.revoice INPUT=data/defended/limo_part.json
# OUTPUT=data/curated/revoice_part.json). Defaults reproduce the PART generation exactly.
GEN_MODULE="${GEN_MODULE:-antidistill.defenses.part}"
INPUT="${INPUT:-data/raw/limo_v2.json}"
OUTPUT="${OUTPUT:-data/defended/limo_part.json}"
if [[ "${PILOT:-0}" == "1" ]]; then
  OUT="${OUTPUT%.json}_pilot.json"; SEL="--pilot 20"
else
  OUT="$OUTPUT"; SEL=""
fi

# Optional second replica, e.g. GPUS2=2,3 (TP=2). Each replica stays inside one NUMA node: on this
# host collectives crossing the socket run ~6x slower (CLAUDE.md §7.3).
GPUS2="${GPUS2:-}"
PORT2="${PORT2:-8004}"
# Requests in flight per replica = 85% of its KV cache / the KV a request held on average in the
# pilot (128 requests filled 53% of 380,240 tokens: ~1,575 each, rounded up). Keeps every replica
# saturated without vLLM preempting requests, which would waste work. Override with CAPS=a,b.
TOKENS_PER_REQUEST="${TOKENS_PER_REQUEST:-1600}"

for g in "$GPUS" "$GPUS2"; do
  case ",$g," in *",0,"*|*",1,"*) echo "GPUs 0 and 1 are reserved -- refusing $g"; exit 1;; esac
done
if [[ ! -f "$REWRITER_DIR/config.json" ]]; then
  echo "rewriter not found at $REWRITER_DIR (no config.json)."
  echo "Download Qwen/Qwen2.5-32B-Instruct there first (~65 GB, root disk). See design doc §8."
  exit 1
fi

mkdir -p logs data/defended
PORTS=("$PORT"); [[ -n "$GPUS2" ]] && PORTS+=("$PORT2")
stop_servers() {
  for p in "${PORTS[@]}"; do
    f="logs/teacher_server_${p}.pid"; [[ -f "$f" ]] && kill "$(cat "$f")" 2>/dev/null; rm -f "$f"
  done
}
[[ "${KEEP_SERVER:-0}" == "1" ]] || trap stop_servers EXIT

serve() {   # serve <gpus> <port>
  echo "[$(date +%H:%M:%S)] serving $SERVED_NAME from $REWRITER_DIR on GPUs $1 :$2"
  MODEL="$REWRITER_DIR" SERVED_NAME="$SERVED_NAME" REASONING_PARSER=none MAX_MODEL_LEN=16384 \
    GPUS="$1" PORT="$2" scripts/40_serve_teacher.sh
}
serve "$GPUS" "$PORT" || { echo "server on $GPUS failed to start"; exit 1; }
[[ -n "$GPUS2" ]] && { serve "$GPUS2" "$PORT2" || { echo "server on $GPUS2 failed to start"; exit 1; }; }

URLS=(); CAPS_AUTO=()
for p in "${PORTS[@]}"; do
  URLS+=(--base-url "http://127.0.0.1:${p}/v1")
  kv=$(grep -oE "GPU KV cache size: [0-9,]+ tokens" "logs/teacher_server_${p}.log" | tail -1 | grep -oE "[0-9,]+" | tr -d ,)
  CAPS_AUTO+=($(( kv * 85 / 100 / TOKENS_PER_REQUEST )))
  echo "  :$p KV cache $kv tokens -> up to ${CAPS_AUTO[-1]} requests in flight"
done
CAPS="${CAPS:-$(IFS=,; echo "${CAPS_AUTO[*]}")}"

LOG="logs/$(basename "${OUT%.json}")_generate.log"
echo "[$(date +%H:%M:%S)] rewriting -> $OUT (log: $LOG), caps $CAPS"
"$PY" -m "$GEN_MODULE" --input "$INPUT" --output "$OUT" \
    "${URLS[@]}" --server-concurrency "$CAPS" --model "$SERVED_NAME" $SEL ${PART_ARGS:-} 2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
echo "[$(date +%H:%M:%S)] generator exit code $RC"
exit "$RC"
