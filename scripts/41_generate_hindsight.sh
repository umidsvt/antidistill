#!/usr/bin/env bash
# Generate the hindsight (no-epistemic-verbalization) dataset with two sharded generators,
# one per teacher replica. Waits for both servers, runs both shards concurrently, merges.
#
# Prereq:  GPUS=0,1,2,3 PORT=8001 scripts/40_serve_teacher.sh
#          GPUS=4,5,6,7 PORT=8002 scripts/40_serve_teacher.sh
set -uo pipefail
cd "$(dirname "$0")/.."
REPO="$PWD"
PY="$REPO/.venv-infer/bin/python"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"

IN="data/raw/limo_v2.json"
OUT="data/defended/limo_hindsight_ds32b.json"
mkdir -p data/defended logs

echo "waiting for both teacher replicas ..."
for port in 8001 8002; do
  for i in $(seq 1 180); do
    curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1 && { echo "  :$port ready"; break; }
    sleep 10
    [[ $i -eq 180 ]] && { echo "  :$port TIMEOUT"; exit 1; }
  done
done

echo "launching 2 sharded generators ..."
$PY -m antidistill.defenses.hindsight --input "$IN" --output "data/defended/_shard0.json" \
    --base-url http://127.0.0.1:8001/v1 --shard 0 --num-shards 2 \
    > logs/hindsight_shard0.log 2>&1 &
P0=$!
$PY -m antidistill.defenses.hindsight --input "$IN" --output "data/defended/_shard1.json" \
    --base-url http://127.0.0.1:8002/v1 --shard 1 --num-shards 2 \
    > logs/hindsight_shard1.log 2>&1 &
P1=$!

echo "  shard0 pid $P0 -> logs/hindsight_shard0.log"
echo "  shard1 pid $P1 -> logs/hindsight_shard1.log"
wait $P0; R0=$?
wait $P1; R1=$?
echo "shard exit codes: $R0 $R1"
[[ $R0 -ne 0 || $R1 -ne 0 ]] && { echo "A SHARD FAILED — not merging."; exit 1; }

echo "merging shards -> $OUT"
$PY - "$OUT" <<'PYEOF'
import json, sys, pathlib
out = pathlib.Path(sys.argv[1])
recs, meta = [], []
for s in (0, 1):
    d = pathlib.Path(f"data/defended/_shard{s}.json")
    m = pathlib.Path(f"data/defended/_shard{s}.meta.json")
    recs += json.loads(d.read_text(encoding="utf-8"))
    meta += json.loads(m.read_text(encoding="utf-8"))
order = sorted(range(len(meta)), key=lambda i: meta[i]["index"])   # restore original order
recs = [recs[i] for i in order]; meta = [meta[i] for i in order]
out.write_text(json.dumps(recs, ensure_ascii=False, indent=2), encoding="utf-8")
out.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
good = sum(1 for m in meta if m["verdict"] == "good")
exh  = sum(1 for m in meta if m["verdict"] == "exhausted")
print(f"merged {len(recs)} traces -> {out}")
print(f"  validated GOOD : {good}")
print(f"  exhausted      : {exh}   <-- fell back to last attempt; may be incorrect")
PYEOF

echo "HINDSIGHT GENERATION COMPLETE"
