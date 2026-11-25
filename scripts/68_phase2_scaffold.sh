#!/usr/bin/env bash
# Phase 2 wave 2a (results/m8_audit_design.md §30): 8 tries per training puzzle with three scaffolds, on one GPU.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/gpu_env.sh
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_OFFLINE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
G=${G:-3}; [[ $G != 2 ]] || exit 1; B=outputs/kk_phase2
idle() { [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $G | tr -d ' ')" ]] \
         && (( $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $G | tr -d ' ') >= 42000 )); }
for st in demo case both; do
  for i in 1 2 3; do
    [[ -f $B/scaffold_$st/summary.json ]] && break
    until idle; do sleep 60; done
    CUDA_VISIBLE_DEVICES=$G $(pin $G) .venv-dg/bin/python scripts/kk_sample.py --pool $B/sets/train_scaffold_$st.jsonl \
        --seed 123 --out $B/scaffold_$st >> $B/scaffold_$st.log 2>&1
  done
  echo "[$(date '+%F %T')] $st: $(cat $B/scaffold_$st/summary.json 2>/dev/null | tr -d '\n ' | cut -c1-200)"
done
.venv-dg/bin/python - <<'PY'
import json, sys
sys.path.insert(0, "scripts")
from kk_build import load_tries
from pathlib import Path
T = {k: v[0]["labels"] for k, v in load_tries(Path("outputs/kk_step2/gen"), "T").items()}
for st in ("demo", "case", "both"):
    p = Path(f"outputs/kk_phase2/scaffold_{st}/tries.jsonl")
    if not p.exists():
        print(st, "missing"); continue
    rows = [json.loads(l) for l in open(p)]
    ans = [r for r in rows if T.get(r["id"]) is not None]
    reach = sum(any(x == T[r["id"]] for x in r["labels"] if x is not None) for r in ans)
    print(f"{st}: teacher's answer reached at least once on {reach} of {len(ans)} puzzles ({100*reach/len(ans):.1f}%)")
PY
echo "SCAFFOLD DONE"
