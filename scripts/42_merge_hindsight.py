#!/usr/bin/env python
"""Merge sharded hindsight generations into the final dataset.

Handles overlapping shards. The run was rebalanced mid-flight — when shard 0 finished it freed a
teacher replica, so a third shard was started on the tail of shard 1's range and shard 1 was
stopped at its next checkpoint. Because shards checkpoint every N traces, shard 1 overshot the
handover point, so a band of indices exists in two shards.

Dedupe policy, in order:
  1. prefer verdict 'good' over 'exhausted'/'prompt_too_long' (exhausted = fell back to a trace
     the validator judged BAD, i.e. possibly a wrong solution);
  2. tie-break on fewer retries;
  3. tie-break on longer output.

Asserts every index 0..N-1 is present exactly once before writing.

Usage:
    .venv-infer/bin/python scripts/42_merge_hindsight.py \
        --shards data/defended/_shard0.json data/defended/_shard1.json data/defended/_shard3.json \
        --output data/defended/limo_hindsight_ds32b.json \
        --expect 800
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RANK = {"good": 0, "exhausted": 1, "prompt_too_long": 2, "unvalidated": 3}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--output", default="data/defended/limo_hindsight_ds32b.json")
    ap.add_argument("--expect", type=int, default=800)
    args = ap.parse_args()

    best: dict[int, tuple[tuple, dict, dict]] = {}
    dupes = 0
    for s in args.shards:
        sp = Path(s) if Path(s).is_absolute() else REPO / s
        mp = sp.with_suffix(".meta.json")
        recs = json.loads(sp.read_text(encoding="utf-8"))
        meta = json.loads(mp.read_text(encoding="utf-8"))
        if len(recs) != len(meta):
            print(f"ERROR: {sp.name} has {len(recs)} records but {len(meta)} meta entries")
            return 2
        for rec, m in zip(recs, meta):
            idx = m["index"]
            key = (RANK.get(m["verdict"], 9), m["retries"], -m["output_chars"])
            if idx in best:
                dupes += 1
                if key >= best[idx][0]:
                    continue
            best[idx] = (key, rec, m)
        print(f"  {sp.name:24s} {len(recs):4d} records, indices "
              f"{min(m['index'] for m in meta)}-{max(m['index'] for m in meta)}")

    missing = [i for i in range(args.expect) if i not in best]
    if missing:
        print(f"\nERROR: {len(missing)} indices missing, e.g. {missing[:10]}")
        return 1

    order = sorted(best)
    recs = [best[i][1] for i in order]
    meta = [best[i][2] for i in order]

    out = Path(args.output) if Path(args.output).is_absolute() else REPO / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(recs, ensure_ascii=False, indent=2), encoding="utf-8")
    out.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                             encoding="utf-8")

    from collections import Counter
    verdicts = Counter(m["verdict"] for m in meta)
    exhausted = [m["index"] for m in meta if m["verdict"] != "good"]
    chars = [m["output_chars"] for m in meta]
    print(f"\nmerged {len(recs)} traces -> {out.relative_to(REPO)}")
    print(f"  duplicate indices resolved : {dupes}")
    print(f"  verdicts                   : {dict(verdicts)}")
    print(f"  non-good indices           : {exhausted}")
    print(f"  output chars  median {sorted(chars)[len(chars)//2]}  mean {sum(chars)//len(chars)}")
    if verdicts.get("good", 0) < len(meta):
        n = len(meta) - verdicts.get("good", 0)
        print(f"\n  NOTE: {n} trace(s) failed validation and fell back to a BAD-judged attempt.")
        print( "        They may contain wrong solutions. Report M3 with and without them if the")
        print( "        count is material (>2%).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
